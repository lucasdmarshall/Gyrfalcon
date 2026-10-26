//! `prepare`, the only normalizer, and the `gyr-text-v2` analyzer: the `gyr-text-v1` token
//! rules, then English stopword removal and Snowball English stemming of ASCII words.

use unicode_normalization::UnicodeNormalization;

/// Pieces longer than this many Unicode scalar values are dropped.
pub const MAX_TOKEN_CHARS: usize = 40;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    /// Index among surviving tokens, starting at 0.
    pub position: usize,
    /// Byte offsets into `embed_text`.
    pub offset_from: usize,
    pub offset_to: usize,
}

/// A chunk's analyzed text as an index needs it: each distinct term once, sorted, with its
/// count, and the token count. Two allocations instead of one `String` per token, so a buffer
/// or a merge holding many chunks costs about as much as their text.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TermCounts {
    /// The distinct terms, sorted, back to back.
    text: Box<str>,
    /// `(end of the term in text, count)` per term.
    entries: Box<[(u32, u32)]>,
    num_tokens: u32,
}

impl TermCounts {
    pub fn from_tokens(tokens: &[Token]) -> TermCounts {
        let mut terms: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        terms.sort_unstable();
        let mut text = String::with_capacity(terms.iter().map(|t| t.len()).sum());
        let mut entries = Vec::new();
        for run in terms.chunk_by(|a, b| a == b) {
            text.push_str(run[0]);
            entries.push((text.len() as u32, run.len() as u32));
        }
        // `text` may hold repeats' capacity; the box keeps only the distinct terms.
        TermCounts {
            text: text.into_boxed_str(),
            entries: entries.into_boxed_slice(),
            num_tokens: u32::try_from(tokens.len()).unwrap_or(u32::MAX),
        }
    }

    /// Token occurrences, repeats included.
    pub fn num_tokens(&self) -> u32 {
        self.num_tokens
    }

    /// Distinct terms.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `(term, count)` in term order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> + '_ {
        let mut start = 0usize;
        self.entries.iter().map(move |&(end, count)| {
            let term = &self.text[start..end as usize];
            start = end as usize;
            (term, count)
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedText {
    /// NFC, trimmed, original case. This is what the embedder receives.
    pub embed_text: String,
    /// `gyr-text-v2` analyzer output of `embed_text`.
    pub tokens: Vec<Token>,
}

/// Letters, combining marks, and numbers belong to words. Marks are kept so that scripts such
/// as Myanmar, whose vowel signs and medials are `\p{M}`, are not split inside a word.
#[inline]
fn is_word_char(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric();
    }
    use unicode_general_category::{get_general_category, GeneralCategory as G};
    matches!(
        get_general_category(c),
        G::UppercaseLetter
            | G::LowercaseLetter
            | G::TitlecaseLetter
            | G::ModifierLetter
            | G::OtherLetter
            | G::NonspacingMark
            | G::SpacingMark
            | G::EnclosingMark
            | G::DecimalNumber
            | G::LetterNumber
            | G::OtherNumber
    )
}

/// NFC-normalizes and trims `source` without changing case, then analyzes it.
pub fn prepare(source: &str) -> PreparedText {
    let embed_text = nfc_trim(source);
    let tokens = analyze(tokenize(&embed_text));
    PreparedText { embed_text, tokens }
}

/// `prepare` for a chunk with a contextual prefix: prefix, a blank line, then the body.
pub fn prepare_with_prefix(prefix: &str, body: &str) -> PreparedText {
    let embed_text = format!("{}\n\n{}", nfc_trim(prefix), nfc_trim(body));
    let tokens = analyze(tokenize(&embed_text));
    PreparedText { embed_text, tokens }
}

/// Lucene's English stop set, sorted.
const STOPWORDS: [&str; 33] = [
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of",
    "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
];

/// `gyr-text-v2` on `gyr-text-v1` tokens: drops English stopwords and Snowball-stems words of
/// ASCII letters only, so other scripts, numbers, and identifiers such as `e0308` keep their
/// v1 form. Positions are renumbered over the surviving tokens.
pub fn analyze(mut tokens: Vec<Token>) -> Vec<Token> {
    // Every stopword has at most 5 bytes, so most tokens skip the search.
    tokens.retain(|token| token.text.len() > 5 || STOPWORDS.binary_search(&token.text.as_str()).is_err());
    STEMS.with_borrow_mut(|stems| {
        for (position, token) in tokens.iter_mut().enumerate() {
            token.position = position;
            if token.text.bytes().all(|b| b.is_ascii_lowercase()) {
                stems.stem(&mut token.text);
            }
        }
    });
    tokens
}

/// Snowball English with a per-thread, direct-mapped cache. Word frequencies are Zipfian, so
/// almost every token hits; a slot holds a word of up to 16 bytes and its stem inline, so a hit
/// is one hash and one compare with no allocation. A miss or a collision runs the stemmer.
struct StemCache {
    stemmer: frostem::Stemmer,
    slots: Vec<Slot>,
}

/// Cache slots per thread (a power of two).
const STEM_SLOTS: usize = 1 << 16;
const SLOT_BYTES: usize = 16;

#[derive(Clone, Copy, Default)]
struct Slot {
    /// 0 marks an empty slot; words are never empty.
    word_len: u8,
    stem_len: u8,
    word: [u8; SLOT_BYTES],
    stem: [u8; SLOT_BYTES],
}

thread_local! {
    static STEMS: std::cell::RefCell<StemCache> = std::cell::RefCell::new(StemCache {
        stemmer: frostem::Stemmer::new(frostem::Algorithm::English),
        slots: vec![Slot::default(); STEM_SLOTS],
    });
}

impl StemCache {
    /// Replaces `word` (ASCII lowercase) with its stem, truncating in place when the stem is a
    /// prefix.
    fn stem(&mut self, word: &mut String) {
        let bytes = word.as_bytes();
        if bytes.len() > SLOT_BYTES {
            if let std::borrow::Cow::Owned(stem) = self.stemmer.stem(word) {
                replace_with_stem(word, &stem);
            }
            return;
        }
        let mut hasher = rustc_hash::FxHasher::default();
        std::hash::Hasher::write(&mut hasher, bytes);
        let slot = &mut self.slots[std::hash::Hasher::finish(&hasher) as usize & (STEM_SLOTS - 1)];
        if usize::from(slot.word_len) == bytes.len() && &slot.word[..bytes.len()] == bytes {
            let stem = &slot.stem[..usize::from(slot.stem_len)];
            // SAFETY: slots hold stems of ASCII words, which are ASCII.
            replace_with_stem(word, unsafe { std::str::from_utf8_unchecked(stem) });
            return;
        }
        let stem = self.stemmer.stem(word).into_owned();
        if stem.len() <= SLOT_BYTES {
            slot.word_len = bytes.len() as u8;
            slot.word[..bytes.len()].copy_from_slice(bytes);
            slot.stem_len = stem.len() as u8;
            slot.stem[..stem.len()].copy_from_slice(stem.as_bytes());
        }
        replace_with_stem(word, &stem);
    }
}

fn replace_with_stem(word: &mut String, stem: &str) {
    if word.len() == stem.len() {
        if word != stem {
            word.clear();
            word.push_str(stem);
        }
    } else if word.starts_with(stem) {
        word.truncate(stem.len());
    } else {
        word.clear();
        word.push_str(stem);
    }
}

fn nfc_trim(source: &str) -> String {
    let trimmed = source.trim();
    // Most text is already NFC (ASCII always is); skip the rewrite then.
    if unicode_normalization::is_nfc_quick(trimmed.chars()) == unicode_normalization::IsNormalized::Yes {
        return trimmed.to_string();
    }
    source.nfc().collect::<String>().trim().to_string()
}

/// The `gyr-text-v1` rules applied to already-normalized text: split on runs of characters
/// outside `\p{L}`, `\p{M}`, `\p{N}`, lowercase, and drop pieces over 40 scalar values.
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(text.len() / 6);
    let mut start: Option<usize> = None;
    let mut chars = 0usize;
    let mut ascii = true;
    let push = |from: usize, to: usize, chars: usize, ascii: bool, tokens: &mut Vec<Token>| {
        if chars > MAX_TOKEN_CHARS {
            return;
        }
        let piece = &text[from..to];
        let lowered = if ascii {
            let mut owned = piece.to_string();
            owned.make_ascii_lowercase();
            owned
        } else {
            piece.to_lowercase()
        };
        tokens.push(Token { text: lowered, position: tokens.len(), offset_from: from, offset_to: to });
    };
    for (at, c) in text.char_indices() {
        if is_word_char(c) {
            if start.is_none() {
                start = Some(at);
                chars = 0;
                ascii = true;
            }
            chars += 1;
            ascii &= c.is_ascii();
        } else if let Some(from) = start.take() {
            push(from, at, chars, ascii, &mut tokens);
        }
    }
    if let Some(from) = start {
        push(from, text.len(), chars, ascii, &mut tokens);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    /// The original regex definition of `gyr-text-v1`; `tokenize` must match it exactly.
    fn reference(text: &str) -> Vec<Token> {
        let separator = Regex::new(r"[^\p{L}\p{M}\p{N}]+").unwrap();
        let mut tokens = Vec::new();
        let mut start = 0;
        let push = |from: usize, to: usize, tokens: &mut Vec<Token>| {
            let piece = &text[from..to];
            if piece.is_empty() || piece.chars().count() > MAX_TOKEN_CHARS {
                return;
            }
            tokens.push(Token { text: piece.to_lowercase(), position: tokens.len(), offset_from: from, offset_to: to });
        };
        for found in separator.find_iter(text) {
            push(start, found.start(), &mut tokens);
            start = found.end();
        }
        push(start, text.len(), &mut tokens);
        tokens
    }

    #[test]
    fn cached_stems_equal_the_stemmer() {
        let stemmer = frostem::Stemmer::new(frostem::Algorithm::English);
        let words = ["running", "runs", "memory", "safety", "agreed", "flies", "dying", "news", "rust", "a", "abc"];
        for _ in 0..3 {
            for word in words {
                let mut cached = word.to_string();
                STEMS.with_borrow_mut(|stems| stems.stem(&mut cached));
                assert_eq!(cached, stemmer.stem(word), "{word}");
            }
        }
    }

    #[test]
    fn the_stem_cache_survives_collisions_and_long_words() {
        // More distinct words than slots, so slots collide and are overwritten.
        let stemmer = frostem::Stemmer::new(frostem::Algorithm::English);
        let suffixes = ["", "s", "ing", "ed", "ly", "ness", "ation", "ies", "y", "ational"];
        let mut state = 17u64;
        let words: Vec<String> = (0..200_000)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let len = 2 + (state % 18) as usize;
                let stem: String = (0..len).map(|k| (b'a' + ((state >> (k % 60)) % 26) as u8) as char).collect();
                format!("{stem}{}", suffixes[i % suffixes.len()])
            })
            .collect();
        for _ in 0..2 {
            for word in &words {
                let mut cached = word.clone();
                STEMS.with_borrow_mut(|stems| stems.stem(&mut cached));
                assert_eq!(cached, stemmer.stem(word), "{word}");
            }
        }
    }

    #[test]
    fn nfc_fast_path_matches_normalizing_everything() {
        for sample in ["  plain ascii  ", " Cafe\u{301} ", "\u{301}x ", " \u{301}", "\tÉ\n", "ﬁ İ", "  "] {
            let expected = sample.nfc().collect::<String>().trim().to_string();
            assert_eq!(nfc_trim(sample), expected, "{sample:?}");
        }
    }

    #[test]
    fn word_chars_match_the_regex_for_every_scalar_value() {
        let word = Regex::new(r"^[\p{L}\p{M}\p{N}]$").unwrap();
        let mut buf = [0u8; 4];
        let mismatches: Vec<u32> = (0..=0x10FFFFu32)
            .filter_map(char::from_u32)
            .filter(|c| is_word_char(*c) != word.is_match(c.encode_utf8(&mut buf)))
            .map(|c| c as u32)
            .collect();
        assert!(
            mismatches.is_empty(),
            "{} mismatches, first {:x?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(10)]
        );
    }

    #[test]
    fn tokenize_matches_the_regex_definition() {
        let samples = [
            "Memory safety in Rust မြန်မာ E0308",
            "  Ünïcödé ΣΊΣΥΦΟΣ İstanbul ß ﬁle \u{301}a",
            "😀 a😀b __x__ 3.14 x-y_z 中文字符 العربية हिन्दी",
            &format!("{} ok", "x".repeat(41)),
            &format!("{} ok", "é".repeat(40)),
            "",
            "   ",
        ];
        for sample in samples {
            assert_eq!(tokenize(sample), reference(sample), "{sample:?}");
        }
        // Pseudo-random strings over a mixed alphabet.
        let alphabet: Vec<char> = "aZ9 .,-_\n\tÉéß😀မြ်中\u{301}\u{200d}İΣ".chars().collect();
        let mut seed = 7u64;
        for _ in 0..2000 {
            let text: String = (0..64)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    alphabet[(seed % alphabet.len() as u64) as usize]
                })
                .collect();
            assert_eq!(tokenize(&text), reference(&text), "{text:?}");
        }
    }
}
