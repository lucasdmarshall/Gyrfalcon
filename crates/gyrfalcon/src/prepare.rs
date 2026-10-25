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
    static STEMMER: std::sync::LazyLock<frostem::Stemmer> =
        std::sync::LazyLock::new(|| frostem::Stemmer::new(frostem::Algorithm::English));
    tokens.retain(|token| STOPWORDS.binary_search(&token.text.as_str()).is_err());
    for (position, token) in tokens.iter_mut().enumerate() {
        token.position = position;
        if token.text.bytes().all(|b| b.is_ascii_lowercase()) {
            if let std::borrow::Cow::Owned(stem) = STEMMER.stem(&token.text) {
                token.text = stem;
            }
        }
    }
    tokens
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
