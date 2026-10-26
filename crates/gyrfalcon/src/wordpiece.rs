//! A fast path for BERT-style WordPiece tokenizers (`BertNormalizer` with lowercasing,
//! `BertPreTokenizer`, `WordPiece`), producing the same ids as the `tokenizers` crate.
//!
//! Text is split on ASCII whitespace, which the BERT pipeline also treats as a word boundary.
//! A blob of printable ASCII is normalized, split on punctuation, and matched here without
//! allocating. A blob with any other byte, or with `[` (an added token such as `[UNK]` could
//! start there), goes through `tokenizers` unchanged, so non-ASCII normalization and special
//! tokens behave exactly as they do there. The vocabulary is read from `tokenizer.json` with a
//! typed parse that borrows the file's strings, and the caller can load `tokenizers` lazily, on
//! the first text that needs it: a process that only embeds ASCII queries never builds it.

use std::borrow::Cow;
use std::collections::HashMap;

/// Parsed `model` section of a WordPiece `tokenizer.json`.
pub struct FastWordPiece {
    /// Tokens that start a word.
    words: HashMap<Box<str>, u32>,
    /// Continuation tokens, keyed without the `##` prefix.
    continuations: HashMap<Box<str>, u32>,
    unk: u32,
    max_input_chars: usize,
    /// Longest vocabulary entry in bytes; bounds the longest-match search.
    max_piece: usize,
    /// Median vocabulary entry length in chars, `##` included, as `model2vec` computes it.
    pub median_token_chars: usize,
    /// The id of `[UNK]`, which pooling skips.
    pub unk_literal: Option<u32>,
}

/// The parts of `tokenizer.json` the fast path reads. Strings borrow from the file's bytes.
#[derive(serde::Deserialize)]
struct TokenizerJson<'a> {
    #[serde(default)]
    normalizer: serde_json::Value,
    #[serde(default)]
    pre_tokenizer: serde_json::Value,
    #[serde(default, borrow)]
    added_tokens: Vec<AddedToken<'a>>,
    #[serde(borrow)]
    model: ModelJson<'a>,
}

#[derive(serde::Deserialize)]
struct AddedToken<'a> {
    #[serde(borrow)]
    content: Cow<'a, str>,
}

#[derive(serde::Deserialize)]
struct ModelJson<'a> {
    #[serde(rename = "type", default, borrow)]
    kind: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    unk_token: Option<Cow<'a, str>>,
    #[serde(default, borrow)]
    continuing_subword_prefix: Option<Cow<'a, str>>,
    #[serde(default)]
    max_input_chars_per_word: Option<u64>,
    #[serde(default, borrow)]
    vocab: Vocab<'a>,
}

/// A WordPiece vocabulary (`{token: id}`); any other shape reads as empty.
#[derive(Default)]
struct Vocab<'a>(Vec<(Cow<'a, str>, u32)>);

impl<'de: 'a, 'a> serde::Deserialize<'de> for Vocab<'a> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Vocab<'de>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a vocabulary")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(self, mut map: M) -> Result<Vocab<'de>, M::Error> {
                let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(entry) = map.next_entry::<Cow<'de, str>, u32>()? {
                    entries.push(entry);
                }
                Ok(Vocab(entries))
            }
            fn visit_seq<S: serde::de::SeqAccess<'de>>(self, mut seq: S) -> Result<Vocab<'de>, S::Error> {
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(Vocab(Vec::new()))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

impl FastWordPiece {
    /// Builds the fast path when `tokenizer.json` (its bytes) is exactly the supported shape,
    /// else `None`.
    pub fn from_tokenizer_json(bytes: &[u8]) -> Option<FastWordPiece> {
        let json: TokenizerJson = serde_json::from_slice(bytes).ok()?;
        let normalizer = &json.normalizer;
        let supported_normalizer = normalizer["type"] == "BertNormalizer"
            && normalizer["clean_text"] == true
            && normalizer["handle_chinese_chars"] == true
            && normalizer["lowercase"] == true
            && (normalizer["strip_accents"].is_null() || normalizer["strip_accents"] == true);
        let model = &json.model;
        if !supported_normalizer
            || json.pre_tokenizer["type"] != "BertPreTokenizer"
            || model.kind.as_deref() != Some("WordPiece")
            || model.continuing_subword_prefix.as_deref() != Some("##")
        {
            return None;
        }
        // Added tokens are matched in raw text; the fast path routes any blob with `[` to the
        // fallback, which is only enough when every added token starts with `[`.
        if !json.added_tokens.iter().all(|t| t.content.starts_with('[')) {
            return None;
        }
        let vocab = &model.vocab.0;
        let unk_token = model.unk_token.as_deref()?;
        let unk = vocab.iter().find(|(token, _)| token == unk_token)?.1;
        let mut words = HashMap::with_capacity(vocab.len());
        let mut continuations = HashMap::new();
        let mut max_piece = 0;
        let mut lengths = Vec::with_capacity(vocab.len());
        for (token, id) in vocab {
            lengths.push(token.chars().count());
            match token.strip_prefix("##") {
                Some(rest) => {
                    max_piece = max_piece.max(rest.len());
                    continuations.insert(rest.into(), *id);
                }
                None => {
                    max_piece = max_piece.max(token.len());
                    words.insert(token.as_ref().into(), *id);
                }
            }
        }
        // `int(np.median([len(token) for token in tokens]))`.
        lengths.sort_unstable();
        let median_token_chars = match lengths.len() {
            0 => 1,
            n if n % 2 == 1 => lengths[n / 2],
            n => (lengths[n / 2 - 1] + lengths[n / 2]) / 2,
        };
        let unk_literal = words.get("[UNK]").copied();
        Some(FastWordPiece {
            words,
            continuations,
            unk,
            max_input_chars: model.max_input_chars_per_word.unwrap_or(100) as usize,
            max_piece,
            median_token_chars,
            unk_literal,
        })
    }

    /// Appends the ids of `text` (no special tokens) to `out`, stopping once `out` holds at
    /// least `limit` ids. The ids before `limit` are the same as a full encode's.
    /// `fallback` appends the ids of text the fast path does not handle.
    pub fn encode<E>(
        &self,
        text: &str,
        mut fallback: impl FnMut(&str, &mut Vec<u32>) -> Result<(), E>,
        out: &mut Vec<u32>,
        limit: usize,
    ) -> Result<(), E> {
        let mut pending = String::new();
        let mut word = Vec::with_capacity(64);
        for blob in text.split([' ', '\t', '\n', '\r']).filter(|b| !b.is_empty()) {
            let fast = blob.bytes().all(|b| b.is_ascii() && b != b'[');
            if !fast {
                if !pending.is_empty() {
                    pending.push(' ');
                }
                pending.push_str(blob);
                continue;
            }
            if !pending.is_empty() {
                fallback(&pending, out)?;
                pending.clear();
            }
            self.encode_ascii_blob(blob.as_bytes(), &mut word, out);
            if out.len() >= limit {
                return Ok(());
            }
        }
        if !pending.is_empty() {
            fallback(&pending, out)?;
        }
        Ok(())
    }

    /// One blob of ASCII without whitespace: drop control bytes, lowercase, isolate
    /// punctuation, and WordPiece each word.
    fn encode_ascii_blob(&self, blob: &[u8], word: &mut Vec<u8>, out: &mut Vec<u32>) {
        word.clear();
        for &byte in blob {
            if byte.is_ascii_control() {
                continue;
            }
            if byte.is_ascii_punctuation() {
                self.encode_word(word, out);
                word.clear();
                self.encode_word(&[byte], out);
            } else {
                word.push(byte.to_ascii_lowercase());
            }
        }
        self.encode_word(word, out);
    }

    /// Greedy longest-match-first WordPiece of one lowercase ASCII word.
    fn encode_word(&self, word: &[u8], out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        if word.len() > self.max_input_chars {
            out.push(self.unk);
            return;
        }
        // SAFETY: the bytes are ASCII.
        let word = unsafe { std::str::from_utf8_unchecked(word) };
        let start_len = out.len();
        let mut start = 0;
        while start < word.len() {
            let map = if start == 0 { &self.words } else { &self.continuations };
            let mut end = word.len().min(start + self.max_piece);
            let mut found = None;
            while end > start {
                if let Some(id) = map.get(&word[start..end]) {
                    found = Some(*id);
                    break;
                }
                end -= 1;
            }
            match found {
                Some(id) => {
                    out.push(id);
                    start = end;
                }
                None => {
                    out.truncate(start_len);
                    out.push(self.unk);
                    return;
                }
            }
        }
    }
}
