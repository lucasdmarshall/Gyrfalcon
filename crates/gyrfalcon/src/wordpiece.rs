//! A fast path for BERT-style WordPiece tokenizers (`BertNormalizer` with lowercasing,
//! `BertPreTokenizer`, `WordPiece`), producing the same ids as the `tokenizers` crate.
//!
//! Text is split on ASCII whitespace, which the BERT pipeline also treats as a word boundary.
//! A blob of printable ASCII is normalized, split on punctuation, and matched here without
//! allocating. A blob with any other byte, or with `[` (an added token such as `[UNK]` could
//! start there), goes through `tokenizers` unchanged, so non-ASCII normalization and special
//! tokens behave exactly as they do there.

use std::collections::HashMap;

use tokenizers::Tokenizer;

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
}

impl FastWordPiece {
    /// Builds the fast path when `tokenizer.json` is exactly the supported shape, else `None`.
    pub fn from_tokenizer_json(json: &serde_json::Value) -> Option<FastWordPiece> {
        let normalizer = &json["normalizer"];
        let supported_normalizer = normalizer["type"] == "BertNormalizer"
            && normalizer["clean_text"] == true
            && normalizer["handle_chinese_chars"] == true
            && normalizer["lowercase"] == true
            && (normalizer["strip_accents"].is_null() || normalizer["strip_accents"] == true);
        let model = &json["model"];
        if !supported_normalizer
            || json["pre_tokenizer"]["type"] != "BertPreTokenizer"
            || model["type"] != "WordPiece"
            || model["continuing_subword_prefix"] != "##"
        {
            return None;
        }
        // Added tokens are matched in raw text; the fast path routes any blob with `[` to the
        // fallback, which is only enough when every added token starts with `[`.
        let added = json["added_tokens"].as_array()?;
        if !added.iter().all(|t| t["content"].as_str().is_some_and(|c| c.starts_with('['))) {
            return None;
        }
        let vocab = model["vocab"].as_object()?;
        let unk = vocab.get(model["unk_token"].as_str()?)?.as_u64()? as u32;
        let mut words = HashMap::new();
        let mut continuations = HashMap::new();
        let mut max_piece = 0;
        for (token, id) in vocab {
            let id = id.as_u64()? as u32;
            match token.strip_prefix("##") {
                Some(rest) => {
                    max_piece = max_piece.max(rest.len());
                    continuations.insert(rest.into(), id);
                }
                None => {
                    max_piece = max_piece.max(token.len());
                    words.insert(token.as_str().into(), id);
                }
            }
        }
        Some(FastWordPiece {
            words,
            continuations,
            unk,
            max_input_chars: model["max_input_chars_per_word"].as_u64().unwrap_or(100) as usize,
            max_piece,
        })
    }

    /// Appends the ids of `text` (no special tokens) to `out`, stopping once `out` holds at
    /// least `limit` ids. The ids before `limit` are the same as a full encode's.
    pub fn encode(&self, text: &str, fallback: &Tokenizer, out: &mut Vec<u32>, limit: usize) -> tokenizers::Result<()> {
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
                out.extend_from_slice(fallback.encode(pending.as_str(), false)?.get_ids());
                pending.clear();
            }
            self.encode_ascii_blob(blob.as_bytes(), &mut word, out);
            if out.len() >= limit {
                return Ok(());
            }
        }
        if !pending.is_empty() {
            out.extend_from_slice(fallback.encode(pending.as_str(), false)?.get_ids());
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
