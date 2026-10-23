//! `prepare`, the only normalizer, and the `gyr-text-v1` token rules.

use std::sync::OnceLock;

use regex::Regex;
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
    /// `gyr-text-v1` analyzer output of `embed_text`.
    pub tokens: Vec<Token>,
}

fn separator() -> &'static Regex {
    static SEPARATOR: OnceLock<Regex> = OnceLock::new();
    // Letters, combining marks, and numbers stay. Marks are kept so that scripts such as
    // Myanmar, whose vowel signs and medials are `\p{M}`, are not split inside a word.
    SEPARATOR.get_or_init(|| Regex::new(r"[^\p{L}\p{M}\p{N}]+").unwrap())
}

/// NFC-normalizes and trims `source` without changing case, then tokenizes it.
pub fn prepare(source: &str) -> PreparedText {
    let embed_text = nfc_trim(source);
    let tokens = tokenize(&embed_text);
    PreparedText { embed_text, tokens }
}

/// `prepare` for a chunk with a contextual prefix: prefix, a blank line, then the body.
pub fn prepare_with_prefix(prefix: &str, body: &str) -> PreparedText {
    let embed_text = format!("{}\n\n{}", nfc_trim(prefix), nfc_trim(body));
    let tokens = tokenize(&embed_text);
    PreparedText { embed_text, tokens }
}

fn nfc_trim(source: &str) -> String {
    source.nfc().collect::<String>().trim().to_string()
}

/// The `gyr-text-v1` rules applied to already-normalized text.
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut start = 0;
    let push = |from: usize, to: usize, tokens: &mut Vec<Token>| {
        let piece = &text[from..to];
        if piece.is_empty() || piece.chars().count() > MAX_TOKEN_CHARS {
            return;
        }
        tokens.push(Token { text: piece.to_lowercase(), position: tokens.len(), offset_from: from, offset_to: to });
    };
    for found in separator().find_iter(text) {
        push(start, found.start(), &mut tokens);
        start = found.end();
    }
    push(start, text.len(), &mut tokens);
    tokens
}
