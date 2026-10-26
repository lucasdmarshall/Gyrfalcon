//! The product chunker: 2048-byte windows with 256 bytes of overlap.

pub const WINDOW: usize = 2048;
pub const OVERLAP: usize = 256;

/// A byte range into the original document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

pub fn chunk(text: &str) -> Vec<Span> {
    chunk_with(text, WINDOW, OVERLAP)
}

/// Walks byte windows of `window`. A cut prefers a blank line, then a newline, then `. ` or
/// `။`, first in the window's back half and then anywhere, else the last char boundary inside
/// the window. A break early in the window (a heading, a short paragraph) would otherwise cut
/// off a fragment and start the next window on most of this one's text. The next window
/// starts `overlap` bytes before the cut, on a char boundary. Every byte is covered and no
/// span splits a UTF-8 code point.
pub fn chunk_with(text: &str, window: usize, overlap: usize) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut start = 0;
    let mut previous_end = 0usize;
    while start < text.len() {
        if text.len() - start <= window {
            spans.push(Span { start, end: text.len() });
            break;
        }
        let limit = floor_boundary(text, start + window);
        let cut = if limit <= start {
            // A single code point wider than the window: take it whole.
            ceil_boundary(text, start + 1)
        } else {
            // A cut inside the overlap would repeat the previous span's tail.
            let min_offset = previous_end.saturating_sub(start);
            preferred_cut(&text[start..limit], min_offset).map(|offset| start + offset).unwrap_or(limit)
        };
        spans.push(Span { start, end: cut });
        previous_end = cut;
        let next = floor_boundary(text, cut.saturating_sub(overlap));
        start = if next > start { next } else { cut };
    }
    spans
}

fn preferred_cut(window: &str, min_offset: usize) -> Option<usize> {
    cut_after(window, min_offset.max(window.len() / 2)).or_else(|| cut_after(window, min_offset))
}

/// The last break of the strongest kind that lies past `min_offset`.
fn cut_after(window: &str, min_offset: usize) -> Option<usize> {
    let after = |cut: Option<usize>| cut.filter(|at| *at > min_offset && *at > 0);
    if let Some(cut) = after(window.rfind("\n\n").map(|at| at + 2)) {
        return Some(cut);
    }
    if let Some(cut) = after(window.rfind('\n').map(|at| at + 1)) {
        return Some(cut);
    }
    let period = after(window.rfind(". ").map(|at| at + 2));
    let myanmar = after(window.rfind('။').map(|at| at + '။'.len_utf8()));
    period.max(myanmar)
}

fn floor_boundary(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at -= 1;
    }
    at
}

fn ceil_boundary(text: &str, mut at: usize) -> usize {
    at = at.min(text.len());
    while !text.is_char_boundary(at) {
        at += 1;
    }
    at
}
