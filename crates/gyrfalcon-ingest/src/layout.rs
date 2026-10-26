//! Page text to paragraphs.
//!
//! A PDF extractor returns each page as lines in reading order. Retrieval wants the document's
//! prose without what repeats on every page, so this layer
//!
//! - drops control and private-use characters (glyphs of fonts with no Unicode mapping), and
//!   removes the marker PDFium leaves where it joined a word split by a soft hyphen;
//! - drops running headers and footers: a line among a page's first or last two lines that,
//!   with digits ignored, recurs on at least 30% of the pages (and on 3 or more), and bare page
//!   numbers there;
//! - rebuilds paragraphs: a line ends one when it is much shorter than the page's lines, or
//!   ends a sentence short of the full width, or when the next line opens a numbered or
//!   upper-case heading. Lines of a paragraph join with a space, across a page break too when
//!   the next page continues in lower case. A word hyphenated at a line end is rejoined when
//!   the whole word occurs elsewhere in the document, and keeps its hyphen otherwise.
//!
//! Paragraphs are separated by a blank line, where the chunker prefers to cut.

use std::collections::{HashMap, HashSet};

/// A document's text and the byte offset in it where each page starts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extracted {
    pub text: String,
    pub page_starts: Vec<usize>,
}

/// Lines among the first and last of a page that may be running headers or footers.
const EDGE_LINES: usize = 2;
/// Below this share of the page's typical line length a line ends its paragraph.
const SHORT_LINE: f32 = 0.6;
/// Below this share a line that ends a sentence ends its paragraph.
const SENTENCE_END_LINE: f32 = 0.85;

pub fn clean(pages: &[String]) -> Extracted {
    let pages: Vec<Vec<String>> = pages.iter().map(|page| lines_of(page)).collect();
    let furniture = running_lines(&pages);
    let vocabulary: HashSet<String> = pages
        .iter()
        .flatten()
        .flat_map(|line| line.split(|c: char| !c.is_alphabetic()))
        .filter(|word| word.len() > 2)
        .map(str::to_lowercase)
        .collect();

    let mut out = Extracted::default();
    // Whether the last paragraph may continue on the next page.
    let mut open = false;
    for (page_no, lines) in pages.iter().enumerate() {
        let lines: Vec<&str> = lines
            .iter()
            .enumerate()
            .filter(|(n, _)| !furniture.contains(&(page_no, *n)))
            .map(|(_, l)| l.as_str())
            .collect();
        let typical = typical_len(&lines);
        out.page_starts.push(out.text.len());
        for (n, line) in lines.iter().enumerate() {
            let continues = if n == 0 { open && starts_lowercase(line) } else { open };
            if continues {
                join_line(&mut out.text, line, &vocabulary);
            } else {
                if !out.text.is_empty() {
                    out.text.push_str("\n\n");
                }
                out.text.push_str(line);
            }
            if n == 0 {
                // The page starts where its first line landed, inside a paragraph that runs on.
                *out.page_starts.last_mut().unwrap() = out.text.len() - line.len();
            }
            open = match lines.get(n + 1) {
                Some(next) => !ends_paragraph(line, typical) && !is_heading(next, typical),
                None => !ends_paragraph(line, typical),
            };
        }
    }
    out
}

/// The page's non-empty lines, trimmed, with runs of spaces collapsed and unwanted characters
/// dropped.
fn lines_of(page: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    let mut space = false;
    for c in page.chars() {
        match c {
            '\n' | '\r' => {
                if !line.is_empty() {
                    lines.push(std::mem::take(&mut line));
                }
                space = false;
            }
            // PDFium's marker of a soft hyphen it removed, and a literal soft hyphen.
            '\u{2}' | '\u{ad}' => {}
            c if c.is_whitespace() || c.is_control() || is_private_use(c) => space = !line.is_empty(),
            c => {
                if space {
                    line.push(' ');
                    space = false;
                }
                line.push(c);
            }
        }
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

fn is_private_use(c: char) -> bool {
    matches!(c as u32, 0xE000..=0xF8FF | 0xF0000..=0xFFFFD | 0x100000..=0x10FFFD | 0xFFFE | 0xFFFF)
}

/// `(page, line)` of every running header, footer, and page number.
fn running_lines(pages: &[Vec<String>]) -> HashSet<(usize, usize)> {
    let edges = |lines: &[String]| -> Vec<usize> {
        let n = lines.len();
        let mut at: Vec<usize> = (0..n.min(EDGE_LINES)).chain(n.saturating_sub(EDGE_LINES)..n).collect();
        at.dedup();
        at
    };
    let mut pages_with: HashMap<String, usize> = HashMap::new();
    for lines in pages {
        let keys: HashSet<String> = edges(lines).into_iter().map(|n| furniture_key(&lines[n])).collect();
        for key in keys {
            *pages_with.entry(key).or_default() += 1;
        }
    }
    let threshold = (pages.len() * 3).div_ceil(10).max(3);
    let mut drop = HashSet::new();
    for (page_no, lines) in pages.iter().enumerate() {
        for n in edges(lines) {
            let recurs = pages_with[&furniture_key(&lines[n])] >= threshold;
            if recurs || (pages.len() > 1 && is_page_number(&lines[n])) {
                drop.insert((page_no, n));
            }
        }
    }
    drop
}

/// The line lower-cased with every digit run read as one `#`, so "Page 3" and "Page 12" match.
fn furniture_key(line: &str) -> String {
    let mut key = String::with_capacity(line.len());
    for c in line.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_digit() {
            if !key.ends_with('#') {
                key.push('#');
            }
        } else {
            key.push(c);
        }
    }
    key
}

/// `12`, `Page 12`, `12 of 30`, `12 / 30`, `- 12 -`, or a lower-case roman numeral.
fn is_page_number(line: &str) -> bool {
    let lower = line.to_lowercase();
    let body = lower.trim_matches(|c: char| c == '-' || c == '–' || c.is_whitespace());
    let body = body.strip_prefix("page").map_or(body, str::trim_start);
    let number = |s: &str| !s.is_empty() && s.len() <= 4 && s.bytes().all(|b| b.is_ascii_digit());
    if number(body) {
        return true;
    }
    for sep in [" of ", "/"] {
        if let Some((a, b)) = body.split_once(sep) {
            if number(a.trim()) && number(b.trim()) {
                return true;
            }
        }
    }
    line.len() <= 6 && line.chars().all(|c| "ivx".contains(c))
}

/// The 80th-percentile line length in chars: the full width of the page's text, robust to a
/// few long or short lines.
fn typical_len(lines: &[&str]) -> f32 {
    let mut lens: Vec<usize> = lines.iter().map(|l| l.chars().count()).collect();
    if lens.is_empty() {
        return 0.0;
    }
    lens.sort_unstable();
    lens[(lens.len() * 4 / 5).min(lens.len() - 1)] as f32
}

fn ends_paragraph(line: &str, typical: f32) -> bool {
    let len = line.chars().count() as f32;
    if len < typical * SHORT_LINE {
        return true;
    }
    let last = line.trim_end_matches(['"', '\'', ')', '”', '’']).chars().last();
    matches!(last, Some('.' | '!' | '?' | ':')) && len < typical * SENTENCE_END_LINE
}

/// `2 Method`, `3.1. Setup`, `IV. RESULTS`, or a short line in capitals such as `REFERENCES`.
fn is_heading(line: &str, typical: f32) -> bool {
    let mut words = line.split(' ');
    let first = words.next().unwrap_or_default();
    let next_starts_upper = words.next().and_then(|w| w.chars().next()).is_some_and(char::is_uppercase);
    let numbered = first
        .trim_end_matches('.')
        .split('.')
        .all(|part| !part.is_empty() && part.len() <= 2 && part.bytes().all(|b| b.is_ascii_digit()));
    let roman = first.ends_with('.') && first.len() > 1 && first[..first.len() - 1].chars().all(|c| "IVX".contains(c));
    if (numbered || roman) && next_starts_upper {
        return true;
    }
    let letters = line.chars().filter(|c| c.is_alphabetic()).count();
    letters >= 4
        && line.chars().filter(|c| c.is_alphabetic()).all(char::is_uppercase)
        && (line.chars().count() as f32) < typical * 0.7
}

fn starts_lowercase(line: &str) -> bool {
    line.chars().next().is_some_and(char::is_lowercase)
}

/// Appends `line` to the paragraph in `text`, rejoining a word hyphenated across the break.
fn join_line(text: &mut String, line: &str, vocabulary: &HashSet<String>) {
    if let Some(stem) = text.strip_suffix('-') {
        let left: String =
            stem.chars().rev().take_while(|c| c.is_alphabetic()).collect::<Vec<_>>().into_iter().rev().collect();
        if !left.is_empty() && starts_lowercase(line) {
            let right: String = line.chars().take_while(|c| c.is_alphabetic()).collect();
            if vocabulary.contains(&format!("{left}{right}").to_lowercase()) {
                text.pop();
            }
            text.push_str(line);
            return;
        }
    }
    text.push(' ');
    text.push_str(line);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|t| t.to_string()).collect()
    }

    const WIDE: &str = "a line of body text that runs the full width of the column here";

    #[test]
    fn running_headers_page_numbers_and_markers_go() {
        let words = ["one", "two", "three", "four"];
        let page = |n: usize| {
            let w = words[n - 1];
            format!("Journal of Things 2024 page {n}\r\nOpener {w} {WIDE}\r\n{WIDE}\r\n{WIDE}\r\nends sentence {w}.\r\n{n}\r\n")
        };
        let mut input: Vec<String> = (1..=4).map(page).collect();
        input[0] =
            format!("The Title\r\nOpener {WIDE}\r\n{WIDE}\r\n{WIDE}\r\nco\u{2}operation and \u{f8ec}glyphs.\r\n1\r\n");
        let out = clean(&input);
        assert!(!out.text.contains("Journal"), "{}", out.text);
        assert!(!out.text.contains("\n1"));
        assert!(out.text.contains("cooperation and glyphs."));
        assert!(out.text.starts_with("The Title"));
        assert_eq!(out.page_starts.len(), 4);
        assert!(out.text[out.page_starts[1]..].starts_with("Opener two"));
    }

    #[test]
    fn paragraphs_split_on_short_lines_and_headings_and_join_otherwise() {
        let page = format!("{WIDE}\n{WIDE}\nshort end.\n{WIDE}\n2.1 Method Details\n{WIDE}\nREFERENCES\n{WIDE}");
        let out = clean(&pages(&[&page]));
        let paragraphs: Vec<&str> = out.text.split("\n\n").collect();
        assert_eq!(paragraphs.len(), 6, "{paragraphs:#?}");
        assert_eq!(paragraphs[0], format!("{WIDE} {WIDE} short end."));
        assert_eq!(paragraphs[2], "2.1 Method Details");
    }

    #[test]
    fn hyphenated_words_rejoin_only_when_the_document_knows_them() {
        let page = format!("{WIDE} cov-\nerage and {WIDE} state-of-the-\nart models with coverage in {WIDE}\n{WIDE}");
        let out = clean(&pages(&[&page]));
        assert!(out.text.contains(" coverage and "), "{}", out.text);
        assert!(out.text.contains(" state-of-the-art models"), "{}", out.text);
    }

    #[test]
    fn a_paragraph_runs_across_a_page_break_in_lower_case() {
        let out = clean(&pages(&[&format!("{WIDE}\n{WIDE}"), &format!("continues here {WIDE}\n{WIDE}")]));
        assert_eq!(out.text.matches("\n\n").count(), 0);
        let out = clean(&pages(&[&format!("{WIDE}\n{WIDE}"), &format!("New start {WIDE}\n{WIDE}")]));
        assert_eq!(out.text.matches("\n\n").count(), 1);
        assert!(out.text[out.page_starts[1]..].starts_with("New start"));
    }

    #[test]
    fn page_numbers_are_recognized() {
        for yes in ["12", "Page 3", "3 of 10", "4 / 9", "- 7 -", "xii"] {
            assert!(is_page_number(yes), "{yes}");
        }
        for no in ["2024 results", "Table 3", "12345", "civic"] {
            assert!(!is_page_number(no), "{no}");
        }
    }
}
