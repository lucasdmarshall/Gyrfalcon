//! HTML to paragraphs.
//!
//! A single pass over the markup, without a DOM: the title comes first, block elements end
//! paragraphs (a blank line, where the chunker prefers to cut), `<br>` ends a line, `<pre>`
//! keeps its whitespace, and elsewhere runs of whitespace collapse to one space. Scripts,
//! styles, navigation, forms' option lists, and embedded media are skipped. Character
//! references are decoded: numeric ones and the named ones common in prose.

/// Elements whose content is not text to index.
const SKIPPED: &[&str] =
    &["script", "style", "noscript", "template", "svg", "math", "nav", "select", "iframe", "canvas", "object", "head"];

/// Elements that end a paragraph where they open and close.
const BLOCKS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "body",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "tbody",
    "thead",
    "tfoot",
    "tr",
    "ul",
    "caption",
];

pub fn to_text(html: &str) -> String {
    let mut out = Writer::default();
    let mut title = String::new();
    let mut in_title = false;
    // The skipped element being passed over, and how deep it nests in itself.
    let mut skipping: Option<(String, usize)> = None;
    let mut pre = 0usize;
    let mut rest = html;
    while !rest.is_empty() {
        let Some(open) = rest.find('<') else {
            text(&mut out, &mut title, in_title, skipping.is_some(), pre > 0, rest);
            break;
        };
        text(&mut out, &mut title, in_title, skipping.is_some(), pre > 0, &rest[..open]);
        rest = &rest[open..];
        if let Some(after) = rest.strip_prefix("<!--") {
            rest = after.find("-->").map_or("", |end| &after[end + 3..]);
            continue;
        }
        if let Some(after) = rest.strip_prefix("<![CDATA[") {
            let end = after.find("]]>").unwrap_or(after.len());
            text(&mut out, &mut title, in_title, skipping.is_some(), pre > 0, &after[..end]);
            rest = after.get(end + 3..).unwrap_or("");
            continue;
        }
        let Some((tag, after)) = parse_tag(rest) else {
            // A lone `<` is text.
            text(&mut out, &mut title, in_title, skipping.is_some(), pre > 0, "<");
            rest = &rest[1..];
            continue;
        };
        rest = after;
        let name = tag.name.as_str();
        if let Some((skipped, depth)) = &mut skipping {
            if name == skipped {
                if tag.closing {
                    *depth -= 1;
                    if *depth == 0 {
                        skipping = None;
                    }
                } else if !tag.self_closing {
                    *depth += 1;
                }
            }
            // The title lives in the skipped head.
            if name == "title" {
                in_title = !tag.closing;
            }
            continue;
        }
        if name == "title" {
            in_title = !tag.closing;
            continue;
        }
        if SKIPPED.contains(&name) {
            if !tag.closing && !tag.self_closing {
                skipping = Some((tag.name.clone(), 1));
            }
            continue;
        }
        match name {
            "br" => out.line_break(),
            "td" | "th" if tag.closing => out.space(),
            "pre" => {
                out.paragraph_break();
                pre = if tag.closing { pre.saturating_sub(1) } else { pre + 1 };
            }
            _ if BLOCKS.contains(&name) => out.paragraph_break(),
            _ => {}
        }
    }
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    let body = out.finish();
    match (title.is_empty(), body.starts_with(&title)) {
        (false, false) => format!("{title}\n\n{body}"),
        _ => body,
    }
}

fn text(out: &mut Writer, title: &mut String, in_title: bool, skipping: bool, pre: bool, raw: &str) {
    if raw.is_empty() {
        return;
    }
    let decoded = decode_entities(raw);
    if in_title {
        title.push_str(&decoded);
    } else if !skipping {
        out.text(&decoded, pre);
    }
}

#[derive(Default)]
struct Writer {
    out: String,
    /// Whitespace seen since the last word, to emit as one space before the next.
    space: bool,
}

impl Writer {
    fn text(&mut self, text: &str, pre: bool) {
        if pre {
            self.out.push_str(text);
            self.space = false;
            return;
        }
        for c in text.chars() {
            if c.is_whitespace() {
                self.space = true;
            } else {
                if self.space && !self.out.is_empty() && !self.out.ends_with(['\n', ' ']) {
                    self.out.push(' ');
                }
                self.space = false;
                self.out.push(c);
            }
        }
    }

    fn space(&mut self) {
        self.space = true;
    }

    fn line_break(&mut self) {
        self.trim_end_spaces();
        if !self.out.is_empty() && !self.out.ends_with('\n') {
            self.out.push('\n');
        }
        self.space = false;
    }

    fn paragraph_break(&mut self) {
        self.trim_end_spaces();
        if !self.out.is_empty() && !self.out.ends_with("\n\n") {
            self.out.push_str(if self.out.ends_with('\n') { "\n" } else { "\n\n" });
        }
        self.space = false;
    }

    fn trim_end_spaces(&mut self) {
        let kept = self.out.trim_end_matches(' ').len();
        self.out.truncate(kept);
    }

    fn finish(self) -> String {
        self.out.trim().to_string()
    }
}

struct Tag {
    name: String,
    closing: bool,
    self_closing: bool,
}

/// The tag at the start of `s` (which starts with `<`) and the text after it. Quoted
/// attribute values may hold `>`. `None` when `<` does not open a tag.
fn parse_tag(s: &str) -> Option<(Tag, &str)> {
    let bytes = s.as_bytes();
    let mut at = 1;
    let closing = bytes.get(at) == Some(&b'/');
    if closing {
        at += 1;
    }
    let declaration = bytes.get(at) == Some(&b'!') || bytes.get(at) == Some(&b'?');
    if !declaration && !bytes.get(at).is_some_and(u8::is_ascii_alphabetic) {
        return None;
    }
    let name_start = at;
    while bytes
        .get(at)
        .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'-' || *b == b':' || *b == b'!' || *b == b'?')
    {
        at += 1;
    }
    let name = s[name_start..at].to_ascii_lowercase();
    let mut quote = None;
    while let Some(&b) = bytes.get(at) {
        match quote {
            Some(q) if b == q => quote = None,
            Some(_) => {}
            None if b == b'"' || b == b'\'' => quote = Some(b),
            None if b == b'>' => {
                let self_closing = at > 0 && bytes[at - 1] == b'/';
                return Some((Tag { name, closing, self_closing }, &s[at + 1..]));
            }
            None => {}
        }
        at += 1;
    }
    // An unterminated tag runs to the end of the input.
    Some((Tag { name, closing, self_closing: false }, ""))
}

fn decode_entities(s: &str) -> std::borrow::Cow<'_, str> {
    if !s.contains('&') {
        return s.into();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        rest = &rest[amp..];
        let end = rest.char_indices().take(12).find(|(_, c)| *c == ';').map(|(i, _)| i);
        let decoded = end.and_then(|end| entity(&rest[1..end]).map(|c| (c, end)));
        match decoded {
            Some((c, end)) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out.into()
}

fn entity(name: &str) -> Option<char> {
    if let Some(number) = name.strip_prefix('#') {
        let code = match number.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => number.parse().ok()?,
        };
        return char::from_u32(code).filter(|c| *c != '\0');
    }
    Some(match name {
        "amp" => '&',
        "lt" => '<',
        "gt" => '>',
        "quot" => '"',
        "apos" => '\'',
        "nbsp" => ' ',
        "ndash" => '–',
        "mdash" => '—',
        "hellip" => '…',
        "lsquo" => '‘',
        "rsquo" => '’',
        "ldquo" => '“',
        "rdquo" => '”',
        "laquo" => '«',
        "raquo" => '»',
        "bull" => '•',
        "middot" => '·',
        "copy" => '©',
        "reg" => '®',
        "trade" => '™',
        "deg" => '°',
        "times" => '×',
        "divide" => '÷',
        "plusmn" => '±',
        "minus" => '−',
        "le" => '≤',
        "ge" => '≥',
        "ne" => '≠',
        "rarr" => '→',
        "larr" => '←',
        "euro" => '€',
        "pound" => '£',
        "yen" => '¥',
        "cent" => '¢',
        "sect" => '§',
        "para" => '¶',
        "shy" => '\u{ad}',
        "zwj" => '\u{200d}',
        "zwnj" => '\u{200c}',
        "eacute" => 'é',
        "egrave" => 'è',
        "aacute" => 'á',
        "agrave" => 'à',
        "uuml" => 'ü',
        "ouml" => 'ö',
        "auml" => 'ä',
        "szlig" => 'ß',
        "ccedil" => 'ç',
        "ntilde" => 'ñ',
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_become_paragraphs_and_inline_text_flows() {
        let html = r#"<!DOCTYPE html><html><head><title>The  Page</title><style>p{color:red}</style>
            <script>var x = "<p>not text</p>";</script></head>
            <body><nav><a href="/">Home</a> <a href="/b">Blog</a></nav>
            <h1>Heading</h1><p>First <b>bold</b>
            paragraph &amp; more.</p><!-- a <p> comment --><div>Line one<br>line two</div>
            <ul><li>item a</li><li>item b</li></ul>
            <table><tr><td>cell 1</td><td>cell 2</td></tr></table>
            <pre>  keep
   this</pre><p title="a > b">Quoted &#x2014; &#8212; &mdash; &unknown; 5 < 6</p></body></html>"#;
        let text = to_text(html);
        assert_eq!(
            text,
            "The Page\n\nHeading\n\nFirst bold paragraph & more.\n\nLine one\nline two\n\nitem a\n\nitem b\n\ncell 1 cell 2\n\n  keep\n   this\n\nQuoted — — — &unknown; 5 < 6"
        );
    }

    #[test]
    fn nested_skipped_elements_and_a_title_repeated_as_heading() {
        let html = "<title>Guide</title><h1>Guide</h1><svg><svg><text>x</text></svg><text>y</text></svg><p>body</p>";
        assert_eq!(to_text(html), "Guide\n\nbody");
    }

    #[test]
    fn entities_decode_and_malformed_ones_stay() {
        assert_eq!(decode_entities("a &lt;b&gt; &#65;&#x42; & c &#0; &amp"), "a <b> AB & c &#0; &amp");
    }
}
