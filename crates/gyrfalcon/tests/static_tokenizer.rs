//! The static embedder's WordPiece fast path must produce the `tokenizers` ids exactly.
//! Runs only when `GYR_STATIC_MODEL_DIR` names a Model2Vec model with a BERT tokenizer whose
//! median token length is 7 (potion-retrieval-32M).

use std::path::Path;

fn corpus_samples() -> Vec<String> {
    let mut texts: Vec<String> = [
        "Memory safety in Rust; E0308!",
        "naïve café — Ünïcödé façade, ΣΊΣΥΦΟΣ İstanbul ß ﬁle",
        "中文字符 and 日本語 mixed with English",
        "[UNK] [CLS]x [SEP] [MASK]ed brackets [not special]",
        "tabs\tnew\nlines\r\nvertical\x0bform\x0cfeed\x00null\x7fdel",
        "\u{a0}non-breaking\u{2003}em\u{3000}ideographic spaces",
        "emoji 😀👍🏽 and zero\u{200b}width\u{200d}joiner",
        "a.b,c;d:e!f?g(h)i[j]k{l}m<n>o@p#q$r%s^t&u*v-w_x+y=z/|\\\\'\"`~",
        "",
        "   ",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    texts.push(format!("{} short", "y".repeat(150)));
    texts.push("supercalifragilisticexpialidocious antidisestablishmentarianism".into());
    for file in
        ["bench/fixtures/text-only/chunks.jsonl", ".data/beir/fiqa/corpus.jsonl", ".data/beir/scifact/corpus.jsonl"]
    {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(file);
        let Ok(text) = std::fs::read_to_string(path) else { continue };
        for line in text.lines().step_by(7).take(3000) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
                texts.push(value["text"].as_str().unwrap_or_default().to_string());
            }
        }
    }
    texts
}

#[test]
fn fast_wordpiece_matches_tokenizers() {
    let Ok(dir) = std::env::var("GYR_STATIC_MODEL_DIR") else {
        eprintln!("GYR_STATIC_MODEL_DIR is not set; skipping");
        return;
    };
    let dir = Path::new(&dir);
    let embedder = gyrfalcon::StaticEmbedder::from_dir(dir, "x").unwrap();
    let mut reference = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    reference.with_padding(None);
    reference.with_truncation(None).unwrap();
    let samples = corpus_samples();
    let mut checked = 0;
    for text in &samples {
        let max_chars = 512 * 7;
        let cut = match text.char_indices().nth(max_chars) {
            Some((at, _)) => &text[..at],
            None => text.as_str(),
        };
        let mut expected = reference.encode(cut, false).unwrap().get_ids().to_vec();
        expected.truncate(512);
        assert_eq!(embedder.token_ids(text).unwrap(), expected, "{:?}", &cut[..cut.len().min(200)]);
        checked += 1;
    }
    eprintln!("{checked} texts match");
}
