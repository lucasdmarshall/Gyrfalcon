//! PDF text extraction benchmark: one extractor over a directory of PDFs, single-threaded.
//!
//!     pdf-bench EXTRACTOR PDF_DIR OUT_DIR
//!
//! Writes `OUT_DIR/EXTRACTOR/{name}.txt` per PDF and prints one JSON line per file with the
//! wall time, the text length, and the error, if any. A panic inside an extractor counts as an
//! error for that file. `pages` as the extractor prints page counts (pdf_oxide) instead.
use std::path::Path;
use std::time::Instant;

fn extract(extractor: &str, path: &Path, bytes: &[u8]) -> Result<String, String> {
    match extractor {
        "pdf_oxide" => {
            let doc = pdf_oxide::PdfDocument::open(path).map_err(|e| e.to_string())?;
            doc.extract_all_text().map_err(|e| e.to_string())
        }
        "unpdf" => {
            let doc = unpdf::parse_bytes(bytes).map_err(|e| e.to_string())?;
            unpdf::render::to_text(&doc, &unpdf::render::RenderOptions::default()).map_err(|e| e.to_string())
        }
        "pdf-extract" => pdf_extract::extract_text_from_mem(bytes).map_err(|e| e.to_string()),
        "oxidize-pdf" => {
            let doc = oxidize_pdf::parser::PdfDocument::open(path).map_err(|e| e.to_string())?;
            let pages = doc.extract_text().map_err(|e| e.to_string())?;
            Ok(pages.into_iter().map(|p| p.text).collect::<Vec<_>>().join("\n"))
        }
        "lopdf" => {
            let doc = lopdf::Document::load_mem(bytes).map_err(|e| e.to_string())?;
            let pages: Vec<u32> = doc.get_pages().keys().copied().collect();
            doc.extract_text(&pages).map_err(|e| e.to_string())
        }
        "pages" => {
            let doc = pdf_oxide::PdfDocument::open(path).map_err(|e| e.to_string())?;
            Ok(doc.page_count().map_err(|e| e.to_string())?.to_string())
        }
        other => Err(format!("unknown extractor {other}")),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (extractor, pdf_dir, out_dir) = (&args[1], Path::new(&args[2]), Path::new(&args[3]));
    let out = out_dir.join(extractor);
    std::fs::create_dir_all(&out).unwrap();
    let mut files: Vec<_> = std::fs::read_dir(pdf_dir)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "pdf"))
        .collect();
    files.sort();
    std::panic::set_hook(Box::new(|_| {}));
    for path in files {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(&path).unwrap();
        let started = Instant::now();
        let result = std::panic::catch_unwind(|| extract(extractor, &path, &bytes))
            .unwrap_or_else(|_| Err("panicked".to_string()));
        let ms = started.elapsed().as_secs_f64() * 1e3;
        let (chars, error) = match &result {
            Ok(text) => {
                std::fs::write(out.join(format!("{name}.txt")), text).unwrap();
                (text.chars().count(), None)
            }
            Err(err) => (0, Some(err.chars().take(200).collect::<String>())),
        };
        let line = serde_json::json!({"extractor": extractor, "file": name, "bytes": bytes.len(), "ms": ms, "chars": chars, "error": error});
        println!("{line}");
    }
}
