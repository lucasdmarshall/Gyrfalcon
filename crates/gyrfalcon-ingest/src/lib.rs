//! Files to text for indexing.
//!
//! `read_file` picks a reader by extension: PDF through PDFium and the layout layer, anything
//! else as UTF-8 text.

pub mod layout;
pub mod pdf;

use std::path::Path;

pub use layout::{clean, Extracted};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDFium unavailable: {0}")]
    PdfiumUnavailable(String),
    #[error("pdf {path}: {message}")]
    Pdf { path: String, message: String },
}

pub type Result<T> = std::result::Result<T, Error>;

/// The text of the file at `path`.
pub fn read_file(path: &Path) -> Result<Extracted> {
    let extension = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match extension.as_str() {
        "pdf" => pdf::extract_file(path),
        _ => {
            let text = std::fs::read_to_string(path)?;
            Ok(Extracted { text, page_starts: Vec::new() })
        }
    }
}
