//! Files to text for indexing.
//!
//! `read_file` picks a reader by extension: PDF through PDFium and the layout layer, HTML
//! through its block structure, anything else as UTF-8 text.

pub mod html;
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
        "html" | "htm" | "xhtml" => {
            let bytes = std::fs::read(path)?;
            Ok(Extracted { text: html::to_text(&String::from_utf8_lossy(&bytes)), page_starts: Vec::new() })
        }
        _ => {
            let text = std::fs::read_to_string(path)?;
            Ok(Extracted { text, page_starts: Vec::new() })
        }
    }
}

/// Extensions a directory walk picks up; a file named explicitly is read whatever its name.
pub const EXTENSIONS: &[&str] = &["pdf", "html", "htm", "xhtml", "txt", "text", "md", "markdown", "rst", "adoc", "org"];

/// Whether a directory walk should read `path`.
pub fn is_supported(path: &Path) -> bool {
    path.extension().is_some_and(|e| EXTENSIONS.contains(&e.to_string_lossy().to_lowercase().as_str()))
}
