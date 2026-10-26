//! PDF text through PDFium.
//!
//! PDFium is the most accurate and the fastest of the extractors measured on 100 arXiv papers
//! (`bench/pdf`): 3.4 ms of CPU per page. Its library is loaded on first use from `GYR_PDFIUM`
//! (the library file or its directory), else beside the executable, else the system library
//! path. Prebuilt libraries: <https://github.com/bblanchon/pdfium-binaries>.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use pdfium_render::prelude::*;

use crate::layout::{clean, Extracted};
use crate::{Error, Result};

fn candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(value) = std::env::var_os("GYR_PDFIUM") {
        let path = PathBuf::from(value);
        paths.push(if path.is_dir() { Pdfium::pdfium_platform_library_name_at_path(&path) } else { path });
    }
    if let Some(dir) = std::env::current_exe().ok().and_then(|exe| exe.parent().map(Path::to_path_buf)) {
        paths.push(Pdfium::pdfium_platform_library_name_at_path(&dir));
    }
    paths
}

/// The process-wide PDFium binding.
pub fn pdfium() -> Result<&'static Pdfium> {
    static PDFIUM: OnceLock<std::result::Result<Pdfium, String>> = OnceLock::new();
    PDFIUM
        .get_or_init(|| {
            let mut tried = Vec::new();
            for path in candidates() {
                match Pdfium::bind_to_library(&path) {
                    Ok(bindings) => return Ok(Pdfium::new(bindings)),
                    // A library that exists but does not load (wrong platform or version) says why.
                    Err(err) if path.exists() => return Err(format!("{} does not load: {err:?}", path.display())),
                    Err(_) => tried.push(path.display().to_string()),
                }
            }
            Pdfium::bind_to_system_library().map(Pdfium::new).map_err(|_| {
                tried.push("the system library path".into());
                format!(
                    "library not found in {}; set GYR_PDFIUM to libpdfium (prebuilt: https://github.com/bblanchon/pdfium-binaries)",
                    tried.join(", ")
                )
            })
        })
        .as_ref()
        .map_err(|message| Error::PdfiumUnavailable(message.clone()))
}

/// Each page's text, in page order.
pub fn page_texts(path: &Path) -> Result<Vec<String>> {
    let failed = |err: PdfiumError| Error::Pdf { path: path.display().to_string(), message: err.to_string() };
    let document = pdfium()?.load_pdf_from_file(path, None).map_err(failed)?;
    document.pages().iter().map(|page| Ok(page.text().map_err(failed)?.all())).collect()
}

pub fn extract_file(path: &Path) -> Result<Extracted> {
    Ok(clean(&page_texts(path)?))
}
