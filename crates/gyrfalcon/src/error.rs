use std::fmt;
use std::path::PathBuf;

/// The manifest field an open request disagreed with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncompatibleField {
    ModelId,
    Dimension,
    Quantization,
    AnalyzerId,
}

impl IncompatibleField {
    pub fn as_str(self) -> &'static str {
        match self {
            IncompatibleField::ModelId => "model_id",
            IncompatibleField::Dimension => "dimension",
            IncompatibleField::Quantization => "quantization",
            IncompatibleField::AnalyzerId => "analyzer_id",
        }
    }
}

impl fmt::Display for IncompatibleField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("incompatible index: {field} is {manifest_value} in the manifest, {requested_value} requested")]
    IncompatibleIndex { field: IncompatibleField, manifest_value: String, requested_value: String },
    #[error("segment corrupt: {0}")]
    SegmentCorrupt(String),
    #[error("index not found: {}", .0.display())]
    IndexNotFound(PathBuf),
    #[error("another writer (pid {pid}) holds the collection")]
    WriterBusy { pid: u32 },
    #[error("empty query")]
    EmptyQuery,
    #[error("model unavailable: {0}")]
    ModelUnavailable(String),
    #[error("llm request failed: {0}")]
    Llm(String),
    #[error("index already exists: {}", .0.display())]
    AlreadyExists(PathBuf),
    #[error("lexical engine: {0}")]
    Lexical(String),
    #[error("vector engine: {0}")]
    Vector(String),
    #[error("embedder: {0}")]
    Embed(String),
    #[error("injected failure before the manifest rename")]
    InjectedFailure,
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    /// HTTP status the server maps this error to.
    pub fn http_status(&self) -> u16 {
        match self {
            Error::IncompatibleIndex { .. } | Error::WriterBusy { .. } | Error::AlreadyExists(_) => 409,
            Error::IndexNotFound(_) => 404,
            Error::EmptyQuery => 400,
            Error::Llm(_) => 502,
            _ => 500,
        }
    }

    /// Stable machine-readable name used in HTTP bodies.
    pub fn code(&self) -> &'static str {
        match self {
            Error::IncompatibleIndex { .. } => "incompatible_index",
            Error::SegmentCorrupt(_) => "segment_corrupt",
            Error::IndexNotFound(_) => "index_not_found",
            Error::WriterBusy { .. } => "writer_busy",
            Error::EmptyQuery => "empty_query",
            Error::ModelUnavailable(_) => "model_unavailable",
            Error::Llm(_) => "llm",
            Error::AlreadyExists(_) => "already_exists",
            _ => "internal",
        }
    }
}

pub(crate) fn mismatch<T>(field: IncompatibleField, manifest_value: &str, requested_value: &str) -> Result<T, Error> {
    Err(Error::IncompatibleIndex {
        field,
        manifest_value: manifest_value.to_string(),
        requested_value: requested_value.to_string(),
    })
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
