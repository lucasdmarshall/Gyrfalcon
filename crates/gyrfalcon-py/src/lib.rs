//! `gyrfalcon._native`: a collection handle over `gyrfalcon::Index`. The Python package in
//! `python/gyrfalcon` finds the models and native libraries and wraps this module; every
//! engine call here releases the GIL.

use std::path::PathBuf;
use std::sync::RwLock;

use gyrfalcon::{Document, Filter, Index, RetrieveOptions, Tier};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

fn engine_error(err: gyrfalcon::Error) -> PyErr {
    match err {
        gyrfalcon::Error::IndexNotFound(_) => PyFileNotFoundError::new_err(err.to_string()),
        gyrfalcon::Error::EmptyQuery => PyValueError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

fn ingest_error(err: gyrfalcon_ingest::Error) -> PyErr {
    match err {
        gyrfalcon_ingest::Error::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
            PyFileNotFoundError::new_err(io.to_string())
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn parse_tier(name: &str) -> PyResult<Tier> {
    Tier::parse(name).ok_or_else(|| PyValueError::new_err(format!("unknown tier {name:?}; use \"lite\" or \"full\"")))
}

type Metadata = Vec<(String, String)>;

/// One retrieved chunk.
#[pyclass(frozen, module = "gyrfalcon")]
struct Hit {
    #[pyo3(get)]
    chunk_id: u64,
    #[pyo3(get)]
    doc_id: String,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    text: String,
    /// Byte offsets of the chunk in its document's text.
    #[pyo3(get)]
    start: u64,
    #[pyo3(get)]
    end: u64,
    pairs: Metadata,
}

#[pymethods]
impl Hit {
    /// The document's metadata: each key's value, or a list when the key repeats.
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        for (key, value) in &self.pairs {
            match dict.get_item(key)? {
                None => dict.set_item(key, value)?,
                Some(existing) => match existing.cast::<pyo3::types::PyList>() {
                    Ok(list) => list.append(value)?,
                    Err(_) => dict.set_item(
                        key,
                        pyo3::types::PyList::new(py, [existing, value.into_pyobject(py)?.into_any()])?,
                    )?,
                },
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        let preview: String = self.text.chars().take(60).collect();
        format!(
            "Hit(doc_id={:?}, chunk_id={}, score={:.4}, text={:?})",
            self.doc_id, self.chunk_id, self.score, preview
        )
    }
}

/// A collection on disk, opened for reading and writing by this process.
#[pyclass(module = "gyrfalcon")]
struct Collection {
    index: RwLock<Option<Index>>,
    path: PathBuf,
    tier: Tier,
}

impl Collection {
    fn read<T>(&self, py: Python<'_>, f: impl FnOnce(&Index) -> gyrfalcon::Result<T> + Send) -> PyResult<T>
    where
        T: Send,
    {
        py.detach(|| {
            let guard = self.index.read().unwrap_or_else(|p| p.into_inner());
            let index = guard.as_ref().ok_or_else(|| PyRuntimeError::new_err("the collection is closed"))?;
            f(index).map_err(engine_error)
        })
    }

    fn write<T>(&self, py: Python<'_>, f: impl FnOnce(&mut Index) -> PyResult<T> + Send) -> PyResult<T>
    where
        T: Send,
    {
        py.detach(|| {
            let mut guard = self.index.write().unwrap_or_else(|p| p.into_inner());
            let index = guard.as_mut().ok_or_else(|| PyRuntimeError::new_err("the collection is closed"))?;
            f(index)
        })
    }
}

#[pymethods]
impl Collection {
    /// Opens the collection at `path`, creating it when `create` is true and it does not exist.
    /// `model_dir` holds the tier's model files. `warm` is `none`, `background`, or `eager`.
    #[new]
    #[pyo3(signature = (path, model_dir, tier = "lite", create = true, warm = "eager"))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        model_dir: PathBuf,
        tier: &str,
        create: bool,
        warm: &str,
    ) -> PyResult<Collection> {
        let tier = parse_tier(tier)?;
        let warm = gyrfalcon::Warmup::parse(warm)
            .ok_or_else(|| PyValueError::new_err("warm must be \"none\", \"background\", or \"eager\""))?;
        let index = py.detach(|| tier.open_warming(&path, &model_dir, create, warm)).map_err(engine_error)?;
        Ok(Collection { index: RwLock::new(Some(index)), path, tier })
    }

    #[getter]
    fn path(&self) -> PathBuf {
        self.path.clone()
    }

    #[getter]
    fn tier(&self) -> &'static str {
        self.tier.name()
    }

    /// Indexes `(doc_id, text)` pairs in one write and returns the number of chunks written.
    /// Writing a doc_id that already exists adds its chunks beside the old ones; use `update`
    /// to replace a document.
    fn add_many(&self, py: Python<'_>, docs: Vec<(String, String, Metadata)>) -> PyResult<usize> {
        let docs: Vec<Document> =
            docs.into_iter().map(|(id, text, metadata)| Document::body(id, text).with_metadata(metadata)).collect();
        self.write(py, move |index| Ok(index.index_documents(&docs).map_err(engine_error)?.chunk_ids.len()))
    }

    /// Indexes one document and returns its chunk ids.
    #[pyo3(signature = (doc_id, text, metadata = Vec::new()))]
    fn add(&self, py: Python<'_>, doc_id: String, text: String, metadata: Metadata) -> PyResult<Vec<u64>> {
        let doc = Document::body(doc_id, text).with_metadata(metadata);
        self.write(py, move |index| Ok(index.index_documents(&[doc]).map_err(engine_error)?.chunk_ids))
    }

    /// Reads a file (PDF, HTML, or text) and indexes it under `doc_id`, by default its path.
    /// Returns the number of chunks written.
    #[pyo3(signature = (path, doc_id = None, metadata = Vec::new()))]
    fn add_file(&self, py: Python<'_>, path: PathBuf, doc_id: Option<String>, metadata: Metadata) -> PyResult<usize> {
        let doc_id = doc_id.unwrap_or_else(|| path.to_string_lossy().into_owned());
        self.write(py, move |index| {
            let text = gyrfalcon_ingest::read_file(&path).map_err(ingest_error)?.text;
            let doc = Document::body(doc_id, text).with_metadata(metadata);
            Ok(index.index_documents(&[doc]).map_err(engine_error)?.chunk_ids.len())
        })
    }

    /// Replaces a document's chunks with those of `text` in one atomic write.
    #[pyo3(signature = (doc_id, text, metadata = Vec::new()))]
    fn update(&self, py: Python<'_>, doc_id: String, text: String, metadata: Metadata) -> PyResult<usize> {
        let doc = Document::body(doc_id, text).with_metadata(metadata);
        self.write(py, move |index| Ok(index.update_document(doc).map_err(engine_error)?.chunk_ids.len()))
    }

    /// Removes a document and returns the number of chunks removed.
    fn delete(&self, py: Python<'_>, doc_id: String) -> PyResult<usize> {
        self.write(py, move |index| Ok(index.delete_document(&doc_id).map_err(engine_error)?.deleted_chunks))
    }

    /// The best `k` chunks for `query`, best first, among the documents matching every
    /// `(key, values)` clause of `filter` (any of the values).
    #[pyo3(signature = (query, k = 10, filter = Vec::new()))]
    fn search(
        &self,
        py: Python<'_>,
        query: String,
        k: usize,
        filter: Vec<(String, Vec<String>)>,
    ) -> PyResult<Vec<Hit>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let filter = filter.into_iter().fold(Filter::new(), |f, (key, values)| f.any(key, values));
        let options = RetrieveOptions { rerank: false, limit: k };
        let hits = self.read(py, move |index| index.retrieve_filtered(&query, options, Some(&filter)))?;
        Ok(hits
            .hits
            .into_iter()
            .map(|hit| Hit {
                chunk_id: hit.chunk_id,
                doc_id: hit.doc_id,
                score: hit.score,
                text: hit.body,
                start: hit.byte_start,
                end: hit.byte_end,
                pairs: hit.metadata,
            })
            .collect())
    }

    /// Seals buffered writes into a segment. Writes are durable without it (they are logged);
    /// flushing makes the next open faster.
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        self.write(py, |index| index.flush().map(|_| ()).map_err(engine_error))
    }

    /// Flushes, waits for background merges, and releases the collection.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let mut guard = self.index.write().unwrap_or_else(|p| p.into_inner());
            if let Some(mut index) = guard.take() {
                index.flush().map_err(engine_error)?;
                index.wait_until_merge_idle();
                index.reap_now();
            }
            Ok(())
        })
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, pyo3::types::PyTuple>) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }

    fn __repr__(&self) -> String {
        format!("Collection({:?}, tier={:?})", self.path.display().to_string(), self.tier.name())
    }
}

/// `(repository, path in it, file name)` for each file of a tier's model directory.
#[pyfunction]
fn tier_files(tier: &str) -> PyResult<Vec<(&'static str, &'static str, &'static str)>> {
    Ok(parse_tier(tier)?.files().iter().map(|f| (f.repo, f.path, f.name)).collect())
}

/// The directory name of a tier's model under a models root.
#[pyfunction]
fn tier_model_dir(tier: &str) -> PyResult<&'static str> {
    Ok(parse_tier(tier)?.model_dir())
}

/// The text Gyrfalcon indexes for a file, and the byte offset of each page (PDF only).
#[pyfunction]
fn read_file(py: Python<'_>, path: PathBuf) -> PyResult<(String, Vec<usize>)> {
    let extracted = py.detach(|| gyrfalcon_ingest::read_file(&path)).map_err(ingest_error)?;
    Ok((extracted.text, extracted.page_starts))
}

/// File extensions `add_path` picks up in a directory.
#[pyfunction]
fn supported_extensions() -> Vec<&'static str> {
    gyrfalcon_ingest::EXTENSIONS.to_vec()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Collection>()?;
    m.add_class::<Hit>()?;
    m.add_function(wrap_pyfunction!(tier_files, m)?)?;
    m.add_function(wrap_pyfunction!(tier_model_dir, m)?)?;
    m.add_function(wrap_pyfunction!(read_file, m)?)?;
    m.add_function(wrap_pyfunction!(supported_extensions, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
