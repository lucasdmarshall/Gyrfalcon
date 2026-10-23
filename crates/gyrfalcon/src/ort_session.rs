//! ONNX Runtime sessions for the embedder and the cross-encoder reranker.
//!
//! `ort` is built with `load-dynamic`: the runtime library is found through `ORT_DYLIB_PATH`
//! or the system library path when the first session is created.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use std::borrow::Cow;

use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::{EncodeInput, Tokenizer};

use crate::error::{Error, Result};

/// Longest token sequence fed to either model.
pub const MAX_TOKENS: usize = 512;

fn model_error(err: impl std::fmt::Display) -> Error {
    Error::ModelUnavailable(err.to_string())
}

/// Intra-op threads: `min(4, physical cores)` unless `GYR_ORT_THREADS` overrides it.
pub fn intra_threads() -> usize {
    std::env::var("GYR_ORT_THREADS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(4))
}

/// The process-wide ORT environment. Loading the runtime library can panic inside `ort`
/// when it is missing; that becomes `ModelUnavailable`.
fn environment() -> Result<&'static ort::environment::Environment> {
    static ENVIRONMENT: OnceLock<ort::environment::Environment> = OnceLock::new();
    if let Some(environment) = ENVIRONMENT.get() {
        return Ok(environment);
    }
    let built = std::panic::catch_unwind(|| ort::init().build())
        .map_err(|_| Error::ModelUnavailable("ONNX Runtime library could not be loaded; set ORT_DYLIB_PATH".into()))?
        .map_err(model_error)?;
    Ok(ENVIRONMENT.get_or_init(|| built))
}

/// A loaded ONNX model and its `tokenizer.json`, from one model directory.
pub struct OrtModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    input_names: Vec<String>,
    /// Last dimension of the first output, `-1` when dynamic.
    output_width: i64,
}

impl OrtModel {
    /// Loads `model.onnx` (or `onnx/model.onnx`) and `tokenizer.json` from `dir`.
    pub fn load(dir: &Path) -> Result<OrtModel> {
        let model_path = ["model.onnx", "onnx/model.onnx"]
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.is_file())
            .ok_or_else(|| Error::ModelUnavailable(format!("no model.onnx under {}", dir.display())))?;
        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(Error::ModelUnavailable(format!("no tokenizer.json under {}", dir.display())));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(model_error)?;
        let session = Session::builder(environment()?)
            .map_err(model_error)?
            .with_intra_threads(intra_threads())
            .map_err(model_error)?
            .commit_from_file(&model_path)
            .map_err(model_error)?;
        let input_names = session.inputs().iter().map(|input| input.name().to_string()).collect();
        let output_width = session
            .outputs()
            .first()
            .and_then(|output| output.dtype().tensor_shape())
            .and_then(|shape| shape.last().copied())
            .unwrap_or(-1);
        Ok(OrtModel { session: Mutex::new(session), tokenizer, input_names, output_width })
    }

    pub fn output_width(&self) -> i64 {
        self.output_width
    }

    /// Tokenizes, pads to the longest sequence, runs the model, and returns the first output
    /// as `(shape, values)`.
    fn run<'a>(&self, inputs: Vec<EncodeInput<'a>>) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut encodings = self.tokenizer.encode_batch(inputs, true).map_err(model_error)?;
        for encoding in &mut encodings {
            encoding.truncate(MAX_TOKENS, 0, tokenizers::TruncationDirection::Right);
        }
        let batch = encodings.len();
        let width = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0).max(1);
        let mut ids = vec![0i64; batch * width];
        let mut mask = vec![0i64; batch * width];
        let mut types = vec![0i64; batch * width];
        for (row, encoding) in encodings.iter().enumerate() {
            for (col, ((id, m), t)) in
                encoding.get_ids().iter().zip(encoding.get_attention_mask()).zip(encoding.get_type_ids()).enumerate()
            {
                ids[row * width + col] = *id as i64;
                mask[row * width + col] = *m as i64;
                types[row * width + col] = *t as i64;
            }
        }
        let shape = [batch, width];
        let mut values = Vec::new();
        for name in &self.input_names {
            let data = match name.as_str() {
                "input_ids" => ids.clone(),
                "attention_mask" => mask.clone(),
                "token_type_ids" => types.clone(),
                other => return Err(Error::ModelUnavailable(format!("unsupported model input {other}"))),
            };
            values.push((
                Cow::from(name.clone()),
                SessionInputValue::from(Tensor::from_array((shape, data)).map_err(model_error)?),
            ));
        }
        let mut session = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let outputs = session.run(values).map_err(model_error)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(model_error)?;
        Ok((shape.to_vec(), data.to_vec()))
    }

    /// Sentence embeddings: CLS pooling of a `[batch, seq, dim]` output (or a `[batch, dim]`
    /// output as is), then L2 normalization.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = texts.iter().map(|text| EncodeInput::from(text.as_str())).collect();
        let (shape, data) = self.run(inputs)?;
        let (batch, stride, dim) = match shape.as_slice() {
            [b, s, d] => (*b as usize, (*s * *d) as usize, *d as usize),
            [b, d] => (*b as usize, *d as usize, *d as usize),
            other => return Err(Error::Embed(format!("unexpected output shape {other:?}"))),
        };
        Ok((0..batch)
            .map(|row| {
                let mut vector = data[row * stride..row * stride + dim].to_vec();
                let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > 0.0 {
                    vector.iter_mut().for_each(|v| *v /= norm);
                }
                vector
            })
            .collect())
    }

    /// Cross-encoder logits for `(query, body)` pairs.
    pub fn score_pairs(&self, query: &str, bodies: &[String]) -> Result<Vec<f32>> {
        if bodies.is_empty() {
            return Ok(Vec::new());
        }
        let inputs = bodies.iter().map(|body| EncodeInput::from((query, body.as_str()))).collect();
        let (shape, data) = self.run(inputs)?;
        let per_row = shape.iter().skip(1).product::<i64>().max(1) as usize;
        Ok(data.chunks(per_row).map(|row| row[0]).collect())
    }
}
