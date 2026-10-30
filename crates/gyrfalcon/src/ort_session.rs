//! ONNX Runtime sessions for the embedder and the cross-encoder reranker.
//!
//! `ort` is built with `load-dynamic`: the runtime library is found through `ORT_DYLIB_PATH`
//! or the system library path when the first session is created. When that library was built
//! with them, a session tries CUDA and then OpenVINO before the CPU provider.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use std::borrow::Cow;

use ort::ep::ExecutionProvider;
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use tokenizers::{EncodeInput, Tokenizer};

use crate::error::{Error, Result};

/// Longest token sequence fed to either model.
pub const MAX_TOKENS: usize = 512;

fn model_error(err: impl std::fmt::Display) -> Error {
    Error::ModelUnavailable(err.to_string())
}

/// Hardware tried before the CPU provider, fastest device first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accelerator {
    /// NVIDIA GPU through ONNX Runtime's CUDA execution provider.
    Cuda,
    /// Intel CPU through ONNX Runtime's OpenVINO execution provider.
    OpenVino,
}

/// Which accelerators a session should register.
///
/// `choice` is `GYR_ORT_EP`: empty or `auto` keeps both, `cuda` or `openvino` keeps one,
/// and `cpu` keeps neither. A provider the loaded library was not built with is skipped.
pub fn accelerator_plan(choice: &str, cuda_available: bool, openvino_available: bool) -> Vec<Accelerator> {
    let (want_cuda, want_openvino) = match choice.trim().to_ascii_lowercase().as_str() {
        "cuda" => (true, false),
        "openvino" => (false, true),
        "cpu" | "none" => (false, false),
        _ => (true, true),
    };
    let mut plan = Vec::new();
    if want_cuda && cuda_available {
        plan.push(Accelerator::Cuda);
    }
    if want_openvino && openvino_available {
        plan.push(Accelerator::OpenVino);
    }
    plan
}

fn execution_providers(plan: &[Accelerator]) -> Vec<ort::ep::ExecutionProviderDispatch> {
    plan.iter()
        .map(|accelerator| match accelerator {
            Accelerator::Cuda => ort::ep::CUDA::default()
                .with_conv_algorithm_search(ort::ep::cuda::ConvAlgorithmSearch::Heuristic)
                .build(),
            Accelerator::OpenVino => ort::ep::OpenVINO::default()
                .with_device_type("CPU")
                .with_num_threads(intra_threads())
                .with_dynamic_shapes(true)
                .build(),
        })
        .collect()
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
    /// Token budget enforced by the tokenizer (longest-first for pairs, keeping the final
    /// `[SEP]`). `None` cuts each encoding at `MAX_TOKENS` after tokenizing.
    max_tokens: Option<usize>,
    /// Providers registered ahead of CPU. Empty when the library has neither, or `GYR_ORT_EP`
    /// asked for CPU only.
    accelerators: Vec<Accelerator>,
}

impl OrtModel {
    /// Loads `model.onnx` (or `onnx/model.onnx`) and `tokenizer.json` from `dir`.
    pub fn load(dir: &Path) -> Result<OrtModel> {
        OrtModel::load_with(dir, None)
    }

    /// `load`, optionally reading the weights sequentially before the session maps them.
    pub fn load_prefetch(dir: &Path, prefetch: bool) -> Result<OrtModel> {
        OrtModel::load_with_prefetch(dir, None, prefetch)
    }

    /// `load` with a tokenizer-enforced token budget per input (a pair counts as one input).
    pub fn load_with(dir: &Path, max_tokens: Option<usize>) -> Result<OrtModel> {
        OrtModel::load_with_prefetch(dir, max_tokens, true)
    }

    fn load_with_prefetch(dir: &Path, max_tokens: Option<usize>, prefetch: bool) -> Result<OrtModel> {
        let model_path = ["model.onnx", "onnx/model.onnx"]
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.is_file())
            .ok_or_else(|| Error::ModelUnavailable(format!("no model.onnx under {}", dir.display())))?;
        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            return Err(Error::ModelUnavailable(format!("no tokenizer.json under {}", dir.display())));
        }
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(model_error)?;
        if let Some(max_length) = max_tokens {
            tokenizer
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length,
                    strategy: tokenizers::TruncationStrategy::LongestFirst,
                    ..Default::default()
                }))
                .map_err(model_error)?;
        }
        let environment = environment()?;
        let accelerators = accelerator_plan(
            &std::env::var("GYR_ORT_EP").unwrap_or_default(),
            ort::ep::CUDA::is_available().unwrap_or(false),
            ort::ep::OpenVINO::is_available().unwrap_or(false),
        );
        let mut builder = Session::builder(environment)
            .map_err(model_error)?
            .with_intra_threads(intra_threads())
            .map_err(model_error)?
            // Idle ort threads would otherwise spin on the cores retrieval needs.
            .with_intra_op_spinning(false)
            .map_err(model_error)?;
        if !accelerators.is_empty() {
            builder = builder.with_execution_providers(execution_providers(&accelerators)).map_err(model_error)?;
        }
        // The session reads the weights as it loads. Doing it sequentially first keeps that
        // read off a cold, random fault path when the runtime maps the file. `none` and
        // `background` skip it: the session load is the read, and it cannot return earlier.
        if prefetch {
            crate::warm::warm_file(&model_path)?;
        }
        let session = builder.commit_from_file(&model_path).map_err(model_error)?;
        let input_names = session.inputs().iter().map(|input| input.name().to_string()).collect();
        let output_width = session
            .outputs()
            .first()
            .and_then(|output| output.dtype().tensor_shape())
            .and_then(|shape| shape.last().copied())
            .unwrap_or(-1);
        Ok(OrtModel { session: Mutex::new(session), tokenizer, input_names, output_width, max_tokens, accelerators })
    }

    /// Providers this session asked for, CUDA before OpenVINO. A library without one of them
    /// leaves it out, and ONNX Runtime still falls back to CPU if registration fails.
    pub fn accelerators(&self) -> &[Accelerator] {
        &self.accelerators
    }

    pub fn output_width(&self) -> i64 {
        self.output_width
    }

    /// Tokenizes, pads to the longest sequence, runs the model, and returns the first output
    /// as `(shape, values)`.
    fn run<'a>(&self, inputs: Vec<EncodeInput<'a>>) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut encodings = self.tokenizer.encode_batch(inputs, true).map_err(model_error)?;
        if self.max_tokens.is_none() {
            for encoding in &mut encodings {
                encoding.truncate(MAX_TOKENS, 0, tokenizers::TruncationDirection::Right);
            }
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

#[cfg(test)]
mod tests {
    use super::{accelerator_plan, Accelerator};

    #[test]
    fn auto_tries_cuda_before_openvino() {
        assert_eq!(accelerator_plan("auto", true, true), vec![Accelerator::Cuda, Accelerator::OpenVino]);
        assert_eq!(accelerator_plan("", true, true), vec![Accelerator::Cuda, Accelerator::OpenVino]);
    }

    #[test]
    fn unavailable_provider_is_left_out() {
        assert_eq!(accelerator_plan("auto", false, true), vec![Accelerator::OpenVino]);
        assert_eq!(accelerator_plan("auto", true, false), vec![Accelerator::Cuda]);
        assert!(accelerator_plan("auto", false, false).is_empty());
    }

    #[test]
    fn explicit_choice_keeps_one_provider() {
        assert_eq!(accelerator_plan("cuda", true, true), vec![Accelerator::Cuda]);
        assert_eq!(accelerator_plan("OpenVINO", true, true), vec![Accelerator::OpenVino]);
        assert!(accelerator_plan("cuda", false, true).is_empty());
        assert!(accelerator_plan("cpu", true, true).is_empty());
        assert!(accelerator_plan("none", true, true).is_empty());
    }
}
