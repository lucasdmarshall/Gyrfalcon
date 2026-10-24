//! `gyr`: index, query, serve, compact, and bench a Gyrfalcon collection.

mod bench;
mod server;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use gyrfalcon::{
    AnswerEvent, AnswerOptions, ChatConfig, Document, Embedder, Error, Fusion, HashEmbedder, HttpEmbedder,
    IncompatibleField, Index, IndexConfig, OrtModel, OrtReranker, Quantization, RetrieveOptions, SessionEmbedder,
    StaticEmbedder,
};

#[derive(Parser)]
#[command(name = "gyr", version, about = "Sealed-segment hybrid retrieval")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create or open a collection and index files (or stdin), then wait for the merger.
    Index {
        dir: PathBuf,
        files: Vec<PathBuf>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Retrieve (or answer) one query and print chunk ids.
    Query {
        dir: PathBuf,
        text: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
        /// Turn the local cross-encoder on.
        #[arg(long)]
        rerank: bool,
        /// Stream an answer from the chat endpoint instead of printing hits.
        #[arg(long)]
        answer: bool,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Serve the HTTP layer over one collection.
    Serve {
        dir: PathBuf,
        #[arg(long, default_value = "127.0.0.1:7700")]
        bind: String,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Run one synchronous tier merge, or rewrite everything with --all.
    Compact {
        dir: PathBuf,
        #[arg(long)]
        all: bool,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Benchmarks. Every report is JSON.
    Bench {
        #[command(subcommand)]
        bench: bench::BenchCommand,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum EmbedderKind {
    /// Local ONNX model from --model-dir.
    Ort,
    /// Model-free feature hashing; for tests and smoke runs.
    Hash,
    /// OpenAI-compatible embeddings endpoint from --embed-url.
    Http,
    /// Static Model2Vec model from --model-dir (model.safetensors, tokenizer.json).
    Static,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum FusionKind {
    Rrf,
    Score,
    Rescore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Tier {
    /// Static potion-retrieval-32M embeddings and rescore fusion: fast on any CPU.
    Lite,
    /// int8 bge-small-en-v1.5 through ONNX Runtime and rescore fusion: best quality.
    Full,
}

/// What a tier resolves to when no flag overrides it.
struct TierDefaults {
    embedder: EmbedderKind,
    model_dir: &'static str,
    model_id: &'static str,
    dense_alpha: f32,
}

impl Tier {
    fn defaults(self) -> TierDefaults {
        match self {
            Tier::Lite => TierDefaults {
                embedder: EmbedderKind::Static,
                model_dir: "potion-retrieval-32M",
                model_id: gyrfalcon::STATIC_MODEL_ID,
                dense_alpha: 0.2,
            },
            Tier::Full => TierDefaults {
                embedder: EmbedderKind::Ort,
                model_dir: "bge-small-en-v1.5-q8",
                model_id: "Xenova/bge-small-en-v1.5-q8",
                dense_alpha: 0.3,
            },
        }
    }
}

#[derive(Args, Clone, Debug)]
pub struct ModelArgs {
    /// Model and fusion preset. Every other flag overrides it.
    #[arg(long, value_enum, env = "GYR_TIER", default_value = "lite")]
    pub tier: Tier,
    /// Directory the tier models live under, one subdirectory per model.
    #[arg(long, env = "GYR_MODELS", default_value = ".models")]
    pub models: PathBuf,
    #[arg(long, value_enum, env = "GYR_EMBEDDER")]
    pub embedder: Option<EmbedderKind>,
    /// Model directory for the embedder; defaults to the tier's model under --models.
    #[arg(long, env = "GYR_MODEL_DIR")]
    pub model_dir: Option<PathBuf>,
    /// Directory with model.onnx and tokenizer.json for the cross-encoder reranker.
    #[arg(long, env = "GYR_RERANKER_DIR")]
    pub reranker_dir: Option<PathBuf>,
    #[arg(long, env = "GYR_EMBED_URL")]
    pub embed_url: Option<String>,
    /// Defaults to the tier's model id, or gyr-hash for the hash embedder.
    #[arg(long, env = "GYR_MODEL_ID")]
    pub model_id: Option<String>,
    /// Defaults to the static model's width, else 384.
    #[arg(long)]
    pub dimension: Option<u32>,
    #[arg(long, default_value = "i8", value_parser = parse_quantization)]
    pub quantization: Quantization,
    /// RRF weight of the BM25 leg.
    #[arg(long, default_value_t = 1.0)]
    pub lexical_weight: f32,
    /// RRF weight of the dense leg.
    #[arg(long, default_value_t = 1.0)]
    pub dense_weight: f32,
    /// How retrieve combines BM25 and dense: rrf, score, or rescore.
    #[arg(long, value_enum, default_value = "rescore")]
    pub fusion: FusionKind,
    /// Dense share for --fusion score or rescore, in [0, 1]; defaults to the tier's.
    #[arg(long)]
    pub dense_alpha: Option<f32>,
    #[arg(long, env = "GYR_CHAT_ENDPOINT")]
    pub chat_endpoint: Option<String>,
    #[arg(long, env = "GYR_CHAT_MODEL", default_value = "default")]
    pub chat_model: String,
}

fn parse_quantization(value: &str) -> Result<Quantization, String> {
    Quantization::parse(value).ok_or_else(|| format!("unknown quantization {value}; use i8, f16, f32, or b1"))
}

impl ModelArgs {
    pub fn embedder(&self) -> EmbedderKind {
        self.embedder.unwrap_or(self.tier.defaults().embedder)
    }

    /// An explicit --model-dir, else the tier's model when the tier's embedder is in use.
    pub fn model_dir(&self) -> Option<PathBuf> {
        let defaults = self.tier.defaults();
        self.model_dir
            .clone()
            .or_else(|| (self.embedder() == defaults.embedder).then(|| self.models.join(defaults.model_dir)))
    }

    pub fn model_id(&self) -> String {
        let defaults = self.tier.defaults();
        self.model_id.clone().unwrap_or_else(|| match self.embedder() {
            EmbedderKind::Hash => "gyr-hash".to_string(),
            kind if kind == defaults.embedder && self.model_dir.is_none() => defaults.model_id.to_string(),
            EmbedderKind::Static => gyrfalcon::STATIC_MODEL_ID.to_string(),
            _ => gyrfalcon::DEFAULT_MODEL_ID.to_string(),
        })
    }

    pub fn fusion(&self) -> Fusion {
        let dense_weight = self.dense_alpha.unwrap_or(self.tier.defaults().dense_alpha);
        match self.fusion {
            FusionKind::Rrf => Fusion::Rrf,
            FusionKind::Score => Fusion::Score { dense_weight },
            FusionKind::Rescore => Fusion::Rescore { dense_weight },
        }
    }

    /// The process config every open compares against.
    pub fn config(&self) -> IndexConfig {
        let model_id = self.model_id();
        let dimension = self.dimension.unwrap_or_else(|| match (self.embedder(), self.model_dir()) {
            // Reads only the safetensors header; the table loads after the open gate.
            (EmbedderKind::Static, Some(dir)) => {
                StaticEmbedder::read_dimension(&dir).unwrap_or(gyrfalcon::DEFAULT_DIMENSION)
            }
            _ => gyrfalcon::DEFAULT_DIMENSION,
        });
        IndexConfig {
            model_id,
            dimension,
            quantization: self.quantization,
            analyzer_id: gyrfalcon::ANALYZER_ID.to_string(),
        }
    }

    fn chat(&self) -> Option<ChatConfig> {
        self.chat_endpoint.as_ref().map(|endpoint| ChatConfig {
            endpoint: endpoint.clone(),
            model: self.chat_model.clone(),
            api_key: std::env::var("GYR_CHAT_API_KEY").ok(),
        })
    }

    /// Loads the embedder the process config names and checks its width.
    pub fn load_embedder(&self, config: &IndexConfig) -> Result<Box<dyn Embedder>, Error> {
        let model_dir = self.model_dir();
        let embedder: Box<dyn Embedder> = match self.embedder() {
            EmbedderKind::Hash => Box::new(HashEmbedder::new(config.dimension)),
            EmbedderKind::Ort => {
                let model_dir = model_dir.as_ref().ok_or_else(|| {
                    Error::ModelUnavailable("--model-dir (or GYR_MODEL_DIR) is required for the ort embedder".into())
                })?;
                Box::new(SessionEmbedder::new(OrtModel::load(model_dir)?, config.model_id.clone()))
            }
            EmbedderKind::Static => {
                let model_dir = model_dir.as_ref().ok_or_else(|| {
                    Error::ModelUnavailable("--model-dir (or GYR_MODEL_DIR) is required for the static embedder".into())
                })?;
                Box::new(StaticEmbedder::from_dir(model_dir, config.model_id.clone())?)
            }
            EmbedderKind::Http => {
                let url = self
                    .embed_url
                    .as_ref()
                    .ok_or_else(|| Error::ModelUnavailable("--embed-url is required for the http embedder".into()))?;
                Box::new(HttpEmbedder::new(url.clone(), config.dimension).with_model(config.model_id.clone()))
            }
        };
        if embedder.dimension() != config.dimension {
            return Err(Error::IncompatibleIndex {
                field: IncompatibleField::Dimension,
                manifest_value: config.dimension.to_string(),
                requested_value: embedder.dimension().to_string(),
            });
        }
        Ok(embedder)
    }

    /// Opens (or creates) the collection with the process config, then loads the models.
    /// The open gate runs before any model is loaded.
    pub fn open(&self, dir: &Path, create: bool) -> Result<Index, Error> {
        let config = self.config();
        let mut index = if create && !dir.join("manifest.json").exists() {
            Index::create(dir, &config)?
        } else {
            Index::open(dir, &config)?
        };
        let embedder = self.load_embedder(&config)?;
        index.set_embedder(embedder);
        index.rrf_weights = [self.lexical_weight, self.dense_weight];
        index.fusion = self.fusion();
        if let Some(dir) = &self.reranker_dir {
            index.set_reranker(Box::new(OrtReranker::from_dir(dir)?));
        }
        if let Some(chat) = self.chat() {
            index.set_chat(chat);
        }
        Ok(index)
    }
}

fn read_documents(files: &[PathBuf]) -> Result<Vec<Document>, Error> {
    if files.is_empty() {
        let mut text = String::new();
        std::io::stdin().read_to_string(&mut text)?;
        return Ok(vec![Document::body("stdin", text)]);
    }
    files.iter().map(|path| Ok(Document::body(path.to_string_lossy(), std::fs::read_to_string(path)?))).collect()
}

fn run(cli: Cli) -> Result<(), Error> {
    match cli.command {
        Command::Index { dir, files, model } => {
            let mut index = model.open(&dir, true)?;
            let mut chunks = 0;
            for doc in read_documents(&files)? {
                chunks += index.index_documents(std::slice::from_ref(&doc))?.chunk_ids.len();
            }
            index.wait_until_merge_idle();
            index.reap_now();
            let manifest = index.manifest();
            let report = serde_json::json!({
                "chunks": chunks,
                "epoch": manifest.epoch,
                "segments": manifest.sealed_segments.len(),
                "warning": index.last_report().warning,
            });
            println!("{report}");
        }
        Command::Query { dir, text, limit, rerank, answer, model } => {
            let index = model.open(&dir, false)?;
            if answer {
                let stream = index.answer(&text, AnswerOptions { rerank, ..AnswerOptions::default() })?;
                let mut out = std::io::stdout().lock();
                for event in stream {
                    match event {
                        AnswerEvent::TimeToRequest { ms } => eprintln!("time_to_request_ms {ms:.3}"),
                        AnswerEvent::Hit(hit) => eprintln!("[{}] chunk {}", hit.doc_id, hit.chunk_id),
                        AnswerEvent::Token(token) => {
                            write!(out, "{token}")?;
                            out.flush()?;
                        }
                        AnswerEvent::Done => writeln!(out)?,
                        AnswerEvent::Error(err) => return Err(err),
                    }
                }
            } else {
                let hits = index.retrieve(&text, RetrieveOptions { rerank, limit })?;
                for hit in hits.hits {
                    println!("{}\t{}\t{:.6}", hit.chunk_id, hit.doc_id, hit.score);
                }
                if let Some(warning) = hits.warning {
                    eprintln!("warning: {warning}");
                }
            }
        }
        Command::Serve { dir, bind, model } => {
            let index = model.open(&dir, true)?;
            server::serve(index, &bind, model.config(), &dir)?;
        }
        Command::Compact { dir, all, model } => {
            let mut index = model.open(&dir, false)?;
            let published = if all { index.compact_all()? } else { index.compact()? };
            index.reap_now();
            println!("{}", serde_json::json!({ "published": published, "segments": index.segment_chunk_counts() }));
        }
        Command::Bench { bench } => bench::run(bench)?,
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err}");
            ExitCode::FAILURE
        }
    }
}
