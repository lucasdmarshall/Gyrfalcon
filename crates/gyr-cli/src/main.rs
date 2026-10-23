//! `gyr`: index, query, serve, compact, and bench a Gyrfalcon collection.

mod bench;
mod server;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand, ValueEnum};
use gyrfalcon::{
    AnswerEvent, AnswerOptions, ChatConfig, Document, Embedder, Error, HashEmbedder, HttpEmbedder, IncompatibleField,
    Index, IndexConfig, OrtModel, OrtReranker, Quantization, RetrieveOptions, SessionEmbedder,
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
}

#[derive(Args, Clone, Debug)]
pub struct ModelArgs {
    #[arg(long, value_enum, env = "GYR_EMBEDDER", default_value = "ort")]
    pub embedder: EmbedderKind,
    /// Directory with model.onnx and tokenizer.json for the embedder.
    #[arg(long, env = "GYR_MODEL_DIR")]
    pub model_dir: Option<PathBuf>,
    /// Directory with model.onnx and tokenizer.json for the cross-encoder reranker.
    #[arg(long, env = "GYR_RERANKER_DIR")]
    pub reranker_dir: Option<PathBuf>,
    #[arg(long, env = "GYR_EMBED_URL")]
    pub embed_url: Option<String>,
    /// Defaults to BAAI/bge-small-en-v1.5, or gyr-hash for the hash embedder.
    #[arg(long, env = "GYR_MODEL_ID")]
    pub model_id: Option<String>,
    #[arg(long, default_value_t = gyrfalcon::DEFAULT_DIMENSION)]
    pub dimension: u32,
    #[arg(long, default_value = "i8", value_parser = parse_quantization)]
    pub quantization: Quantization,
    #[arg(long, env = "GYR_CHAT_ENDPOINT")]
    pub chat_endpoint: Option<String>,
    #[arg(long, env = "GYR_CHAT_MODEL", default_value = "default")]
    pub chat_model: String,
}

fn parse_quantization(value: &str) -> Result<Quantization, String> {
    Quantization::parse(value).ok_or_else(|| format!("unknown quantization {value}; use i8, f16, f32, or b1"))
}

impl ModelArgs {
    /// The process config every open compares against.
    pub fn config(&self) -> IndexConfig {
        let model_id = self.model_id.clone().unwrap_or_else(|| match self.embedder {
            EmbedderKind::Hash => "gyr-hash".to_string(),
            _ => gyrfalcon::DEFAULT_MODEL_ID.to_string(),
        });
        IndexConfig {
            model_id,
            dimension: self.dimension,
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

    /// Opens (or creates) the collection with the process config, then loads the models.
    /// The open gate runs before any model is loaded.
    pub fn open(&self, dir: &Path, create: bool) -> Result<Index, Error> {
        let config = self.config();
        let mut index = if create && !dir.join("manifest.json").exists() {
            Index::create(dir, &config)?
        } else {
            Index::open(dir, &config)?
        };
        let embedder: Box<dyn Embedder> = match self.embedder {
            EmbedderKind::Hash => Box::new(HashEmbedder::new(config.dimension)),
            EmbedderKind::Ort => {
                let model_dir = self.model_dir.as_ref().ok_or_else(|| {
                    Error::ModelUnavailable("--model-dir (or GYR_MODEL_DIR) is required for the ort embedder".into())
                })?;
                Box::new(SessionEmbedder::new(OrtModel::load(model_dir)?, config.model_id.clone()))
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
        index.set_embedder(embedder);
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
