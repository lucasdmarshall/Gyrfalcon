//! The two product tiers: which model a collection embeds with, where its files come from, and
//! how retrieval fuses the legs. The CLI and the Python package open collections through here,
//! so a collection made by one opens in the other.

use std::path::Path;

use crate::embed::{Embedder, SessionEmbedder};
use crate::error::{Error, Result};
use crate::fuse::{Fusion, FULL_RESCORE_DENSE_WEIGHT, LITE_DENSE_WEIGHT};
use crate::index::Index;
use crate::manifest::{IndexConfig, Quantization, ANALYZER_ID, DEFAULT_DIMENSION, STATIC_MODEL_ID};
use crate::ort_session::OrtModel;
use crate::static_embed::StaticEmbedder;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Static potion-retrieval-32M embeddings: no runtime, fast on any CPU.
    Lite,
    /// int8 bge-small-en-v1.5 through ONNX Runtime: best quality.
    Full,
}

/// A file a tier's model directory needs: the Hugging Face repository, the path in it, and the
/// name it takes in the model directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile {
    pub repo: &'static str,
    pub path: &'static str,
    pub name: &'static str,
}

impl Tier {
    pub fn parse(name: &str) -> Option<Tier> {
        match name {
            "lite" => Some(Tier::Lite),
            "full" => Some(Tier::Full),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Tier::Lite => "lite",
            Tier::Full => "full",
        }
    }

    /// The model directory's name under a models root.
    pub fn model_dir(self) -> &'static str {
        match self {
            Tier::Lite => "potion-retrieval-32M",
            Tier::Full => "bge-small-en-v1.5-q8",
        }
    }

    /// The model id a collection of this tier records in its manifest.
    pub fn model_id(self) -> &'static str {
        match self {
            Tier::Lite => STATIC_MODEL_ID,
            Tier::Full => "Xenova/bge-small-en-v1.5-q8",
        }
    }

    /// How retrieval fuses the legs, chosen on validation splits per tier. Static cosines
    /// spread widely, so the lite tier mixes them raw with BM25 over its query maximum; bge's
    /// cosines sit in a narrow band and need min-max stretching.
    pub fn fusion(self) -> Fusion {
        match self {
            Tier::Lite => Fusion::RescoreMax { dense_weight: LITE_DENSE_WEIGHT },
            Tier::Full => Fusion::Rescore { dense_weight: FULL_RESCORE_DENSE_WEIGHT },
        }
    }

    pub fn files(self) -> &'static [ModelFile] {
        const LITE: &[ModelFile] = &[
            ModelFile { repo: "minishlab/potion-retrieval-32M", path: "config.json", name: "config.json" },
            ModelFile { repo: "minishlab/potion-retrieval-32M", path: "model.safetensors", name: "model.safetensors" },
            ModelFile { repo: "minishlab/potion-retrieval-32M", path: "tokenizer.json", name: "tokenizer.json" },
        ];
        const FULL: &[ModelFile] = &[
            ModelFile { repo: "Xenova/bge-small-en-v1.5", path: "tokenizer.json", name: "tokenizer.json" },
            ModelFile { repo: "Xenova/bge-small-en-v1.5", path: "onnx/model_quantized.onnx", name: "model.onnx" },
        ];
        match self {
            Tier::Lite => LITE,
            Tier::Full => FULL,
        }
    }

    /// The identity a collection of this tier is created with and checked against on open.
    /// `model_dir` holds the tier's files; only the lite model's header is read here.
    pub fn config(self, model_dir: &Path) -> IndexConfig {
        let dimension = match self {
            Tier::Lite => StaticEmbedder::read_dimension(model_dir).unwrap_or(DEFAULT_DIMENSION),
            Tier::Full => DEFAULT_DIMENSION,
        };
        IndexConfig {
            model_id: self.model_id().to_string(),
            dimension,
            quantization: Quantization::I8,
            analyzer_id: ANALYZER_ID.to_string(),
        }
    }

    /// Loads the tier's embedder from `model_dir`.
    pub fn embedder(self, model_dir: &Path) -> Result<Box<dyn Embedder>> {
        let embedder: Box<dyn Embedder> = match self {
            Tier::Lite => Box::new(StaticEmbedder::from_dir(model_dir, self.model_id())?),
            Tier::Full => Box::new(SessionEmbedder::new(OrtModel::load(model_dir)?, self.model_id())),
        };
        embedder.warm();
        Ok(embedder)
    }

    /// Opens the collection at `dir`, creating it when `create` is set and it does not exist,
    /// with this tier's identity, embedder, and fusion. The open gate runs before the model
    /// loads, so a collection of the other tier is refused without loading anything.
    pub fn open(self, dir: &Path, model_dir: &Path, create: bool) -> Result<Index> {
        self.open_warming(dir, model_dir, create, crate::warm::Warmup::default())
    }

    /// `open` with an explicit page-cache warmup. The model is loaded before `warmup` runs, so
    /// a background read includes the static table. An ONNX session still loads on this thread:
    /// it cannot answer a query until its weights are read.
    pub fn open_warming(
        self,
        dir: &Path,
        model_dir: &Path,
        create: bool,
        warmup: crate::warm::Warmup,
    ) -> Result<Index> {
        if !model_dir.is_dir() {
            return Err(Error::ModelUnavailable(format!(
                "the {} tier's model is not at {}",
                self.name(),
                model_dir.display()
            )));
        }
        let config = self.config(model_dir);
        let mut index = if create && !dir.join("manifest.json").exists() {
            Index::create(dir, &config)?
        } else {
            Index::open_with_warmup(dir, &config, crate::warm::Warmup::None)?
        };
        let embedder: Box<dyn Embedder> = match self {
            Tier::Lite => Box::new(StaticEmbedder::from_dir(model_dir, self.model_id())?),
            Tier::Full => Box::new(SessionEmbedder::new(
                OrtModel::load_prefetch(model_dir, warmup == crate::warm::Warmup::Eager)?,
                self.model_id(),
            )),
        };
        if embedder.dimension() != config.dimension {
            return Err(Error::Embed(format!(
                "the model makes {}-dimensional vectors, the collection holds {}",
                embedder.dimension(),
                config.dimension
            )));
        }
        index.set_embedder(embedder);
        index.fusion = self.fusion();
        index.warmup(warmup)?;
        Ok(index)
    }
}
