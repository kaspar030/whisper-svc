use std::path::PathBuf;

use anyhow::{Context as _, anyhow};
use once_cell::sync::OnceCell;
use tracing::debug;
use whisper_rs::{WhisperContext, WhisperContextParameters};

use serde::{Deserialize, Serialize};

use crate::{WhisperAudio, WhisperError, WhisperLanguage, WhisperResult};

static MODEL_TINY: OnceCell<WhisperContext> = OnceCell::new();
static MODEL_BASE: OnceCell<WhisperContext> = OnceCell::new();
static MODEL_SMALL: OnceCell<WhisperContext> = OnceCell::new();
static MODEL_MEDIUM: OnceCell<WhisperContext> = OnceCell::new();
static MODEL_LARGE: OnceCell<WhisperContext> = OnceCell::new();
static MODEL_LARGE_V3_TURBO: OnceCell<WhisperContext> = OnceCell::new();

fn params<'a>() -> WhisperContextParameters<'a> {
    let mut params = WhisperContextParameters::new();
    params.flash_attn(true);
    params
}

static MODEL_PATH: OnceCell<PathBuf> = OnceCell::new();

pub fn set_model_directory(path: &str) -> Result<(), anyhow::Error> {
    MODEL_PATH
        .set(PathBuf::from(path))
        .map_err(|_| anyhow!("model path already set"))?;
    Ok(())
}

fn open_model(name: &str) -> Result<WhisperContext, anyhow::Error> {
    let model_path = MODEL_PATH.get_or_try_init(|| {
        let mut model_path =
            dirs::data_dir().ok_or(anyhow!("could not determine models directory"))?;
        model_path.push("whisper-models");
        Ok::<PathBuf, anyhow::Error>(model_path)
    })?;

    let mut model_path = model_path.join(name);
    model_path.set_extension("bin");

    WhisperContext::new_with_params(model_path.to_str().unwrap(), params())
        .context("opening model file")
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Hash, Serialize, Deserialize)]
pub enum WhisperModel {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    #[default]
    LargeV3Turbo,
}

impl WhisperModel {
    fn instance(&self) -> Result<&WhisperContext, WhisperError> {
        let (once_cell, name) = match self {
            WhisperModel::Tiny => (&MODEL_TINY, "ggml-tiny"),
            WhisperModel::Base => (&MODEL_BASE, "ggml-base"),
            WhisperModel::Small => (&MODEL_SMALL, "ggml-small"),
            WhisperModel::Medium => (&MODEL_MEDIUM, "ggml-medium"),
            WhisperModel::Large => (&MODEL_LARGE, "ggml-large"),
            WhisperModel::LargeV3Turbo => (&MODEL_LARGE_V3_TURBO, "ggml-large-v3-turbo"),
        };

        once_cell.get_or_try_init(|| open_model(name)).map_err(|e| {
            debug!("error loading model file `{name}`: {e}");
            WhisperError::LoadingModelFailed
        })
    }

    pub(crate) fn process(
        &self,
        audio_data: &WhisperAudio,
        language: Option<WhisperLanguage>,
    ) -> Result<WhisperResult, WhisperError> {
        use whisper_rs::{FullParams, SamplingStrategy};
        debug!("audio length: {:.2?}", audio_data.len());

        debug!("loading model...");
        let model = self.instance()?;

        debug!("creating state...");
        let state = &mut model
            .create_state()
            .expect("failed to create whisper state");

        debug!("model state ready, starting inference");
        let mut params = FullParams::new(SamplingStrategy::default());
        params.set_n_threads(num_cpus::get_physical() as i32);
        params.set_translate(false);

        if let Some(ref language) = language {
            let s: &str = language.into();
            debug!("forcing language `{s}`");
        }
        params.set_language(language.map(std::convert::Into::into));

        let start = std::time::Instant::now();

        state
            .full(params, &audio_data[..])
            .expect("failed to run model");

        // fetch the results
        let num_segments = state.full_n_segments().expect("getting segments");
        if num_segments == 0 {
            debug!("no voice found");
            Err(WhisperError::NoSpeech)
        } else {
            let mut out = Vec::new();
            for i in 0..num_segments {
                let segment = state
                    .full_get_segment_text(i)
                    .expect("failed to get segment");
                let start_timestamp = state.full_get_segment_t0(i).expect("getting segment t0");
                let end_timestamp = state.full_get_segment_t1(i).expect("getting segment t1");
                debug!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
                out.push(segment.trim().to_string());
            }
            let transcription = out.join("\n");
            let inference_time = start.elapsed();
            debug!(
                "inference done. took {:.2?} (ratio: {:.2?})",
                inference_time,
                inference_time.as_secs_f64() / audio_data.len().as_secs_f64()
            );
            Ok(WhisperResult {
                text: transcription,
            })
        }
    }
}

impl TryFrom<&str> for WhisperModel {
    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        Ok(match value {
            "tiny" => WhisperModel::Tiny,
            "base" => WhisperModel::Base,
            "small" => WhisperModel::Small,
            "medium" => WhisperModel::Medium,
            "large" => WhisperModel::Large,
            "large-v3-turbo" => WhisperModel::LargeV3Turbo,
            _ => return Err(anyhow::anyhow!("unknown model {value}")),
        })
    }

    type Error = anyhow::Error;
}
