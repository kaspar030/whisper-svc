use std::time::Duration;
use std::{io::Cursor, ops::Deref};

use anyhow::{Error, Result, anyhow};
use once_cell::sync::Lazy;
use redlux::Decoder;
use thiserror::Error;
use tracing::{debug, info};
use whisper_rs::install_logging_hooks;

pub use models::WhisperModel;

mod models;
mod resample;

pub type WhisperChannel = flume::Sender<WhisperRequest>;

pub struct WhisperRequest {
    audio_data: WhisperAudio,
    model: Option<WhisperModel>,
    language: Option<WhisperLanguage>,
    reply_channel: flume::Sender<Result<WhisperResult, WhisperError>>,
}

#[derive(Error, Debug)]
pub enum WhisperError {
    #[error("no speech found in audio data")]
    NoSpeech,
    #[error("could not load model")]
    LoadingModelFailed,
}

pub struct WhisperResult {
    text: String,
}

static WHISPER_THREADS: Lazy<WhisperChannel> = Lazy::new(init);

fn init() -> WhisperChannel {
    install_logging_hooks();

    debug!("whisper-simple: init()");

    let (out, input) = flume::bounded(0);

    std::thread::spawn(move || whisper_loop(input));

    out
}

pub struct WhisperAudio {
    // audio data in whisper's format
    audio_data: Vec<f32>,
}

impl Deref for WhisperAudio {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        &self.audio_data
    }
}

impl WhisperAudio {
    pub fn from_aac(audio_data: &[u8]) -> Result<Self> {
        use redlux::Decoder;
        let decoder = Decoder::new_aac(std::io::Cursor::new(&audio_data));
        Self::from_aac_inner(decoder)
    }

    pub fn from_m4a(audio_data: &[u8]) -> Result<Self> {
        use redlux::Decoder;
        let file_size = audio_data.len() as u64;
        let decoder = Decoder::new_mpeg4(std::io::Cursor::new(&audio_data), file_size)?;
        Self::from_aac_inner(decoder)
    }

    fn from_aac_inner(mut decoder: Decoder<Cursor<&&[u8]>>) -> Result<Self> {
        // read one sample to populate metadata (sample rate, channels)
        decoder.decode_next_sample().unwrap();

        let audio_sample_rate = decoder.sample_rate();
        let audio_channels = decoder.channels();

        info!("audio sample rate: {audio_sample_rate} channels: {audio_channels}");

        match audio_channels {
            0 | 3.. => return Err(anyhow!("invalid channel number")),
            2 => {
                decoder.decode_next_sample().unwrap();
            }
            _ => (),
        }

        let mut decoded: Vec<f32> = decoder
            .map(|sample| sample as f32 / i16::MAX as f32)
            .collect();

        if audio_channels == 2 {
            decoded =
                whisper_rs::convert_stereo_to_mono_audio(&decoded[..]).map_err(|s| anyhow!(s))?;
        }

        let decoded = resample::resample(&decoded[..], audio_sample_rate as u64, 16000u64);

        Ok(Self {
            audio_data: decoded,
        })
    }

    fn len(&self) -> Duration {
        Duration::from_millis((self.audio_data.len() / 16) as u64)
    }
}

#[derive(Default)]
pub enum WhisperLanguage {
    #[default]
    Unknown,
    German,
    English,
    Spanish,
}

impl From<&WhisperLanguage> for &'static str {
    fn from(val: &WhisperLanguage) -> Self {
        match val {
            WhisperLanguage::Unknown => "auto",
            WhisperLanguage::German => "de",
            WhisperLanguage::English => "en",
            WhisperLanguage::Spanish => "es",
        }
    }
}

impl From<WhisperLanguage> for &'static str {
    fn from(val: WhisperLanguage) -> Self {
        Self::from(&val)
    }
}

impl TryFrom<&str> for WhisperLanguage {
    type Error = Error;

    fn try_from(val: &str) -> Result<Self> {
        match val {
            "de" => Ok(WhisperLanguage::German),
            "en" => Ok(WhisperLanguage::English),
            "es" => Ok(WhisperLanguage::Spanish),
            "auto" => Ok(WhisperLanguage::Unknown),
            _ => Err(anyhow!("unrecognized language {val}")),
        }
    }
}

fn whisper_loop(input: flume::Receiver<WhisperRequest>) {
    loop {
        let request = input.recv();
        if request.is_err() {
            debug!("whisper_loop(): exiting");
            break;
        }
        let request = request.unwrap();
        let model = request.model.unwrap_or_default();

        let result = model.process(&request.audio_data, request.language);

        // ignore if the reply channel has been closed
        let _ = request.reply_channel.send(result);
    }
}

pub async fn transcribe(
    model: WhisperModel,
    audio_data: WhisperAudio,
    language: Option<WhisperLanguage>,
) -> Result<String, Error> {
    debug!(
        "processing audio data (size: {:.2})",
        audio_data.len().as_secs_f32()
    );

    let (tx, rx) = flume::bounded(0);

    let request = WhisperRequest {
        model: Some(model),
        audio_data,
        language,
        reply_channel: tx,
    };

    WHISPER_THREADS.send(request)?;
    Ok(rx.recv_async().await?.map(|result| result.text)?)
}
