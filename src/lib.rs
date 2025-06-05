use std::time::Duration;
use std::{io::Cursor, ops::Deref};

use anyhow::{Error, Result, anyhow};
use once_cell::sync::Lazy;
use redlux::Decoder;
use symphonia::core::audio::SampleBuffer;
use thiserror::Error;
use tracing::debug;
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

        debug!("decoding AAC, sample rate: {audio_sample_rate} channels: {audio_channels}");

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

    pub fn from_mp3(audio_data: &[u8]) -> Result<Self> {
        use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
        use symphonia::core::errors::Error;
        use symphonia::core::formats::FormatOptions;
        use symphonia::core::io::MediaSourceStream;
        use symphonia::core::meta::MetadataOptions;
        use symphonia::core::probe::Hint;

        // create owned boxed audio data
        let audio_data = Vec::from(audio_data);

        // Create the media source stream.
        let mss = MediaSourceStream::new(Box::new(Cursor::new(audio_data)), Default::default());

        // Create a probe hint using the file's extension. [Optional]
        let mut hint = Hint::new();
        hint.with_extension("mp3");

        // Use the default options for metadata and format readers.
        let meta_opts: MetadataOptions = Default::default();
        let fmt_opts: FormatOptions = Default::default();

        // Probe the media source.
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .expect("unsupported format");

        // Get the instantiated format reader.
        let mut format = probed.format;

        // Find the first audio track with a known (decodeable) codec.
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .expect("no supported audio tracks");

        let audio_sample_rate = track.codec_params.sample_rate.unwrap_or_default();
        let audio_channels = track.codec_params.channels.unwrap_or_default().count();

        debug!("decoding MP3, sample rate: {audio_sample_rate} channels: {audio_channels}");

        // Use the default options for the decoder.
        let dec_opts: DecoderOptions = Default::default();

        // Create a decoder for the track.
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .expect("unsupported codec");

        // Store the track identifier, it will be used to filter packets.
        let track_id = track.id;

        let mut sample_count = 0;
        let mut sample_buf = None;

        let mut decoded: Vec<f32> = Vec::new();

        loop {
            // Get the next packet from the format reader.
            let packet = format.next_packet();
            if packet.is_err() {
                break;
            }

            let packet = packet?;

            // If the packet does not belong to the selected track, skip it.
            if packet.track_id() != track_id {
                continue;
            }

            // Decode the packet into audio samples, ignoring any decode errors.
            match decoder.decode(&packet) {
                Ok(audio_buf) => {
                    // The decoded audio samples may now be accessed via the audio buffer if per-channel
                    // slices of samples in their native decoded format is desired. Use-cases where
                    // the samples need to be accessed in an interleaved order or converted into
                    // another sample format, or a byte buffer is required, are covered by copying the
                    // audio buffer into a sample buffer or raw sample buffer, respectively. In the
                    // example below, we will copy the audio buffer into a sample buffer in an
                    // interleaved order while also converting to a f32 sample format.

                    // If this is the *first* decoded packet, create a sample buffer matching the
                    // decoded audio buffer format.
                    if sample_buf.is_none() {
                        // Get the audio buffer specification.
                        let spec = *audio_buf.spec();

                        // Get the capacity of the decoded buffer. Note: This is capacity, not length!
                        let duration = audio_buf.capacity() as u64;

                        // Create the f32 sample buffer.
                        sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
                    }

                    // Copy the decoded audio buffer into the sample buffer in an interleaved format.
                    if let Some(buf) = &mut sample_buf {
                        buf.copy_interleaved_ref(audio_buf);

                        // The samples may now be access via the `samples()` function.
                        sample_count += buf.samples().len();
                        print!("\rDecoded {} samples", sample_count);
                        decoded.extend_from_slice(buf.samples());
                    }
                }
                Err(Error::DecodeError(_)) => (),
                Err(_) => break,
            }
        }

        match audio_channels {
            2 => {
                println!("converting to mono...");
                decoded =
                    whisper_rs::convert_stereo_to_mono_audio(&decoded).map_err(|s| anyhow!(s))?;
            }
            1 => (),
            _ => return Err(anyhow!("unexpected channel count")),
        }

        let decoded = resample::resample(&decoded, audio_sample_rate as u64, 16000u64);

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
