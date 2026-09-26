//! Property-based tests for the voice identity byte formats: the 16 kHz mono
//! WAV recordings and the owner voiceprint file. Arbitrary bytes never panic
//! either parser, and valid inputs round-trip.
#![cfg(feature = "voice-id")]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use enton_adapters::voice_id::{
    EXPECTED_SAMPLE_RATE, VOICEPRINT_MAGIC, Voiceprint, WavAudio, WavError, load_voiceprint,
    parse_voiceprint_bytes, parse_wav_bytes, save_voiceprint,
};
use proptest::collection::vec;
use proptest::prelude::*;
use proptest::sample::Index;
use proptest::test_runner::{Config, TestCaseError};

/// Explicit case counts keep the suite fast in debug builds; failures are not
/// written to `proptest-regressions` files next to the sources.
fn config(cases: u32) -> Config {
    Config {
        cases,
        failure_persistence: None,
        ..Config::default()
    }
}

// ---------------------------------------------------------------------------
// WAV
// ---------------------------------------------------------------------------

const PCM: u16 = 1;
const IEEE_FLOAT: u16 = 3;

fn len32(len: usize) -> u32 {
    u32::try_from(len).unwrap_or(u32::MAX)
}

/// One RIFF chunk: its id, the size its header declares, and its payload,
/// padded to an even length as RIFF requires.
fn chunk(id: [u8; 4], declared: u32, payload: &[u8]) -> Vec<u8> {
    let mut bytes = id.to_vec();
    bytes.extend_from_slice(&declared.to_le_bytes());
    bytes.extend_from_slice(payload);
    if payload.len() % 2 == 1 {
        bytes.push(0);
    }
    bytes
}

/// A RIFF/WAVE container around already encoded chunks.
fn riff(chunks: &[Vec<u8>]) -> Vec<u8> {
    let body: Vec<u8> = chunks.concat();
    let mut wav = b"RIFF".to_vec();
    wav.extend_from_slice(&len32(body.len() + 4).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend(body);
    wav
}

/// The `fmt ` chunk of mono 16 kHz audio in `format` at `bits` per sample.
fn fmt_chunk(format: u16, bits: u16) -> Vec<u8> {
    let block_align = bits / 8;
    let byte_rate = EXPECTED_SAMPLE_RATE * u32::from(block_align);
    let payload = [
        format.to_le_bytes().as_slice(),
        &1_u16.to_le_bytes(),
        &EXPECTED_SAMPLE_RATE.to_le_bytes(),
        &byte_rate.to_le_bytes(),
        &block_align.to_le_bytes(),
        &bits.to_le_bytes(),
    ]
    .concat();
    chunk(*b"fmt ", 16, &payload)
}

fn data_chunk(data: &[u8]) -> Vec<u8> {
    chunk(*b"data", len32(data.len()), data)
}

/// A well-formed mono 16 kHz WAV holding `data` in `format` at `bits`.
fn wav(format: u16, bits: u16, data: &[u8]) -> Vec<u8> {
    riff(&[fmt_chunk(format, bits), data_chunk(data)])
}

fn le_bytes<const N: usize, T: Copy>(samples: &[T], encode: fn(T) -> [u8; N]) -> Vec<u8> {
    samples.iter().flat_map(|sample| encode(*sample)).collect()
}

/// What the parser promises about anything it accepts.
fn check_accepted(audio: &WavAudio) -> Result<(), TestCaseError> {
    prop_assert_eq!(audio.sample_rate, EXPECTED_SAMPLE_RATE);
    prop_assert_eq!(audio.channels, 1);
    prop_assert!(!audio.samples.is_empty(), "accepted audio holds samples");
    Ok(())
}

/// A chunk id: the two the parser reads, a common one it skips, or anything.
fn chunk_id() -> impl Strategy<Value = [u8; 4]> {
    prop_oneof![
        Just(*b"fmt "),
        Just(*b"data"),
        Just(*b"LIST"),
        any::<[u8; 4]>(),
    ]
}

/// A chunk whose header may lie about its size.
fn any_chunk() -> impl Strategy<Value = Vec<u8>> {
    (
        chunk_id(),
        vec(any::<u8>(), 0..64),
        prop_oneof![3 => Just(None), 1 => any::<u32>().prop_map(Some)],
    )
        .prop_map(|(id, payload, lie)| {
            chunk(id, lie.unwrap_or_else(|| len32(payload.len())), &payload)
        })
}

/// A chunk the parser must skip: never `fmt ` or `data`, any length, odd included.
fn foreign_chunk() -> impl Strategy<Value = Vec<u8>> {
    (
        any::<[u8; 4]>().prop_filter("not a chunk the parser reads", |id| {
            id != b"fmt " && id != b"data"
        }),
        vec(any::<u8>(), 0..33),
    )
        .prop_map(|(id, payload)| chunk(id, len32(payload.len()), &payload))
}

proptest! {
    #![proptest_config(config(512))]

    /// Arbitrary bytes never panic the parser, and whatever it accepts is
    /// 16 kHz mono audio with samples.
    #[test]
    fn wav_parser_survives_arbitrary_bytes(bytes in vec(any::<u8>(), 0..256)) {
        if let Ok(audio) = parse_wav_bytes(&bytes, "fuzz.wav") {
            check_accepted(&audio)?;
        }
    }

    /// The same for RIFF containers of arbitrary chunks, some lying about their size.
    #[test]
    fn wav_parser_survives_arbitrary_chunks(chunks in vec(any_chunk(), 0..6)) {
        if let Ok(audio) = parse_wav_bytes(&riff(&chunks), "chunks.wav") {
            check_accepted(&audio)?;
        }
    }

    /// 16-bit PCM reads back exactly as each sample over 32768.
    #[test]
    fn pcm16_round_trips_exactly(samples in vec(any::<i16>(), 1..1_024)) {
        let audio = parse_wav_bytes(&wav(PCM, 16, &le_bytes(&samples, i16::to_le_bytes)), "pcm16.wav")
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        check_accepted(&audio)?;
        let expected: Vec<u32> = samples.iter().map(|s| (f32::from(*s) / 32_768.0).to_bits()).collect();
        let parsed: Vec<u32> = audio.samples.iter().map(|s| s.to_bits()).collect();
        prop_assert_eq!(parsed, expected);
        prop_assert!(audio.samples.iter().all(|s| (-1.0..1.0).contains(s)));
    }

    /// 32-bit float samples read back bit for bit.
    #[test]
    fn float32_round_trips_exactly(samples in vec(-1.0_f32..=1.0, 1..1_024)) {
        let audio = parse_wav_bytes(&wav(IEEE_FLOAT, 32, &le_bytes(&samples, f32::to_le_bytes)), "float.wav")
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        check_accepted(&audio)?;
        let expected: Vec<u32> = samples.iter().map(|s| s.to_bits()).collect();
        let parsed: Vec<u32> = audio.samples.iter().map(|s| s.to_bits()).collect();
        prop_assert_eq!(parsed, expected);
    }

    /// Chunks the parser does not read (LIST, fact, anything, odd lengths with
    /// their pad byte) change nothing, wherever they sit.
    #[test]
    fn foreign_chunks_are_skipped(
        samples in vec(any::<i16>(), 1..64),
        before in vec(foreign_chunk(), 0..3),
        between in vec(foreign_chunk(), 0..3),
        after in vec(foreign_chunk(), 0..3),
    ) {
        let data = le_bytes(&samples, i16::to_le_bytes);
        let plain = parse_wav_bytes(&wav(PCM, 16, &data), "plain.wav");
        let mut chunks = before;
        chunks.push(fmt_chunk(PCM, 16));
        chunks.extend(between);
        chunks.push(data_chunk(&data));
        chunks.extend(after);
        let padded = parse_wav_bytes(&riff(&chunks), "padded.wav");
        prop_assert!(plain.is_ok());
        prop_assert_eq!(padded, plain);
    }

    /// A recording cut short anywhere is rejected, never read as shorter audio.
    #[test]
    fn a_truncated_wav_is_rejected(samples in vec(any::<i16>(), 1..64), cut in any::<Index>()) {
        let bytes = wav(PCM, 16, &le_bytes(&samples, i16::to_le_bytes));
        let prefix = &bytes[..cut.index(bytes.len())];
        prop_assert!(parse_wav_bytes(prefix, "cut.wav").is_err(), "{} of {} bytes parsed", prefix.len(), bytes.len());
    }

    /// Bug: the parser accepts 32-bit integer PCM (`format 1, bits 32`) but reads
    /// each sample's bits as an IEEE float, so a 32-bit PCM recording decodes to
    /// denormals, huge values and NaN instead of `sample / 2^31`. Shrunk
    /// reproducers (one sample each): `-1` reads back as NaN; `2148` reads back
    /// as 3.0e-42 instead of 1.0e-6.
    #[test]
    fn pcm32_round_trips_within_quantization(samples in vec(any::<i32>(), 1..256)) {
        let audio = parse_wav_bytes(&wav(PCM, 32, &le_bytes(&samples, i32::to_le_bytes)), "pcm32.wav")
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        check_accepted(&audio)?;
        prop_assert_eq!(audio.samples.len(), samples.len());
        for (parsed, sample) in audio.samples.iter().zip(&samples) {
            let expected = f64::from(*sample) / 2_147_483_648.0;
            prop_assert!(
                (f64::from(*parsed) - expected).abs() <= 1e-6,
                "PCM32 sample {} read as {}, expected {}", sample, parsed, expected
            );
        }
    }

    /// Bug: `WavAudio` is "decoded, validated" audio, yet an IEEE float WAV holding
    /// NaN or an infinity is accepted as is, and `extract_embedding` hands the
    /// samples to the extractor unchecked. Shrunk reproducer: a float WAV with the
    /// single sample NaN parses to `samples: [NaN]`.
    #[test]
    fn accepted_float_samples_are_finite(samples in vec(prop::num::f32::ANY, 1..64)) {
        if let Ok(audio) = parse_wav_bytes(&wav(IEEE_FLOAT, 32, &le_bytes(&samples, f32::to_le_bytes)), "float.wav") {
            prop_assert!(audio.samples.iter().all(|s| s.is_finite()), "{:?}", audio.samples);
        }
    }
}

/// A truncation the parser must name as such, not as some other defect.
#[test]
fn a_data_chunk_running_past_the_end_is_truncated() {
    let mut bytes = wav(PCM, 16, &[0, 1, 2, 3]);
    bytes.truncate(bytes.len() - 1);
    assert!(matches!(
        parse_wav_bytes(&bytes, "short.wav"),
        Err(WavError::TruncatedData { .. })
    ));
}

// ---------------------------------------------------------------------------
// Voiceprint
// ---------------------------------------------------------------------------

/// Fixed voiceprint header: magic, model SHA-256 and the 16-bit dimension.
const HEADER: usize = 6 + 32 + 2;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

/// A scratch directory, removed on drop, with a separate "repository" the
/// biometric guard must keep the voiceprint out of.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> std::io::Result<Self> {
        let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "enton-voiceprint-properties-{}-{id}",
            std::process::id()
        ));
        std::fs::create_dir_all(path.join("repo"))?;
        Ok(Self(path))
    }

    fn repo(&self) -> PathBuf {
        self.0.join("repo")
    }

    fn voiceprint(&self) -> PathBuf {
        self.0.join("owner.voiceprint")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // Best effort: a leftover temporary directory must not fail a property.
        drop(std::fs::remove_dir_all(&self.0));
    }
}

fn bits(voiceprint: &Voiceprint) -> Vec<u32> {
    voiceprint
        .embedding
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn save_and_read(
    scratch: &Scratch,
    embedding: &[f32],
    model: &[u8; 32],
) -> Result<Vec<u8>, TestCaseError> {
    let path = scratch.voiceprint();
    save_voiceprint(&path, embedding, model, &scratch.repo())
        .map_err(|error| TestCaseError::fail(error.to_string()))?;
    std::fs::read(Path::new(&path)).map_err(|error| TestCaseError::fail(error.to_string()))
}

proptest! {
    #![proptest_config(config(512))]

    /// Arbitrary bytes never panic the parser, and whatever it accepts has the
    /// exact length its header declares.
    #[test]
    fn voiceprint_parser_survives_arbitrary_bytes(bytes in vec(any::<u8>(), 0..128)) {
        if let Ok(voiceprint) = parse_voiceprint_bytes(&bytes, "fuzz.voiceprint") {
            prop_assert_eq!(bytes.len(), HEADER + 4 * voiceprint.embedding.len());
        }
    }

    /// The same behind a valid magic, where the header fields are reached.
    #[test]
    fn voiceprint_parser_survives_arbitrary_headers(tail in vec(any::<u8>(), 0..128)) {
        let bytes = [VOICEPRINT_MAGIC.as_slice(), &tail].concat();
        if let Ok(voiceprint) = parse_voiceprint_bytes(&bytes, "fuzz.voiceprint") {
            prop_assert!(!voiceprint.embedding.is_empty());
            prop_assert_eq!(bytes.len(), HEADER + 4 * voiceprint.embedding.len());
            prop_assert_eq!(&voiceprint.model_sha256[..], &tail[..32]);
        }
    }
}

proptest! {
    #![proptest_config(config(48))]

    /// A saved voiceprint loads back bit for bit, for every f32 bit pattern,
    /// and neither a cut nor an extra byte is ever read as a voiceprint.
    #[test]
    fn a_saved_voiceprint_loads_back_exactly(
        embedding in vec(any::<u32>().prop_map(f32::from_bits), 1..=512),
        model in any::<[u8; 32]>(),
        cut in any::<Index>(),
        extra in any::<u8>(),
    ) {
        let scratch = Scratch::new().map_err(|error| TestCaseError::fail(error.to_string()))?;
        let bytes = save_and_read(&scratch, &embedding, &model)?;
        prop_assert_eq!(bytes.len(), HEADER + 4 * embedding.len());

        let loaded = load_voiceprint(&scratch.voiceprint())
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let original = Voiceprint { model_sha256: model, embedding };
        prop_assert_eq!(loaded.model_sha256, original.model_sha256);
        prop_assert_eq!(bits(&loaded), bits(&original));
        let parsed = parse_voiceprint_bytes(&bytes, "saved.voiceprint")
            .map_err(|error| TestCaseError::fail(error.to_string()))?;
        prop_assert_eq!(bits(&parsed), bits(&original));

        let prefix = &bytes[..cut.index(bytes.len())];
        prop_assert!(parse_voiceprint_bytes(prefix, "cut.voiceprint").is_err());
        let longer = [bytes.as_slice(), &[extra]].concat();
        prop_assert!(parse_voiceprint_bytes(&longer, "long.voiceprint").is_err());
    }
}
