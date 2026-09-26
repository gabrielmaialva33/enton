//! 16 kHz mono WAV parsing for enrollment and scoring recordings.

use super::EXPECTED_SAMPLE_RATE;

/// Decoded, validated mono audio.
#[derive(Debug, Clone, PartialEq)]
pub struct WavAudio {
    /// Sample rate in hertz.
    pub sample_rate: u32,
    /// Channel count (one, after validation).
    pub channels: u16,
    /// Samples normalized to [-1, 1].
    pub samples: Vec<f32>,
}

/// Why a WAV buffer was rejected.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum WavError {
    /// The buffer is shorter than a WAV header.
    #[error("file '{file}' too small for WAV header ({bytes} bytes)")]
    TooSmall {
        /// File the error refers to.
        file: String,
        /// Length of the input, in bytes.
        bytes: usize,
    },
    /// The buffer does not start with a RIFF header.
    #[error("file '{file}' has invalid RIFF header")]
    InvalidRiff {
        /// File the error refers to.
        file: String,
    },
    /// The RIFF container is not WAVE.
    #[error("file '{file}' has invalid WAVE identifier")]
    InvalidWave {
        /// File the error refers to.
        file: String,
    },
    /// No `fmt ` chunk was found.
    #[error("file '{file}' is missing 'fmt ' chunk")]
    MissingFmtChunk {
        /// File the error refers to.
        file: String,
    },
    /// No `data` chunk was found.
    #[error("file '{file}' is missing 'data' chunk")]
    MissingDataChunk {
        /// File the error refers to.
        file: String,
    },
    /// The `data` chunk precedes the `fmt ` chunk.
    #[error("file '{file}' has 'data' chunk before 'fmt ' chunk")]
    DataBeforeFmt {
        /// File the error refers to.
        file: String,
    },
    /// The sample format is neither PCM nor IEEE float.
    #[error(
        "unsupported audio format {format} in '{file}': only PCM (1) or IEEE Float (3) supported"
    )]
    UnsupportedFormat {
        /// File the error refers to.
        file: String,
        /// Format tag found.
        format: u16,
    },
    /// The audio is not mono.
    #[error("unsupported channels {channels} in '{file}': only mono (1 channel) is supported")]
    UnsupportedChannels {
        /// File the error refers to.
        file: String,
        /// Channel count found.
        channels: u16,
    },
    /// The audio is not 16 kHz.
    #[error("unsupported sample rate {rate} Hz in '{file}': only 16000 Hz is supported")]
    UnsupportedSampleRate {
        /// File the error refers to.
        file: String,
        /// Sample rate found, in hertz.
        rate: u32,
    },
    /// The sample width is unsupported.
    #[error(
        "unsupported bits per sample {bits} in '{file}': only 16-bit PCM or 32-bit float supported"
    )]
    UnsupportedBitsPerSample {
        /// File the error refers to.
        file: String,
        /// Bits per sample found.
        bits: u16,
    },
    /// A chunk runs past the end of the buffer.
    #[error("truncated chunk or data in '{file}'")]
    TruncatedData {
        /// File the error refers to.
        file: String,
    },
    /// The `data` chunk holds no samples.
    #[error("WAV file '{file}' contains zero audio samples")]
    EmptyAudio {
        /// File the error refers to.
        file: String,
    },
    /// The audio data contains non-finite sample values (NaN or infinity).
    #[error("file '{file}' contains non-finite audio samples (NaN or infinity)")]
    NonFiniteSample {
        /// File the error refers to.
        file: String,
    },
}
/// The `N` bytes at `at`, or a truncation error naming `file`.
fn field<const N: usize>(data: &[u8], at: usize, file: &str) -> Result<[u8; N], WavError> {
    data.get(at..at.saturating_add(N))
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| WavError::TruncatedData {
            file: file.to_owned(),
        })
}

/// Parse and validate a 16 kHz mono WAV byte buffer.
pub fn parse_wav_bytes(bytes: &[u8], file_name: &str) -> Result<WavAudio, WavError> {
    let file = file_name.to_string();
    if bytes.len() < 12 {
        return Err(WavError::TooSmall {
            file,
            bytes: bytes.len(),
        });
    }
    if bytes.get(0..4) != Some(b"RIFF") {
        return Err(WavError::InvalidRiff { file });
    }
    if bytes.get(8..12) != Some(b"WAVE") {
        return Err(WavError::InvalidWave { file });
    }

    let mut cursor = 12_usize;
    let mut fmt_info: Option<(u16, u16, u32, u16)> = None;
    let mut samples: Option<Vec<f32>> = None;

    while cursor.saturating_add(8) <= bytes.len() {
        let id: [u8; 4] = field(bytes, cursor, &file)?;
        let sz = u32::from_le_bytes(field(bytes, cursor + 4, &file)?) as usize;
        let start = cursor + 8;
        let end = start.saturating_add(sz);
        if end > bytes.len() {
            return Err(WavError::TruncatedData { file });
        }
        let data = bytes
            .get(start..end)
            .ok_or_else(|| WavError::TruncatedData { file: file.clone() })?;

        if &id == b"fmt " {
            if data.len() < 16 {
                return Err(WavError::TruncatedData { file });
            }
            let fmt = u16::from_le_bytes(field(data, 0, &file)?);
            let ch = u16::from_le_bytes(field(data, 2, &file)?);
            let rate = u32::from_le_bytes(field(data, 4, &file)?);
            let bits = u16::from_le_bytes(field(data, 14, &file)?);

            if fmt != 1 && fmt != 3 {
                return Err(WavError::UnsupportedFormat { file, format: fmt });
            }
            if ch != 1 {
                return Err(WavError::UnsupportedChannels { file, channels: ch });
            }
            if rate != EXPECTED_SAMPLE_RATE {
                return Err(WavError::UnsupportedSampleRate { file, rate });
            }
            if (fmt == 1 && bits != 16 && bits != 32) || (fmt == 3 && bits != 32) {
                return Err(WavError::UnsupportedBitsPerSample { file, bits });
            }
            fmt_info = Some((fmt, ch, rate, bits));
        } else if &id == b"data" {
            let Some((fmt, _, _, bits)) = fmt_info else {
                return Err(WavError::DataBeforeFmt { file });
            };
            let mut out = Vec::new();
            match (fmt, bits) {
                (1, 16) => {
                    out.reserve(data.len() / 2);
                    for &c in data.as_chunks::<2>().0 {
                        let val = i16::from_le_bytes(c);
                        out.push(f32::from(val) / 32768.0);
                    }
                }
                (1, 32) => {
                    out.reserve(data.len() / 4);
                    for &c in data.as_chunks::<4>().0 {
                        let val = i32::from_le_bytes(c);
                        out.push(val as f32 / 2_147_483_648.0);
                    }
                }
                (3, 32) => {
                    out.reserve(data.len() / 4);
                    for &c in data.as_chunks::<4>().0 {
                        let val = f32::from_le_bytes(c);
                        if !val.is_finite() {
                            return Err(WavError::NonFiniteSample { file });
                        }
                        out.push(val);
                    }
                }
                _ => {}
            }
            samples = Some(out);
        }
        cursor = end.saturating_add(sz & 1);
    }

    let Some((_, channels, sample_rate, _)) = fmt_info else {
        return Err(WavError::MissingFmtChunk { file });
    };
    let Some(parsed_samples) = samples else {
        return Err(WavError::MissingDataChunk { file });
    };
    if parsed_samples.is_empty() {
        return Err(WavError::EmptyAudio { file });
    }
    Ok(WavAudio {
        sample_rate,
        channels,
        samples: parsed_samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::cast_possible_truncation)]
    fn make_test_wav_from_bytes(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        format: u16,
        data_bytes: &[u8],
    ) -> Vec<u8> {
        let byte_rate = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
        let block_align = channels * bits_per_sample / 8;
        let data_len = data_bytes.len() as u32;
        let fmt_len = 16_u32;
        let file_size_minus_8 = 4 + (8 + fmt_len) + (8 + data_len);

        let mut wav = Vec::with_capacity((file_size_minus_8 + 8) as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&file_size_minus_8.to_le_bytes());
        wav.extend_from_slice(b"WAVE");

        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&fmt_len.to_le_bytes());
        wav.extend_from_slice(&format.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&byte_rate.to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&bits_per_sample.to_le_bytes());

        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(data_bytes);

        wav
    }

    // The fixture encodes small synthetic signals whose sizes and levels fit by construction.
    #[allow(clippy::cast_possible_truncation)]
    fn make_test_wav(
        sample_rate: u32,
        channels: u16,
        bits_per_sample: u16,
        format: u16,
        samples: &[f32],
    ) -> Vec<u8> {
        let mut data_bytes = Vec::new();
        if format == 1 && bits_per_sample == 16 {
            for &s in samples {
                let clamped = s.clamp(-1.0, 1.0);
                let val = (clamped * 32767.0) as i16;
                data_bytes.extend_from_slice(&val.to_le_bytes());
            }
        } else if format == 1 && bits_per_sample == 32 {
            for &s in samples {
                let clamped = s.clamp(-1.0, 1.0);
                let val = (clamped * 2_147_483_647.0) as i32;
                data_bytes.extend_from_slice(&val.to_le_bytes());
            }
        } else if format == 3 && bits_per_sample == 32 {
            for &s in samples {
                data_bytes.extend_from_slice(&s.to_le_bytes());
            }
        }
        make_test_wav_from_bytes(sample_rate, channels, bits_per_sample, format, &data_bytes)
    }

    #[test]
    fn wav_header_validation_valid_16k_mono_pcm16() {
        let input_samples = vec![0.0_f32, 0.5, -0.5, 0.99];
        let bytes = make_test_wav(16_000, 1, 16, 1, &input_samples);

        let parsed = parse_wav_bytes(&bytes, "test_pcm16.wav").expect("should parse valid WAV");
        assert_eq!(parsed.sample_rate, 16_000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.samples.len(), input_samples.len());
        assert!((parsed.samples[1] - 0.5).abs() < 1e-4);
        assert!((parsed.samples[2] - (-0.5)).abs() < 1e-4);
    }

    #[test]
    fn wav_header_validation_valid_16k_mono_float32() {
        let input_samples = vec![0.123_f32, -0.456, 0.789];
        let bytes = make_test_wav(16_000, 1, 32, 3, &input_samples);

        let parsed =
            parse_wav_bytes(&bytes, "test_float.wav").expect("should parse valid float WAV");
        assert_eq!(parsed.sample_rate, 16_000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.samples.len(), input_samples.len());
        assert!((parsed.samples[0] - 0.123).abs() < 1e-6);
    }

    #[test]
    fn wav_header_validation_rejects_stereo() {
        let bytes = make_test_wav(16_000, 2, 16, 1, &[0.0, 0.0]);
        let err = parse_wav_bytes(&bytes, "stereo.wav").unwrap_err();
        assert_eq!(
            err,
            WavError::UnsupportedChannels {
                file: "stereo.wav".to_string(),
                channels: 2,
            }
        );
    }

    #[test]
    fn wav_header_validation_rejects_non_16k_rate() {
        let bytes_44k = make_test_wav(44_100, 1, 16, 1, &[0.0, 0.0]);
        let err_44k = parse_wav_bytes(&bytes_44k, "rate44k.wav").unwrap_err();
        assert_eq!(
            err_44k,
            WavError::UnsupportedSampleRate {
                file: "rate44k.wav".to_string(),
                rate: 44_100,
            }
        );

        let bytes_48k = make_test_wav(48_000, 1, 16, 1, &[0.0, 0.0]);
        let err_48k = parse_wav_bytes(&bytes_48k, "rate48k.wav").unwrap_err();
        assert_eq!(
            err_48k,
            WavError::UnsupportedSampleRate {
                file: "rate48k.wav".to_string(),
                rate: 48_000,
            }
        );
    }

    #[test]
    fn wav_header_validation_rejects_corrupted_headers() {
        let bad_riff = b"RIFX0000WAVEfmt ";
        assert!(matches!(
            parse_wav_bytes(bad_riff, "corrupt.wav"),
            Err(WavError::InvalidRiff { .. })
        ));

        let bad_wave = b"RIFF\x04\0\0\0WAV_";
        assert!(matches!(
            parse_wav_bytes(bad_wave, "corrupt.wav"),
            Err(WavError::InvalidWave { .. })
        ));

        let too_short = b"RIFF";
        assert!(matches!(
            parse_wav_bytes(too_short, "short.wav"),
            Err(WavError::TooSmall { .. })
        ));
    }

    #[test]
    fn wav_header_validation_valid_16k_mono_pcm32() {
        let input_samples = vec![0.0_f32, 0.5, -0.5, 0.99];
        let bytes = make_test_wav(16_000, 1, 32, 1, &input_samples);

        let parsed = parse_wav_bytes(&bytes, "test_pcm32.wav").expect("should parse valid WAV");
        assert_eq!(parsed.sample_rate, 16_000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.samples.len(), input_samples.len());
        assert!((parsed.samples[1] - 0.5).abs() < 1e-4);
        assert!((parsed.samples[2] - (-0.5)).abs() < 1e-4);
    }

    #[test]
    fn wav_pcm32_reproducers() {
        // Reproducer A1: -1 sample must not decode to NaN; it scales by 1/2^31.
        let bytes_neg_one = make_test_wav_from_bytes(16_000, 1, 32, 1, &(-1_i32).to_le_bytes());
        let parsed_neg_one = parse_wav_bytes(&bytes_neg_one, "pcm32_neg_one.wav")
            .expect("should parse valid 32-bit PCM -1");
        assert_eq!(parsed_neg_one.samples.len(), 1);
        let s0 = parsed_neg_one.samples[0];
        assert!(!s0.is_nan(), "sample -1 must not be NaN");
        let expected_neg_one = -1.0_f32 / 2_147_483_648.0;
        assert!((s0 - expected_neg_one).abs() < 1e-12);

        // Reproducer A2: 2148 sample must scale by 1/2^31 (approx 1.0e-6), not ~3.0e-42.
        let bytes_2148 = make_test_wav_from_bytes(16_000, 1, 32, 1, &2148_i32.to_le_bytes());
        let parsed_2148 = parse_wav_bytes(&bytes_2148, "pcm32_2148.wav")
            .expect("should parse valid 32-bit PCM 2148");
        assert_eq!(parsed_2148.samples.len(), 1);
        let s1 = parsed_2148.samples[0];
        let expected_2148 = 2148.0_f32 / 2_147_483_648.0;
        assert!((s1 - expected_2148).abs() < 1e-12);
        assert!((s1 - 1.0e-6).abs() < 1e-8);
    }

    #[test]
    fn wav_float_rejects_non_finite_samples() {
        // Reproducer B1: float NaN sample must be rejected.
        let nan_bytes = make_test_wav(16_000, 1, 32, 3, &[f32::NAN]);
        let err_nan = parse_wav_bytes(&nan_bytes, "nan.wav").unwrap_err();
        assert_eq!(
            err_nan,
            WavError::NonFiniteSample {
                file: "nan.wav".to_string(),
            }
        );

        // Reproducer B2: float infinity sample must be rejected.
        let inf_bytes = make_test_wav(16_000, 1, 32, 3, &[f32::INFINITY]);
        let err_inf = parse_wav_bytes(&inf_bytes, "inf.wav").unwrap_err();
        assert_eq!(
            err_inf,
            WavError::NonFiniteSample {
                file: "inf.wav".to_string(),
            }
        );

        let neg_inf_bytes = make_test_wav(16_000, 1, 32, 3, &[f32::NEG_INFINITY]);
        let err_neg_inf = parse_wav_bytes(&neg_inf_bytes, "neg_inf.wav").unwrap_err();
        assert_eq!(
            err_neg_inf,
            WavError::NonFiniteSample {
                file: "neg_inf.wav".to_string(),
            }
        );
    }
}
