//! Owner voice probe: speaker verification measurement on the real microphone.
//!
//! Evaluates CAM++ speaker embedding model from 3D-Speaker (16 kHz) through sherpa-onnx.
//! Measures EER, d', FAR, FRR, throughput, and creates owner voiceprints with mode 0600.

// CLI tool talks to the terminal by design.
#![allow(clippy::print_stdout, clippy::print_stderr)]
// Numerical audio parsing and biometric metric algorithms are self-contained.
#![allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

#[cfg(not(feature = "voice-id"))]
fn main() {
    eprintln!(
        "Enable voice-id feature: cargo run -p enton-adapters --features voice-id --example owner_probe -- <command>"
    );
}

#[cfg(feature = "voice-id")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    enabled::run()
}

#[cfg(feature = "voice-id")]
mod enabled {
    use std::{
        fs::{self, OpenOptions},
        io::Write,
        path::{Path, PathBuf},
        time::{Duration, Instant},
    };

    use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

    pub(crate) const EXPECTED_SAMPLE_RATE: u32 = 16_000;
    pub(crate) const MODEL_FILENAME: &str = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";
    pub(crate) const VOICEPRINT_MAGIC_V1: &[u8; 6] = b"ENVP\x01\0";
    pub(crate) const VOICEPRINT_MAGIC: &[u8; 6] = b"ENVP\x02\0";
    pub(crate) const MIN_FILES_PER_CLASS: usize = 20;

    /// Enrolled owner voiceprint holding model SHA-256 identity and centroid embedding.
    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct Voiceprint {
        /// SHA-256 digest of the model file used during enrollment.
        pub model_sha256: [u8; 32],
        /// L2-normalized centroid embedding vector.
        pub embedding: Vec<f32>,
    }

    /// Errors occurring during voiceprint loading, saving, or parsing.
    #[derive(Debug, thiserror::Error, PartialEq)]
    pub(crate) enum VoiceprintError {
        #[error("voiceprint file '{file}' is too small ({bytes} bytes)")]
        TooSmall { file: String, bytes: usize },
        #[error(
            "outdated voiceprint format (version 1) in '{file}': voiceprint lacks model SHA-256 identity; \
             please re-enroll with 'owner_probe enroll <dir>' to generate a version 2 voiceprint"
        )]
        OutdatedVersion1 { file: String },
        #[error("voiceprint file '{file}' has invalid magic bytes (expected ENVP version 2)")]
        InvalidMagic { file: String },
        #[error(
            "voiceprint file '{file}' size mismatch: expected {expected} bytes for dim {dim}, got {got}"
        )]
        SizeMismatch {
            file: String,
            expected: usize,
            dim: usize,
            got: usize,
        },
        #[error("voiceprint file '{file}' contains zero embedding dimension")]
        ZeroDimension { file: String },
        #[error("cannot save empty voiceprint embedding")]
        EmptyEmbedding,
        #[error("embedding dimension {dim} exceeds u16 limit")]
        DimensionOverflow { dim: usize },
        #[error("failed to read voiceprint file '{file}': {message}")]
        IoRead { file: String, message: String },
        #[error("failed to write voiceprint file '{file}': {message}")]
        IoWrite { file: String, message: String },
        #[error(
            "Biometric safety guard: voiceprint path '{path}' is inside the repository ('{repo_root}'). \
             Biometric voiceprint data must never be saved inside the repository. \
             Please specify an external path such as ~/.cache/enton/owner.voiceprint or /tmp/owner.voiceprint."
        )]
        PathInRepo { path: String, repo_root: String },
        #[error(
            "Biometric safety guard: cannot resolve '{path}' ({message}); refusing to save the voiceprint."
        )]
        GuardUnresolved { path: String, message: String },
    }

    /// Error returned when scoring or evaluating with a model differing from the enrolled voiceprint.
    #[derive(Debug, thiserror::Error, PartialEq)]
    pub(crate) enum ModelMismatchError {
        #[error(
            "model mismatch: voiceprint was enrolled with model '{enrolled_model}' (SHA-256 {enrolled_sha256}), \
             but scoring model is '{scoring_model}' (SHA-256 {scoring_sha256})"
        )]
        Mismatch {
            enrolled_model: String,
            enrolled_sha256: String,
            scoring_model: String,
            scoring_sha256: String,
        },
    }

    /// Errors occurring during cosine similarity computation.
    #[derive(Debug, thiserror::Error, PartialEq)]
    pub(crate) enum CosineSimilarityError {
        #[error(
            "dimension mismatch in cosine similarity: vector 'a' has dim {len_a}, but vector 'b' has dim {len_b}"
        )]
        DimensionMismatch { len_a: usize, len_b: usize },
        #[error("empty vector passed to cosine similarity")]
        EmptyVector,
        #[error(
            "zero norm in cosine similarity: vector norm is zero or near zero (<= f32::EPSILON)"
        )]
        ZeroNorm,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct WavAudio {
        pub sample_rate: u32,
        pub channels: u16,
        pub samples: Vec<f32>,
    }

    #[derive(Debug, thiserror::Error, PartialEq)]
    pub(crate) enum WavError {
        #[error("file '{file}' too small for WAV header ({bytes} bytes)")]
        TooSmall { file: String, bytes: usize },
        #[error("file '{file}' has invalid RIFF header")]
        InvalidRiff { file: String },
        #[error("file '{file}' has invalid WAVE identifier")]
        InvalidWave { file: String },
        #[error("file '{file}' is missing 'fmt ' chunk")]
        MissingFmtChunk { file: String },
        #[error("file '{file}' is missing 'data' chunk")]
        MissingDataChunk { file: String },
        #[error("file '{file}' has 'data' chunk before 'fmt ' chunk")]
        DataBeforeFmt { file: String },
        #[error(
            "unsupported audio format {format} in '{file}': only PCM (1) or IEEE Float (3) supported"
        )]
        UnsupportedFormat { file: String, format: u16 },
        #[error("unsupported channels {channels} in '{file}': only mono (1 channel) is supported")]
        UnsupportedChannels { file: String, channels: u16 },
        #[error("unsupported sample rate {rate} Hz in '{file}': only 16000 Hz is supported")]
        UnsupportedSampleRate { file: String, rate: u32 },
        #[error(
            "unsupported bits per sample {bits} in '{file}': only 16-bit PCM or 32-bit float supported"
        )]
        UnsupportedBitsPerSample { file: String, bits: u16 },
        #[error("truncated chunk or data in '{file}'")]
        TruncatedData { file: String },
        #[error("WAV file '{file}' contains zero audio samples")]
        EmptyAudio { file: String },
    }
    /// Parse and validate a 16 kHz mono WAV byte buffer.
    pub(crate) fn parse_wav_bytes(bytes: &[u8], file_name: &str) -> Result<WavAudio, WavError> {
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
            let id = bytes
                .get(cursor..cursor + 4)
                .ok_or_else(|| WavError::TruncatedData { file: file.clone() })?;
            let sz = u32::from_le_bytes(
                bytes
                    .get(cursor + 4..cursor + 8)
                    .ok_or_else(|| WavError::TruncatedData { file: file.clone() })?
                    .try_into()
                    .map_err(|_| WavError::TruncatedData { file: file.clone() })?,
            ) as usize;
            let start = cursor + 8;
            let end = start.saturating_add(sz);
            if end > bytes.len() {
                return Err(WavError::TruncatedData { file });
            }
            let data = bytes
                .get(start..end)
                .ok_or_else(|| WavError::TruncatedData { file: file.clone() })?;

            if id == b"fmt " {
                if data.len() < 16 {
                    return Err(WavError::TruncatedData { file });
                }
                let fmt = u16::from_le_bytes(
                    data.get(0..2)
                        .ok_or_else(|| WavError::TruncatedData { file: file.clone() })?
                        .try_into()
                        .map_err(|_| WavError::TruncatedData { file: file.clone() })?,
                );
                let ch = u16::from_le_bytes(
                    data.get(2..4)
                        .ok_or_else(|| WavError::TruncatedData { file: file.clone() })?
                        .try_into()
                        .map_err(|_| WavError::TruncatedData { file: file.clone() })?,
                );
                let rate = u32::from_le_bytes(
                    data.get(4..8)
                        .ok_or_else(|| WavError::TruncatedData { file: file.clone() })?
                        .try_into()
                        .map_err(|_| WavError::TruncatedData { file: file.clone() })?,
                );
                let bits = u16::from_le_bytes(
                    data.get(14..16)
                        .ok_or_else(|| WavError::TruncatedData { file: file.clone() })?
                        .try_into()
                        .map_err(|_| WavError::TruncatedData { file: file.clone() })?,
                );

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
            } else if id == b"data" {
                let Some((fmt, _, _, bits)) = fmt_info else {
                    return Err(WavError::DataBeforeFmt { file });
                };
                let mut out = Vec::new();
                if fmt == 1 && bits == 16 {
                    out.reserve(data.len() / 2);
                    for &c in data.as_chunks::<2>().0 {
                        let val = i16::from_le_bytes(c);
                        out.push(f32::from(val) / 32768.0);
                    }
                } else if (fmt == 3 || fmt == 1) && bits == 32 {
                    out.reserve(data.len() / 4);
                    for &c in data.as_chunks::<4>().0 {
                        out.push(f32::from_le_bytes(c));
                    }
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
    /// Compute L2 norm of a slice.
    #[must_use]
    pub(crate) fn l2_norm(vec: &[f32]) -> f32 {
        let sum_sq: f32 = vec.iter().map(|&x| x * x).sum();
        sum_sq.sqrt()
    }

    /// Normalize a vector to unit length in place.
    pub(crate) fn normalize_l2(vec: &mut [f32]) {
        let norm = l2_norm(vec);
        if norm > f32::EPSILON {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Cosine similarity between two vectors.
    ///
    /// Returns `Ok(score)` clamped to `[-1.0, 1.0]`, or [`CosineSimilarityError`] if dimensions mismatch,
    /// either vector is empty, or either vector has zero norm.
    pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, CosineSimilarityError> {
        if a.is_empty() || b.is_empty() {
            return Err(CosineSimilarityError::EmptyVector);
        }
        if a.len() != b.len() {
            return Err(CosineSimilarityError::DimensionMismatch {
                len_a: a.len(),
                len_b: b.len(),
            });
        }
        let norm_a = l2_norm(a);
        let norm_b = l2_norm(b);
        if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
            return Err(CosineSimilarityError::ZeroNorm);
        }
        let denom = norm_a * norm_b;
        if denom <= f32::EPSILON {
            return Err(CosineSimilarityError::ZeroNorm);
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        Ok((dot / denom).clamp(-1.0, 1.0))
    }

    /// Centroid of L2-normalized embeddings, itself L2-normalized.
    pub(crate) fn compute_centroid(embeddings: &[Vec<f32>]) -> Result<Vec<f32>, String> {
        let first = embeddings
            .first()
            .ok_or_else(|| "cannot compute centroid of empty embedding list".to_string())?;
        let dim = first.len();
        if dim == 0 {
            return Err("embedding dimension cannot be zero".to_string());
        }
        let mut sum = vec![0.0_f32; dim];
        for emb in embeddings {
            if emb.len() != dim {
                return Err(format!(
                    "embedding dimension mismatch: expected {dim}, got {}",
                    emb.len()
                ));
            }
            let mut norm_emb = emb.clone();
            normalize_l2(&mut norm_emb);
            for (s, &val) in sum.iter_mut().zip(norm_emb.iter()) {
                *s += val;
            }
        }
        normalize_l2(&mut sum);
        Ok(sum)
    }

    /// Verify target voiceprint path is not located within the repository.
    ///
    /// Fails closed: a repository root or target ancestor that cannot be
    /// resolved is an error, never a pass.
    pub(crate) fn check_path_not_in_repo(
        path: &Path,
        repo_root: &Path,
    ) -> Result<(), VoiceprintError> {
        let unresolved = |at: &Path, error: std::io::Error| VoiceprintError::GuardUnresolved {
            path: at.display().to_string(),
            message: error.to_string(),
        };
        let canonical_repo = repo_root
            .canonicalize()
            .map_err(|error| unresolved(repo_root, error))?;
        let target = if path.is_absolute() {
            path.to_path_buf()
        } else {
            repo_root.join(path)
        };

        let mut check_ancestor = target.as_path();
        while !check_ancestor.exists() {
            match check_ancestor.parent() {
                Some(parent) => check_ancestor = parent,
                None => break,
            }
        }

        let canonical_ancestor = check_ancestor
            .canonicalize()
            .map_err(|error| unresolved(check_ancestor, error))?;
        if canonical_ancestor.starts_with(&canonical_repo) {
            return Err(VoiceprintError::PathInRepo {
                path: path.display().to_string(),
                repo_root: canonical_repo.display().to_string(),
            });
        }
        Ok(())
    }

    /// Save voiceprint to disk with strict Unix mode 0600.
    pub(crate) fn save_voiceprint(
        path: &Path,
        embedding: &[f32],
        model_sha256: &[u8; 32],
        repo_root: &Path,
    ) -> Result<(), VoiceprintError> {
        check_path_not_in_repo(path, repo_root)?;

        if embedding.is_empty() {
            return Err(VoiceprintError::EmptyEmbedding);
        }
        let Ok(dim) = u16::try_from(embedding.len()) else {
            return Err(VoiceprintError::DimensionOverflow {
                dim: embedding.len(),
            });
        };

        let mut bytes = Vec::with_capacity(40 + embedding.len() * 4);
        bytes.extend_from_slice(VOICEPRINT_MAGIC);
        bytes.extend_from_slice(model_sha256);
        bytes.extend_from_slice(&dim.to_le_bytes());
        for val in embedding {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|e| VoiceprintError::IoWrite {
                file: parent.display().to_string(),
                message: e.to_string(),
            })?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let mut options = OpenOptions::new();
            options.write(true).create(true).truncate(true).mode(0o600);
            let mut file = options.open(path).map_err(|e| VoiceprintError::IoWrite {
                file: path.display().to_string(),
                message: format!("failed to open with mode 0600: {e}"),
            })?;
            file.write_all(&bytes)
                .map_err(|e| VoiceprintError::IoWrite {
                    file: path.display().to_string(),
                    message: e.to_string(),
                })?;
            file.sync_all().map_err(|e| VoiceprintError::IoWrite {
                file: path.display().to_string(),
                message: format!("sync failed: {e}"),
            })?;
        }

        #[cfg(not(unix))]
        {
            fs::write(path, &bytes).map_err(|e| VoiceprintError::IoWrite {
                file: path.display().to_string(),
                message: e.to_string(),
            })?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = fs::metadata(path).map_err(|e| VoiceprintError::IoRead {
                file: path.display().to_string(),
                message: e.to_string(),
            })?;
            let mode = meta.permissions().mode() & 0o777;
            if mode != 0o600 {
                fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|e| {
                    VoiceprintError::IoWrite {
                        file: path.display().to_string(),
                        message: format!("failed to enforce 0600 permissions: {e}"),
                    }
                })?;
            }
        }

        Ok(())
    }

    /// Load voiceprint from file.
    pub(crate) fn load_voiceprint(path: &Path) -> Result<Voiceprint, VoiceprintError> {
        let bytes = fs::read(path).map_err(|e| VoiceprintError::IoRead {
            file: path.display().to_string(),
            message: e.to_string(),
        })?;
        parse_voiceprint_bytes(&bytes, &path.display().to_string())
    }

    /// Parse serialized voiceprint bytes.
    pub(crate) fn parse_voiceprint_bytes(
        bytes: &[u8],
        name: &str,
    ) -> Result<Voiceprint, VoiceprintError> {
        let file = name.to_string();
        if bytes.len() < 6 {
            return Err(VoiceprintError::TooSmall {
                file,
                bytes: bytes.len(),
            });
        }
        let magic = bytes.get(0..6).ok_or_else(|| VoiceprintError::TooSmall {
            file: file.clone(),
            bytes: bytes.len(),
        })?;
        if magic == VOICEPRINT_MAGIC_V1 {
            return Err(VoiceprintError::OutdatedVersion1 { file });
        }
        if magic != VOICEPRINT_MAGIC {
            return Err(VoiceprintError::InvalidMagic { file });
        }
        if bytes.len() < 40 {
            return Err(VoiceprintError::TooSmall {
                file,
                bytes: bytes.len(),
            });
        }
        let model_sha256: [u8; 32] = bytes
            .get(6..38)
            .ok_or_else(|| VoiceprintError::TooSmall {
                file: file.clone(),
                bytes: bytes.len(),
            })?
            .try_into()
            .map_err(|_| VoiceprintError::TooSmall {
                file: file.clone(),
                bytes: bytes.len(),
            })?;
        let dim_bytes: [u8; 2] = bytes
            .get(38..40)
            .ok_or_else(|| VoiceprintError::TooSmall {
                file: file.clone(),
                bytes: bytes.len(),
            })?
            .try_into()
            .map_err(|_| VoiceprintError::TooSmall {
                file: file.clone(),
                bytes: bytes.len(),
            })?;
        let dim = u16::from_le_bytes(dim_bytes) as usize;
        if dim == 0 {
            return Err(VoiceprintError::ZeroDimension { file });
        }
        let expected_len = 40 + dim * 4;
        if bytes.len() != expected_len {
            return Err(VoiceprintError::SizeMismatch {
                file,
                expected: expected_len,
                dim,
                got: bytes.len(),
            });
        }
        let mut out = Vec::with_capacity(dim);
        let payload = bytes.get(40..).ok_or_else(|| VoiceprintError::TooSmall {
            file: file.clone(),
            bytes: bytes.len(),
        })?;
        for &chunk in payload.as_chunks::<4>().0 {
            out.push(f32::from_le_bytes(chunk));
        }
        Ok(Voiceprint {
            model_sha256,
            embedding: out,
        })
    }

    /// Minimal, zero-dependency SHA-256 implementation conforming to FIPS 180-4.
    #[derive(Debug, Clone)]
    pub(crate) struct Sha256 {
        state: [u32; 8],
        buffer: [u8; 64],
        buf_len: usize,
        total_len: u64,
    }

    const SHA256_K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    const SHA256_H0: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    impl Sha256 {
        /// Create a new SHA-256 hasher initialized to standard initial state.
        #[must_use]
        pub(crate) fn new() -> Self {
            Self {
                state: SHA256_H0,
                buffer: [0u8; 64],
                buf_len: 0,
                total_len: 0,
            }
        }

        #[allow(clippy::many_single_char_names)] // Standard FIPS 180-4 variable names (a..h, w)
        fn compress_block(&mut self, block: &[u8; 64]) {
            let mut w = [0u32; 64];
            for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
                if let Some(slot) = w.get_mut(i) {
                    *slot = u32::from_be_bytes(*chunk);
                }
            }
            for i in 16..64 {
                let w_i_2 = w.get(i - 2).copied().unwrap_or(0);
                let w_i_7 = w.get(i - 7).copied().unwrap_or(0);
                let w_i_15 = w.get(i - 15).copied().unwrap_or(0);
                let w_i_16 = w.get(i - 16).copied().unwrap_or(0);

                let s1 = w_i_2.rotate_right(17) ^ w_i_2.rotate_right(19) ^ (w_i_2 >> 10);
                let s0 = w_i_15.rotate_right(7) ^ w_i_15.rotate_right(18) ^ (w_i_15 >> 3);
                if let Some(slot) = w.get_mut(i) {
                    *slot = s1.wrapping_add(w_i_7).wrapping_add(s0).wrapping_add(w_i_16);
                }
            }

            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;

            for (&k_t, &w_t) in SHA256_K.iter().zip(w.iter()) {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let ch = (e & f) ^ ((!e) & g);
                let temp1 = h
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(k_t)
                    .wrapping_add(w_t);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let maj = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(maj);

                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }

            let [s0, s1, s2, s3, s4, s5, s6, s7] = self.state;
            self.state = [
                s0.wrapping_add(a),
                s1.wrapping_add(b),
                s2.wrapping_add(c),
                s3.wrapping_add(d),
                s4.wrapping_add(e),
                s5.wrapping_add(f),
                s6.wrapping_add(g),
                s7.wrapping_add(h),
            ];
        }
        /// Update the hasher state with a slice of input bytes.
        pub(crate) fn update(&mut self, data: &[u8]) {
            self.total_len = self.total_len.saturating_add(data.len() as u64);
            let mut offset = 0;
            while offset < data.len() {
                let available = 64 - self.buf_len;
                let to_copy = (data.len() - offset).min(available);
                if let (Some(dest), Some(src)) = (
                    self.buffer.get_mut(self.buf_len..self.buf_len + to_copy),
                    data.get(offset..offset + to_copy),
                ) {
                    dest.copy_from_slice(src);
                }
                self.buf_len += to_copy;
                offset += to_copy;
                if self.buf_len == 64 {
                    let block = self.buffer;
                    self.compress_block(&block);
                    self.buf_len = 0;
                }
            }
        }

        /// Finalize the hash computation, returning the 32-byte digest.
        #[must_use]
        pub(crate) fn finalize(mut self) -> [u8; 32] {
            let bit_len = self.total_len.saturating_mul(8);
            self.update(&[0x80]);
            while self.buf_len != 56 {
                self.update(&[0x00]);
            }
            self.update(&bit_len.to_be_bytes());
            let mut out = [0u8; 32];
            for (i, word) in self.state.iter().enumerate() {
                let bytes = word.to_be_bytes();
                if let Some(slot) = out.get_mut(i * 4..i * 4 + 4) {
                    slot.copy_from_slice(&bytes);
                }
            }
            out
        }

        /// One-shot computation of SHA-256 digest over byte slice.
        #[cfg(test)]
        #[must_use]
        pub(crate) fn digest(data: &[u8]) -> [u8; 32] {
            let mut hasher = Self::new();
            hasher.update(data);
            hasher.finalize()
        }
    }

    impl Default for Sha256 {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Compute SHA-256 digest of a file on disk.
    pub(crate) fn hash_file(path: &Path) -> Result<[u8; 32], std::io::Error> {
        let file = fs::File::open(path)?;
        let mut reader = std::io::BufReader::with_capacity(64 * 1024, file);
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 8192];
        loop {
            use std::io::Read;
            let n = reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            if let Some(chunk) = buf.get(..n) {
                hasher.update(chunk);
            }
        }
        Ok(hasher.finalize())
    }

    /// Format byte slice as lowercase hexadecimal string.
    #[must_use]
    pub(crate) fn hex_encode(bytes: &[u8]) -> String {
        const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            if let Some(&hi) = HEX_DIGITS.get((b >> 4) as usize) {
                s.push(hi as char);
            }
            if let Some(&lo) = HEX_DIGITS.get((b & 0x0f) as usize) {
                s.push(lo as char);
            }
        }
        s
    }

    /// Identify known CAM++ model variants by their SHA-256 digest hex string.
    #[must_use]
    pub(crate) fn known_model_name(sha256_hex: &str) -> Option<&'static str> {
        match sha256_hex {
            "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11" => {
                Some("CAM++ zh-cn_16k-common (3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx)")
            }
            "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b" => {
                Some("CAM++ en_voxceleb_16k (3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx)")
            }
            "aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2" => Some(
                "CAM++ zh_en_16k-common_advanced (3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx)",
            ),
            _ => None,
        }
    }

    /// Describe model identity using known model names or file path.
    #[must_use]
    pub(crate) fn describe_model(path: Option<&Path>, sha256: &[u8; 32]) -> String {
        let hex = hex_encode(sha256);
        if let Some(known) = known_model_name(&hex) {
            if let Some(p) = path {
                format!("{known} at '{}'", p.display())
            } else {
                known.to_string()
            }
        } else if let Some(p) = path {
            format!("'{}' (SHA-256: {hex})", p.display())
        } else {
            format!("unknown model (SHA-256: {hex})")
        }
    }

    /// Verify that the scoring model matches the model used at voiceprint enrollment.
    pub(crate) fn verify_model_identity(
        enrolled_sha256: &[u8; 32],
        scoring_model_path: &Path,
        scoring_sha256: &[u8; 32],
    ) -> Result<(), ModelMismatchError> {
        if enrolled_sha256 != scoring_sha256 {
            let enrolled_hex = hex_encode(enrolled_sha256);
            let scoring_hex = hex_encode(scoring_sha256);
            let enrolled_model = describe_model(None, enrolled_sha256);
            let scoring_model = describe_model(Some(scoring_model_path), scoring_sha256);
            return Err(ModelMismatchError::Mismatch {
                enrolled_model,
                enrolled_sha256: enrolled_hex,
                scoring_model,
                scoring_sha256: scoring_hex,
            });
        }
        Ok(())
    }

    /// Inverse standard normal CDF (Acklam's algorithm).
    #[must_use]
    pub(crate) fn standard_normal_inv_cdf(p: f64) -> f64 {
        const A: [f64; 6] = [
            -3.969_683_028_665_376e+01,
            2.209_460_984_245_205e+02,
            -2.759_285_104_469_687e+02,
            1.383_577_518_672_69e2,
            -3.066_479_806_614_716e+01,
            2.506_628_277_459_239e+00,
        ];
        const B: [f64; 5] = [
            -5.447_609_879_822_406e+01,
            1.615_858_368_580_409e+02,
            -1.556_989_798_598_866e+02,
            6.680_131_188_771_972e+01,
            -1.328_068_155_288_572e+01,
        ];
        const C: [f64; 6] = [
            -7.784_894_002_430_293e-03,
            -3.223_964_580_411_365e-01,
            -2.400_758_277_161_838e+00,
            -2.549_732_539_343_734e+00,
            4.374_664_141_464_968e+00,
            2.938_163_982_698_783e+00,
        ];
        const D: [f64; 4] = [
            7.784_695_709_041_462e-03,
            3.224_671_290_700_398e-01,
            2.445_134_137_142_996e+00,
            3.754_408_661_907_416e+00,
        ];

        let p_clamped = p.clamp(1e-15, 1.0 - 1e-15);
        let p_low = 0.02425_f64;
        let p_high = 1.0 - p_low;

        if p_clamped < p_low {
            let q = (-2.0 * p_clamped.ln()).sqrt();
            let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
            let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
            num / den
        } else if p_clamped <= p_high {
            let q = p_clamped - 0.5;
            let r = q * q;
            let num = (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q;
            let den = ((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0;
            num / den
        } else {
            let q = (-2.0 * (1.0 - p_clamped).ln()).sqrt();
            let num = ((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5];
            let den = (((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0;
            -num / den
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    pub(crate) struct MetricsReport {
        pub target_count: usize,
        pub nontarget_count: usize,
        pub target_mean: f32,
        pub target_std: f32,
        pub target_min: f32,
        pub target_max: f32,
        pub nontarget_mean: f32,
        pub nontarget_std: f32,
        pub nontarget_min: f32,
        pub nontarget_max: f32,
        pub empirical_dprime: f32,
        pub eer: f32,
        pub eer_threshold: f32,
        pub gaussian_dprime: f32,
        pub frr_at_far_2_pct: f32,
        pub frr_at_far_2_pct_thresh: f32,
        pub frr_at_far_6_7_pct: f32,
        pub frr_at_far_6_7_pct_thresh: f32,
        pub far_at_frr_1_pct: f32,
        pub far_at_frr_1_pct_thresh: f32,
        pub total_audio_duration_s: f64,
        pub total_embedding_duration_s: f64,
        pub mean_embedding_time_per_sec_ms: f64,
        pub rtf: f64,
    }

    impl MetricsReport {
        #[must_use]
        pub(crate) fn format_report(&self) -> String {
            let pass = self.gaussian_dprime >= 3.83 && self.eer <= 0.0305;
            let (status_line, interp_line) = if pass {
                (
                    format!(
                        "  Result: PASSED (EER {:.2}% <= 3.0%, d' {:.2} >= 3.83)",
                        self.eer * 100.0,
                        self.gaussian_dprime
                    ),
                    "  Interpretation: Voice verification alone is viable for authenticated attention on this mic.".to_string(),
                )
            } else {
                (
                    format!(
                        "  Result: DEFICIT (EER {:.2}% > 3.0% or d' {:.2} < 3.83)",
                        self.eer * 100.0,
                        self.gaussian_dprime
                    ),
                    "  Interpretation: Voice alone is insufficient. Organism v3 requires multi-modal fusion (temporal prior / DoA) to reach d' >= 3.83.".to_string(),
                )
            };

            let lines = [
                "================================================================================",
                "Enton Owner Voice Probe — Speaker Verification Report (Task 0013)",
                "================================================================================",
                "Counts:",
                &format!("  Target (genuine owner):      {} files", self.target_count),
                &format!(
                    "  Non-target (distractors):    {} files",
                    self.nontarget_count
                ),
                "",
                "Timing & Throughput:",
                &format!(
                    "  Total audio duration:        {:.2} s",
                    self.total_audio_duration_s
                ),
                &format!(
                    "  Total embedding compute:     {:.2} s",
                    self.total_embedding_duration_s
                ),
                &format!(
                    "  Mean compute time / sec:     {:.2} ms / s audio  (RTF = {:.4})",
                    self.mean_embedding_time_per_sec_ms, self.rtf
                ),
                "",
                "Cosine Score Statistics:",
                &format!(
                    "  Target:      mean = {:.4},  std = {:.4}  (min = {:.4}, max = {:.4})",
                    self.target_mean, self.target_std, self.target_min, self.target_max
                ),
                &format!(
                    "  Non-target:  mean = {:.4},  std = {:.4}  (min = {:.4}, max = {:.4})",
                    self.nontarget_mean, self.nontarget_std, self.nontarget_min, self.nontarget_max
                ),
                "",
                "Discriminability & Error Rates:",
                &format!(
                    "  EER (Equal Error Rate):      {:.2}%  (cosine threshold = {:.4})",
                    self.eer * 100.0,
                    self.eer_threshold
                ),
                &format!("  Gaussian d' (from EER):      {:.2}", self.gaussian_dprime),
                &format!(
                    "  Empirical d' (from mean/std): {:.2}",
                    self.empirical_dprime
                ),
                "",
                "Operational Risk Points:",
                &format!(
                    "  FRR at FAR = 2.0%:           {:.2}%  (operational threshold = {:.4})",
                    self.frr_at_far_2_pct * 100.0,
                    self.frr_at_far_2_pct_thresh
                ),
                &format!(
                    "  FRR at FAR = 6.7%:           {:.2}%  (operational threshold = {:.4})",
                    self.frr_at_far_6_7_pct * 100.0,
                    self.frr_at_far_6_7_pct_thresh
                ),
                &format!(
                    "  FAR at FRR = 1.0%:           {:.2}%  (operational threshold = {:.4})",
                    self.far_at_frr_1_pct * 100.0,
                    self.far_at_frr_1_pct_thresh
                ),
                "",
                "Organism v3 Feasibility Gate:",
                "  Target criterion: combined d' >= 3.83 (voice alone: EER <= 3.0%)",
                &status_line,
                &interp_line,
                "================================================================================",
            ];

            let mut out = lines.join("\n");
            out.push('\n');
            out
        }
    }

    fn interpolate_roc(roc: &[(f32, f32, f32)], target: f32, at_far: bool) -> (f32, f32) {
        if roc.is_empty() {
            return (0.0, 0.0);
        }
        let best_thresh = roc
            .iter()
            .find(|p| if at_far { p.1 <= target } else { p.2 >= target })
            .map_or(0.0, |p| p.0);

        for i in 0..roc.len().saturating_sub(1) {
            let Some(&(_, far1, frr1)) = roc.get(i) else {
                break;
            };
            let Some(&(_, far2, frr2)) = roc.get(i + 1) else {
                break;
            };
            let (v1, v2, o1, o2) = if at_far {
                (far1, far2, frr1, frr2)
            } else {
                (frr1, frr2, far1, far2)
            };
            if (v1 >= target && v2 <= target) || (v1 <= target && v2 >= target) {
                let d = v2 - v1;
                if d.abs() > 1e-7 {
                    let alpha = (target - v1) / d;
                    return ((o1 + alpha * (o2 - o1)).clamp(0.0, 1.0), best_thresh);
                }
                return (o1, best_thresh);
            }
        }
        let fallback = if at_far {
            if target >= 1.0 { 0.0 } else { 1.0 }
        } else if target <= 0.0 {
            1.0
        } else {
            0.0
        };
        (fallback, best_thresh)
    }

    fn calc_stats(scores: &[f32]) -> (f32, f32, f32, f32) {
        let n = scores.len();
        let sum: f64 = scores.iter().map(|&x| f64::from(x)).sum();
        let mean = (sum / n as f64) as f32;
        let var: f64 = scores
            .iter()
            .map(|&x| {
                let d = f64::from(x) - f64::from(mean);
                d * d
            })
            .sum::<f64>()
            / (n - 1) as f64;
        let std = var.sqrt() as f32;
        let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (mean, std, min, max)
    }

    /// Calculate verification metrics: EER, empirical and Gaussian d', FAR, FRR.
    pub(crate) fn calculate_verification_metrics(
        target_scores: &[f32],
        nontarget_scores: &[f32],
        total_audio_s: f64,
        total_compute_s: f64,
    ) -> Result<MetricsReport, String> {
        let t_len = target_scores.len();
        let nt_len = nontarget_scores.len();

        if t_len < MIN_FILES_PER_CLASS || nt_len < MIN_FILES_PER_CLASS {
            return Err(format!(
                "Refusing to report: fewer than {MIN_FILES_PER_CLASS} files per class. \
                 Target count: {t_len}, Nontarget count: {nt_len}. Minimum required is {MIN_FILES_PER_CLASS} per class."
            ));
        }

        let (t_mean, t_std, t_min, t_max) = calc_stats(target_scores);
        let (nt_mean, nt_std, nt_min, nt_max) = calc_stats(nontarget_scores);

        let pooled_std = f32::midpoint(t_std * t_std, nt_std * nt_std).sqrt();
        let empirical_dprime = if pooled_std > 1e-6 {
            (t_mean - nt_mean) / pooled_std
        } else {
            0.0
        };

        let mut thresholds: Vec<f32> = Vec::with_capacity(t_len + nt_len + 2);
        thresholds.push(-1.01);
        for &s in target_scores {
            thresholds.push(s);
        }
        for &s in nontarget_scores {
            thresholds.push(s);
        }
        thresholds.push(1.01);
        thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        thresholds.dedup();

        let mut roc: Vec<(f32, f32, f32)> = Vec::with_capacity(thresholds.len());
        for &thresh in &thresholds {
            let far_count = nontarget_scores.iter().filter(|&&s| s >= thresh).count();
            let frr_count = target_scores.iter().filter(|&&s| s < thresh).count();
            let far = far_count as f32 / nt_len as f32;
            let frr = frr_count as f32 / t_len as f32;
            roc.push((thresh, far, frr));
        }

        let mut eer = 0.0_f32;
        let mut eer_threshold = 0.0_f32;
        if t_min > nt_max {
            eer = 0.0;
            eer_threshold = f32::midpoint(nt_max, t_min);
        } else {
            for i in 0..roc.len().saturating_sub(1) {
                let Some(&(th1, far1, frr1)) = roc.get(i) else {
                    break;
                };
                let Some(&(th2, far2, frr2)) = roc.get(i + 1) else {
                    break;
                };
                let diff1 = far1 - frr1;
                let diff2 = far2 - frr2;
                if diff1 >= 0.0 && diff2 <= 0.0 {
                    let denom = (far2 - far1) - (frr2 - frr1);
                    if denom.abs() > 1e-7 {
                        let num = far2 * frr1 - far1 * frr2;
                        let cand_eer = num / denom;
                        eer = cand_eer.clamp(0.0, 1.0);
                        let span = far2 - far1;
                        if span.abs() > 1e-7 {
                            let alpha = (eer - far1) / span;
                            eer_threshold = th1 + alpha * (th2 - th1);
                        } else {
                            eer_threshold = f32::midpoint(th1, th2);
                        }
                    } else {
                        eer = far1;
                        eer_threshold = th1;
                    }
                    break;
                }
            }
        }

        let gaussian_dprime = (2.0 * standard_normal_inv_cdf(1.0 - f64::from(eer))) as f32;

        let (frr_at_far_2_pct, frr_at_far_2_pct_thresh) = interpolate_roc(&roc, 0.02, true);
        let (frr_at_far_6_7_pct, frr_at_far_6_7_pct_thresh) = interpolate_roc(&roc, 0.067, true);
        let (far_at_frr_1_pct, far_at_frr_1_pct_thresh) = interpolate_roc(&roc, 0.01, false);

        let mean_embedding_time_per_sec_ms = if total_audio_s > 0.0 {
            (total_compute_s / total_audio_s) * 1000.0
        } else {
            0.0
        };
        let rtf = if total_audio_s > 0.0 {
            total_compute_s / total_audio_s
        } else {
            0.0
        };

        Ok(MetricsReport {
            target_count: t_len,
            nontarget_count: nt_len,
            target_mean: t_mean,
            target_std: t_std,
            target_min: t_min,
            target_max: t_max,
            nontarget_mean: nt_mean,
            nontarget_std: nt_std,
            nontarget_min: nt_min,
            nontarget_max: nt_max,
            empirical_dprime,
            eer,
            eer_threshold,
            gaussian_dprime,
            frr_at_far_2_pct,
            frr_at_far_2_pct_thresh,
            frr_at_far_6_7_pct,
            frr_at_far_6_7_pct_thresh,
            far_at_frr_1_pct,
            far_at_frr_1_pct_thresh,
            total_audio_duration_s: total_audio_s,
            total_embedding_duration_s: total_compute_s,
            mean_embedding_time_per_sec_ms,
            rtf,
        })
    }

    /// Gather sorted `.wav` files from a directory.
    pub(crate) fn gather_wav_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
        if !dir.exists() {
            return Err(format!("directory '{}' does not exist", dir.display()));
        }
        if !dir.is_dir() {
            return Err(format!("path '{}' is not a directory", dir.display()));
        }
        let mut files = Vec::new();
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("failed to read directory '{}': {e}", dir.display()))?;
        for entry in entries {
            let entry = entry.map_err(|e| format!("failed to read directory entry: {e}"))?;
            let path = entry.path();
            if path.is_file()
                && let Some(ext) = path.extension()
                && ext.eq_ignore_ascii_case("wav")
            {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }

    /// Resolve CAM++ speaker embedding model file path.
    pub(crate) fn resolve_model_path(model_override: Option<&Path>) -> Result<PathBuf, String> {
        if let Some(path) = model_override {
            if path.exists() {
                return Ok(path.to_path_buf());
            }
            return Err(format!(
                "specified model path does not exist: '{}'",
                path.display()
            ));
        }
        if let Ok(env_path) = std::env::var("ENTON_SPEAKER_MODEL") {
            let p = PathBuf::from(env_path);
            if p.exists() {
                return Ok(p);
            }
            return Err(format!(
                "ENTON_SPEAKER_MODEL path does not exist: '{}'",
                p.display()
            ));
        }
        let home = std::env::var("HOME")
            .map_err(|_| "HOME environment variable is not set".to_string())?;
        let default_path = PathBuf::from(home)
            .join(".cache/enton/models")
            .join(MODEL_FILENAME);
        if default_path.exists() {
            return Ok(default_path);
        }

        Err(format!(
            "CAM++ speaker embedding model not found at:\n  {}\n\n\
             Please download the sherpa-onnx 3D-Speaker CAM++ model (16 kHz):\n\
             mkdir -p ~/.cache/enton/models\n\
             curl -L -o ~/.cache/enton/models/{MODEL_FILENAME} \\\n\
               https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{MODEL_FILENAME}\n\n\
             Model metadata:\n\
             • File:    {MODEL_FILENAME}\n\
             • Size:    28,281,138 bytes (~28.3 MB)\n\
             • SHA-256: f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11\n\
             • License: Apache-2.0\n",
            default_path.display()
        ))
    }

    /// Initialize `SpeakerEmbeddingExtractor`.
    pub(crate) fn create_extractor(model_path: &Path) -> Result<SpeakerEmbeddingExtractor, String> {
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model_path.to_string_lossy().into_owned()),
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        };
        SpeakerEmbeddingExtractor::create(&config).ok_or_else(|| {
            format!(
                "failed to initialize SpeakerEmbeddingExtractor with model at '{}'",
                model_path.display()
            )
        })
    }

    /// Extract an L2-normalized embedding and track duration and computation time.
    pub(crate) fn extract_embedding(
        extractor: &SpeakerEmbeddingExtractor,
        wav_path: &Path,
    ) -> Result<(Vec<f32>, f64, Duration), Box<dyn std::error::Error>> {
        let wav_bytes = fs::read(wav_path)
            .map_err(|e| format!("failed to read WAV file {}: {e}", wav_path.display()))?;
        let audio = parse_wav_bytes(&wav_bytes, &wav_path.to_string_lossy())?;

        let audio_duration_sec = audio.samples.len() as f64 / f64::from(audio.sample_rate);

        let stream = extractor
            .create_stream()
            .ok_or("failed to create sherpa-onnx online stream")?;

        let start_time = Instant::now();
        stream.accept_waveform(audio.sample_rate.cast_signed(), &audio.samples);
        stream.input_finished();

        if !extractor.is_ready(&stream) {
            return Err(format!(
                "audio file {} duration ({:.2} s) is too short for CAM++ embedding extraction (< 0.4 s)",
                wav_path.display(),
                audio_duration_sec
            )
            .into());
        }

        let mut raw_emb = extractor
            .compute(&stream)
            .ok_or_else(|| format!("failed to compute embedding for {}", wav_path.display()))?;
        let elapsed = start_time.elapsed();

        normalize_l2(&mut raw_emb);
        Ok((raw_emb, audio_duration_sec, elapsed))
    }

    pub(crate) fn default_voiceprint_path() -> Result<PathBuf, String> {
        let home = std::env::var("HOME")
            .map_err(|_| "HOME environment variable is not set".to_string())?;
        Ok(PathBuf::from(home).join(".cache/enton/owner.voiceprint"))
    }

    pub(crate) fn find_repo_root() -> PathBuf {
        let current = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut dir = current.as_path();
        loop {
            let manifest = dir.join("Cargo.toml");
            if manifest.exists()
                && let Ok(contents) = fs::read_to_string(&manifest)
                && contents.contains("[workspace]")
            {
                return dir.to_path_buf();
            }
            match dir.parent() {
                Some(parent) => dir = parent,
                None => break,
            }
        }
        current
    }

    fn print_usage() {
        println!("enton owner_probe — speaker verification measurement tool");
        println!();
        println!("USAGE:");
        println!("  owner_probe enroll <dir> [--out <voiceprint_file>] [--model <model_file>]");
        println!("  owner_probe score --owner <voiceprint_file> <wav>... [--model <model_file>]");
        println!(
            "  owner_probe eer --owner <voiceprint_file> --target <target_dir> --nontarget <nontarget_dir> [--model <model_file>]"
        );
        println!();
        println!("COMMANDS:");
        println!(
            "  enroll <dir>       Compute centroid of L2-normalized embeddings of WAV files in <dir>,"
        );
        println!(
            "                     writing the voiceprint with permissions 0600 (printed path only)."
        );
        println!(
            "  score              Compute cosine similarity against the enrolled owner voiceprint for WAV files."
        );
        println!("  eer                Compute Equal Error Rate (EER), Gaussian and empirical d',");
        println!(
            "                     FRR at FAR=2% and 6.7%, FAR at FRR=1%, throughput, and feasibility."
        );
        println!("                     (Refuses to report with fewer than 20 files per class).");
        println!();
        println!("OPTIONS:");
        println!(
            "  --owner <path>     Path to enrolled owner voiceprint file (e.g. ~/.cache/enton/owner.voiceprint)."
        );
        println!(
            "  --out <path>       Output path for enrolled voiceprint (must be outside the repository)."
        );
        println!(
            "  --target <dir>     Directory containing genuine owner test WAV clips (>= 20 files)."
        );
        println!(
            "  --nontarget <dir>  Directory containing non-owner distractor WAV clips (>= 20 files)."
        );
        println!(
            "  --model <path>     Path to 3D-Speaker CAM++ ONNX model file (default: ~/.cache/enton/models/{MODEL_FILENAME})."
        );
        println!("  -h, --help         Print this help message.");
    }

    pub(crate) fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
            print_usage();
            return Ok(());
        }

        let repo_root = find_repo_root();
        let command = args.first().map_or("", String::as_str);

        match command {
            "enroll" => {
                let mut dir = None;
                let mut out = None;
                let mut model = None;

                let mut idx = 1;
                while idx < args.len() {
                    let arg = args.get(idx).map_or("", String::as_str);
                    if arg == "--out" {
                        idx += 1;
                        out = args.get(idx).map(PathBuf::from);
                    } else if arg == "--model" {
                        idx += 1;
                        model = args.get(idx).map(PathBuf::from);
                    } else if !arg.starts_with('-') && dir.is_none() {
                        dir = Some(PathBuf::from(arg));
                    } else if !arg.starts_with('-') && out.is_none() {
                        out = Some(PathBuf::from(arg));
                    }
                    idx += 1;
                }

                let dir_path =
                    dir.ok_or("enroll requires a directory argument containing WAV files")?;
                run_enroll(&dir_path, out.as_deref(), model.as_deref(), &repo_root)
            }
            "score" => {
                let mut owner = None;
                let mut model = None;
                let mut wavs = Vec::new();

                let mut idx = 1;
                while idx < args.len() {
                    let arg = args.get(idx).map_or("", String::as_str);
                    if arg == "--owner" {
                        idx += 1;
                        owner = args.get(idx).map(PathBuf::from);
                    } else if arg == "--model" {
                        idx += 1;
                        model = args.get(idx).map(PathBuf::from);
                    } else if !arg.starts_with('-') {
                        wavs.push(PathBuf::from(arg));
                    }
                    idx += 1;
                }

                let owner_path = owner.ok_or("score requires --owner <voiceprint_file>")?;
                if wavs.is_empty() {
                    return Err(
                        "score requires at least one WAV file or directory argument".into(),
                    );
                }
                run_score(&owner_path, &wavs, model.as_deref())
            }
            "eer" => {
                let mut owner = None;
                let mut target = None;
                let mut nontarget = None;
                let mut model = None;

                let mut idx = 1;
                while idx < args.len() {
                    let arg = args.get(idx).map_or("", String::as_str);
                    if arg == "--owner" {
                        idx += 1;
                        owner = args.get(idx).map(PathBuf::from);
                    } else if arg == "--target" {
                        idx += 1;
                        target = args.get(idx).map(PathBuf::from);
                    } else if arg == "--nontarget" {
                        idx += 1;
                        nontarget = args.get(idx).map(PathBuf::from);
                    } else if arg == "--model" {
                        idx += 1;
                        model = args.get(idx).map(PathBuf::from);
                    }
                    idx += 1;
                }

                let owner_path = owner.ok_or("eer requires --owner <voiceprint_file>")?;
                let target_dir = target.ok_or("eer requires --target <target_dir>")?;
                let nontarget_dir =
                    nontarget.ok_or("eer requires --nontarget <nontarget_dir>")?;

                run_eer(&owner_path, &target_dir, &nontarget_dir, model.as_deref())
            }
            other => Err(format!(
                "Unknown command '{other}'. Expected 'enroll', 'score', or 'eer'. Run with --help for usage."
            )
            .into()),
        }
    }

    fn run_enroll(
        dir_path: &Path,
        out_path: Option<&Path>,
        model_override: Option<&Path>,
        repo_root: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let wav_files = gather_wav_files(dir_path)?;
        if wav_files.is_empty() {
            return Err(format!("no WAV files found in directory '{}'", dir_path.display()).into());
        }

        let default_out = default_voiceprint_path()?;
        let out = out_path.unwrap_or(&default_out);
        check_path_not_in_repo(out, repo_root)?;

        let model_path = resolve_model_path(model_override)?;
        let model_sha256 = hash_file(&model_path).map_err(|e| {
            format!(
                "failed to compute SHA-256 for model file '{}': {e}",
                model_path.display()
            )
        })?;
        let extractor = create_extractor(&model_path)?;

        let model_desc = describe_model(Some(&model_path), &model_sha256);
        println!("Enrollment model: {model_desc}");
        println!(
            "Enrolling owner voice from {} files in '{}'...",
            wav_files.len(),
            dir_path.display()
        );

        let mut embeddings = Vec::with_capacity(wav_files.len());
        for file in &wav_files {
            let (emb, duration, elapsed) = extract_embedding(&extractor, file)?;
            let file_display = file.file_name().unwrap_or_default().to_string_lossy();
            println!(
                "  Processed: {file_display} ({duration:.2} s, embedding extracted in {:.1} ms)",
                elapsed.as_secs_f64() * 1000.0
            );
            embeddings.push(emb);
        }

        let centroid = compute_centroid(&embeddings)?;
        save_voiceprint(out, &centroid, &model_sha256, repo_root)?;

        println!("\nEnrolled {} utterances.", wav_files.len());
        println!("Voiceprint written to: {}", out.display());
        Ok(())
    }

    fn run_score(
        owner_path: &Path,
        wav_paths: &[PathBuf],
        model_override: Option<&Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let voiceprint = load_voiceprint(owner_path)?;
        let model_path = resolve_model_path(model_override)?;
        let model_sha256 = hash_file(&model_path).map_err(|e| {
            format!(
                "failed to compute SHA-256 for model file '{}': {e}",
                model_path.display()
            )
        })?;
        verify_model_identity(&voiceprint.model_sha256, &model_path, &model_sha256)?;

        let extractor = create_extractor(&model_path)?;

        let model_desc = describe_model(Some(&model_path), &model_sha256);
        println!("Scoring model: {model_desc}");
        println!("{:<8}  File", "Score");
        println!("{:-<8}  {:-<50}", "", "");

        for path in wav_paths {
            if path.is_dir() {
                let files = gather_wav_files(path)?;
                for file in files {
                    let (emb, _, _) = extract_embedding(&extractor, &file)?;
                    let s = cosine_similarity(&emb, &voiceprint.embedding)?;
                    println!("{s:<8.4}  {}", file.display());
                }
            } else {
                let (emb, _, _) = extract_embedding(&extractor, path)?;
                let s = cosine_similarity(&emb, &voiceprint.embedding)?;
                println!("{s:<8.4}  {}", path.display());
            }
        }
        Ok(())
    }

    fn run_eer(
        owner_path: &Path,
        target_dir: &Path,
        nontarget_dir: &Path,
        model_override: Option<&Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let target_files = gather_wav_files(target_dir)?;
        let nontarget_files = gather_wav_files(nontarget_dir)?;

        if target_files.len() < MIN_FILES_PER_CLASS || nontarget_files.len() < MIN_FILES_PER_CLASS {
            return Err(format!(
                "Refusing to report EER: fewer than {MIN_FILES_PER_CLASS} files per class. \
                 Found {} target files in '{}' and {} nontarget files in '{}'. \
                 Task 0013 mandates >= {MIN_FILES_PER_CLASS} files per class for statistical validity.",
                target_files.len(),
                target_dir.display(),
                nontarget_files.len(),
                nontarget_dir.display()
            )
            .into());
        }

        let voiceprint = load_voiceprint(owner_path)?;
        let model_path = resolve_model_path(model_override)?;
        let model_sha256 = hash_file(&model_path).map_err(|e| {
            format!(
                "failed to compute SHA-256 for model file '{}': {e}",
                model_path.display()
            )
        })?;
        verify_model_identity(&voiceprint.model_sha256, &model_path, &model_sha256)?;

        let extractor = create_extractor(&model_path)?;

        let model_desc = describe_model(Some(&model_path), &model_sha256);
        println!("Evaluation model: {model_desc}");
        println!("Evaluating owner voice verification (Task 0013)...");
        println!(
            "Target files:    {} in '{}'",
            target_files.len(),
            target_dir.display()
        );
        println!(
            "Nontarget files: {} in '{}'",
            nontarget_files.len(),
            nontarget_dir.display()
        );

        let mut target_scores = Vec::with_capacity(target_files.len());
        let mut total_audio_s = 0.0_f64;
        let mut total_compute_s = 0.0_f64;

        for file in &target_files {
            let (emb, dur, elapsed) = extract_embedding(&extractor, file)?;
            total_audio_s += dur;
            total_compute_s += elapsed.as_secs_f64();
            let s = cosine_similarity(&emb, &voiceprint.embedding)?;
            target_scores.push(s);
        }

        let mut nontarget_scores = Vec::with_capacity(nontarget_files.len());
        for file in &nontarget_files {
            let (emb, dur, elapsed) = extract_embedding(&extractor, file)?;
            total_audio_s += dur;
            total_compute_s += elapsed.as_secs_f64();
            let s = cosine_similarity(&emb, &voiceprint.embedding)?;
            nontarget_scores.push(s);
        }

        let report = calculate_verification_metrics(
            &target_scores,
            &nontarget_scores,
            total_audio_s,
            total_compute_s,
        )?;

        println!("\n{}", report.format_report());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::enabled::{
        CosineSimilarityError, ModelMismatchError, Sha256, VOICEPRINT_MAGIC, VOICEPRINT_MAGIC_V1,
        VoiceprintError, WavError, calculate_verification_metrics, check_path_not_in_repo,
        compute_centroid, cosine_similarity, find_repo_root, hex_encode, l2_norm, normalize_l2,
        parse_voiceprint_bytes, parse_wav_bytes, save_voiceprint, standard_normal_inv_cdf,
        verify_model_identity,
    };
    use std::fs;

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
        } else if (format == 3 || format == 1) && bits_per_sample == 32 {
            for &s in samples {
                data_bytes.extend_from_slice(&s.to_le_bytes());
            }
        }

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
        wav.extend_from_slice(&data_bytes);

        wav
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
    fn standard_normal_inv_cdf_known_values() {
        let z_half = standard_normal_inv_cdf(0.5);
        assert!(z_half.abs() < 1e-12);

        let z_8871 = standard_normal_inv_cdf(0.8871);
        let dprime_8871 = 2.0 * z_8871;
        assert!((dprime_8871 - 2.422).abs() < 0.005);

        let dprime_05 = 2.0 * standard_normal_inv_cdf(0.95);
        assert!((dprime_05 - 3.29).abs() < 0.01);

        let dprime_03 = 2.0 * standard_normal_inv_cdf(0.97);
        assert!((dprime_03 - 3.76).abs() < 0.01);

        let dprime_01 = 2.0 * standard_normal_inv_cdf(0.99);
        assert!((dprime_01 - 4.65).abs() < 0.01);
    }
    #[test]
    fn eer_and_dprime_synthetic_perfect_separation() {
        let targets = vec![0.85_f32; 25];
        let nontargets = vec![0.15_f32; 25];

        let report = calculate_verification_metrics(&targets, &nontargets, 50.0, 1.0)
            .expect("should calculate report");

        assert_eq!(report.target_count, 25);
        assert_eq!(report.nontarget_count, 25);
        assert!(report.eer.abs() < f32::EPSILON);
        assert!(report.frr_at_far_2_pct.abs() < f32::EPSILON);
        assert!(report.frr_at_far_6_7_pct.abs() < f32::EPSILON);
        assert!(report.far_at_frr_1_pct.abs() < f32::EPSILON);
        assert!(report.gaussian_dprime > 5.0);
    }

    #[test]
    fn eer_synthetic_known_overlap() {
        let mut targets = vec![0.60_f32; 20];
        targets.extend(vec![0.80_f32; 20]);

        let mut nontargets = vec![0.40_f32; 20];
        nontargets.extend(vec![0.60_f32; 20]);

        let report = calculate_verification_metrics(&targets, &nontargets, 80.0, 1.0)
            .expect("should calculate report");

        assert!((report.eer - 0.25).abs() < 0.01);
        assert!((report.gaussian_dprime - 1.35).abs() < 0.05);
    }

    #[test]
    fn refuse_fewer_than_20_files_per_class() {
        let targets_19 = vec![0.8_f32; 19];
        let nontargets_20 = vec![0.2_f32; 20];

        let err =
            calculate_verification_metrics(&targets_19, &nontargets_20, 10.0, 0.1).unwrap_err();
        assert!(err.contains("fewer than 20 files per class"));

        let targets_20 = vec![0.8_f32; 20];
        let nontargets_19 = vec![0.2_f32; 19];

        let err2 =
            calculate_verification_metrics(&targets_20, &nontargets_19, 10.0, 0.1).unwrap_err();
        assert!(err2.contains("fewer than 20 files per class"));
    }

    #[test]
    fn repo_path_guard_fails_closed_on_an_unresolvable_root() {
        let missing_root =
            std::env::temp_dir().join(format!("enton_missing_repo_{}", std::process::id()));
        assert!(!missing_root.exists());
        let err = check_path_not_in_repo(&missing_root.join("owner.voiceprint"), &missing_root)
            .unwrap_err();
        assert!(matches!(err, VoiceprintError::GuardUnresolved { .. }));
    }

    #[test]
    fn file_mode_0600_and_repo_path_guard() {
        let tmp_dir = std::env::temp_dir().join(format!("enton_test_probe_{}", std::process::id()));
        fs::create_dir_all(&tmp_dir).unwrap();
        let vp_path = tmp_dir.join("test_owner.voiceprint");

        // Resolved at run time like the CLI does: a compile-time CARGO_MANIFEST_DIR
        // goes stale when the checkout moves and Cargo reuses the cached test binary.
        let repo_root = find_repo_root();

        let in_repo_path = repo_root.join("test_voiceprint_in_repo.bin");
        let guard_err = check_path_not_in_repo(&in_repo_path, &repo_root).unwrap_err();
        assert!(guard_err.to_string().contains("Biometric safety guard"));

        let dummy_model_hash = [0x5au8; 32];
        let embedding = vec![0.5_f32; 192];
        save_voiceprint(&vp_path, &embedding, &dummy_model_hash, &repo_root)
            .expect("should save outside repo");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let meta = fs::metadata(&vp_path).expect("read metadata");
            let mode = meta.permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "file mode must be exactly 0600");
        }

        let loaded = super::enabled::load_voiceprint(&vp_path).expect("should load voiceprint");
        assert_eq!(loaded.embedding.len(), 192);
        assert_eq!(loaded.embedding, embedding);
        assert_eq!(loaded.model_sha256, dummy_model_hash);

        drop(fs::remove_file(&vp_path));
        drop(fs::remove_dir(&tmp_dir));
    }

    #[test]
    fn l2_normalization_and_cosine_similarity() {
        let mut v = vec![3.0_f32, 4.0_f32];
        assert!((l2_norm(&v) - 5.0).abs() < 1e-6);

        normalize_l2(&mut v);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);

        let identical = cosine_similarity(&v, &v).expect("identical vectors");
        assert!((identical - 1.0).abs() < 1e-6);

        let orthogonal = vec![-0.8_f32, 0.6_f32];
        assert!(
            cosine_similarity(&v, &orthogonal)
                .expect("orthogonal")
                .abs()
                < 1e-6
        );

        let opposite = vec![-0.6_f32, -0.8_f32];
        assert!((cosine_similarity(&v, &opposite).expect("opposite") - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn centroid_computation() {
        let v1 = vec![1.0_f32, 0.0_f32];
        let v2 = vec![0.0_f32, 1.0_f32];
        let centroid = compute_centroid(&[v1, v2]).expect("centroid of two orthogonal vectors");

        assert_eq!(centroid.len(), 2);
        assert!((l2_norm(&centroid) - 1.0).abs() < 1e-6);
        let expected = 1.0_f32 / 2.0_f32.sqrt();
        assert!((centroid[0] - expected).abs() < 1e-6);
        assert!((centroid[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn voiceprint_serialization_roundtrip() {
        let dim = 192_usize;
        let original: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
        let dummy_hash = [0x77u8; 32];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(VOICEPRINT_MAGIC);
        bytes.extend_from_slice(&dummy_hash);
        bytes.extend_from_slice(&u16::try_from(dim).unwrap().to_le_bytes());
        for &val in &original {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        let parsed =
            parse_voiceprint_bytes(&bytes, "test.vp").expect("should parse voiceprint bytes");
        assert_eq!(parsed.model_sha256, dummy_hash);
        assert_eq!(parsed.embedding, original);

        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            parse_voiceprint_bytes(&bad_magic, "bad.vp"),
            Err(VoiceprintError::InvalidMagic { .. })
        ));

        let truncated = &bytes[0..bytes.len() - 4];
        assert!(matches!(
            parse_voiceprint_bytes(truncated, "trunc.vp"),
            Err(VoiceprintError::SizeMismatch { .. })
        ));
    }
    #[test]
    fn dim_mismatch_and_zero_norm_are_errors() {
        let a = [1.0_f32, 2.0_f32];
        let b = [1.0_f32, 2.0_f32, 3.0_f32];
        assert_eq!(
            cosine_similarity(&a, &b),
            Err(CosineSimilarityError::DimensionMismatch { len_a: 2, len_b: 3 })
        );

        let zero_vec = [0.0_f32, 0.0_f32];
        let norm_vec = [1.0_f32, 2.0_f32];
        assert_eq!(
            cosine_similarity(&zero_vec, &norm_vec),
            Err(CosineSimilarityError::ZeroNorm)
        );
        assert_eq!(
            cosine_similarity(&norm_vec, &zero_vec),
            Err(CosineSimilarityError::ZeroNorm)
        );

        assert_eq!(
            cosine_similarity(&[], &[]),
            Err(CosineSimilarityError::EmptyVector)
        );
    }

    #[test]
    fn old_format_rejected() {
        let dim = 192_u16;
        let mut old_bytes = Vec::new();
        old_bytes.extend_from_slice(VOICEPRINT_MAGIC_V1);
        old_bytes.extend_from_slice(&dim.to_le_bytes());
        for _ in 0..dim {
            old_bytes.extend_from_slice(&0.5_f32.to_le_bytes());
        }

        let err = parse_voiceprint_bytes(&old_bytes, "legacy_owner.voiceprint").unwrap_err();
        assert!(matches!(err, VoiceprintError::OutdatedVersion1 { .. }));
        let msg = err.to_string();
        assert!(msg.contains("version 1"));
        assert!(msg.contains("lacks model SHA-256 identity"));
        assert!(msg.contains("re-enroll"));
    }

    #[test]
    fn model_mismatch_rejected() {
        let tmp_dir =
            std::env::temp_dir().join(format!("enton_test_mismatch_{}", std::process::id()));
        fs::create_dir_all(&tmp_dir).unwrap();
        let model_a_path = tmp_dir.join("model_a.onnx");
        let model_b_path = tmp_dir.join("model_b.onnx");
        fs::write(&model_a_path, b"dummy model A weights 12345").unwrap();
        fs::write(&model_b_path, b"dummy model B weights 67890").unwrap();

        let hash_a = super::enabled::hash_file(&model_a_path).expect("hash model a");
        let hash_b = super::enabled::hash_file(&model_b_path).expect("hash model b");
        assert_ne!(hash_a, hash_b);

        let vp_path = tmp_dir.join("owner.voiceprint");
        let repo_root = find_repo_root();

        let embedding = vec![0.1_f32; 192];
        save_voiceprint(&vp_path, &embedding, &hash_a, &repo_root).expect("save voiceprint");

        let loaded = super::enabled::load_voiceprint(&vp_path).expect("load voiceprint");
        assert_eq!(loaded.model_sha256, hash_a);

        // Scoring with matching model succeeds
        assert!(verify_model_identity(&loaded.model_sha256, &model_a_path, &hash_a).is_ok());

        // Scoring with different model must return typed ModelMismatch error naming both models
        let err = verify_model_identity(&loaded.model_sha256, &model_b_path, &hash_b).unwrap_err();
        match &err {
            ModelMismatchError::Mismatch {
                enrolled_model: _,
                enrolled_sha256,
                scoring_model,
                scoring_sha256,
            } => {
                assert_eq!(enrolled_sha256, &hex_encode(&hash_a));
                assert_eq!(scoring_sha256, &hex_encode(&hash_b));
                assert!(scoring_model.contains(&model_b_path.display().to_string()));
            }
        }
        let err_msg = err.to_string();
        assert!(err_msg.contains("model mismatch"));
        assert!(err_msg.contains(&hex_encode(&hash_a)));
        assert!(err_msg.contains(&hex_encode(&hash_b)));

        drop(fs::remove_file(&vp_path));
        drop(fs::remove_file(&model_a_path));
        drop(fs::remove_file(&model_b_path));
        drop(fs::remove_dir(&tmp_dir));
    }

    #[test]
    fn sha256_standard_vectors() {
        let empty_hash = Sha256::digest(b"");
        assert_eq!(
            hex_encode(&empty_hash),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );

        let abc_hash = Sha256::digest(b"abc");
        assert_eq!(
            hex_encode(&abc_hash),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let msg56 = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
        let msg56_hash = Sha256::digest(msg56);
        assert_eq!(
            hex_encode(&msg56_hash),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }
}
