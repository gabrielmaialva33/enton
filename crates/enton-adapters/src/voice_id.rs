//! Owner voice identity (speaker verification) on top of CAM++ embeddings.
//!
//! The `owner_probe` example is the command line over this module: it enrolls a
//! voiceprint, scores recordings against it and measures EER and d' on the real
//! microphone. The same pieces will feed `SpeechCue::speaker_sim` at runtime.

mod embedding;
mod metrics;
mod model;
mod voiceprint;
mod wav;

pub use embedding::{
    CosineSimilarityError, compute_centroid, cosine_similarity, create_extractor,
    extract_embedding, l2_norm, normalize_l2,
};
pub use metrics::{
    MIN_FILES_PER_CLASS, MetricsReport, calculate_verification_metrics, standard_normal_inv_cdf,
};
pub use model::{
    MODEL_FILENAME, ModelMismatchError, describe_model, hash_file, hex_encode, known_model_name,
    verify_model_identity,
};
pub use voiceprint::{
    VOICEPRINT_MAGIC, Voiceprint, VoiceprintError, check_path_not_in_repo, load_voiceprint,
    parse_voiceprint_bytes, save_voiceprint,
};
pub use wav::{WavAudio, WavError, parse_wav_bytes};

/// Sample rate every recording and embedding works at, in hertz.
pub const EXPECTED_SAMPLE_RATE: u32 = 16_000;
