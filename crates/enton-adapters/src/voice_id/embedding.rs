//! Speaker embeddings: CAM++ extraction through sherpa-onnx and vector arithmetic.

use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

use sherpa_onnx::{SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig};

use super::wav::parse_wav_bytes;

/// Errors occurring during cosine similarity computation.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CosineSimilarityError {
    /// The vectors differ in length.
    #[error(
        "dimension mismatch in cosine similarity: vector 'a' has dim {len_a}, but vector 'b' has dim {len_b}"
    )]
    DimensionMismatch {
        /// Length of the first vector.
        len_a: usize,
        /// Length of the second vector.
        len_b: usize,
    },
    /// A vector is empty.
    #[error("empty vector passed to cosine similarity")]
    EmptyVector,
    /// A vector has zero (or near zero) norm.
    #[error("zero norm in cosine similarity: vector norm is zero or near zero (<= f32::EPSILON)")]
    ZeroNorm,
}

/// Compute L2 norm of a slice.
#[must_use]
pub fn l2_norm(vec: &[f32]) -> f32 {
    let sum_sq: f32 = vec.iter().map(|&x| x * x).sum();
    sum_sq.sqrt()
}

/// Normalize a vector to unit length in place.
pub fn normalize_l2(vec: &mut [f32]) {
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
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, CosineSimilarityError> {
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
pub fn compute_centroid(embeddings: &[Vec<f32>]) -> Result<Vec<f32>, String> {
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

/// Initialize `SpeakerEmbeddingExtractor`.
pub fn create_extractor(model_path: &Path) -> Result<SpeakerEmbeddingExtractor, String> {
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
pub fn extract_embedding(
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
