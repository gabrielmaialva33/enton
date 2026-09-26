//! Model identity: a voiceprint is only comparable with the model that produced it.

use std::{fs, path::Path};

use sha2::{Digest, Sha256};

/// File name of the default CAM++ model (3D-Speaker, 16 kHz, zh-cn common).
pub const MODEL_FILENAME: &str = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx";

/// Error returned when scoring or evaluating with a model differing from the enrolled voiceprint.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ModelMismatchError {
    /// The enrolled and scoring models differ.
    #[error(
        "model mismatch: voiceprint was enrolled with model '{enrolled_model}' (SHA-256 {enrolled_sha256}), \
             but scoring model is '{scoring_model}' (SHA-256 {scoring_sha256})"
    )]
    Mismatch {
        /// Model used at enrollment.
        enrolled_model: String,
        /// SHA-256 of the enrollment model, in hex.
        enrolled_sha256: String,
        /// Model used for scoring.
        scoring_model: String,
        /// SHA-256 of the scoring model, in hex.
        scoring_sha256: String,
    },
}

/// Compute SHA-256 digest of a file on disk.
pub fn hash_file(path: &Path) -> Result<[u8; 32], std::io::Error> {
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
    Ok(hasher.finalize().into())
}

/// Format byte slice as lowercase hexadecimal string.
#[must_use]
pub fn hex_encode(bytes: &[u8]) -> String {
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
pub fn known_model_name(sha256_hex: &str) -> Option<&'static str> {
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
pub fn describe_model(path: Option<&Path>, sha256: &[u8; 32]) -> String {
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
pub fn verify_model_identity(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice_id::{load_voiceprint, save_voiceprint};
    use std::fs;

    #[test]
    fn model_mismatch_rejected() {
        let tmp_dir =
            std::env::temp_dir().join(format!("enton_test_mismatch_{}", std::process::id()));
        fs::create_dir_all(&tmp_dir).unwrap();
        let model_a_path = tmp_dir.join("model_a.onnx");
        let model_b_path = tmp_dir.join("model_b.onnx");
        fs::write(&model_a_path, b"dummy model A weights 12345").unwrap();
        fs::write(&model_b_path, b"dummy model B weights 67890").unwrap();

        let hash_a = hash_file(&model_a_path).expect("hash model a");
        let hash_b = hash_file(&model_b_path).expect("hash model b");
        assert_ne!(hash_a, hash_b);

        let vp_path = tmp_dir.join("owner.voiceprint");
        let repo_root = tmp_dir.join("repo");
        fs::create_dir_all(&repo_root).unwrap();

        let embedding = vec![0.1_f32; 192];
        save_voiceprint(&vp_path, &embedding, &hash_a, &repo_root).expect("save voiceprint");

        let loaded = load_voiceprint(&vp_path).expect("load voiceprint");
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
    fn hash_file_matches_the_standard_vector() {
        let path = std::env::temp_dir().join(format!("enton_test_sha_{}", std::process::id()));
        fs::write(&path, b"abc").unwrap();
        assert_eq!(
            hex_encode(&hash_file(&path).unwrap()),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        fs::remove_file(&path).unwrap();
    }
}
