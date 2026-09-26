//! The owner voiceprint file: format, a mode-0600 writer, and a guard that refuses
//! to store biometric data inside the repository.

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
};

/// Magic bytes of the obsolete version 1 format, which lacked the model identity.
const VOICEPRINT_MAGIC_V1: &[u8; 6] = b"ENVP\x01\0";
/// Magic bytes of the current (version 2) voiceprint format.
pub const VOICEPRINT_MAGIC: &[u8; 6] = b"ENVP\x02\0";

/// Enrolled owner voiceprint holding model SHA-256 identity and centroid embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct Voiceprint {
    /// SHA-256 digest of the model file used during enrollment.
    pub model_sha256: [u8; 32],
    /// L2-normalized centroid embedding vector.
    pub embedding: Vec<f32>,
}

/// Errors occurring during voiceprint loading, saving, or parsing.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum VoiceprintError {
    /// The file is shorter than a voiceprint header.
    #[error("voiceprint file '{file}' is too small ({bytes} bytes)")]
    TooSmall {
        /// File the error refers to.
        file: String,
        /// Length of the input, in bytes.
        bytes: usize,
    },
    /// A version 1 voiceprint, which lacks the model identity.
    #[error(
        "outdated voiceprint format (version 1) in '{file}': voiceprint lacks model SHA-256 identity; \
             please re-enroll with 'owner_probe enroll <dir>' to generate a version 2 voiceprint"
    )]
    OutdatedVersion1 {
        /// File the error refers to.
        file: String,
    },
    /// The file does not start with the voiceprint magic bytes.
    #[error("voiceprint file '{file}' has invalid magic bytes (expected ENVP version 2)")]
    InvalidMagic {
        /// File the error refers to.
        file: String,
    },
    /// The file length does not match its declared dimension.
    #[error(
        "voiceprint file '{file}' size mismatch: expected {expected} bytes for dim {dim}, got {got}"
    )]
    SizeMismatch {
        /// File the error refers to.
        file: String,
        /// Expected file length, in bytes.
        expected: usize,
        /// Declared or attempted embedding dimension.
        dim: usize,
        /// Actual file length, in bytes.
        got: usize,
    },
    /// The header declares an empty embedding.
    #[error("voiceprint file '{file}' contains zero embedding dimension")]
    ZeroDimension {
        /// File the error refers to.
        file: String,
    },
    /// An empty embedding cannot be saved.
    #[error("cannot save empty voiceprint embedding")]
    EmptyEmbedding,
    /// The embedding is too long for the 16-bit dimension field.
    #[error("embedding dimension {dim} exceeds u16 limit")]
    DimensionOverflow {
        /// Declared or attempted embedding dimension.
        dim: usize,
    },
    /// Reading the file failed.
    #[error("failed to read voiceprint file '{file}': {message}")]
    IoRead {
        /// File the error refers to.
        file: String,
        /// Underlying error message.
        message: String,
    },
    /// Writing the file failed.
    #[error("failed to write voiceprint file '{file}': {message}")]
    IoWrite {
        /// File the error refers to.
        file: String,
        /// Underlying error message.
        message: String,
    },
    /// The target path lies inside the repository.
    #[error(
        "Biometric safety guard: voiceprint path '{path}' is inside the repository ('{repo_root}'). \
             Biometric voiceprint data must never be saved inside the repository. \
             Please specify an external path such as ~/.cache/enton/owner.voiceprint or /tmp/owner.voiceprint."
    )]
    PathInRepo {
        /// Path the guard examined.
        path: String,
        /// Canonical repository root.
        repo_root: String,
    },
    /// The repository guard could not resolve a path, so it refused.
    #[error(
        "Biometric safety guard: cannot resolve '{path}' ({message}); refusing to save the voiceprint."
    )]
    GuardUnresolved {
        /// Path the guard examined.
        path: String,
        /// Underlying error message.
        message: String,
    },
}

/// Verify target voiceprint path is not located within the repository.
///
/// Fails closed: a repository root or target ancestor that cannot be
/// resolved is an error, never a pass.
pub fn check_path_not_in_repo(path: &Path, repo_root: &Path) -> Result<(), VoiceprintError> {
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
pub fn save_voiceprint(
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
pub fn load_voiceprint(path: &Path) -> Result<Voiceprint, VoiceprintError> {
    let bytes = fs::read(path).map_err(|e| VoiceprintError::IoRead {
        file: path.display().to_string(),
        message: e.to_string(),
    })?;
    parse_voiceprint_bytes(&bytes, &path.display().to_string())
}

/// Parse serialized voiceprint bytes.
pub fn parse_voiceprint_bytes(bytes: &[u8], name: &str) -> Result<Voiceprint, VoiceprintError> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
        let repo_root = tmp_dir.join("repo");
        fs::create_dir_all(&repo_root).unwrap();

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

        let loaded = load_voiceprint(&vp_path).expect("should load voiceprint");
        assert_eq!(loaded.embedding.len(), 192);
        assert_eq!(loaded.embedding, embedding);
        assert_eq!(loaded.model_sha256, dummy_model_hash);

        drop(fs::remove_file(&vp_path));
        drop(fs::remove_dir(&tmp_dir));
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
}
