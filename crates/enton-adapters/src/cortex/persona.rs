//! The persona: the system prompt that gives the cortex Enton's voice.
//!
//! By default it is [`DEFAULT_SYSTEM_PROMPT`], compiled into the binary. The
//! owner can replace it with a Markdown file they edit by hand (the binary looks
//! for `PERSONA.md` in its config directory), capped at [`MAX_PERSONA_BYTES`]:
//! a larger file is refused with a clear error, never truncated.
//!
//! # Security: Enton only ever reads the persona
//!
//! The persona is read once, at startup, and the cortex keeps its own copy of
//! the text for the whole run. Nothing in Enton writes the file: no code path
//! opens it for writing, no tool exposes it, and nothing the cortex says can
//! reach it. This is deliberate. `OpenClaw` injects a user-editable `SOUL.md` into
//! every turn, and its agent can rewrite that file too; attackers exploited
//! this: a zero-click prompt injection rewrote `SOUL.md` every two minutes to
//! keep control of the agent, and a bundled hook could swap the file silently.
//! A persona the agent can write lets whoever reaches its input stay in charge
//! of every later turn. Enton's soul is its hash-chained event log, not this
//! file, and the log records which persona each thought was asked with as a
//! SHA-256 hash, so a changed persona shows up in `enton why`.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::config::DEFAULT_SYSTEM_PROMPT;

/// The largest persona file Enton reads, in bytes: 8 KiB, about 2,000 tokens,
/// half of the cortex's default 4,096-token context, so the conversation still fits.
pub const MAX_PERSONA_BYTES: u64 = 8 * 1024;

/// Where the text of a persona came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersonaOrigin {
    /// [`DEFAULT_SYSTEM_PROMPT`], compiled into the binary.
    BuiltIn,
    /// A file the owner wrote.
    File(PathBuf),
}

/// A persona, loaded once and never written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Persona {
    text: String,
    origin: PersonaOrigin,
    sha256: [u8; 32],
    bytes: u64,
}

/// Why a persona file could not be used.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PersonaError {
    /// Nothing is at the path.
    #[error("no persona file at {}", .0.display())]
    Missing(PathBuf),
    /// The path names a directory, a pipe or a device instead of a file.
    #[error("the persona at {} is not a regular file", .0.display())]
    NotAFile(PathBuf),
    /// The file is over [`MAX_PERSONA_BYTES`].
    #[error(
        "the persona file {} is {bytes} bytes, over the {MAX_PERSONA_BYTES}-byte (8 KiB) cap: shorten it (Enton never truncates a persona)",
        path.display()
    )]
    TooLarge {
        /// The file.
        path: PathBuf,
        /// Its size.
        bytes: u64,
    },
    /// The file is not UTF-8 text.
    #[error("the persona file {} is not UTF-8 text", .0.display())]
    NotText(PathBuf),
    /// The file holds nothing but whitespace.
    #[error(
        "the persona file {} is empty: write a persona in it, or delete it to use the built-in one",
        .0.display()
    )]
    Empty(PathBuf),
    /// Reading the file failed.
    #[error("cannot read the persona file {}: {source}", path.display())]
    Unreadable {
        /// The file.
        path: PathBuf,
        /// What failed.
        source: std::io::Error,
    },
}

impl Persona {
    /// The built-in persona, [`DEFAULT_SYSTEM_PROMPT`].
    #[must_use]
    pub fn built_in() -> Self {
        let bytes = DEFAULT_SYSTEM_PROMPT.as_bytes();
        Self {
            text: DEFAULT_SYSTEM_PROMPT.to_owned(),
            origin: PersonaOrigin::BuiltIn,
            sha256: Sha256::digest(bytes).into(),
            bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        }
    }

    /// Read the persona at `path`, opened read-only.
    ///
    /// The hash covers the file's bytes exactly as stored, so `sha256sum` prints
    /// the same one; the cortex gets the text without its trailing whitespace.
    /// Fails with [`PersonaError::Missing`] when nothing is there,
    /// [`PersonaError::TooLarge`] past [`MAX_PERSONA_BYTES`], and
    /// [`PersonaError::NotText`] or [`PersonaError::Empty`] when the file holds
    /// no usable text.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PersonaError> {
        let path = path.as_ref();
        let unreadable = |source| PersonaError::Unreadable {
            path: path.to_path_buf(),
            source,
        };
        // Checked before opening: opening a named pipe would block until a writer comes.
        let meta = std::fs::metadata(path).map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                PersonaError::Missing(path.to_path_buf())
            } else {
                unreadable(err)
            }
        })?;
        if !meta.is_file() {
            return Err(PersonaError::NotAFile(path.to_path_buf()));
        }
        let too_large = |bytes| PersonaError::TooLarge {
            path: path.to_path_buf(),
            bytes,
        };
        if meta.len() > MAX_PERSONA_BYTES {
            return Err(too_large(meta.len()));
        }
        let mut raw = Vec::new();
        File::open(path)
            .and_then(|file| file.take(MAX_PERSONA_BYTES + 1).read_to_end(&mut raw))
            .map_err(unreadable)?;
        let bytes = u64::try_from(raw.len()).unwrap_or(u64::MAX);
        // The file grew between the size check and the read.
        if bytes > MAX_PERSONA_BYTES {
            return Err(too_large(bytes));
        }
        let sha256 = Sha256::digest(&raw).into();
        let text = String::from_utf8(raw).map_err(|_| PersonaError::NotText(path.to_path_buf()))?;
        let text = text.trim_end();
        if text.trim_start().is_empty() {
            return Err(PersonaError::Empty(path.to_path_buf()));
        }
        Ok(Self {
            text: text.to_owned(),
            origin: PersonaOrigin::File(path.to_path_buf()),
            sha256,
            bytes,
        })
    }

    /// The persona at `path`, or the built-in one when no file is there. Any
    /// other failure is an error: a persona the owner wrote is never skipped
    /// silently.
    pub fn from_file_or_built_in(path: impl AsRef<Path>) -> Result<Self, PersonaError> {
        match Self::from_file(path) {
            Err(PersonaError::Missing(_)) => Ok(Self::built_in()),
            loaded => loaded,
        }
    }

    /// The text the cortex is given as its system prompt.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Where the text came from.
    #[must_use]
    pub const fn origin(&self) -> &PersonaOrigin {
        &self.origin
    }

    /// The SHA-256 of the persona's bytes: of the file as stored, or of
    /// [`DEFAULT_SYSTEM_PROMPT`].
    #[must_use]
    pub const fn sha256(&self) -> [u8; 32] {
        self.sha256
    }

    /// The length of those bytes.
    #[must_use]
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// What the soul keeps of a persona: its hash, length and origin, never its text.
#[cfg(feature = "soul")]
impl From<&Persona> for crate::soul::PersonaDigest {
    fn from(persona: &Persona) -> Self {
        Self {
            sha256: persona.sha256,
            bytes: persona.bytes,
            source: match persona.origin {
                PersonaOrigin::BuiltIn => crate::soul::PersonaSource::BuiltIn,
                PersonaOrigin::File(_) => crate::soul::PersonaSource::File,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT: AtomicU64 = AtomicU64::new(0);

    /// A scratch directory, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new() -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            let path =
                std::env::temp_dir().join(format!("enton-persona-{}-{id}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn file(&self, contents: &[u8]) -> PathBuf {
            let path = self.0.join("PERSONA.md");
            std::fs::write(&path, contents).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            drop(std::fs::remove_dir_all(&self.0));
        }
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    #[test]
    fn without_a_file_the_built_in_persona_speaks() {
        let scratch = Scratch::new();
        let absent = scratch.0.join("PERSONA.md");
        let persona = Persona::from_file_or_built_in(&absent).unwrap();
        assert_eq!(persona, Persona::built_in());
        assert_eq!(persona.origin(), &PersonaOrigin::BuiltIn);
        assert_eq!(persona.text(), DEFAULT_SYSTEM_PROMPT);
        assert_eq!(persona.sha256(), sha256(DEFAULT_SYSTEM_PROMPT.as_bytes()));
        assert_eq!(persona.bytes(), DEFAULT_SYSTEM_PROMPT.len() as u64);
        // Asked for by name, a missing file is an error, not the default.
        assert!(matches!(
            Persona::from_file(&absent),
            Err(PersonaError::Missing(path)) if path == absent
        ));
        assert!(!absent.exists(), "looking created the file");
    }

    #[test]
    fn a_file_is_hashed_as_stored_and_sent_without_trailing_whitespace() {
        let scratch = Scratch::new();
        let contents = "Você é o Enton.\n\nResponda curto.\n\n".as_bytes();
        let path = scratch.file(contents);
        let persona = Persona::from_file_or_built_in(&path).unwrap();
        assert_eq!(persona.origin(), &PersonaOrigin::File(path.clone()));
        assert_eq!(persona.text(), "Você é o Enton.\n\nResponda curto.");
        assert_eq!(persona.sha256(), sha256(contents));
        assert_eq!(persona.bytes(), contents.len() as u64);
        assert_ne!(persona.sha256(), Persona::built_in().sha256());
        // Reading never changes the file.
        assert_eq!(std::fs::read(&path).unwrap(), contents);
    }

    #[test]
    fn a_persona_over_the_cap_is_refused_not_truncated() {
        let scratch = Scratch::new();
        let cap = usize::try_from(MAX_PERSONA_BYTES).unwrap();
        let exact = scratch.file(&vec![b'a'; cap]);
        assert_eq!(
            Persona::from_file(&exact).unwrap().bytes(),
            MAX_PERSONA_BYTES
        );

        let over = scratch.file(&vec![b'a'; cap + 1]);
        let err = Persona::from_file_or_built_in(&over).unwrap_err();
        assert!(
            matches!(err, PersonaError::TooLarge { bytes, .. } if bytes == MAX_PERSONA_BYTES + 1),
            "{err}"
        );
        let message = err.to_string();
        assert!(message.contains("8193 bytes"), "{message}");
        assert!(message.contains("8192-byte (8 KiB) cap"), "{message}");
        assert!(message.contains("never truncates"), "{message}");
    }

    #[test]
    fn a_file_without_usable_text_is_an_error() {
        let scratch = Scratch::new();
        assert!(matches!(
            Persona::from_file_or_built_in(scratch.file(b" \n\t\n")),
            Err(PersonaError::Empty(_))
        ));
        assert!(matches!(
            Persona::from_file_or_built_in(scratch.file(&[0xff, 0xfe, b'a'])),
            Err(PersonaError::NotText(_))
        ));
        assert!(matches!(
            Persona::from_file_or_built_in(&scratch.0),
            Err(PersonaError::NotAFile(_))
        ));
    }
}
