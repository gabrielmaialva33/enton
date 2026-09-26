use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use cpal::SampleFormat;

/// Errors from voice configuration, synthesis, and playback.
#[derive(Debug, thiserror::Error)]
pub enum VoiceError {
    /// A required model asset is missing or has the wrong file type.
    #[error(
        "voice error: {asset} not found: {path}. See the README (Voice and microphone) for the model layout."
    )]
    MissingAsset {
        /// Name of the model asset.
        asset: &'static str,
        /// Configured file or directory path.
        path: PathBuf,
    },
    /// A required native model path is not UTF-8.
    #[error("voice error: invalid UTF-8 in {0}")]
    InvalidPath(&'static str),
    /// A voice model name other than `kokoro` or `piper`.
    #[error("voice error: unknown voice model {0:?} (expected kokoro or piper)")]
    UnknownModel(String),
    /// Sentence queue capacity is outside 1..=64.
    #[error("voice error: queue_capacity must be between 1 and 64 (got {0})")]
    QueueCapacity(usize),
    /// Native TTS initialization failed.
    #[error("voice error: failed to create sherpa-onnx OfflineTts instance")]
    Initialization,
    /// Native TTS generation failed; no transcript is included in the error.
    #[error("voice error: sherpa-onnx TTS generation returned no audio")]
    Synthesis,
    /// A resampler could not be created for the requested rates.
    #[error("voice error: failed to create resampler from {input_hz} to {output_hz} Hz")]
    Resampler {
        /// Native model sample rate in hertz.
        input_hz: i32,
        /// Requested output sample rate in hertz.
        output_hz: i32,
    },
    /// The audio backend failed during a named operation.
    #[error("voice error: {operation}: {source}")]
    Device {
        /// Backend operation that failed.
        operation: &'static str,
        /// Original backend error.
        #[source]
        source: cpal::Error,
    },
    /// No matching output device is available; `None` denotes the default.
    #[error("voice error: output device not found: {0:?}")]
    NoOutputDevice(Option<String>),
    /// The output device's sample format is unsupported.
    #[error("voice error: unsupported output sample format: {0:?}")]
    SampleFormat(SampleFormat),
    /// The synthesis worker has disconnected from its bounded job queue.
    #[error("voice error: synthesis queue closed")]
    QueueClosed,
    /// The bounded submission or playback queue is full; the new job is rejected.
    #[error("voice error: queue full; new utterance rejected")]
    QueueFull,
    /// Empty or whitespace-only text cannot identify an utterance.
    #[error("voice error: speech text is empty")]
    EmptyText,
    /// All 16 subscriber slots are occupied.
    #[error("voice error: event subscriber limit reached")]
    SubscriberLimit,
}

/// Kokoro fp32 export: directory under the models directory and its model file.
///
/// Only Kokoro v1.0 speaks Portuguese. v1.1 covers Chinese and English only, so
/// its speaker 42 is not `pf_dora`, and discovery never picks it.
const KOKORO_FP32: (&str, &str) = ("kokoro-multi-lang-v1_0", "model.onnx");
/// Kokoro int8 export, the fallback for machines that only have it. On x86 it is
/// about 4x slower than fp32 and sounded worst in a listening test.
const KOKORO_INT8: (&str, &str) = ("kokoro-int8-multi-lang-v1_0", "model.int8.onnx");
/// Piper `pt_BR` faber-medium (VITS): directory and model file.
const PIPER_FABER: (&str, &str) = ("vits-piper-pt_BR-faber-medium", "pt_BR-faber-medium.onnx");

/// The neural voice to speak with, as chosen on the command line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VoiceModel {
    /// Kokoro multi-lang v1.0: 24 kHz, many speakers selected by ID.
    Kokoro,
    /// Piper `pt_BR` faber-medium (VITS): 22.05 kHz, a single speaker.
    Piper,
}

impl VoiceModel {
    /// The voice a hardware profile speaks with unless told otherwise.
    ///
    /// The desktop speaks with Kokoro (fp32, speaker 42 `pf_dora`). T1-ref speaks
    /// with Piper because it is small (63 MB) and fast (0.10 s for 6.7 s of audio
    /// on the desktop's i9-13900K); its cost on the ARM board still has to be measured.
    #[must_use]
    pub fn for_profile(profile_name: &str) -> Self {
        if profile_name == "desktop" {
            Self::Kokoro
        } else {
            Self::Piper
        }
    }

    /// The name used on the command line.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Kokoro => "kokoro",
            Self::Piper => "piper",
        }
    }

    /// Whether the voice has several speakers to choose from by ID.
    #[must_use]
    pub fn is_multi_speaker(self) -> bool {
        matches!(self, Self::Kokoro)
    }
}

impl fmt::Display for VoiceModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for VoiceModel {
    type Err = VoiceError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "kokoro" => Ok(Self::Kokoro),
            "piper" => Ok(Self::Piper),
            other => Err(VoiceError::UnknownModel(other.to_owned())),
        }
    }
}

/// Assets and speaker of a Kokoro multi-lang voice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KokoroVoice {
    /// Path to the ONNX model file (`model.onnx` or `model.int8.onnx`).
    pub model_path: PathBuf,
    /// Path to the speaker voices binary (`voices.bin`).
    pub voices_path: PathBuf,
    /// Path to the phoneme tokens file (`tokens.txt`).
    pub tokens_path: PathBuf,
    /// Path to the espeak-ng-data directory.
    pub data_dir: PathBuf,
    /// Optional path to the dict directory (`dict/`).
    pub dict_dir: Option<PathBuf>,
    /// Optional path to a custom lexicon file.
    pub lexicon_path: Option<PathBuf>,
    /// Language code passed to Kokoro (e.g. `"pt-br"` or `"pt"`).
    pub lang: String,
    /// Speaker ID within voices.bin (`42` for `pf_dora`, `43` for `pm_alex`).
    pub speaker_id: i32,
}

impl KokoroVoice {
    /// Finds Kokoro v1.0 under `models_dir`: the fp32 export when present, else
    /// the int8 one. When neither is present the paths name the fp32 layout, so
    /// validation reports the file to download. Speaks `pt-br` as speaker 42.
    #[must_use]
    pub fn discover(models_dir: &Path) -> Self {
        let (dir, model) = [KOKORO_FP32, KOKORO_INT8]
            .into_iter()
            .find(|(dir, model)| models_dir.join(dir).join(model).is_file())
            .unwrap_or(KOKORO_FP32);
        let base = models_dir.join(dir);
        let dict = base.join("dict");
        Self {
            model_path: base.join(model),
            voices_path: base.join("voices.bin"),
            tokens_path: base.join("tokens.txt"),
            data_dir: base.join("espeak-ng-data"),
            dict_dir: dict.is_dir().then_some(dict),
            lexicon_path: None,
            lang: "pt-br".to_owned(),
            speaker_id: 42, // pf_dora (Brazilian Portuguese female)
        }
    }
}

/// Assets of a single-speaker Piper (VITS) voice.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PiperVoice {
    /// Path to the ONNX model file (e.g. `pt_BR-faber-medium.onnx`).
    pub model_path: PathBuf,
    /// Path to the phoneme tokens file (`tokens.txt`).
    pub tokens_path: PathBuf,
    /// Path to the espeak-ng-data directory.
    pub data_dir: PathBuf,
}

impl PiperVoice {
    /// The `pt_BR` faber-medium layout under `models_dir`.
    #[must_use]
    pub fn discover(models_dir: &Path) -> Self {
        let (dir, model) = PIPER_FABER;
        let base = models_dir.join(dir);
        Self {
            model_path: base.join(model),
            tokens_path: base.join("tokens.txt"),
            data_dir: base.join("espeak-ng-data"),
        }
    }
}

/// The synthesis engine and the model assets it loads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceEngine {
    /// Kokoro multi-lang through sherpa-onnx's Kokoro model config.
    Kokoro(KokoroVoice),
    /// Piper through sherpa-onnx's VITS model config.
    Piper(PiperVoice),
}

impl VoiceEngine {
    /// Discovers the assets of `model` under `models_dir`.
    #[must_use]
    pub fn discover(model: VoiceModel, models_dir: &Path) -> Self {
        match model {
            VoiceModel::Kokoro => Self::Kokoro(KokoroVoice::discover(models_dir)),
            VoiceModel::Piper => Self::Piper(PiperVoice::discover(models_dir)),
        }
    }

    /// Which voice model this engine runs.
    #[must_use]
    pub fn model(&self) -> VoiceModel {
        match self {
            Self::Kokoro(_) => VoiceModel::Kokoro,
            Self::Piper(_) => VoiceModel::Piper,
        }
    }

    /// Speaker ID passed to the synthesizer: the chosen Kokoro speaker, or 0 for
    /// single-speaker Piper.
    #[must_use]
    pub fn speaker_id(&self) -> i32 {
        match self {
            Self::Kokoro(kokoro) => kokoro.speaker_id,
            Self::Piper(_) => 0,
        }
    }

    /// Path to the ONNX model file.
    #[must_use]
    pub fn model_path(&self) -> &Path {
        match self {
            Self::Kokoro(kokoro) => &kokoro.model_path,
            Self::Piper(piper) => &piper.model_path,
        }
    }

    fn validate(&self) -> Result<(), VoiceError> {
        require_file("model file", self.model_path())?;
        let (tokens, data_dir) = match self {
            Self::Kokoro(kokoro) => {
                require_file("voices file", &kokoro.voices_path)?;
                (&kokoro.tokens_path, &kokoro.data_dir)
            }
            Self::Piper(piper) => (&piper.tokens_path, &piper.data_dir),
        };
        require_file("tokens file", tokens)?;
        if !data_dir.is_dir() {
            return Err(VoiceError::MissingAsset {
                asset: "espeak-ng-data directory",
                path: data_dir.clone(),
            });
        }
        Ok(())
    }
}

impl fmt::Display for VoiceEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kokoro(kokoro) => write!(
                f,
                "Kokoro {} speaker #{}, {}",
                kokoro.lang,
                kokoro.speaker_id,
                ShortPath(&kokoro.model_path)
            ),
            Self::Piper(piper) => write!(f, "Piper, {}", ShortPath(&piper.model_path)),
        }
    }
}

/// A model path shown as its directory and file name only.
struct ShortPath<'a>(&'a Path);

impl fmt::Display for ShortPath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = self.0.file_name().unwrap_or(self.0.as_os_str());
        match self.0.parent().and_then(Path::file_name) {
            Some(dir) => write!(f, "{}/{}", dir.display(), file.display()),
            None => write!(f, "{}", file.display()),
        }
    }
}

fn require_file(asset: &'static str, path: &Path) -> Result<(), VoiceError> {
    if path.is_file() {
        Ok(())
    } else {
        Err(VoiceError::MissingAsset {
            asset,
            path: path.to_path_buf(),
        })
    }
}

/// `~/.cache/enton/models`, where voice models are expected; the current
/// directory when `HOME` is unset.
fn default_models_dir() -> PathBuf {
    std::env::var_os("HOME").map_or_else(
        || PathBuf::from("."),
        |home| PathBuf::from(home).join(".cache/enton/models"),
    )
}

/// Configuration for the local neural voice synthesizer (sherpa-onnx).
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    /// The engine that speaks and the model assets it loads.
    pub engine: VoiceEngine,
    /// Playback speed factor (default 1.0).
    pub speed: f32,
    /// Number of worker threads for TTS inference (default 2).
    pub num_threads: i32,
    /// Bound (1..=64) for each synthesis job queue and the ready playback queue.
    /// New jobs are rejected when full; ready-queue overflow emits `Failed`.
    pub queue_capacity: usize,
    /// Specific CPAL device identifier string; `None` uses default output device.
    pub device_id: Option<String>,
}

impl Default for VoiceConfig {
    /// Kokoro discovered under `~/.cache/enton/models`, 2 threads.
    fn default() -> Self {
        Self::with_engine(VoiceEngine::discover(
            VoiceModel::Kokoro,
            &default_models_dir(),
        ))
    }
}

impl VoiceConfig {
    /// A configuration for `engine`: speed 1.0, 2 threads, queues of 8 and the
    /// default output device.
    #[must_use]
    pub fn with_engine(engine: VoiceEngine) -> Self {
        Self {
            engine,
            speed: 1.0,
            num_threads: 2,
            queue_capacity: 8,
            device_id: None,
        }
    }

    /// Creates a voice configuration tailored for a specific hardware profile name.
    ///
    /// Speaks with the profile's voice ([`VoiceModel::for_profile`]) and configures
    /// 8 TTS worker threads for `"desktop"` and 2 worker threads for `"t1-ref"`.
    #[must_use]
    pub fn for_profile(profile_name: &str) -> Self {
        Self::for_profile_with_model(profile_name, VoiceModel::for_profile(profile_name))
    }

    /// Like [`VoiceConfig::for_profile`], but speaking with `model`, whose assets
    /// are discovered under `~/.cache/enton/models`.
    #[must_use]
    pub fn for_profile_with_model(profile_name: &str, model: VoiceModel) -> Self {
        let num_threads = if profile_name == "desktop" { 8 } else { 2 };
        Self {
            num_threads,
            ..Self::with_engine(VoiceEngine::discover(model, &default_models_dir()))
        }
    }

    /// Validates that model files exist on disk.
    ///
    /// # Errors
    /// Returns [`VoiceError`] if any required model file or directory is missing.
    pub fn validate(&self) -> Result<(), VoiceError> {
        self.engine.validate()?;
        if self.queue_capacity == 0 || self.queue_capacity > 64 {
            return Err(VoiceError::QueueCapacity(self.queue_capacity));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    /// A scratch models directory, removed on drop.
    struct TempModels(PathBuf);

    impl TempModels {
        fn new() -> Self {
            static COUNT: AtomicU32 = AtomicU32::new(0);
            let path = std::env::temp_dir().join(format!(
                "enton_voice_models_{}_{}",
                std::process::id(),
                COUNT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        /// Create `relative` as an empty file, with its parent directories.
        fn file(&self, relative: &str) -> &Self {
            let path = self.0.join(relative);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, b"").unwrap();
            self
        }

        fn dir(&self, relative: &str) -> &Self {
            fs::create_dir_all(self.0.join(relative)).unwrap();
            self
        }

        /// A complete Kokoro layout in `dir` with `model` as its model file.
        fn kokoro(&self, dir: &str, model: &str) -> &Self {
            self.file(&format!("{dir}/{model}"))
                .file(&format!("{dir}/voices.bin"))
                .file(&format!("{dir}/tokens.txt"))
                .dir(&format!("{dir}/espeak-ng-data"))
        }
    }

    impl Drop for TempModels {
        fn drop(&mut self) {
            // Best effort: a leftover scratch directory must not fail the test.
            fs::remove_dir_all(&self.0).ok();
        }
    }

    fn missing_path(config: &VoiceConfig) -> PathBuf {
        match config.validate() {
            Err(VoiceError::MissingAsset { path, .. }) => path,
            other => panic!("expected a missing asset, got {other:?}"),
        }
    }

    #[test]
    fn voice_config_validation_catches_missing_files() {
        let config = VoiceConfig::with_engine(VoiceEngine::discover(
            VoiceModel::Kokoro,
            Path::new("/non/existent"),
        ));
        assert_eq!(
            missing_path(&config),
            Path::new("/non/existent/kokoro-multi-lang-v1_0/model.onnx")
        );
        let config = VoiceConfig::with_engine(VoiceEngine::discover(
            VoiceModel::Piper,
            Path::new("/non/existent"),
        ));
        assert_eq!(
            missing_path(&config),
            Path::new("/non/existent/vits-piper-pt_BR-faber-medium/pt_BR-faber-medium.onnx")
        );
    }

    #[test]
    fn kokoro_discovery_prefers_fp32_over_int8() {
        let models = TempModels::new();
        models
            .kokoro("kokoro-int8-multi-lang-v1_0", "model.int8.onnx")
            .kokoro("kokoro-multi-lang-v1_0", "model.onnx")
            .dir("kokoro-multi-lang-v1_0/dict");
        let VoiceEngine::Kokoro(kokoro) = VoiceEngine::discover(VoiceModel::Kokoro, &models.0)
        else {
            panic!("expected Kokoro");
        };
        let fp32 = models.0.join("kokoro-multi-lang-v1_0");
        assert_eq!(kokoro.model_path, fp32.join("model.onnx"));
        assert_eq!(kokoro.voices_path, fp32.join("voices.bin"));
        assert_eq!(kokoro.tokens_path, fp32.join("tokens.txt"));
        assert_eq!(kokoro.data_dir, fp32.join("espeak-ng-data"));
        assert_eq!(kokoro.dict_dir, Some(fp32.join("dict")));
        assert_eq!((kokoro.lang.as_str(), kokoro.speaker_id), ("pt-br", 42));
        let config = VoiceConfig::with_engine(VoiceEngine::Kokoro(kokoro));
        assert!(config.validate().is_ok());
    }

    #[test]
    fn kokoro_discovery_falls_back_to_int8() {
        let models = TempModels::new();
        models.kokoro("kokoro-int8-multi-lang-v1_0", "model.int8.onnx");
        let engine = VoiceEngine::discover(VoiceModel::Kokoro, &models.0);
        assert_eq!(
            engine.model_path(),
            models.0.join("kokoro-int8-multi-lang-v1_0/model.int8.onnx")
        );
        let VoiceEngine::Kokoro(kokoro) = &engine else {
            panic!("expected Kokoro");
        };
        assert!(kokoro.dict_dir.is_none());
        assert!(VoiceConfig::with_engine(engine).validate().is_ok());
    }

    #[test]
    fn kokoro_discovery_never_picks_v1_1_which_has_no_portuguese() {
        let models = TempModels::new();
        models
            .kokoro("kokoro-multi-lang-v1_1", "model.onnx")
            .kokoro("kokoro-int8-multi-lang-v1_1", "model.int8.onnx")
            .kokoro("kokoro-int8-multi-lang-v1_0", "model.int8.onnx");
        assert_eq!(
            VoiceEngine::discover(VoiceModel::Kokoro, &models.0).model_path(),
            models.0.join("kokoro-int8-multi-lang-v1_0/model.int8.onnx")
        );

        let only_v1_1 = TempModels::new();
        only_v1_1
            .kokoro("kokoro-multi-lang-v1_1", "model.onnx")
            .kokoro("kokoro-int8-multi-lang-v1_1", "model.int8.onnx");
        let config =
            VoiceConfig::with_engine(VoiceEngine::discover(VoiceModel::Kokoro, &only_v1_1.0));
        assert_eq!(
            missing_path(&config),
            only_v1_1.0.join("kokoro-multi-lang-v1_0/model.onnx")
        );
    }

    #[test]
    fn piper_discovery_uses_the_faber_layout_with_one_speaker() {
        let models = TempModels::new();
        models
            .file("vits-piper-pt_BR-faber-medium/pt_BR-faber-medium.onnx")
            .dir("vits-piper-pt_BR-faber-medium/espeak-ng-data");
        let engine = VoiceEngine::discover(VoiceModel::Piper, &models.0);
        let base = models.0.join("vits-piper-pt_BR-faber-medium");
        assert_eq!(
            engine,
            VoiceEngine::Piper(PiperVoice {
                model_path: base.join("pt_BR-faber-medium.onnx"),
                tokens_path: base.join("tokens.txt"),
                data_dir: base.join("espeak-ng-data"),
            })
        );
        assert_eq!(engine.model(), VoiceModel::Piper);
        assert_eq!(engine.speaker_id(), 0);
        let config = VoiceConfig::with_engine(engine);
        assert_eq!(missing_path(&config), base.join("tokens.txt"));
        models.file("vits-piper-pt_BR-faber-medium/tokens.txt");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn profiles_pick_their_voice_and_threads() {
        let desktop = VoiceConfig::for_profile("desktop");
        assert_eq!(desktop.engine.model(), VoiceModel::Kokoro);
        assert_eq!(desktop.engine.speaker_id(), 42);
        assert_eq!(desktop.num_threads, 8);
        let t1 = VoiceConfig::for_profile("t1-ref");
        assert_eq!(t1.engine.model(), VoiceModel::Piper);
        assert_eq!(t1.num_threads, 2);
        let t1_kokoro = VoiceConfig::for_profile_with_model("t1-ref", VoiceModel::Kokoro);
        assert_eq!(t1_kokoro.engine.model(), VoiceModel::Kokoro);
        assert_eq!(t1_kokoro.num_threads, 2);
        assert_eq!(VoiceConfig::default().engine.model(), VoiceModel::Kokoro);
    }

    #[test]
    fn voice_model_names_round_trip_and_unknown_names_are_rejected() {
        for model in [VoiceModel::Kokoro, VoiceModel::Piper] {
            assert_eq!(model.name().parse::<VoiceModel>().unwrap(), model);
            assert_eq!(model.to_string(), model.name());
        }
        assert!(VoiceModel::Kokoro.is_multi_speaker());
        assert!(!VoiceModel::Piper.is_multi_speaker());
        for bad in ["", "Kokoro", "vits", "kokoro-int8"] {
            assert!(matches!(
                bad.parse::<VoiceModel>(),
                Err(VoiceError::UnknownModel(name)) if name == bad
            ));
        }
    }

    #[test]
    fn engine_display_names_the_model_file_and_speaker() {
        let models = Path::new("/models");
        assert_eq!(
            VoiceEngine::discover(VoiceModel::Kokoro, models).to_string(),
            "Kokoro pt-br speaker #42, kokoro-multi-lang-v1_0/model.onnx"
        );
        assert_eq!(
            VoiceEngine::discover(VoiceModel::Piper, models).to_string(),
            "Piper, vits-piper-pt_BR-faber-medium/pt_BR-faber-medium.onnx"
        );
    }
}
