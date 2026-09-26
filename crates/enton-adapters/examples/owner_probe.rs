//! Owner voice probe: speaker verification measurement on the real microphone.
//!
//! Evaluates CAM++ speaker embedding model from 3D-Speaker (16 kHz) through sherpa-onnx.
//! Measures EER, d', FAR, FRR, throughput, and creates owner voiceprints with mode 0600.

// CLI tool talks to the terminal by design.
#![allow(clippy::print_stdout, clippy::print_stderr)]
// The subcommands print long, flat reports; splitting them would only scatter the output.
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
        fs,
        path::{Path, PathBuf},
    };

    use enton_adapters::voice_id::{
        MIN_FILES_PER_CLASS, MODEL_FILENAME, calculate_verification_metrics,
        check_path_not_in_repo, compute_centroid, cosine_similarity, create_extractor,
        describe_model, extract_embedding, hash_file, load_voiceprint, save_voiceprint,
        verify_model_identity,
    };

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
        println!("enton owner_probe: speaker verification measurement tool");
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
