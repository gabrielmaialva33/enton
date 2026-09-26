//! `enton why` against a small soul written with the soul's own API: the binary
//! replays it read-only and explains each speech decision.

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

use enton_adapters::cortex::Persona;
use enton_adapters::soul::PersonaDigest;
use enton_adapters::{Soul, SoulConfig};
use enton_core::{Event, Millis, Profile, SpeechCue, ThoughtId, UtteranceId};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Enton's own reply is stored with the thought's event; `why` must never print it.
const REPLY: &str = "a reply that enton why must never print";

const MEDIA_LINE: &str =
    "Abstained (Media): the tagger heard a loudspeaker, -2.4 nats; the TV had been on for 3 min.";

/// A data directory removed on drop.
#[derive(Debug)]
struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new() -> Result<Self, Box<dyn Error>> {
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("enton-why-{}-{count}", std::process::id()));
        std::fs::create_dir_all(path.join("enton"))?;
        Ok(Self { path })
    }

    /// Where `enton why --profile t1-ref` looks with `XDG_DATA_HOME` set here.
    fn default_soul(&self) -> PathBuf {
        self.path.join("enton").join("soul-t1-ref.sqlite")
    }

    fn listing(&self) -> Result<Vec<String>, Box<dyn Error>> {
        let mut names = std::fs::read_dir(self.path.join("enton"))?
            .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        names.sort();
        Ok(names)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.path));
    }
}

/// A line from a TV: only the audio tagger described it, and it heard a loudspeaker.
fn tv_line() -> SpeechCue {
    SpeechCue {
        energy: 0.6,
        duration_ms: 1_500,
        vad_confidence: 0.9,
        media: Some(0.8),
        ..SpeechCue::default()
    }
}

/// "Enton, what time is it?": the name and a whole request in one segment.
fn request() -> SpeechCue {
    SpeechCue {
        energy: 0.8,
        duration_ms: 1_500,
        vad_confidence: 0.95,
        keyword: true,
        ..SpeechCue::default()
    }
}

/// A soul in which the owner asked Enton something (the cortex was down), then a TV
/// talked for four minutes, and Enton shut down cleanly two minutes
/// after its last line, snapshotting on the way out.
fn write_soul(path: &Path) -> Result<(), Box<dyn Error>> {
    let soul = Soul::open(path, SoulConfig::default())?;
    soul.append_event(&Event::Tick { now: Millis(1_000) })?;
    let asked = soul.append_event(&Event::Speech {
        now: Millis(5_000),
        cue: request(),
    })?;
    soul.record_pending(ThoughtId(1), asked, &built_in())?;
    soul.mark_failed(ThoughtId(1), r#"{"reason":"cortex unavailable"}"#)?;
    soul.append_event(&Event::CortexReply {
        now: Millis(6_000),
        thought: ThoughtId(1),
        text: REPLY.to_owned(),
    })?;
    for line in 1..=25 {
        soul.append_event(&Event::Speech {
            now: Millis(10_000 + line * 10_000),
            cue: tv_line(),
        })?;
    }
    let mut last = 0;
    for second in (270..=380).step_by(10) {
        last = soul.append_event(&Event::Tick {
            now: Millis(second * 1_000),
        })?;
    }
    // After a clean shutdown the tail past the latest snapshot is empty: `why`
    // must replay from the start of the log, not from there.
    let (organism, _) = soul.replay_organism(&Profile::t1_ref())?;
    soul.save_organism_snapshot(last, &organism)?;
    Ok(())
}

/// The persona the thought in [`write_soul`] was asked with.
fn built_in() -> PersonaDigest {
    (&Persona::built_in()).into()
}

fn why(data: &TempDir, args: &[&str]) -> Result<Output, Box<dyn Error>> {
    Ok(Command::new(env!("CARGO_BIN_EXE_enton"))
        .arg("why")
        .args(args)
        .env("XDG_DATA_HOME", &data.path)
        .output()?)
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

#[test]
fn why_explains_an_abstention_from_the_default_soul() {
    let data = TempDir::new().unwrap();
    write_soul(&data.default_soul()).unwrap();
    let before = data.listing().unwrap();

    let output = why(&data, &[]).unwrap();
    let text = stdout(&output);
    assert!(output.status.success(), "{output:?}");
    assert!(
        text.contains("Replayed 40 events, seq 1 to 40, from a fresh organism."),
        "{text}"
    );
    assert!(
        text.contains("The last 10 of 26 speech cues; times count back on Enton's clock from the last recorded event (t = 380000 ms)."),
        "{text}"
    );
    assert!(
        text.contains("2 min ago (t = 260000 ms, seq 28): Abstain: Media (Speech, salience"),
        "{text}"
    );
    assert!(
        text.contains("  evidence  live over reproduced -2.40, owner-live -2.40 nats\n"),
        "{text}"
    );
    assert!(text.contains("TV on (0."), "{text}");
    assert!(
        text.contains(&format!("  why       {MEDIA_LINE}\n")),
        "{text}"
    );
    assert!(!text.contains(REPLY), "{text}");
    // The audit is read-only: no file appears, none disappears.
    assert_eq!(data.listing().unwrap(), before);
}

#[test]
fn why_reports_a_thought_the_cortex_never_answered() {
    let data = TempDir::new().unwrap();
    let soul = data.path.join("elsewhere.sqlite");
    write_soul(&soul).unwrap();

    let soul = soul.to_string_lossy().into_owned();
    let output = why(
        &data,
        &["--soul", &soul, "--profile=t1-ref", "--last", "30"],
    )
    .unwrap();
    let text = stdout(&output);
    assert!(output.status.success(), "{output:?}");
    assert!(text.contains("All 26 speech cues"), "{text}");
    assert!(
        text.contains("6 min ago (t = 5000 ms, seq 2): Think #1 (Keyword, salience"),
        "{text}"
    );
    assert!(
        text.contains("Thought #1 (Keyword): Enton was called by name, but the thought failed: cortex unavailable."),
        "{text}"
    );
    assert!(!text.contains(REPLY), "{text}");
    // The persona it was asked with, by the hash the startup line shows.
    let persona = built_in();
    let name = format!(
        "{} (built-in default, {} bytes)",
        persona.short_hex(),
        persona.bytes
    );
    assert!(
        text.contains(&format!("Persona: {name} first used at seq 2.\n")),
        "{text}"
    );
    assert!(
        text.contains(&format!("  persona   {name}\n  why       Thought #1 ")),
        "{text}"
    );
    assert!(!text.contains("  warning"), "{text}");
    assert_eq!(text.matches("  persona   ").count(), 1, "{text}");
}

#[test]
fn why_prints_machine_readable_records() {
    let data = TempDir::new().unwrap();
    write_soul(&data.default_soul()).unwrap();

    let output = why(&data, &["--json", "--last", "2"]).unwrap();
    assert!(output.status.success(), "{output:?}");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["profile"], "t1-ref");
    assert_eq!(report["events"], 40);
    assert_eq!(report["speech_cues"], 26);
    assert_eq!(report["last_event_ms"], 380_000);
    assert!(report["from_snapshot"].is_null());
    let cues = report["cues"].as_array().unwrap();
    assert_eq!(cues.len(), 2);

    let last = &cues[1];
    assert_eq!(last["seq"], 28);
    assert_eq!(last["at_ms"], 260_000);
    assert_eq!(last["ago_ms"], 120_000);
    assert_eq!(last["decision"]["kind"], "abstain");
    assert_eq!(last["decision"]["why"], "Media");
    assert_eq!(last["decision"]["reason"], "Speech");
    // Evidence only for the sensor that ran: the tagger.
    let evidence = last["evidence"].as_object().unwrap();
    let mut sensors: Vec<&str> = evidence.keys().map(String::as_str).collect();
    sensors.sort_unstable();
    assert_eq!(sensors, ["live_over_reproduced", "owner_live"]);
    let llr = evidence["live_over_reproduced"].as_f64().unwrap();
    assert!((llr + 2.4).abs() < 1e-5, "{llr}");
    assert_eq!(last["context"]["tv_on_for_ms"], 200_000);
    assert_eq!(last["context"]["name_pending"], false);
    assert_eq!(last["context"]["torpor"], false);
    assert!(last["context"]["attention_left_ms"].is_null());
    assert_eq!(
        last["cue"]["media"]
            .as_f64()
            .map(|m| (m - 0.8).abs() < 1e-6),
        Some(true)
    );
    assert_eq!(last["explanation"], MEDIA_LINE);
    assert!(!String::from_utf8_lossy(&output.stdout).contains(REPLY));

    // The failed thought, with the reason the runtime stored.
    let output = why(&data, &["--json", "--last", "26"]).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let asked = &report["cues"][0];
    assert_eq!(asked["decision"]["kind"], "think");
    assert_eq!(asked["decision"]["thought"], 1);
    assert_eq!(asked["decision"]["reason"], "Keyword");
    assert_eq!(asked["decision"]["fate"]["status"], "failed");
    assert_eq!(asked["decision"]["fate"]["failure"], "cortex unavailable");
    assert_eq!(asked["decision"]["persona"]["sha256"], built_in().hex());
    assert_eq!(asked["decision"]["persona"]["source"], "built_in");
    assert_eq!(asked["decision"]["persona"]["first_seq"], 2);
    assert!(asked["decision"].get("persona_changed").is_none());
    assert_eq!(report["personas"][0]["bytes"], built_in().bytes);
    assert_eq!(asked["evidence"], serde_json::json!({}));

    // `--since` counts back from the last event: nothing was said in its last minute.
    let output = why(&data, &["--json", "--since", "1m"]).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["cues"], serde_json::json!([]));
    let output = why(&data, &["--since=1m"]).unwrap();
    assert!(
        stdout(&output).contains("No speech cues in the last 1 min (26 in the soul)."),
        "{output:?}"
    );
}

#[test]
fn why_without_a_soul_says_so() {
    let data = TempDir::new().unwrap();
    let output = why(&data, &[]).unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no soul at"), "{stderr}");
    assert!(stderr.contains("soul-t1-ref.sqlite"), "{stderr}");
    // Looking must not create one.
    assert!(data.listing().unwrap().is_empty());

    let output = why(&data, &["--last", "0"]).unwrap();
    assert!(!output.status.success());
}

#[test]
fn why_refuses_to_replay_under_another_profile() {
    let data = TempDir::new().unwrap();
    let soul = data.default_soul();
    write_soul(&soul).unwrap();
    let soul = soul.to_string_lossy().into_owned();
    let output = why(&data, &["--soul", &soul, "--profile", "desktop"]).unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("profile differs"), "{stderr}");
}

/// The owner asked, Enton answered aloud, and the owner called again over its second
/// sentence: the player cut that sentence and the one queued after it.
fn write_cut_soul(path: &Path) -> Result<(), Box<dyn Error>> {
    let soul = Soul::open(path, SoulConfig::default())?;
    let played = |now, utterance, interrupted| Event::PlaybackFinished {
        now: Millis(now),
        utterance: UtteranceId(utterance),
        interrupted,
    };
    soul.append_event(&Event::Speech {
        now: Millis(1_000),
        cue: request(),
    })?;
    soul.append_event(&Event::CortexReply {
        now: Millis(2_000),
        thought: ThoughtId(1),
        text: REPLY.to_owned(),
    })?;
    soul.append_event(&Event::PlaybackStarted {
        now: Millis(2_100),
        utterance: UtteranceId(1),
    })?;
    soul.append_event(&played(3_000, 1, false))?;
    soul.append_event(&Event::PlaybackStarted {
        now: Millis(3_010),
        utterance: UtteranceId(2),
    })?;
    // Loud: over a reply with its own name in it, only that interrupts Enton.
    soul.append_event(&Event::Speech {
        now: Millis(4_000),
        cue: SpeechCue {
            energy: 0.95,
            ..request()
        },
    })?;
    soul.append_event(&played(4_010, 2, true))?;
    soul.append_event(&played(4_010, 3, true))?;
    Ok(())
}

#[test]
fn why_says_which_cue_cut_enton_off() {
    let data = TempDir::new().unwrap();
    let path = data.path.join("cut.sqlite");
    write_cut_soul(&path).unwrap();
    let soul = path.to_string_lossy().into_owned();

    let output = why(&data, &["--soul", &soul]).unwrap();
    assert!(output.status.success(), "{output:?}");
    let text = stdout(&output);
    let cut = "  playback  cut off: Enton stopped speaking for this cue";
    assert_eq!(text.matches(cut).count(), 1, "{text}");
    // Under the second request, which interrupted Enton, not the first.
    let second = text.find("(t = 4000 ms, seq 6)").unwrap();
    assert!(text.find(cut).unwrap() > second, "{text}");
    assert!(!text.contains(REPLY), "{text}");

    let output = why(&data, &["--soul", &soul, "--json"]).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let cut_off: Vec<bool> = report["cues"]
        .as_array()
        .unwrap()
        .iter()
        .map(|cue| cue.get("cut_off").is_some_and(|flag| flag == true))
        .collect();
    assert_eq!(cut_off, [false, true]);
}
