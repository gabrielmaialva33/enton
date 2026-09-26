use enton_core::{
    Abstention, Action, BodySignals, DirectedModel, Event, Millis, Profile, Reason, Senses,
    SourceModel, SpeechCue, ThoughtId, TurnModel, VoiceModel,
};

pub(crate) fn speech(now: u64, keyword: bool) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms: 1_500,
            vad_confidence: 1.0,
            keyword,
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
        },
    }
}

pub(crate) fn body(now: u64, temperature_c: Option<f32>, battery: Option<f32>) -> Event {
    Event::Body {
        now: Millis(now),
        signals: BodySignals {
            temperature_c,
            battery,
            cpu_load: 0.0,
        },
    }
}

pub(crate) fn voice(now: u64, keyword: bool, duration_ms: u32, speaker_sim: f32) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 1.0,
            duration_ms,
            vad_confidence: 1.0,
            keyword,
            speaker_sim: Some(speaker_sim),
            media: None,
            turn_complete: None,
            directed: None,
        },
    }
}

pub(crate) fn tv_cue(now: u64) -> Event {
    Event::Speech {
        now: Millis(now),
        cue: SpeechCue {
            energy: 0.8,
            duration_ms: 1_000,
            vad_confidence: 0.9,
            keyword: false,
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
        },
    }
}

/// A clean lab calibration: every sensor tells its classes apart decisively, so
/// tests about the reducer's mechanics do not depend on what a far-field
/// microphone can or cannot tell apart.
pub(crate) fn lab_profile() -> Profile {
    let mut profile = Profile::t1_ref();
    profile.senses = Senses {
        voice: VoiceModel {
            owner: [0.85; 3],
            other: [0.25; 3],
            reproduced: [0.2; 3],
            sd: 0.1,
        },
        source: SourceModel {
            live: [0.1; 3],
            reproduced: [0.9; 3],
            sd: 0.1,
        },
        turn: TurnModel {
            llr: [[-4.0, -2.0, 2.0, 4.0]; 3],
        },
        directed: DirectedModel {
            llr: [-4.0, 0.0, 4.0],
        },
        max_llr: 6.0,
    };
    profile
}

pub(crate) fn sensitive_profile() -> Profile {
    {
        let mut profile = Profile::t1_ref();
        profile.ignition.threshold = 0.01;
        profile.ignition.hysteresis = 0.002;
        profile.ignition.ema_alpha = 1.0;
        profile.ignition.cooldown_ms = 0;
        profile
    }
}

pub(crate) fn assert_thought(actions: &[Action], expected: u64, expected_reason: &Reason) {
    assert!(
        matches!(actions, [Action::Think { thought, reason, .. }]
            if *thought == ThoughtId(expected) && reason == expected_reason),
        "expected thought {expected} for {expected_reason:?}, got {actions:?}"
    );
}

pub(crate) fn assert_abstention(actions: &[Action], expected: Abstention) {
    assert!(
        matches!(actions, [Action::Abstain { why, .. }] if *why == expected),
        "expected {expected:?}, got {actions:?}"
    );
}
