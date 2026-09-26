use enton_core::{
    Abstention, Action, BodySignals, Event, Millis, Profile, Reason, SpeechCue, ThoughtId,
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
        },
    }
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
