//! Hardware-specific policy, grouped by the part of the brainstem it tunes.
//!
//! Every group is `#[serde(flatten)]`ed into [`Profile`], so the serialized form
//! stays one flat object: snapshots written before the grouping still restore.

use serde::{Deserialize, Serialize};

use crate::Senses;

/// Hardware-specific scaffold policy. Prices are provisional budget units.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// Human-readable identifier for this profile.
    pub name: String,
    /// When the body forces torpor.
    #[serde(flatten)]
    pub body: BodyLimits,
    /// What a thought costs and how fast each account refills.
    #[serde(flatten)]
    pub budgets: BudgetPolicy,
    /// When drives and overheard speech ignite a thought.
    #[serde(flatten)]
    pub ignition: IgnitionPolicy,
    /// How long Enton stays addressed, and who may continue a turn.
    #[serde(flatten)]
    pub attention: AttentionPolicy,
    /// How a speech cue turns into salience.
    #[serde(flatten)]
    pub salience: SaliencePolicy,
    /// How repetition mutes familiar cues.
    #[serde(flatten)]
    pub habituation: HabituationPolicy,
    /// How Enton tells its own voice from someone interrupting it.
    #[serde(flatten)]
    pub echo: EchoPolicy,
    /// Which sounds count as a live voice at all.
    #[serde(flatten)]
    pub source: SourcePolicy,
    /// Logged exploration near the evidence thresholds, for offline evaluation.
    /// Off unless `explore_probability` is above zero.
    #[serde(flatten)]
    pub exploration: ExplorationPolicy,
    /// What each sensor's reading is worth: the calibration of this body's
    /// microphone and models.
    #[serde(default)]
    pub senses: Senses,
}

/// Which sounds count as a live voice in the room.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SourcePolicy {
    /// Evidence for a loudspeaker over a live voice, in nats, at or beyond which a
    /// cue is reproduced media (TV, radio, music): never a follow-up, a barge-in or
    /// overheard speech worth a thought. Saying Enton's name is always heard. Cues
    /// without an audio tagger pass.
    #[serde(default = "default_media_llr")]
    pub media_llr: f32,
    /// Evidence for a loudspeaker voice over the owner's, in nats, at or beyond
    /// which an overheard line counts as the TV (or radio) talking.
    #[serde(default = "default_tv_line_llr")]
    pub tv_line_llr: f32,
    /// How far one such line moves the belief that a TV is on toward certainty,
    /// from zero to one.
    #[serde(default = "default_tv_line_weight")]
    pub tv_line_weight: f32,
    /// Half-life, in milliseconds, of the belief that a TV is on.
    #[serde(default = "default_tv_half_life_ms")]
    pub tv_half_life_ms: u64,
    /// Belief at or above which the TV counts as on.
    #[serde(default = "default_tv_on_level")]
    pub tv_on_level: f32,
    /// While the TV is on, a window is stricter by this many nats: another voice
    /// or a loudspeaker needs that much less evidence to be turned away. With the
    /// TV on, most voices in a window are the TV's.
    #[serde(default = "default_tv_caution_llr")]
    pub tv_caution_llr: f32,
    /// Half-life, in milliseconds, of the overheard TV lines that teach Enton where the
    /// TV is: long enough to outlive a pause between shows, short enough to follow a TV
    /// (or a device) that was moved. A cue's direction is weighed only while the TV is on
    /// and never during Enton's own playback.
    #[serde(default = "default_tv_direction_half_life_ms")]
    pub tv_direction_half_life_ms: u64,
    /// How many TV lines, each discounted by its age, must have taught the TV's
    /// direction before a cue's direction of arrival is weighed against it. Ten minutes
    /// and twenty lines were chosen with E1's calibration seeds; between three and twenty
    /// lines and ten to sixty minutes, E1 moves by a few requests in 3200.
    #[serde(default = "default_tv_direction_min_lines")]
    pub tv_direction_min_lines: f32,
    /// When, for a cue whose direction was weighed against the TV's, the TV caution
    /// applies to the loudspeaker alternative alone (voice, tagger and direction
    /// together) rather than to every sensor on its own and to another person's voice.
    #[serde(default)]
    pub tv_caution_confinement: TvCautionConfinement,
}

/// When a direction of arrival confines the TV caution to the loudspeaker alternative.
///
/// Without a direction, voice and tagger cannot tell the owner from the TV well
/// enough, so the caution raises every bar, including the one against another
/// person's voice. A direction gives the TV a sensor of its own, but says nothing
/// about another person: that is the directedness detector's job. Chosen with E1's
/// calibration seeds: with the array alone, confining lets other people in and costs
/// more calls than E1's floor allows; with a directedness detector as well, it serves
/// about a third more of the owner's turns with the TV on within the floor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TvCautionConfinement {
    /// Never: the direction only adds evidence and never loosens a bar.
    Never,
    /// When a directedness detector also judged the cue, since it answers for other
    /// people.
    #[default]
    WithDirectedness,
    /// Whenever the direction speaks.
    Always,
}

/// Logged exploration: how often, and how close to a threshold, a cue that an evidence
/// objection turns away gets a thought anyway, so that an offline estimator can learn what
/// the abstention cost.
///
/// A cue is borderline when an evidence objection (a loudspeaker, another voice, speech
/// addressed to someone else, or a voice the longer window did not verify) turns it away,
/// every sensor that objects is at most `explore_margin_nats` past its threshold, and
/// without the objection it would have bought a thought. Only such a cue explores, never
/// during Enton's own playback, never in torpor, never for a keyword, and never unless
/// the discretionary account can pay: exploration spends only that account. The draw
/// comes from a generator in the organism's state, seeded here and snapshotted with it,
/// so replay decides every coin flip the same way.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExplorationPolicy {
    /// Probability, from zero to one, that a borderline cue thinks anyway. Zero, the
    /// default, never explores and never draws, so decisions are those of a profile
    /// without exploration. The generator draws 24 bits, so the probability applied, and
    /// logged as the thought's propensity, is this one rounded up to a multiple of 2^-24.
    #[serde(default)]
    pub explore_probability: f32,
    /// How far past its threshold, in nats, every objecting sensor may be for a cue to
    /// count as borderline. An offline estimate is only supported for candidate thresholds
    /// that stay within this margin of the logging profile's. One nat was chosen with E1's
    /// calibration seeds.
    #[serde(default = "default_explore_margin_nats")]
    pub explore_margin_nats: f32,
    /// Seed of the exploration generator.
    #[serde(default)]
    pub explore_seed: u64,
}

impl Default for ExplorationPolicy {
    fn default() -> Self {
        Self {
            explore_probability: 0.0,
            explore_margin_nats: default_explore_margin_nats(),
            explore_seed: 0,
        }
    }
}

/// Body signals that force torpor.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BodyLimits {
    /// Enter torpor at or above this temperature, in degrees Celsius.
    pub fever_c: f32,
    /// Enter torpor at or below this normalized battery charge.
    pub lethargy_battery: f32,
}

/// The price of a thought and the two accounts that pay for it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BudgetPolicy {
    /// Cost to buy one thought.
    pub think_cost: f32,
    /// Hourly budget refill rate and capacity for directed obligation turns (`Keyword`, `FollowUp`, timeout `Attend`).
    pub obligation_budget_per_hour: f32,
    /// Hourly budget refill rate and capacity for discretionary thoughts (`Drive`, undirected `Speech`).
    pub discretionary_budget_per_hour: f32,
}

/// Ignition thresholds, hysteresis and the shared thought cooldown.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct IgnitionPolicy {
    /// Minimum salience required to ignite a thought.
    pub threshold: f32,
    /// Threshold for discretionary non-addressed speech (`Reason::Speech`).
    pub discretionary_threshold: f32,
    /// Amount to subtract from threshold on firing, prevents cascade.
    pub hysteresis: f32,
    /// Milliseconds between allowable ignitions of different thoughts.
    pub cooldown_ms: u64,
    /// Exponential moving average smoothing factor.
    pub ema_alpha: f32,
}

/// The attention window after being addressed, and what may continue a turn.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AttentionPolicy {
    /// Duration in milliseconds for an active attention window when addressed.
    pub attention_ms: u64,
    /// Cues with keyword shorter than this are treated as keyword-only turns (ms).
    pub keyword_only_ms: u32,
    /// Minimal VAD confidence required to trigger a follow-up inside attention window.
    pub follow_up_min_vad: f32,
    /// Evidence that the owner is speaking live, in nats, at or above which a cue is
    /// the owner's verified voice: it may use the longer window and interrupt with a
    /// smaller margin.
    #[serde(default = "default_verified_voice_llr")]
    pub verified_voice_llr: f32,
    /// Evidence against the owner speaking live, in nats, at or beyond which a cue
    /// is someone else's (or a loudspeaker's): it neither continues nor interrupts a
    /// turn. Cues without speaker verification or a tagger say nothing and pass.
    #[serde(default = "default_other_voice_llr")]
    pub other_voice_llr: f32,
    /// Longer attention window, in milliseconds, for the owner's verified voice:
    /// other voices cannot use it, so a pause to think does not end
    /// the conversation. Counted from the end of Enton's reply, 10 s covers an answer
    /// given about 12 s after the previous turn; longer only gives impostors more chances.
    #[serde(default = "default_verified_attention_ms")]
    pub verified_attention_ms: u64,
    /// How much a keyword cue's length says about holding a whole request, in nats
    /// per second away from `keyword_only_ms`. Added to the end-of-turn evidence, a
    /// positive sum is answered at once and a negative one waits for the rest.
    /// Without an end-of-turn model this is the `keyword_only_ms` duration rule.
    #[serde(default = "default_whole_request_llr_per_s")]
    pub whole_request_llr_per_s: f32,
    /// Longest silence, in milliseconds, between an unfinished "Enton..." and a finished
    /// turn that starts right after it for the two to be one request: that closeness is
    /// stronger evidence than a single segment's voice or media score.
    #[serde(default = "default_continuation_gap_ms")]
    pub continuation_gap_ms: u32,
    /// Evidence that speech is addressed to someone else, in nats, at or beyond which
    /// a cue inside an attention window is not for Enton (the owner talking to someone
    /// in the room, or others talking to each other): it neither continues a turn nor
    /// extends the window. Cues without a directedness detector pass. Against the
    /// shipped calibration only the clearly undirected band reaches 1.5 nats: an
    /// ambiguous reading, which one follow-up in ten gets, does not.
    #[serde(default = "default_undirected_llr")]
    pub undirected_llr: f32,
    /// Whether speech clearly addressed to Enton may use the longer window too, like
    /// the owner's verified voice, as long as its voice does not rule the owner out.
    /// Off, directedness only ever turns cues away.
    #[serde(default = "default_directed_extends_window")]
    pub directed_extends_window: bool,
    /// Evidence that speech is addressed to Enton, in nats, at or above which a cue
    /// counts as clearly addressed for `directed_extends_window`. Against the shipped
    /// calibration, 1.5 nats takes the clearly addressed band and nothing else.
    #[serde(default = "default_directed_window_llr")]
    pub directed_window_llr: f32,
}

/// Weights that turn a speech cue into salience, plus the novelty bonus.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SaliencePolicy {
    /// Weight of VAD confidence in base speech salience.
    pub salience_vad_weight: f32,
    /// Weight of signal energy in base speech salience.
    pub salience_energy_weight: f32,
    /// Weight of duration in base speech salience.
    pub salience_duration_weight: f32,
    /// Maximum duration normalization cap in milliseconds.
    pub salience_duration_max_ms: u32,
    /// Weight of prediction error in novelty salience bonus.
    pub novelty_weight: f32,
    /// Maximum novelty bonus added to salience.
    pub novelty_max: f32,
}

/// Fast and slow habituation to repeated, similar non-addressed cues.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HabituationPolicy {
    /// Half-life in milliseconds for habituation decay on tick.
    pub habituation_decay_half_life_ms: u64,
    /// Step increase in habituation on similar non-addressed cues.
    pub habituation_step: f32,
    /// Cues with similarity above this cutoff contribute to habituation.
    #[serde(default = "default_similarity_cutoff")]
    pub similarity_cutoff: f32,
    /// Learning rate for updating the running cue expectation on new cues.
    #[serde(default = "default_expectation_coefficient")]
    pub expectation_coefficient: f32,
    /// Half-life in milliseconds of long-term habituation, which survives quiet gaps
    /// that erase the fast component (a TV that pauses for a minute is still a TV).
    #[serde(default = "default_slow_habituation_half_life_ms")]
    pub slow_habituation_half_life_ms: u64,
    /// Fraction of each fast habituation increment that also accrues long-term.
    #[serde(default = "default_slow_habituation_rate")]
    pub slow_habituation_rate: f32,
}

/// The self-echo model: hangover, barge-in margins and the playback watchdog.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EchoPolicy {
    /// Duration in milliseconds for echo reverberation hangover after playback finishes.
    #[serde(default = "default_echo_hangover_ms")]
    pub echo_hangover_ms: u64,
    /// Conservative initial speaker-to-mic coupling ceiling before empirical adaptation.
    #[serde(default = "default_echo_initial_energy")]
    pub echo_initial_energy: f32,
    /// Required energy excess (near-end acoustic dominance) for double-talk detection over loudspeaker output.
    #[serde(default = "default_echo_barge_in_margin")]
    pub echo_barge_in_margin: f32,
    /// Smaller energy margin for a barge-in in the caller's verified, live voice: the
    /// verification already tells it from Enton's own echo.
    #[serde(default = "default_verified_barge_in_margin")]
    pub verified_barge_in_margin: f32,
    /// Minimum energy margin over expected echo for unpredicted keywords, rejecting TTS phonetic false positives.
    #[serde(default = "default_keyword_barge_in_margin")]
    pub keyword_barge_in_margin: f32,
    /// Exponential moving average coefficient alpha for tracking dynamic volume changes across an utterance.
    #[serde(default = "default_echo_learning_rate")]
    pub echo_learning_rate: f32,
    /// Watchdog timeout bounds maximum unbroken vocalization before forcing playback termination.
    #[serde(default = "default_max_playback_ms")]
    pub max_playback_ms: u64,
}

fn default_verified_voice_llr() -> f32 {
    0.1
}

fn default_other_voice_llr() -> f32 {
    1.0
}

fn default_verified_attention_ms() -> u64 {
    10_000
}

fn default_whole_request_llr_per_s() -> f32 {
    3.0
}

fn default_continuation_gap_ms() -> u32 {
    1_000
}

fn default_undirected_llr() -> f32 {
    1.5
}

fn default_directed_extends_window() -> bool {
    true
}

fn default_directed_window_llr() -> f32 {
    1.5
}

/// 2^24: the exploration generator draws 24 uniform bits, an integer that f32 holds exactly.
pub(crate) const DRAW_SCALE: f32 = 16_777_216.0;

fn default_explore_margin_nats() -> f32 {
    1.0
}

fn default_verified_barge_in_margin() -> f32 {
    0.05
}

fn default_media_llr() -> f32 {
    1.0
}

fn default_tv_line_llr() -> f32 {
    1.0
}

fn default_tv_line_weight() -> f32 {
    0.3
}

fn default_tv_half_life_ms() -> u64 {
    30_000
}

fn default_tv_on_level() -> f32 {
    0.5
}

fn default_tv_caution_llr() -> f32 {
    2.0
}

fn default_tv_direction_half_life_ms() -> u64 {
    600_000
}

fn default_tv_direction_min_lines() -> f32 {
    20.0
}

fn default_similarity_cutoff() -> f32 {
    0.4
}

fn default_expectation_coefficient() -> f32 {
    0.25
}

fn default_slow_habituation_half_life_ms() -> u64 {
    1_200_000
}

fn default_slow_habituation_rate() -> f32 {
    0.1
}

fn default_echo_hangover_ms() -> u64 {
    200
}

fn default_echo_initial_energy() -> f32 {
    0.75
}

fn default_echo_barge_in_margin() -> f32 {
    0.15
}

fn default_keyword_barge_in_margin() -> f32 {
    0.0
}

fn default_echo_learning_rate() -> f32 {
    0.20
}

fn default_max_playback_ms() -> u64 {
    15_000
}

impl Profile {
    /// Conservative defaults for the resource-constrained Acer reference.
    /// Thermal limits must be calibrated before deployment on another device.
    #[must_use]
    pub fn t1_ref() -> Self {
        Self {
            name: "t1-ref".to_owned(),
            body: BodyLimits {
                fever_c: 80.0,
                lethargy_battery: 0.1,
            },
            budgets: BudgetPolicy {
                think_cost: 1.0,
                obligation_budget_per_hour: 120.0,
                discretionary_budget_per_hour: 12.0,
            },
            ignition: IgnitionPolicy {
                threshold: 0.7,
                discretionary_threshold: 0.7,
                hysteresis: 0.1,
                cooldown_ms: 10_000,
                ema_alpha: 0.1,
            },
            attention: AttentionPolicy {
                attention_ms: 5_000,
                keyword_only_ms: 900,
                follow_up_min_vad: 0.5,
                verified_voice_llr: default_verified_voice_llr(),
                other_voice_llr: default_other_voice_llr(),
                verified_attention_ms: 10_000,
                whole_request_llr_per_s: default_whole_request_llr_per_s(),
                continuation_gap_ms: 1_000,
                undirected_llr: default_undirected_llr(),
                directed_extends_window: default_directed_extends_window(),
                directed_window_llr: default_directed_window_llr(),
            },
            salience: SaliencePolicy {
                salience_vad_weight: 0.60,
                salience_energy_weight: 0.25,
                salience_duration_weight: 0.15,
                salience_duration_max_ms: 1_000,
                novelty_weight: 0.10,
                novelty_max: 0.05,
            },
            habituation: HabituationPolicy {
                habituation_decay_half_life_ms: 30_000,
                habituation_step: 0.35,
                similarity_cutoff: 0.4,
                expectation_coefficient: 0.25,
                slow_habituation_half_life_ms: 1_200_000,
                slow_habituation_rate: 0.1,
            },
            echo: EchoPolicy {
                echo_hangover_ms: 200,
                echo_initial_energy: 0.75,
                echo_barge_in_margin: 0.15,
                verified_barge_in_margin: 0.05,
                keyword_barge_in_margin: 0.0,
                echo_learning_rate: 0.20,
                max_playback_ms: 15_000,
            },
            source: SourcePolicy {
                media_llr: default_media_llr(),
                tv_line_llr: default_tv_line_llr(),
                tv_line_weight: default_tv_line_weight(),
                tv_half_life_ms: default_tv_half_life_ms(),
                tv_on_level: default_tv_on_level(),
                tv_caution_llr: default_tv_caution_llr(),
                tv_direction_half_life_ms: default_tv_direction_half_life_ms(),
                tv_direction_min_lines: default_tv_direction_min_lines(),
                tv_caution_confinement: TvCautionConfinement::default(),
            },
            exploration: ExplorationPolicy::default(),
            senses: Senses::calibrated(),
        }
    }

    /// Provisional desktop policy with a larger thought allowance.
    #[must_use]
    pub fn desktop() -> Self {
        Self {
            name: "desktop".to_owned(),
            body: BodyLimits {
                fever_c: 90.0,
                lethargy_battery: 0.05,
            },
            budgets: BudgetPolicy {
                think_cost: 1.0,
                obligation_budget_per_hour: 300.0,
                discretionary_budget_per_hour: 60.0,
            },
            ignition: IgnitionPolicy {
                threshold: 0.65,
                discretionary_threshold: 0.65,
                hysteresis: 0.1,
                cooldown_ms: 5_000,
                ema_alpha: 0.2,
            },
            attention: AttentionPolicy {
                attention_ms: 5_000,
                keyword_only_ms: 900,
                follow_up_min_vad: 0.45,
                verified_voice_llr: default_verified_voice_llr(),
                other_voice_llr: default_other_voice_llr(),
                verified_attention_ms: 10_000,
                whole_request_llr_per_s: default_whole_request_llr_per_s(),
                continuation_gap_ms: 1_000,
                undirected_llr: default_undirected_llr(),
                directed_extends_window: default_directed_extends_window(),
                directed_window_llr: default_directed_window_llr(),
            },
            salience: SaliencePolicy {
                salience_vad_weight: 0.60,
                salience_energy_weight: 0.25,
                salience_duration_weight: 0.15,
                salience_duration_max_ms: 1_000,
                novelty_weight: 0.10,
                novelty_max: 0.05,
            },
            habituation: HabituationPolicy {
                habituation_decay_half_life_ms: 15_000,
                habituation_step: 0.20,
                similarity_cutoff: 0.4,
                expectation_coefficient: 0.25,
                slow_habituation_half_life_ms: 600_000,
                slow_habituation_rate: 0.1,
            },
            echo: EchoPolicy {
                echo_hangover_ms: 150,
                echo_initial_energy: 0.70,
                echo_barge_in_margin: 0.15,
                verified_barge_in_margin: 0.05,
                keyword_barge_in_margin: 0.0,
                echo_learning_rate: 0.20,
                max_playback_ms: 20_000,
            },
            source: SourcePolicy {
                media_llr: default_media_llr(),
                tv_line_llr: default_tv_line_llr(),
                tv_line_weight: default_tv_line_weight(),
                tv_half_life_ms: default_tv_half_life_ms(),
                tv_on_level: default_tv_on_level(),
                tv_caution_llr: default_tv_caution_llr(),
                tv_direction_half_life_ms: default_tv_direction_half_life_ms(),
                tv_direction_min_lines: default_tv_direction_min_lines(),
                tv_caution_confinement: TvCautionConfinement::default(),
            },
            exploration: ExplorationPolicy::default(),
            senses: Senses::calibrated(),
        }
    }

    /// Validates that all profile parameters are within allowable ranges.
    ///
    /// # Errors
    ///
    /// Returns [`InvalidProfile`] if any parameter is out of range.
    pub fn validate(&self) -> Result<(), InvalidProfile> {
        let valid = self.body.is_valid()
            && self.budgets.is_valid()
            && self.ignition.is_valid()
            && self.attention.is_valid()
            && self.salience.is_valid()
            && self.habituation.is_valid()
            && self.echo.is_valid()
            && self.source.is_valid()
            && self.exploration.is_valid()
            && self.senses.is_valid();
        if valid {
            Ok(())
        } else {
            Err(InvalidProfile {
                name: self.name.clone(),
            })
        }
    }
}

impl SourcePolicy {
    fn is_valid(&self) -> bool {
        [self.media_llr, self.tv_line_llr]
            .iter()
            .all(|llr| llr.is_finite() && *llr > 0.0)
            && self.tv_line_weight > 0.0
            && self.tv_line_weight <= 1.0
            && self.tv_half_life_ms > 0
            && self.tv_on_level > 0.0
            && self.tv_on_level < 1.0
            && self.tv_caution_llr.is_finite()
            && self.tv_caution_llr >= 0.0
            && self.tv_direction_half_life_ms > 0
            && self.tv_direction_min_lines.is_finite()
            && self.tv_direction_min_lines > 0.0
    }
}

impl ExplorationPolicy {
    /// The probability exploration actually applies, and logs as an explored thought's
    /// propensity: `explore_probability` rounded up to a multiple of 2^-24, the resolution
    /// of the generator's draws. An abstention kept by the coin logs one minus this, which
    /// f32 holds exactly too.
    #[must_use]
    pub fn applied_probability(&self) -> f32 {
        (self.explore_probability * DRAW_SCALE).ceil() / DRAW_SCALE
    }

    fn is_valid(&self) -> bool {
        (0.0..=1.0).contains(&self.explore_probability)
            && self.explore_margin_nats.is_finite()
            && self.explore_margin_nats >= 0.0
    }
}

impl BodyLimits {
    fn is_valid(self) -> bool {
        self.fever_c.is_finite() && (0.0..=1.0).contains(&self.lethargy_battery)
    }
}

impl BudgetPolicy {
    fn is_valid(&self) -> bool {
        [
            self.think_cost,
            self.obligation_budget_per_hour,
            self.discretionary_budget_per_hour,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    }
}

impl IgnitionPolicy {
    fn is_valid(&self) -> bool {
        self.threshold.is_finite()
            && self.threshold > 0.0
            && self.discretionary_threshold.is_finite()
            && self.discretionary_threshold > 0.0
            && (0.0..self.threshold).contains(&self.hysteresis)
            && self.ema_alpha > 0.0
            && self.ema_alpha <= 1.0
    }
}

impl AttentionPolicy {
    fn is_valid(&self) -> bool {
        self.attention_ms > 0
            && self.keyword_only_ms > 0
            && (0.0..=1.0).contains(&self.follow_up_min_vad)
            && [
                self.verified_voice_llr,
                self.other_voice_llr,
                self.undirected_llr,
                self.directed_window_llr,
            ]
            .iter()
            .all(|llr| llr.is_finite() && *llr > 0.0)
            && self.verified_attention_ms >= self.attention_ms
            && self.whole_request_llr_per_s.is_finite()
            && self.whole_request_llr_per_s > 0.0
            && self.continuation_gap_ms > 0
    }
}

impl SaliencePolicy {
    fn is_valid(&self) -> bool {
        self.salience_vad_weight.is_finite()
            && self.salience_vad_weight > 0.0
            && self.salience_energy_weight.is_finite()
            && self.salience_energy_weight >= 0.0
            && self.salience_duration_weight.is_finite()
            && self.salience_duration_weight >= 0.0
            && self.salience_duration_max_ms > 0
            && self.novelty_weight.is_finite()
            && self.novelty_weight >= 0.0
            && self.novelty_max.is_finite()
            && self.novelty_max >= 0.0
    }
}

impl HabituationPolicy {
    fn is_valid(&self) -> bool {
        self.habituation_decay_half_life_ms > 0
            && self.habituation_step.is_finite()
            && self.habituation_step >= 0.0
            && self.slow_habituation_half_life_ms > 0
            && (0.0..=1.0).contains(&self.slow_habituation_rate)
            && (0.0..=1.0).contains(&self.similarity_cutoff)
            && (0.0..=1.0).contains(&self.expectation_coefficient)
            && self.expectation_coefficient > 0.0
    }
}

impl EchoPolicy {
    fn is_valid(&self) -> bool {
        self.echo_hangover_ms > 0
            && (0.0..=1.0).contains(&self.echo_initial_energy)
            && self.echo_barge_in_margin.is_finite()
            && self.echo_barge_in_margin >= 0.0
            && (0.0..=self.echo_barge_in_margin).contains(&self.verified_barge_in_margin)
            && self.keyword_barge_in_margin.is_finite()
            && self.keyword_barge_in_margin >= 0.0
            && (0.0..=1.0).contains(&self.echo_learning_rate)
            && self.echo_learning_rate > 0.0
            && self.max_playback_ms > 0
    }
}

/// Error returned when an organism profile fails validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidProfile {
    /// Name of the invalid profile.
    pub name: String,
}

impl std::fmt::Display for InvalidProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid organism profile '{}': a parameter is out of range",
            self.name
        )
    }
}

impl std::error::Error for InvalidProfile {}
