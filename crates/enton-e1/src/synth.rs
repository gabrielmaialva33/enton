//! Protocol 3.3.0 populations. Deliberately does not import the cognitive Profile.

use crate::Error;
use crate::tape::{
    Annotation, Distance, EpisodeId, Interval, Record, RoomCondition, SegmentId, Stimulus, Tape,
    TapeKind, Turn, TurnId, TvBackground, TvContent,
};
use enton_core::{Event, Millis, SpeechCue};

/// Vigna's reference `SplitMix64`: fixed-increment state and mixed output.
/// Reference: <https://prng.di.unimi.it/splitmix64.c> (public domain).
#[derive(Debug, Clone, Copy)]
pub struct SplitMix64(u64);
impl SplitMix64 {
    /// Initialize the 64-bit state with the supplied seed.
    #[must_use]
    pub const fn with_seed(seed: u64) -> Self {
        Self(seed)
    }
    /// Advance the state and return the mixed 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn range(&mut self, low: u32, high: u32) -> u32 {
        let width = u64::from(high - low) + 1;
        let reject_below = width.wrapping_neg() % width;
        loop {
            let value = self.next_u64();
            if value >= reject_below {
                // The remainder is at most high-low, which fits u32.
                return low + u32::try_from(value % width).unwrap_or(0);
            }
        }
    }
    fn real(&mut self, low: f32, high: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / 16_777_216.0;
        low + (high - low) * unit
    }
    /// Box-Muller standard normal sampler scaled to the requested mean and standard deviation.
    pub fn normal(&mut self, mean: f32, sd: f32) -> f32 {
        let u1 = loop {
            let val = self.real(0.0, 1.0);
            if val > 0.0 {
                break val;
            }
        };
        let u2 = self.real(0.0, 1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        mean + sd * (r * theta.cos())
    }
    /// An angle in radians, uniform around the circle.
    fn angle(&mut self) -> f64 {
        std::f64::consts::TAU * f64::from(self.real(0.0, 1.0))
    }
    /// A von Mises angle in radians around zero with concentration `kappa`, by Best and
    /// Fisher's rejection sampler (Applied Statistics 28, 1979).
    pub fn von_mises(&mut self, kappa: f64) -> f64 {
        let tau = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
        let rho = (tau - (2.0 * tau).sqrt()) / (2.0 * kappa);
        let r = (1.0 + rho * rho) / (2.0 * rho);
        loop {
            let u1 = f64::from(self.real(0.0, 1.0));
            let u2 = f64::from(self.real(0.0, 1.0));
            let u3 = f64::from(self.real(0.0, 1.0));
            let z = (std::f64::consts::PI * u1).cos();
            let f = (1.0 + r * z) / (r + z);
            let c = kappa * (r - f);
            if c * (2.0 - c) - u2 > 0.0 || (c / u2).ln() + 1.0 - c >= 0.0 {
                let angle = f.clamp(-1.0, 1.0).acos();
                return if u3 < 0.5 { -angle } else { angle };
            }
        }
    }
}

/// Lower bound of the speaker similarity range for non-speech acoustic segments.
pub const NON_SPEECH_SPEAKER_SIM_MIN: f32 = 0.0;
/// Upper bound of the speaker similarity range for non-speech acoustic segments.
pub const NON_SPEECH_SPEAKER_SIM_MAX: f32 = 0.3;

/// Stream salt for the independent speaker-similarity RNG stream.
const SPEAKER_STREAM_SALT: u64 = 0x5350_4541_4b45_5231;

/// Lower bound of the media likelihood range for non-speech acoustic segments.
pub const NON_SPEECH_MEDIA_MIN: f32 = 0.0;
/// Upper bound of the media likelihood range for non-speech acoustic segments.
pub const NON_SPEECH_MEDIA_MAX: f32 = 0.3;

/// Stream salt for the independent media-tagger RNG stream.
const MEDIA_STREAM_SALT: u64 = 0x4d45_4449_415f_5331;

/// Stream salt for the independent turn-completion RNG stream.
const TURN_STREAM_SALT: u64 = 0x5455_524e_5f53_3130;

/// Spread of a TV line's energy around its show's level.
pub const TV_ENERGY_JITTER: f32 = 0.08;
/// Spread of a TV line's voice activity confidence around its show's level.
pub const TV_VAD_JITTER: f32 = 0.05;

/// Stream salt for the independent environmental and talker condition RNG stream.
const CONDITION_STREAM_SALT: u64 = 0x434f_4e44_5f53_3130;

/// Stream salt for the independent device-directedness RNG stream. The per-owner
/// factor, the per-block habits and every cue's reading are drawn from it alone, so
/// adding the sensor moved no reading of the other three.
const DIRECTED_STREAM_SALT: u64 = 0x4449_5245_4354_5331;

/// Scores that split the simulated directedness detector's output into three bands:
/// clearly addressed to someone else below the first, ambiguous up to the second and
/// clearly addressed to Enton from it on. A score is uniform inside its band.
pub const DIRECTED_BAND_EDGES: [f32; 2] = [0.30, 0.75];
/// Chance that the owner's asides in a three-minute block are entangled with the
/// conversation with Enton ("anota aí também o sabão em pó"): command-shaped, so a
/// detector reads them as addressed to it.
pub const ENTANGLED_BLOCK_RATE: f32 = 0.25;
/// Share of an entangled block's asides that score as clearly addressed to Enton.
pub const ENTANGLED_ASIDE_DIRECTED: f32 = 0.35;
/// Share of any other block's asides that score as clearly addressed to Enton.
pub const PLAIN_ASIDE_DIRECTED: f32 = 0.04;
/// Chance that the owner talks to Enton casually in a block, as to a person in the room.
pub const CASUAL_BLOCK_RATE: f32 = 0.2;
/// Share of a casual block's 1 to 3 s requests that do not score as clearly addressed.
pub const CASUAL_REQUEST_MISSED: f32 = 0.35;
/// Share of any other block's 1 to 3 s requests that do not score as clearly addressed.
pub const PLAIN_REQUEST_MISSED: f32 = 0.10;
/// Spread, in natural-log units, of the per-owner factor on the owner's confusable
/// shares: some people talk to devices and to people alike.
pub const OWNER_CONFUSION_SD: f32 = 0.35;
/// Clearly-directed share of a TV line that the media tagger missed (media below
/// 0.5): dialogue that passes for a live voice passes for a request more often too.
pub const MISSED_TV_DIRECTED: f32 = 0.15;
/// Share of 1 to 3 s requests that do not score as clearly addressed, over all blocks:
/// the casual and plain rates average to it, and other durations scale by the same ratio.
const REQUEST_MISSED: f32 = 0.15;

/// Stream salt for the independent direction-of-arrival RNG stream. The TV's place, the
/// seats of each block and every cue's reading are drawn from it alone, so adding the
/// sensor moved no reading of the other four.
const DIRECTION_STREAM_SALT: u64 = 0x4449_5245_4354_4e31;

/// Width, in degrees, of the bins of [`OWNER_TV_ANGLE_PERCENT`].
pub const OWNER_TV_ANGLE_BIN_DEG: f64 = 15.0;
/// Angle between the owner and the TV, seen from the device, folded to 0 to 180 degrees
/// in 15-degree bins: percent of three-minute blocks, measured on 240 simulated living
/// rooms. The side is a coin flip.
pub const OWNER_TV_ANGLE_PERCENT: [f64; 12] = [
    11.2, 13.8, 16.2, 19.1, 16.4, 9.2, 5.6, 3.4, 2.2, 1.6, 1.0, 0.4,
];
/// Concentration of a persistent direction offset: the owner's in each block, the TV's in
/// each tape (reflections and array geometry bias an estimate the same way for a while).
pub const POSITION_KAPPA: f64 = 20.0;
/// Chance that a persistent offset lands anywhere at all instead.
pub const POSITION_OUTLIER: f32 = 0.05;
/// Concentration of one live segment's reading around its talker's direction.
pub const LIVE_SEGMENT_KAPPA: f64 = 60.0;
/// Chance that a live segment's reading lands anywhere at all, with the TV off, moderate
/// and loud.
pub const LIVE_OUTLIER: [f32; 3] = [0.03, 0.08, 0.12];
/// Chance that a live segment's reading points at the TV instead (the TV drowned the
/// talker), with the TV moderate and loud.
pub const TV_CAPTURE: [f32; 2] = [0.10, 0.25];
/// Concentration of a TV line's reading around the TV's direction.
pub const TV_SEGMENT_KAPPA: f64 = 15.0;
/// Chance that a TV line's reading lands anywhere at all.
pub const TV_OUTLIER: f32 = 0.15;

/// Where the talkers of one three-minute block are, as the array sees them, in radians.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct Seats {
    /// The owner's direction, persistent offset included.
    owner: f64,
    /// Where other live people talk from in this block.
    other: f64,
}

/// The unit vector of an azimuth in radians.
fn unit(azimuth: f64) -> [f32; 2] {
    let (sin, cos) = azimuth.sin_cos();
    // A unit vector's components lie in [-1, 1]: narrowing loses precision only.
    #[allow(clippy::cast_possible_truncation)]
    [cos as f32, sin as f32]
}

/// Who a speech cue is addressed to, as a directedness detector is scored on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Addressee {
    /// The owner talking to Enton: requests, follow-ups, barge-ins and the name itself.
    Enton,
    /// The owner talking to someone else in the room.
    Aside,
    /// Another person, including one who says the name to someone else.
    OtherPerson,
    /// A TV (or radio) voice.
    Tv,
    /// Not speech: household noise, a motor, ventilation.
    NoSpeech,
}

impl Addressee {
    fn of(source: Stimulus) -> Self {
        match source {
            Stimulus::Request(_) | Stimulus::BargeIn(_) => Self::Enton,
            Stimulus::Aside => Self::Aside,
            Stimulus::OtherSpeech | Stimulus::FalseKeyword => Self::OtherPerson,
            Stimulus::Tv => Self::Tv,
            // Enton's own echo never reaches a tape; the runner gives it a fixed reading.
            Stimulus::Noise | Stimulus::Motor | Stimulus::Ventilation | Stimulus::SelfEcho => {
                Self::NoSpeech
            }
        }
    }

    /// Which bands, low to high, are the owner's mistakes for this addressee.
    fn confusable(self) -> Option<[bool; 3]> {
        match self {
            Self::Enton => Some([true, true, false]),
            Self::Aside => Some([false, true, true]),
            Self::OtherPerson | Self::Tv | Self::NoSpeech => None,
        }
    }
}

/// Hidden habits of one three-minute block that correlate directedness errors in it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct AddressHabits {
    /// The owner's asides are entangled with the conversation with Enton.
    entangled: bool,
    /// The owner talks to Enton casually.
    casual: bool,
}

impl AddressHabits {
    fn draw(rng: &mut SplitMix64) -> Self {
        let entangled = rng.real(0.0, 1.0) < ENTANGLED_BLOCK_RATE;
        let casual = rng.real(0.0, 1.0) < CASUAL_BLOCK_RATE;
        Self { entangled, casual }
    }
}

/// Band shares, low to high (clearly addressed to someone else, ambiguous, clearly
/// addressed to Enton), of the conservative directedness profile for one cue, before
/// the per-owner factor. Rows depend on duration: under 1 s, 1 to 3 s, over 3 s.
fn directed_shares(
    addressee: Addressee,
    duration_ms: u32,
    habits: AddressHabits,
    media: f32,
) -> [f32; 3] {
    let short = duration_ms < 1_000;
    let long = duration_ms > 3_000;
    match addressee {
        Addressee::Enton => {
            let [undirected, ambiguous, _] = if short {
                [0.07, 0.18, 0.75]
            } else {
                [0.05, 0.10, 0.85]
            };
            let missed = if habits.casual {
                CASUAL_REQUEST_MISSED
            } else {
                PLAIN_REQUEST_MISSED
            };
            let scale = missed / REQUEST_MISSED;
            [
                undirected * scale,
                ambiguous * scale,
                1.0 - (undirected + ambiguous) * scale,
            ]
        }
        Addressee::Aside => {
            let row = if short {
                [0.60, 0.28, 0.12]
            } else if long {
                [0.80, 0.08, 0.12]
            } else {
                [0.70, 0.18, 0.12]
            };
            let directed = if habits.entangled {
                ENTANGLED_ASIDE_DIRECTED
            } else {
                PLAIN_ASIDE_DIRECTED
            };
            with_directed_share(row, directed)
        }
        Addressee::OtherPerson => {
            if long {
                [0.85, 0.07, 0.08]
            } else {
                [0.75, 0.17, 0.08]
            }
        }
        Addressee::Tv => {
            let row = [0.80, 0.12, 0.08];
            if media < 0.5 {
                with_directed_share(row, MISSED_TV_DIRECTED)
            } else {
                row
            }
        }
        Addressee::NoSpeech => [1.0, 0.0, 0.0],
    }
}

/// The shares with the clearly-directed one set to `directed` and the other two
/// scaled in proportion to fill the rest.
fn with_directed_share(shares: [f32; 3], directed: f32) -> [f32; 3] {
    let [undirected, ambiguous, _] = shares;
    let scale = (1.0 - directed) / (undirected + ambiguous);
    [undirected * scale, ambiguous * scale, directed]
}

/// The shares with the owner's confusable ones multiplied by `factor`, renormalized.
fn with_owner_factor(shares: [f32; 3], addressee: Addressee, factor: f32) -> [f32; 3] {
    let Some(confusable) = addressee.confusable() else {
        return shares;
    };
    let mut scaled = shares;
    for (share, confused) in scaled.iter_mut().zip(confusable) {
        if confused {
            *share *= factor;
        }
    }
    let total: f32 = scaled.iter().sum();
    scaled.map(|share| share / total)
}

/// Linearly interpolate in `log2(duration_ms)` between anchors at 750, 1500 and 3000 ms,
/// holding end values outside [750, 3000].
#[must_use]
pub fn interpolate_log2(duration_ms: u32, y: [f32; 3]) -> f32 {
    let d = duration_ms as f32;
    if d <= 750.0 {
        y[0]
    } else if d <= 1500.0 {
        let w = (d / 750.0).log2();
        y[0] + w * (y[1] - y[0])
    } else if d <= 3000.0 {
        let w = (d / 1500.0).log2();
        y[1] + w * (y[2] - y[1])
    } else {
        y[2]
    }
}

/// A persistent direction offset in radians: von Mises, or anywhere at all now and then.
fn persistent_offset(rng: &mut SplitMix64) -> f64 {
    if rng.real(0.0, 1.0) < POSITION_OUTLIER {
        rng.angle()
    } else {
        rng.von_mises(POSITION_KAPPA)
    }
}

/// The owner's angle from the TV in radians: a bin of [`OWNER_TV_ANGLE_PERCENT`], a
/// uniform point in it, and a side.
fn owner_tv_angle(rng: &mut SplitMix64) -> f64 {
    let total: f64 = OWNER_TV_ANGLE_PERCENT.iter().sum();
    let mut roll = f64::from(rng.real(0.0, 1.0)) * total;
    let mut bin = OWNER_TV_ANGLE_PERCENT.len() - 1;
    for (index, share) in OWNER_TV_ANGLE_PERCENT.iter().enumerate() {
        if roll < *share {
            bin = index;
            break;
        }
        roll -= share;
    }
    let degrees = (bin as f64 + f64::from(rng.real(0.0, 1.0))) * OWNER_TV_ANGLE_BIN_DEG;
    let side = if rng.real(0.0, 1.0) < 0.5 { -1.0 } else { 1.0 };
    side * degrees.to_radians()
}

impl Seats {
    /// The owner at a measured angle from the TV (which stands at `tv`), with this
    /// block's persistent offset; other people anywhere around the room.
    fn draw(rng: &mut SplitMix64, tv: f64) -> Self {
        let owner = tv + owner_tv_angle(rng) + persistent_offset(rng);
        let other = rng.angle();
        Self { owner, other }
    }
}

fn draw_block_condition(rng: &mut SplitMix64) -> RoomCondition {
    let distance = if rng.real(0.0, 1.0) < 0.25 {
        Distance::Near
    } else {
        Distance::Far
    };
    let tv_roll = rng.real(0.0, 1.0);
    let tv = if tv_roll < 0.50 {
        TvBackground::Off
    } else if tv_roll < 0.85 {
        TvBackground::Moderate
    } else {
        TvBackground::Loud
    };
    let tv_content = if rng.real(0.0, 1.0) < 0.60 {
        TvContent::Dialogue
    } else {
        TvContent::Music
    };
    let acoustics = rng.normal(0.0, 1.0);
    let show = rng.normal(0.0, 0.08);
    RoomCondition {
        distance,
        tv,
        tv_content,
        acoustics,
        show,
    }
}

/// Synthesized turn role classifying segment completion expectations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnRole {
    /// Incomplete user turn segment preceding a pause before the final segment.
    Incomplete,
    /// Final or only segment of a user turn.
    Complete,
    /// Non-user audio (TV, background noise, or other talkers) without turn structure.
    Other,
}

struct Builder {
    rng: SplitMix64,
    condition_rng: SplitMix64,
    speaker_rng: SplitMix64,
    media_rng: SplitMix64,
    turn_rng: SplitMix64,
    directed_rng: SplitMix64,
    direction_rng: SplitMix64,
    /// Where the TV's readings cluster, in radians: its place plus the tape's offset.
    tv_azimuth: f64,
    /// Per-block seats, drawn on the direction stream.
    seats: Vec<Seats>,
    owner_offset: f32,
    /// Per-owner factor on the owner's confusable directedness shares.
    owner_confusion: f32,
    conditions: Vec<RoomCondition>,
    /// Per-block directedness habits, drawn on the directedness stream.
    habits: Vec<AddressHabits>,
    records: Vec<Record>,
    turns: Vec<Turn>,
    next_segment: u32,
    next_turn: u32,
}
impl Builder {
    fn new(seed: u64, duration_ms: u64) -> Self {
        let mut condition_rng = SplitMix64::with_seed(seed ^ CONDITION_STREAM_SALT);
        let owner_offset = condition_rng.normal(0.0, 0.05);
        let num_blocks = usize::try_from(duration_ms.div_ceil(180_000)).unwrap_or(0);
        let mut conditions = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            conditions.push(draw_block_condition(&mut condition_rng));
        }
        let mut directed_rng = SplitMix64::with_seed(seed ^ DIRECTED_STREAM_SALT);
        let owner_confusion = directed_rng.normal(0.0, OWNER_CONFUSION_SD).exp();
        let habits = (0..num_blocks)
            .map(|_| AddressHabits::draw(&mut directed_rng))
            .collect();
        let mut direction_rng = SplitMix64::with_seed(seed ^ DIRECTION_STREAM_SALT);
        let tv = direction_rng.angle();
        let tv_azimuth = tv + persistent_offset(&mut direction_rng);
        let seats = (0..num_blocks)
            .map(|_| Seats::draw(&mut direction_rng, tv))
            .collect();
        Self {
            rng: SplitMix64::with_seed(seed),
            condition_rng,
            speaker_rng: SplitMix64::with_seed(seed ^ SPEAKER_STREAM_SALT),
            media_rng: SplitMix64::with_seed(seed ^ MEDIA_STREAM_SALT),
            turn_rng: SplitMix64::with_seed(seed ^ TURN_STREAM_SALT),
            directed_rng,
            direction_rng,
            tv_azimuth,
            seats,
            owner_offset,
            owner_confusion,
            conditions,
            habits,
            records: Vec::new(),
            turns: Vec::new(),
            next_segment: 0,
            next_turn: 0,
        }
    }
    fn draw_speaker_sim(&mut self, source: Stimulus, duration_ms: u32, cond: RoomCondition) -> f32 {
        match source {
            Stimulus::Noise | Stimulus::Motor | Stimulus::Ventilation | Stimulus::SelfEcho => self
                .speaker_rng
                .real(NON_SPEECH_SPEAKER_SIM_MIN, NON_SPEECH_SPEAKER_SIM_MAX),
            Stimulus::Request(_) | Stimulus::BargeIn(_) | Stimulus::Aside => {
                let (means, sds) = match (cond.distance, cond.tv) {
                    (Distance::Near, TvBackground::Off) => ([0.53, 0.72, 0.80], [0.10, 0.08, 0.06]),
                    (Distance::Near, TvBackground::Moderate) => {
                        ([0.53 - 0.08, 0.72 - 0.10, 0.80 - 0.10], [0.11, 0.11, 0.11])
                    }
                    (Distance::Near, TvBackground::Loud) => {
                        ([0.53 - 0.14, 0.72 - 0.20, 0.80 - 0.20], [0.12, 0.12, 0.12])
                    }
                    (Distance::Far, TvBackground::Off) => ([0.40, 0.57, 0.63], [0.11, 0.11, 0.10]),
                    (Distance::Far, TvBackground::Moderate) => {
                        ([0.32, 0.47, 0.53], [0.11, 0.11, 0.11])
                    }
                    (Distance::Far, TvBackground::Loud) => ([0.26, 0.37, 0.43], [0.12, 0.12, 0.12]),
                };
                let mean = interpolate_log2(duration_ms, means);
                let row_sd = interpolate_log2(duration_ms, sds);
                let correlated = self.owner_offset - 0.04 * cond.acoustics;
                let total_sd = row_sd + 0.02;
                let latent_var = 0.05_f32.powi(2) + 0.04_f32.powi(2);
                let residual_sd = (total_sd.powi(2) - latent_var).max(0.02_f32.powi(2)).sqrt();
                let residual = self.speaker_rng.normal(0.0, residual_sd);
                (mean + correlated + residual).clamp(0.0, 1.0)
            }
            Stimulus::OtherSpeech | Stimulus::FalseKeyword => {
                let (means, sds) = match (cond.distance, cond.tv) {
                    (Distance::Near, TvBackground::Off) => ([0.43, 0.56, 0.64], [0.12, 0.11, 0.10]),
                    (Distance::Near, TvBackground::Moderate) => {
                        ([0.43 - 0.05, 0.56 - 0.08, 0.64 - 0.06], [0.12, 0.12, 0.12])
                    }
                    (Distance::Near, TvBackground::Loud) => {
                        ([0.43 - 0.08, 0.56 - 0.11, 0.64 - 0.13], [0.12, 0.12, 0.12])
                    }
                    (Distance::Far, TvBackground::Off) => ([0.34, 0.46, 0.51], [0.13, 0.13, 0.13]),
                    (Distance::Far, TvBackground::Moderate) => {
                        ([0.29, 0.38, 0.45], [0.12, 0.12, 0.12])
                    }
                    (Distance::Far, TvBackground::Loud) => ([0.26, 0.35, 0.38], [0.12, 0.12, 0.12]),
                };
                let mean = interpolate_log2(duration_ms, means);
                let row_sd = interpolate_log2(duration_ms, sds);
                let correlated = -0.02 * cond.acoustics;
                let total_sd = row_sd + 0.02;
                let latent_var = 0.02_f32.powi(2);
                let residual_sd = (total_sd.powi(2) - latent_var).max(0.02_f32.powi(2)).sqrt();
                let residual = self.speaker_rng.normal(0.0, residual_sd);
                (mean + correlated + residual).clamp(0.0, 1.0)
            }
            Stimulus::Tv => {
                let means = [0.24, 0.34, 0.38];
                let sds = [0.12, 0.13, 0.13];
                let mean = interpolate_log2(duration_ms, means);
                let row_sd = interpolate_log2(duration_ms, sds);
                let correlated = -0.02 * cond.acoustics;
                let total_sd = row_sd + 0.02;
                let latent_var = 0.02_f32.powi(2);
                let residual_sd = (total_sd.powi(2) - latent_var).max(0.02_f32.powi(2)).sqrt();
                let residual = self.speaker_rng.normal(0.0, residual_sd);
                (mean + correlated + residual).clamp(0.0, 1.0)
            }
        }
    }
    fn draw_media(&mut self, source: Stimulus, duration_ms: u32, cond: RoomCondition) -> f32 {
        match source {
            Stimulus::Noise | Stimulus::Motor | Stimulus::Ventilation | Stimulus::SelfEcho => self
                .media_rng
                .real(NON_SPEECH_MEDIA_MIN, NON_SPEECH_MEDIA_MAX),
            Stimulus::Request(_)
            | Stimulus::BargeIn(_)
            | Stimulus::Aside
            | Stimulus::OtherSpeech
            | Stimulus::FalseKeyword => {
                let means = match (cond.distance, cond.tv) {
                    (Distance::Near, TvBackground::Off) => [0.28, 0.24, 0.20],
                    (Distance::Far, TvBackground::Off) => [0.32, 0.28, 0.24],
                    (_, TvBackground::Moderate | TvBackground::Loud) => [0.42, 0.38, 0.34],
                };
                let mean = interpolate_log2(duration_ms, means);
                let correlated = 0.05 * cond.acoustics;
                let latent_var = 0.05_f32.powi(2);
                let residual_sd = (0.20_f32.powi(2) - latent_var).max(0.0).sqrt();
                let residual = self.media_rng.normal(0.0, residual_sd);
                (mean + correlated + residual).clamp(0.0, 1.0)
            }
            Stimulus::Tv => {
                let means = match cond.tv_content {
                    TvContent::Dialogue => [0.52, 0.56, 0.60],
                    TvContent::Music => [0.68, 0.75, 0.80],
                };
                let mean = interpolate_log2(duration_ms, means);
                let correlated = -0.05 * cond.acoustics + cond.show;
                let latent_var = 0.05_f32.powi(2) + 0.08_f32.powi(2);
                let residual_sd = (0.20_f32.powi(2) - latent_var).max(0.0).sqrt();
                let residual = self.media_rng.normal(0.0, residual_sd);
                (mean + correlated + residual).clamp(0.0, 1.0)
            }
        }
    }
    fn draw_turn_complete(
        &mut self,
        role: TurnRole,
        duration_ms: u32,
        cond: RoomCondition,
        pause_style: Option<f32>,
    ) -> f32 {
        let q_prime = if cond.tv == TvBackground::Off {
            cond.acoustics
        } else {
            cond.acoustics.max(1.0)
        };
        match (role, pause_style) {
            (TurnRole::Incomplete, Some(a)) => {
                let m = if duration_ms < 1000 {
                    -4.3
                } else if duration_ms <= 2000 {
                    -1.4
                } else {
                    0.0
                };
                let shift = 0.5 * q_prime;
                let e = self.turn_rng.normal(0.0, 4.2);
                let l = m + a + shift + e;
                (1.0 / (1.0 + (-l).exp())).clamp(0.0, 1.0)
            }
            (TurnRole::Complete, Some(a)) => {
                let m = if duration_ms < 1000 {
                    1.9
                } else if duration_ms <= 2000 {
                    5.1
                } else {
                    4.9
                };
                let shift = -0.5 * q_prime;
                let e = self.turn_rng.normal(0.0, 4.2);
                let l = m + a + shift + e;
                (1.0 / (1.0 + (-l).exp())).clamp(0.0, 1.0)
            }
            _ => {
                let l = self.turn_rng.normal(0.9, 5.0);
                (1.0 / (1.0 + (-l).exp())).clamp(0.0, 1.0)
            }
        }
    }
    /// A directedness score: a band from the cue's shares, then a uniform point in it.
    /// `media` is the tagger's reading of the same cue, for the TV lines it missed.
    fn draw_directed(
        &mut self,
        source: Stimulus,
        duration_ms: u32,
        media: f32,
        block: usize,
    ) -> f32 {
        let addressee = Addressee::of(source);
        let habits = self.habits.get(block).copied().unwrap_or_default();
        let shares = with_owner_factor(
            directed_shares(addressee, duration_ms, habits, media),
            addressee,
            self.owner_confusion,
        );
        let [undirected, ambiguous, _] = shares;
        let [low_edge, high_edge] = DIRECTED_BAND_EDGES;
        let roll = self.directed_rng.real(0.0, 1.0);
        let (low, high) = if roll < undirected {
            (0.0, low_edge)
        } else if roll < undirected + ambiguous {
            (low_edge, high_edge)
        } else {
            (high_edge, 1.0)
        };
        let score = self.directed_rng.real(low, high);
        // A lower band is half-open: rounding must not carry a score onto its upper edge.
        if high < 1.0 {
            score.min(high.next_down())
        } else {
            score
        }
    }
    /// A direction-of-arrival reading as a unit vector: a TV line around the TV, a live
    /// talker around their seat (or, with the TV on, sometimes at the TV), anything else
    /// anywhere.
    fn draw_direction(&mut self, source: Stimulus, cond: RoomCondition, block: usize) -> [f32; 2] {
        let seats = self.seats.get(block).copied().unwrap_or_default();
        let azimuth = match source {
            Stimulus::Tv => self.tv_reading(),
            Stimulus::Request(_) | Stimulus::BargeIn(_) | Stimulus::Aside => {
                self.live_reading(seats.owner, cond.tv)
            }
            Stimulus::OtherSpeech | Stimulus::FalseKeyword => {
                self.live_reading(seats.other, cond.tv)
            }
            // Enton's own echo never reaches a tape; the runner gives it no reading.
            Stimulus::Noise | Stimulus::Motor | Stimulus::Ventilation | Stimulus::SelfEcho => {
                self.direction_rng.angle()
            }
        };
        unit(azimuth)
    }
    fn tv_reading(&mut self) -> f64 {
        if self.direction_rng.real(0.0, 1.0) < TV_OUTLIER {
            self.direction_rng.angle()
        } else {
            self.tv_azimuth + self.direction_rng.von_mises(TV_SEGMENT_KAPPA)
        }
    }
    fn live_reading(&mut self, seat: f64, tv: TvBackground) -> f64 {
        let [quiet, moderate, loud] = LIVE_OUTLIER;
        let [moderate_capture, loud_capture] = TV_CAPTURE;
        let (capture, outlier) = match tv {
            TvBackground::Off => (0.0, quiet),
            TvBackground::Moderate => (moderate_capture, moderate),
            TvBackground::Loud => (loud_capture, loud),
        };
        if self.direction_rng.real(0.0, 1.0) < capture {
            return self.tv_reading();
        }
        if self.direction_rng.real(0.0, 1.0) < outlier {
            self.direction_rng.angle()
        } else {
            seat + self.direction_rng.von_mises(LIVE_SEGMENT_KAPPA)
        }
    }
    fn segment(
        &mut self,
        end: u64,
        mut cue: SpeechCue,
        episode: EpisodeId,
        source: Stimulus,
        role: TurnRole,
        pause_style: Option<f32>,
    ) -> SegmentId {
        let block_idx = usize::try_from(end / 180_000)
            .unwrap_or(0)
            .min(self.conditions.len().saturating_sub(1));
        let cond = self.conditions.get(block_idx).copied().unwrap_or_default();
        cue.speaker_sim = Some(self.draw_speaker_sim(source, cue.duration_ms, cond));
        let media = self.draw_media(source, cue.duration_ms, cond);
        cue.media = Some(media);
        cue.turn_complete = Some(self.draw_turn_complete(role, cue.duration_ms, cond, pause_style));
        cue.directed = Some(self.draw_directed(source, cue.duration_ms, media, block_idx));
        cue.direction = Some(self.draw_direction(source, cond, block_idx));
        let id = SegmentId(self.next_segment);
        // All generator loops are protocol-bounded to fewer than MAX_EVENTS entries.
        self.next_segment += 1;
        self.records.push(Record {
            event: Event::Speech {
                now: Millis(end),
                cue,
            },
            annotation: Annotation::Speech {
                segment: id,
                episode: Some(episode),
                source,
                pause_style,
            },
        });
        id
    }
    fn user_cue(&mut self, duration_ms: u32, keyword: bool) -> SpeechCue {
        SpeechCue {
            duration_ms,
            keyword,
            energy: self.rng.real(0.65, 1.0),
            vad_confidence: self.rng.real(0.65, 1.0),
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
            direction: None,
        }
    }
    fn request(
        &mut self,
        start: u64,
        episode: EpisodeId,
        duration: u32,
        keyword: bool,
        barge_in: bool,
        meta: (crate::tape::TurnKind, Option<u32>),
    ) -> u64 {
        let (kind, gap) = meta;
        let id = TurnId(self.next_turn);
        self.next_turn += 1;
        let end = start + u64::from(duration);
        let cue = self.user_cue(duration, keyword);
        let source = if barge_in {
            Stimulus::BargeIn(id)
        } else {
            Stimulus::Request(id)
        };
        let pause_style = self.condition_rng.normal(0.0, 2.7);
        let segment = self.segment(
            end,
            cue,
            episode,
            source,
            TurnRole::Complete,
            Some(pause_style),
        );
        self.turns.push(Turn {
            id,
            episode,
            segments: vec![segment],
            available_at: Millis(end),
            deadline: Millis(end + 10_000),
            kind,
            gap,
            pause_style: Some(pause_style),
        });
        end
    }
    fn split(&mut self, start: u64, episode: EpisodeId, distractor: Option<EpisodeId>) -> u64 {
        let id = TurnId(self.next_turn);
        self.next_turn += 1;
        let pause_style = self.condition_rng.normal(0.0, 2.7);
        let name_duration = self.rng.range(250, 650);
        let name_end = start + u64::from(name_duration);
        let name = self.user_cue(name_duration, true);
        let first = self.segment(
            name_end,
            name,
            episode,
            Stimulus::Request(id),
            TurnRole::Incomplete,
            Some(pause_style),
        );
        let content_base = if let Some(distractor_episode) = distractor {
            let delay = self.rng.range(50, 300);
            let distractor_duration = self.rng.range(300, 600);
            let distractor_end = name_end + u64::from(delay) + u64::from(distractor_duration);
            self.other(distractor_end, distractor_episode, distractor_duration);
            distractor_end
        } else {
            name_end
        };
        let pause = self.rng.range(150, 700);
        let duration = self.rng.range(700, 1600);
        let end = content_base + u64::from(pause) + u64::from(duration);
        let cue = self.user_cue(duration, false);
        let second = self.segment(
            end,
            cue,
            episode,
            Stimulus::Request(id),
            TurnRole::Complete,
            Some(pause_style),
        );
        self.turns.push(Turn {
            id,
            episode,
            segments: vec![first, second],
            available_at: Millis(end),
            deadline: Millis(end + 10_000),
            kind: crate::tape::TurnKind::Split,
            gap: None,
            pause_style: Some(pause_style),
        });
        name_end
    }
    fn conversation(&mut self, start: u64, episode: EpisodeId, block: u32) {
        const GAPS: [u32; 7] = [1500, 3000, 4900, 5000, 5100, 9000, 12_000];
        let duration = self.rng.range(1000, 1800);
        let mut end = self.request(
            start,
            episode,
            duration,
            true,
            false,
            (crate::tape::TurnKind::Conversation, None),
        );
        for turn in 0..4 {
            let gap = GAPS
                .get(((block * 4 + turn) % 7) as usize)
                .copied()
                .unwrap_or(1500);
            let jitter = self.rng.range(0, 200);
            let next_end = end + u64::from(gap - 100 + jitter);
            let duration = self.rng.range(600, 1200);
            end = self.request(
                next_end - u64::from(duration),
                episode,
                duration,
                false,
                false,
                (crate::tape::TurnKind::Conversation, Some(gap)),
            );
        }
    }
    fn other(&mut self, end: u64, episode: EpisodeId, duration: u32) {
        let cue = SpeechCue {
            duration_ms: duration,
            keyword: false,
            energy: self.rng.real(0.55, 0.95),
            vad_confidence: self.rng.real(0.65, 1.0),
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
            direction: None,
        };
        self.segment(
            end,
            cue,
            episode,
            Stimulus::OtherSpeech,
            TurnRole::Other,
            None,
        );
    }
    fn aside(&mut self, end: u64, episode: EpisodeId, duration: u32) {
        let pause_style = self.condition_rng.normal(0.0, 2.7);
        let cue = self.user_cue(duration, false);
        self.segment(
            end,
            cue,
            episode,
            Stimulus::Aside,
            TurnRole::Complete,
            Some(pause_style),
        );
    }
    fn noise(&mut self, start: u64, episode: EpisodeId) {
        let duration_ms = self.rng.range(250, 1000);
        let cue = SpeechCue {
            duration_ms,
            keyword: false,
            energy: self.rng.real(0.1, 0.8),
            vad_confidence: self.rng.real(0.02, 0.35),
            speaker_sim: None,
            media: None,
            turn_complete: None,
            directed: None,
            direction: None,
        };
        self.segment(
            start + u64::from(duration_ms),
            cue,
            episode,
            Stimulus::Noise,
            TurnRole::Other,
            None,
        );
    }
    fn tv_on(&self, block: u32) -> bool {
        usize::try_from(block)
            .ok()
            .and_then(|block| self.conditions.get(block))
            .is_some_and(|condition| condition.tv != TvBackground::Off)
    }
    fn television(&mut self, start: u64, episode: EpisodeId) {
        let span = u64::from(self.rng.range(120_000, 165_000));
        let energy = self.rng.real(0.75, 0.95);
        let vad = self.rng.real(0.80, 0.98);
        let mut now = start;
        while now + 2000 <= start + span {
            let duration_ms = self.rng.range(1000, 2000);
            // The set's volume is fixed, but speech level moves from line to line and
            // speaker to speaker. Drawn on the condition stream, so timing is untouched.
            let level = (energy + self.condition_rng.normal(0.0, TV_ENERGY_JITTER)).clamp(0.0, 1.0);
            let voicing = (vad + self.condition_rng.normal(0.0, TV_VAD_JITTER)).clamp(0.0, 1.0);
            self.segment(
                now + u64::from(duration_ms),
                SpeechCue {
                    duration_ms,
                    keyword: false,
                    energy: level,
                    vad_confidence: voicing,
                    speaker_sim: None,
                    media: None,
                    turn_complete: None,
                    directed: None,
                    direction: None,
                },
                episode,
                Stimulus::Tv,
                TurnRole::Other,
                None,
            );
            now += u64::from(self.rng.range(3000, 6000));
        }
    }
    fn finish(
        mut self,
        kind: TapeKind,
        seed: u64,
        duration: u64,
        noise: Vec<Interval>,
    ) -> Result<Tape, Error> {
        // Insert ticks first at equal timestamps, without changing speech order.
        let ticks = (0..=duration).step_by(1000).map(|now| Record {
            event: Event::Tick { now: Millis(now) },
            annotation: Annotation::Clock,
        });
        let mut records: Vec<_> = ticks.collect();
        records.append(&mut self.records);
        records.sort_by_key(|record| record.event.now());
        Tape::new(
            kind,
            seed,
            Millis(duration),
            records,
            self.turns,
            noise,
            self.conditions,
        )
    }
}

/// Generate the declared E1a mixed population; no controller is run or consulted.
pub fn e1a(seed: u64) -> Result<Tape, Error> {
    let mut b = Builder::new(seed, 3_600_000);
    for block in 0u32..20 {
        let origin = u64::from(block) * 180_000;
        let episode = block * 5;
        let distractor = 100 + block * 7;
        let mut starts = [0u64; 5];
        for (value, offset) in starts
            .iter_mut()
            .zip([5000, 30_000, 55_000, 90_000, 120_000])
        {
            *value = origin + offset + u64::from(b.rng.range(0, 2000));
        }
        let [single, split, conversation, short, interruption] = starts;
        let duration = b.rng.range(1000, 1800);
        let first_end = b.request(
            single,
            EpisodeId(episode),
            duration,
            true,
            false,
            (crate::tape::TurnKind::Single, None),
        );
        let split_distractor = if block.is_multiple_of(2) {
            None
        } else {
            Some(EpisodeId(distractor + 2))
        };
        let name_end = b.split(split, EpisodeId(episode + 1), split_distractor);
        b.conversation(conversation, EpisodeId(episode + 2), block);
        let duration = b.rng.range(250, 700);
        let short_end = b.request(
            short,
            EpisodeId(episode + 3),
            duration,
            true,
            false,
            (crate::tape::TurnKind::Short, None),
        );
        let duration = b.rng.range(1000, 1800);
        let end = b.request(
            interruption,
            EpisodeId(episode + 4),
            duration,
            true,
            false,
            (crate::tape::TurnKind::Interruption, None),
        );
        b.request(
            end + 600,
            EpisodeId(episode + 4),
            600,
            false,
            true,
            (crate::tape::TurnKind::Interruption, None),
        );
        // The show plays only where the block's TV is on, the same TV that degrades
        // the owner's sensors there.
        if b.tv_on(block) {
            b.television(origin, EpisodeId(distractor));
        }
        b.other(first_end + 1000, EpisodeId(distractor + 1), 700);
        if block.is_multiple_of(2) {
            b.segment(
                name_end + 400,
                SpeechCue {
                    energy: 0.8,
                    vad_confidence: 0.8,
                    duration_ms: 250,
                    keyword: true,
                    speaker_sim: None,
                    media: None,
                    turn_complete: None,
                    directed: None,
                    direction: None,
                },
                EpisodeId(distractor + 2),
                Stimulus::FalseKeyword,
                TurnRole::Other,
                None,
            );
        }
        for n in 0..5 {
            b.noise(origin + 60_000 + n * 3000, EpisodeId(distractor + 3));
        }
        let duration = b.rng.range(700, 1600);
        b.other(origin + 110_000, EpisodeId(distractor + 4), duration);
        let aside_end = first_end + u64::from(b.rng.range(5000, 9000));
        let aside_duration = b.rng.range(800, 1600);
        b.aside(aside_end, EpisodeId(distractor + 5), aside_duration);
        let in_window_end = short_end + u64::from(b.rng.range(3000, 6000));
        let in_window_duration = b.rng.range(700, 1400);
        b.other(in_window_end, EpisodeId(distractor + 6), in_window_duration);
    }
    b.finish(TapeKind::E1a, seed, 3_600_000, vec![])
}

/// Generate ten commands and fifty minutes of household noise cues over one hour.
pub fn e1b(seed: u64) -> Result<Tape, Error> {
    let mut b = Builder::new(seed, 3_600_000);
    let mut intervals = Vec::with_capacity(10);
    for block in 0u32..10 {
        let origin = u64::from(block) * 360_000;
        intervals.push(Interval {
            start: Millis(origin),
            end: Millis(origin + 300_000),
        });
        for n in 0u32..100 {
            let start = origin + u64::from(n) * 3000 + u64::from(b.rng.range(0, 500));
            let (energy, vad, duration_ms, source) = match n % 4 {
                0 => (
                    b.rng.real(0.3, 0.8),
                    b.rng.real(0.05, 0.3),
                    b.rng.range(250, 1000),
                    Stimulus::Motor,
                ),
                1 => (
                    b.rng.real(0.1, 0.4),
                    b.rng.real(0.02, 0.2),
                    b.rng.range(250, 1000),
                    Stimulus::Ventilation,
                ),
                2 => (
                    b.rng.real(0.7, 0.95),
                    b.rng.real(0.8, 0.98),
                    b.rng.range(1200, 2400),
                    Stimulus::Tv,
                ),
                _ => (
                    b.rng.real(0.55, 0.95),
                    b.rng.real(0.65, 1.0),
                    b.rng.range(700, 1600),
                    Stimulus::OtherSpeech,
                ),
            };
            b.segment(
                start + u64::from(duration_ms),
                SpeechCue {
                    energy,
                    vad_confidence: vad,
                    duration_ms,
                    keyword: false,
                    speaker_sim: None,
                    media: None,
                    turn_complete: None,
                    directed: None,
                    direction: None,
                },
                EpisodeId(100 + block),
                source,
                TurnRole::Other,
                None,
            );
        }
        let start = origin + 300_000 + u64::from(b.rng.range(1000, 54_000));
        let duration = b.rng.range(1000, 1800);
        b.request(
            start,
            EpisodeId(block),
            duration,
            true,
            false,
            (crate::tape::TurnKind::Command, None),
        );
    }
    b.finish(TapeKind::E1b, seed, 3_600_000, intervals)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_c_known_answer_vectors() {
        // Compiled Vigna's linked reference C on 2026-09-25, setting its x to
        // each seed. These literals are not calculated by this Rust code.
        for (seed, expected) in [
            (
                0,
                [
                    0xe220_a839_7b1d_cdaf,
                    0x6e78_9e6a_a1b9_65f4,
                    0x06c4_5d18_8009_454f,
                    0xf88b_b8a8_724c_81ec,
                    0x1b39_896a_51a8_749b,
                    0x53cb_9f0c_747e_a2ea,
                    0x2c82_9abe_1f45_32e1,
                    0xc584_133a_c916_ab3c,
                ],
            ),
            (
                256,
                [
                    0x6602_d201_e324_653f,
                    0xf648_421a_27ca_8ae0,
                    0x7a63_9a53_5657_02ad,
                    0x963d_391a_95bf_a6ac,
                    0xa014_bb22_a044_6111,
                    0x8498_2fdb_4d57_da0a,
                    0xa9bd_6932_63ce_3676,
                    0x66e8_bcf0_c694_e249,
                ],
            ),
        ] {
            let mut rng = SplitMix64::with_seed(seed);
            for value in expected {
                assert_eq!(rng.next_u64(), value);
            }
        }
    }
    #[test]
    fn old_colliding_seeds_now_produce_diverse_stimuli() {
        let a = e1a(0).unwrap();
        let b = e1a(256).unwrap();
        let features = |t: &Tape| {
            t.records()
                .iter()
                .filter_map(|r| match r.event {
                    Event::Speech { cue, .. } => Some(cue),
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let af = features(&a);
        let bf = features(&b);
        let different = af.iter().zip(&bf).filter(|(a, b)| a != b).count();
        assert!(different * 10 > af.len() * 8);
        assert_eq!(a, e1a(0).unwrap());
    }
    #[test]
    fn calibration_populations_have_full_counts_and_noise() {
        // Never cross the calibration boundary, including under the full gate.
        for seed in [0, 1, 7, 42, 256, 999] {
            let a = e1a(seed).unwrap();
            let b = e1b(seed).unwrap();
            let relevant: std::collections::BTreeSet<_> =
                a.turns().iter().map(|t| t.episode).collect();
            let distractors: std::collections::BTreeSet<_> = a
                .records()
                .iter()
                .filter_map(|r| match r.annotation {
                    Annotation::Speech {
                        episode, source, ..
                    } if source.turn().is_none() => episode,
                    _ => None,
                })
                .collect();
            assert_eq!(relevant.len(), 100);
            let shows = a
                .conditions()
                .iter()
                .filter(|condition| condition.tv != TvBackground::Off)
                .count();
            assert_eq!(distractors.len(), 120 + shows);
            assert_eq!(a.turns().len(), 200);
            assert_eq!(b.turns().len(), 10);
            assert_eq!(
                b.noise_intervals()
                    .iter()
                    .map(|s| s.end.0 - s.start.0)
                    .sum::<u64>(),
                3_000_000
            );
            assert_eq!(
                b.records()
                    .iter()
                    .filter(|r| matches!(r.event, Event::Speech { .. })
                        && r.annotation.turn().is_none())
                    .count(),
                1000
            );
            assert!(a.records().windows(3).any(|rows| {
                rows.iter().any(|r| r.annotation.turn().is_some())
                    && rows.iter().any(|r| {
                        matches!(
                            r.annotation.source(),
                            Some(Stimulus::Tv | Stimulus::OtherSpeech)
                        )
                    })
            }));
        }
    }
    #[test]
    fn normal_sampler_known_answer_and_distribution() {
        let mut rng = SplitMix64::with_seed(42);
        let sample0 = rng.normal(0.0, 1.0);
        let sample1 = rng.normal(0.0, 1.0);
        assert!((sample0 - 0.414_719_72).abs() < 1e-4);
        assert!((sample1 - (-0.891_886_1)).abs() < 1e-4);

        let mut draws = Vec::with_capacity(20_000);
        for _ in 0..20_000 {
            draws.push(rng.normal(2.5, 0.75));
        }
        let mean = draws.iter().copied().sum::<f32>() / draws.len() as f32;
        let var = draws.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (draws.len() - 1) as f32;
        let sd = var.sqrt();
        assert!(
            (mean - 2.5).abs() < 0.03,
            "mean was {mean}, expected near 2.5"
        );
        assert!((sd - 0.75).abs() < 0.03, "sd was {sd}, expected near 0.75");
    }

    #[test]
    fn duration_interpolation_log2() {
        let y = [0.10, 0.50, 0.90];
        assert!((interpolate_log2(750, y) - 0.10).abs() < f32::EPSILON);
        assert!((interpolate_log2(1500, y) - 0.50).abs() < f32::EPSILON);
        assert!((interpolate_log2(3000, y) - 0.90).abs() < f32::EPSILON);
        assert!((interpolate_log2(6000, y) - 0.90).abs() < f32::EPSILON);
        assert!((interpolate_log2(500, y) - 0.10).abs() < f32::EPSILON);
        let at_1060 = interpolate_log2(1060, y);
        let expected_1060 = 0.10 + (1060.0_f32 / 750.0).log2() * (0.50 - 0.10);
        assert!((at_1060 - expected_1060).abs() < 1e-5);
    }

    #[test]
    fn conditions_per_block_and_frequencies_over_seeds() {
        let mut total_blocks = 0;
        let mut near_count = 0;
        let mut tv_off = 0;
        let mut tv_mod = 0;
        let mut tv_loud = 0;
        let mut tv_dialogue = 0;
        let mut acoustics_sum = 0.0_f32;
        let mut show_sum = 0.0_f32;

        for seed in 0..=31 {
            let tape = e1a(seed).unwrap();
            let conditions = tape.conditions();
            assert_eq!(
                conditions.len(),
                20,
                "each 1-hour tape must have 20 conditions"
            );
            for cond in conditions {
                total_blocks += 1;
                if cond.distance == Distance::Near {
                    near_count += 1;
                }
                match cond.tv {
                    TvBackground::Off => tv_off += 1,
                    TvBackground::Moderate => tv_mod += 1,
                    TvBackground::Loud => tv_loud += 1,
                }
                if cond.tv_content == TvContent::Dialogue {
                    tv_dialogue += 1;
                }
                acoustics_sum += cond.acoustics;
                show_sum += cond.show;
            }
        }

        assert_eq!(total_blocks, 640);
        let near_frac = near_count as f32 / total_blocks as f32;
        let off_frac = tv_off as f32 / total_blocks as f32;
        let mod_frac = tv_mod as f32 / total_blocks as f32;
        let loud_frac = tv_loud as f32 / total_blocks as f32;
        let dia_frac = tv_dialogue as f32 / total_blocks as f32;
        let acoustics_mean = acoustics_sum / total_blocks as f32;
        let show_mean = show_sum / total_blocks as f32;

        assert!(
            (0.18..=0.32).contains(&near_frac),
            "near_frac was {near_frac}, expected ~0.25"
        );
        assert!(
            (0.44..=0.56).contains(&off_frac),
            "off_frac was {off_frac}, expected ~0.50"
        );
        assert!(
            (0.29..=0.41).contains(&mod_frac),
            "mod_frac was {mod_frac}, expected ~0.35"
        );
        assert!(
            (0.10..=0.20).contains(&loud_frac),
            "loud_frac was {loud_frac}, expected ~0.15"
        );
        assert!(
            (0.54..=0.66).contains(&dia_frac),
            "dia_frac was {dia_frac}, expected ~0.60"
        );
        assert!(
            acoustics_mean.abs() < 0.15,
            "acoustics mean was {acoustics_mean}"
        );
        assert!(show_mean.abs() < 0.02, "show mean was {show_mean}");
    }

    #[test]
    fn owner_far_tv_off_1500_ms_draws() {
        let mut b = Builder::new(42, 3_600_000);
        let mut draws = Vec::with_capacity(20_000);
        let cond_base = RoomCondition {
            distance: Distance::Far,
            tv: TvBackground::Off,
            tv_content: TvContent::Dialogue,
            acoustics: 0.0,
            show: 0.0,
        };
        for _ in 0..20_000 {
            b.owner_offset = b.condition_rng.normal(0.0, 0.05);
            let cond = RoomCondition {
                acoustics: b.condition_rng.normal(0.0, 1.0),
                ..cond_base
            };
            let sim = b.draw_speaker_sim(Stimulus::Request(TurnId(1)), 1500, cond);
            draws.push(sim);
        }
        let mean = draws.iter().copied().sum::<f32>() / draws.len() as f32;
        let var = draws.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (draws.len() - 1) as f32;
        let sd = var.sqrt();
        let below_06 = draws.iter().filter(|&&s| s < 0.6).count() as f32 / draws.len() as f32;
        assert!(
            (mean - 0.57).abs() < 0.03,
            "mean was {mean}, expected near 0.57"
        );
        assert!((sd - 0.13).abs() < 0.03, "sd was {sd}, expected near 0.13");
        assert!(
            (0.52..=0.68).contains(&below_06),
            "below 0.6 was {below_06}, expected in 0.52..=0.68"
        );
    }

    #[test]
    fn relative_far_tv_off_3000_ms_draws() {
        let mut b = Builder::new(43, 3_600_000);
        let mut draws = Vec::with_capacity(20_000);
        let cond_base = RoomCondition {
            distance: Distance::Far,
            tv: TvBackground::Off,
            tv_content: TvContent::Dialogue,
            acoustics: 0.0,
            show: 0.0,
        };
        for _ in 0..20_000 {
            let cond = RoomCondition {
                acoustics: b.condition_rng.normal(0.0, 1.0),
                ..cond_base
            };
            let sim = b.draw_speaker_sim(Stimulus::OtherSpeech, 3000, cond);
            draws.push(sim);
        }
        let mean = draws.iter().copied().sum::<f32>() / draws.len() as f32;
        let var = draws.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (draws.len() - 1) as f32;
        let sd = var.sqrt();
        let ge_06 = draws.iter().filter(|&&s| s >= 0.6).count() as f32 / draws.len() as f32;
        assert!(
            (mean - 0.51).abs() < 0.03,
            "mean was {mean}, expected near 0.51"
        );
        assert!((sd - 0.15).abs() < 0.03, "sd was {sd}, expected near 0.15");
        assert!(
            (0.18..=0.32).contains(&ge_06),
            "ge 0.6 was {ge_06}, expected in 0.18..=0.32"
        );
    }

    #[test]
    fn tv_dialogue_1500_ms_media_draws() {
        let mut b = Builder::new(44, 3_600_000);
        let mut draws = Vec::with_capacity(20_000);
        let cond_base = RoomCondition {
            distance: Distance::Near,
            tv: TvBackground::Loud,
            tv_content: TvContent::Dialogue,
            acoustics: 0.0,
            show: 0.0,
        };
        for _ in 0..20_000 {
            let cond = RoomCondition {
                acoustics: b.condition_rng.normal(0.0, 1.0),
                show: b.condition_rng.normal(0.0, 0.08),
                ..cond_base
            };
            let media = b.draw_media(Stimulus::Tv, 1500, cond);
            draws.push(media);
        }
        let mean = draws.iter().copied().sum::<f32>() / draws.len() as f32;
        let var = draws.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (draws.len() - 1) as f32;
        let sd = var.sqrt();
        let below_05 = draws.iter().filter(|&&m| m < 0.5).count() as f32 / draws.len() as f32;
        assert!(
            (mean - 0.56).abs() < 0.03,
            "mean was {mean}, expected near 0.56"
        );
        assert!((sd - 0.20).abs() < 0.03, "sd was {sd}, expected near 0.20");
        assert!(
            (0.30..=0.46).contains(&below_05),
            "below 0.5 was {below_05}, expected in 0.30..=0.46"
        );
    }

    #[test]
    fn incomplete_below_1000_ms_turn_complete_draws() {
        let mut b = Builder::new(45, 3_600_000);
        let mut draws = Vec::with_capacity(20_000);
        let cond_base = RoomCondition {
            distance: Distance::Near,
            tv: TvBackground::Off,
            tv_content: TvContent::Dialogue,
            acoustics: 0.0,
            show: 0.0,
        };
        for _ in 0..20_000 {
            let cond = RoomCondition {
                acoustics: b.condition_rng.normal(0.0, 1.0),
                ..cond_base
            };
            let pause_style = b.condition_rng.normal(0.0, 2.7);
            let score = b.draw_turn_complete(TurnRole::Incomplete, 500, cond, Some(pause_style));
            draws.push(score);
        }
        let ge_05 = draws.iter().filter(|&&t| t >= 0.5).count() as f32 / draws.len() as f32;
        assert!(
            (0.15..=0.26).contains(&ge_05),
            "ge 0.5 was {ge_05}, expected in 0.15..=0.26"
        );
    }

    #[test]
    fn complete_below_1000_ms_turn_complete_draws() {
        let mut b = Builder::new(46, 3_600_000);
        let mut draws = Vec::with_capacity(20_000);
        let cond_base = RoomCondition {
            distance: Distance::Near,
            tv: TvBackground::Off,
            tv_content: TvContent::Dialogue,
            acoustics: 0.0,
            show: 0.0,
        };
        for _ in 0..20_000 {
            let cond = RoomCondition {
                acoustics: b.condition_rng.normal(0.0, 1.0),
                ..cond_base
            };
            let pause_style = b.condition_rng.normal(0.0, 2.7);
            let score = b.draw_turn_complete(TurnRole::Complete, 500, cond, Some(pause_style));
            draws.push(score);
        }
        let lt_05 = draws.iter().filter(|&&t| t < 0.5).count() as f32 / draws.len() as f32;
        assert!(
            (0.28..=0.42).contains(&lt_05),
            "lt 0.5 was {lt_05}, expected in 0.28..=0.42"
        );
    }

    #[test]
    fn condition_draws_do_not_alter_other_streams() {
        for seed in [0, 1, 7, 42] {
            let mut b1 = Builder::new(seed, 3_600_000);
            let mut b2 = Builder::new(seed, 3_600_000);
            for _ in 0..100 {
                b2.condition_rng.next_u64();
                b2.condition_rng.normal(0.0, 1.0);
            }
            for _ in 0..100 {
                assert_eq!(b1.rng.next_u64(), b2.rng.next_u64());
                assert_eq!(b1.speaker_rng.next_u64(), b2.speaker_rng.next_u64());
                assert_eq!(b1.media_rng.next_u64(), b2.media_rng.next_u64());
                assert_eq!(b1.turn_rng.next_u64(), b2.turn_rng.next_u64());
            }
        }
    }

    #[test]
    fn directedness_draws_do_not_alter_other_streams() {
        for seed in [0, 1, 7, 42] {
            let mut b1 = Builder::new(seed, 3_600_000);
            let mut b2 = Builder::new(seed, 3_600_000);
            for _ in 0..100 {
                b2.draw_directed(Stimulus::Aside, 1_200, 0.3, 0);
                b2.directed_rng.next_u64();
            }
            for _ in 0..100 {
                assert_eq!(b1.rng.next_u64(), b2.rng.next_u64());
                assert_eq!(b1.condition_rng.next_u64(), b2.condition_rng.next_u64());
                assert_eq!(b1.speaker_rng.next_u64(), b2.speaker_rng.next_u64());
                assert_eq!(b1.media_rng.next_u64(), b2.media_rng.next_u64());
                assert_eq!(b1.turn_rng.next_u64(), b2.turn_rng.next_u64());
            }
        }
    }

    #[test]
    fn direction_draws_do_not_alter_other_streams() {
        for seed in [0, 1, 7, 42] {
            let mut b1 = Builder::new(seed, 3_600_000);
            let mut b2 = Builder::new(seed, 3_600_000);
            for _ in 0..100 {
                b2.draw_direction(Stimulus::Tv, RoomCondition::default(), 0);
                b2.draw_direction(Stimulus::Aside, RoomCondition::default(), 3);
                b2.direction_rng.next_u64();
            }
            for _ in 0..100 {
                assert_eq!(b1.rng.next_u64(), b2.rng.next_u64());
                assert_eq!(b1.condition_rng.next_u64(), b2.condition_rng.next_u64());
                assert_eq!(b1.speaker_rng.next_u64(), b2.speaker_rng.next_u64());
                assert_eq!(b1.media_rng.next_u64(), b2.media_rng.next_u64());
                assert_eq!(b1.turn_rng.next_u64(), b2.turn_rng.next_u64());
                assert_eq!(b1.directed_rng.next_u64(), b2.directed_rng.next_u64());
            }
        }
    }

    /// Mean resultant length of `draws` von Mises angles: I1(kappa) / I0(kappa).
    fn resultant(kappa: f64, draws: u32) -> f64 {
        let mut rng = SplitMix64::with_seed(48);
        let (mut x, mut y) = (0.0, 0.0);
        for _ in 0..draws {
            let (sin, cos) = rng.von_mises(kappa).sin_cos();
            x += cos;
            y += sin;
        }
        (x * x + y * y).sqrt() / f64::from(draws)
    }

    #[test]
    fn von_mises_draws_have_their_concentration() {
        // I1(kappa) / I0(kappa) from the asymptotic series 1 - 1/(2k) - 1/(8k^2) - 1/(8k^3).
        for (kappa, expected) in [(15.0, 0.966_07), (20.0, 0.974_67), (60.0, 0.991_63)] {
            let got = resultant(kappa, 40_000);
            assert!((got - expected).abs() < 0.002, "kappa {kappa}: {got}");
        }
        let mut rng = SplitMix64::with_seed(49);
        for _ in 0..10_000 {
            let angle = rng.von_mises(15.0);
            assert!((-std::f64::consts::PI..=std::f64::consts::PI).contains(&angle));
        }
    }

    #[test]
    fn owner_angles_from_the_tv_follow_the_measured_bins() {
        let draws = 50_000_u32;
        let mut rng = SplitMix64::with_seed(50);
        let mut bins = [0_u32; 12];
        let mut left = 0_u32;
        for _ in 0..draws {
            let angle = owner_tv_angle(&mut rng);
            left += u32::from(angle < 0.0);
            let bin = (angle.abs().to_degrees() / OWNER_TV_ANGLE_BIN_DEG).floor();
            // A bin index between zero and twelve: exact in f64 and in u8.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let bin = usize::from(bin as u8);
            bins[bin.min(11)] += 1;
        }
        let total: f64 = OWNER_TV_ANGLE_PERCENT.iter().sum();
        for (count, percent) in bins.iter().zip(OWNER_TV_ANGLE_PERCENT) {
            let share = f64::from(*count) / f64::from(draws);
            assert!((share - percent / total).abs() < 0.006, "{bins:?}");
        }
        assert!((f64::from(left) / f64::from(draws) - 0.5).abs() < 0.01);
    }

    /// Share of `draws` live readings from a seat opposite the TV that land within 30
    /// degrees of the TV.
    fn at_the_tv(tv: TvBackground, draws: u32) -> f64 {
        let mut b = Builder::new(51, 3_600_000);
        let seat = b.tv_azimuth + std::f64::consts::PI;
        let near = (0..draws)
            .filter(|_| {
                let (sin, cos) = (b.live_reading(seat, tv) - b.tv_azimuth).sin_cos();
                sin.atan2(cos).abs() < 30_f64.to_radians()
            })
            .count();
        near as f64 / f64::from(draws)
    }

    #[test]
    fn a_loud_tv_captures_live_readings() {
        // Capture times a TV reading within 30 degrees (0.87), plus the outliers that
        // land there by chance (one sixth of them).
        for (tv, expected) in [
            (TvBackground::Off, 0.005),
            (TvBackground::Moderate, 0.099),
            (TvBackground::Loud, 0.233),
        ] {
            let share = at_the_tv(tv, 20_000);
            assert!((share - expected).abs() < 0.015, "{tv}: {share}");
        }
    }

    #[test]
    fn tv_lines_cluster_around_the_tv_and_every_cue_carries_a_unit_direction() {
        for tape in [e1a(3).unwrap(), e1b(3).unwrap()] {
            let b = Builder::new(3, 3_600_000);
            let (mut tv_lines, mut near_tv) = (0_u32, 0_u32);
            for record in tape.records() {
                let Event::Speech { cue, .. } = record.event else {
                    continue;
                };
                let [x, y] = cue.direction.expect("every cue is read");
                assert!(((x * x + y * y).sqrt() - 1.0).abs() < 1e-6);
                if record.annotation.source() == Some(Stimulus::Tv) {
                    tv_lines += 1;
                    let off = (f64::from(y).atan2(f64::from(x)) - b.tv_azimuth).sin_cos();
                    near_tv += u32::from(off.0.atan2(off.1).abs() < 20_f64.to_radians());
                }
            }
            // 85% within about 20 degrees (von Mises 15: 90%), plus outliers by chance.
            let share = f64::from(near_tv) / f64::from(tv_lines);
            assert!((0.68..=0.88).contains(&share), "{share} of {tv_lines}");
        }
    }

    /// Band shares, low to high, of `draws` readings for one kind of cue, with the
    /// owner factor at one and each draw's block habits drawn afresh at their rates.
    fn band_shares(source: Stimulus, duration_ms: u32, media: f32) -> [f32; 3] {
        let draws = 20_000u32;
        let mut b = Builder::new(47, 3_600_000);
        b.owner_confusion = 1.0;
        let mut counts = [0u32; 3];
        for _ in 0..draws {
            b.habits = vec![AddressHabits::draw(&mut b.condition_rng)];
            let score = b.draw_directed(source, duration_ms, media, 0);
            assert!((0.0..=1.0).contains(&score), "{score}");
            let band = DIRECTED_BAND_EDGES
                .iter()
                .filter(|edge| score >= **edge)
                .count();
            counts[band] += 1;
        }
        counts.map(|count| count as f32 / draws as f32)
    }

    #[test]
    fn directedness_band_shares_follow_the_conservative_model() {
        let request = Stimulus::Request(TurnId(1));
        for (what, source, duration_ms, media, expected) in [
            ("request", request, 1_500, 0.3, [0.05, 0.10, 0.85]),
            ("short request", request, 500, 0.3, [0.07, 0.18, 0.75]),
            (
                "barge-in",
                Stimulus::BargeIn(TurnId(2)),
                1_500,
                0.3,
                [0.05, 0.10, 0.85],
            ),
            ("aside", Stimulus::Aside, 1_200, 0.3, [0.70, 0.18, 0.12]),
            ("short aside", Stimulus::Aside, 800, 0.3, [0.60, 0.28, 0.12]),
            (
                "long aside",
                Stimulus::Aside,
                3_500,
                0.3,
                [0.80, 0.08, 0.12],
            ),
            (
                "other person",
                Stimulus::OtherSpeech,
                1_000,
                0.3,
                [0.75, 0.17, 0.08],
            ),
            (
                "long other person",
                Stimulus::OtherSpeech,
                3_500,
                0.3,
                [0.85, 0.07, 0.08],
            ),
            (
                "false keyword",
                Stimulus::FalseKeyword,
                250,
                0.3,
                [0.75, 0.17, 0.08],
            ),
            ("tagged TV", Stimulus::Tv, 1_500, 0.8, [0.80, 0.12, 0.08]),
            ("missed TV", Stimulus::Tv, 1_500, 0.3, [0.739, 0.111, 0.15]),
            ("noise", Stimulus::Noise, 500, 0.1, [1.0, 0.0, 0.0]),
            ("motor", Stimulus::Motor, 500, 0.1, [1.0, 0.0, 0.0]),
        ] {
            let shares = band_shares(source, duration_ms, media);
            for (share, want) in shares.iter().zip(expected) {
                assert!(
                    (share - want).abs() < 0.015,
                    "{what}: {shares:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn block_habits_and_the_owner_correlate_directedness_errors() {
        let set = AddressHabits {
            entangled: true,
            casual: true,
        };
        let plain = AddressHabits::default();
        let directed = |addressee, habits| directed_shares(addressee, 1_500, habits, 0.8)[2];
        assert!((directed(Addressee::Aside, set) - ENTANGLED_ASIDE_DIRECTED).abs() < 1e-6);
        assert!((directed(Addressee::Aside, plain) - PLAIN_ASIDE_DIRECTED).abs() < 1e-6);
        assert!((1.0 - directed(Addressee::Enton, set) - CASUAL_REQUEST_MISSED).abs() < 1e-6);
        assert!((1.0 - directed(Addressee::Enton, plain) - PLAIN_REQUEST_MISSED).abs() < 1e-6);
        // Habits are the owner's: other people and the TV do not have them.
        assert_eq!(
            directed_shares(Addressee::OtherPerson, 1_500, set, 0.8).map(f32::to_bits),
            directed_shares(Addressee::OtherPerson, 1_500, plain, 0.8).map(f32::to_bits)
        );

        // An owner twice as confusable doubles the mistaken shares before renormalizing.
        let base = [0.05, 0.10, 0.85];
        let owner = with_owner_factor(base, Addressee::Enton, 2.0);
        let expected = [0.10 / 1.15, 0.20 / 1.15, 0.85 / 1.15];
        for (got, want) in owner.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "{owner:?}");
        }
        let aside = with_owner_factor([0.70, 0.18, 0.12], Addressee::Aside, 0.5);
        assert!((aside[0] - 0.70 / 0.85).abs() < 1e-6, "{aside:?}");
        assert_eq!(
            with_owner_factor(base, Addressee::Tv, 2.0).map(f32::to_bits),
            base.map(f32::to_bits)
        );
    }

    #[test]
    fn habit_rates_and_the_owner_factor_follow_their_distributions() {
        let (mut blocks, mut entangled, mut casual) = (0u32, 0u32, 0u32);
        let mut logs = Vec::new();
        for seed in 0..400 {
            let b = Builder::new(seed, 3_600_000);
            logs.push(b.owner_confusion.ln());
            for habits in &b.habits {
                blocks += 1;
                entangled += u32::from(habits.entangled);
                casual += u32::from(habits.casual);
            }
        }
        assert_eq!(blocks, 8_000);
        let entangled = entangled as f32 / blocks as f32;
        let casual = casual as f32 / blocks as f32;
        assert!(
            (entangled - ENTANGLED_BLOCK_RATE).abs() < 0.02,
            "{entangled}"
        );
        assert!((casual - CASUAL_BLOCK_RATE).abs() < 0.02, "{casual}");
        let mean = logs.iter().sum::<f32>() / logs.len() as f32;
        let sd =
            (logs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (logs.len() - 1) as f32).sqrt();
        assert!(mean.abs() < 0.06, "log owner factor mean {mean}");
        assert!(
            (sd - OWNER_CONFUSION_SD).abs() < 0.05,
            "log owner factor sd {sd}"
        );
    }

    #[test]
    fn every_speech_cue_carries_a_directedness_reading() {
        for tape in [e1a(3).unwrap(), e1b(3).unwrap()] {
            for record in tape.records() {
                if let Event::Speech { cue, .. } = record.event {
                    let score = cue.directed.expect("every cue is read");
                    assert!((0.0..=1.0).contains(&score));
                    if matches!(
                        record.annotation.source(),
                        Some(Stimulus::Noise | Stimulus::Motor | Stimulus::Ventilation)
                    ) {
                        assert!(score < DIRECTED_BAND_EDGES[0], "{score}");
                    }
                }
            }
        }
    }

    #[test]
    fn turn_segments_share_identical_pause_style() {
        for seed in [0, 1, 7, 42] {
            let tape_a = e1a(seed).unwrap();
            let tape_b = e1b(seed).unwrap();
            for tape in [&tape_a, &tape_b] {
                for turn in tape.turns() {
                    let expected_style = turn
                        .pause_style
                        .expect("every user turn must have a pause_style");
                    assert!(
                        !turn.segments.is_empty(),
                        "turn must have at least one segment"
                    );
                    for &seg_id in &turn.segments {
                        let record = tape
                            .records()
                            .iter()
                            .find(|r| match r.annotation {
                                Annotation::Speech { segment, .. } => segment == seg_id,
                                Annotation::Clock => false,
                            })
                            .expect("matching record for turn segment must exist");
                        assert_eq!(
                            record.annotation.pause_style(),
                            Some(expected_style),
                            "segment pause_style must match turn pause_style exactly"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod coverage_tests {
    use super::*;
    #[test]
    fn sustained_tv_interruptions_and_both_sides_of_the_window_are_present() {
        let tape = e1a(7).unwrap();
        let mut tv = std::collections::BTreeMap::<EpisodeId, (u64, u64)>::new();
        for record in tape.records() {
            if let Annotation::Speech {
                episode: Some(episode),
                source: Stimulus::Tv,
                ..
            } = record.annotation
            {
                let time = record.event.now().0;
                let span = tv.entry(episode).or_insert((time, time));
                span.1 = time;
            }
        }
        let shows = tape
            .conditions()
            .iter()
            .filter(|condition| condition.tv != TvBackground::Off)
            .count();
        assert!(shows > 0, "seed 7 has blocks with the TV on");
        assert_eq!(tv.len(), shows);
        assert!(tv.values().all(|(start, end)| end - start > 110_000));
        let mut conversations = std::collections::BTreeMap::<EpisodeId, Vec<u64>>::new();
        for turn in tape.turns() {
            conversations
                .entry(turn.episode)
                .or_default()
                .push(turn.available_at.0);
        }
        let gaps: Vec<_> = conversations
            .values()
            .filter(|ends| ends.len() == 5)
            .flat_map(|ends| ends.windows(2).map(|pair| pair[1] - pair[0]))
            .collect();
        assert!(gaps.iter().any(|gap| (4800..5000).contains(gap)));
        assert!(gaps.iter().any(|gap| (5001..=5200).contains(gap)));
        assert!(gaps.iter().any(|gap| *gap >= 8900));
        for episode in (0..100).step_by(5) {
            let request = tape
                .turns()
                .iter()
                .find(|turn| turn.episode == EpisodeId(episode))
                .unwrap();
            assert!(tape.records().iter().any(|record| record.event.now().0
                == request.available_at.0 + 1000
                && record.annotation.source() == Some(Stimulus::OtherSpeech)));
        }
    }
    #[test]
    fn the_tv_plays_only_in_blocks_where_it_is_on() {
        for seed in [0, 1, 7, 42] {
            let tape = e1a(seed).unwrap();
            let mut blocks_with_tv = std::collections::BTreeSet::new();
            for record in tape.records() {
                if matches!(record.annotation.source(), Some(Stimulus::Tv)) {
                    blocks_with_tv.insert(record.event.now().0 / 180_000);
                }
            }
            for (block, condition) in (0u64..).zip(tape.conditions()) {
                // A show that starts in a TV block may run into the next one.
                if condition.tv == TvBackground::Off {
                    let earlier_on = block > 0
                        && tape
                            .conditions()
                            .get(usize::try_from(block - 1).unwrap())
                            .is_some_and(|previous| previous.tv != TvBackground::Off);
                    assert!(
                        earlier_on || !blocks_with_tv.contains(&block),
                        "seed {seed} block {block}"
                    );
                } else {
                    assert!(blocks_with_tv.contains(&block), "seed {seed} block {block}");
                }
            }
        }
    }
    #[test]
    fn tv_lines_vary_in_level_around_their_show() {
        let tape = e1a(3).unwrap();
        let mut by_show: std::collections::BTreeMap<u32, Vec<f32>> =
            std::collections::BTreeMap::new();
        for record in tape.records() {
            if let (
                Event::Speech { cue, .. },
                Annotation::Speech {
                    episode: Some(episode),
                    source: Stimulus::Tv,
                    ..
                },
            ) = (&record.event, &record.annotation)
            {
                by_show.entry(episode.0).or_default().push(cue.energy);
            }
        }
        let shows = tape
            .conditions()
            .iter()
            .filter(|condition| condition.tv != TvBackground::Off)
            .count();
        assert_eq!(by_show.len(), shows);
        for levels in by_show.values() {
            let distinct: std::collections::BTreeSet<u32> =
                levels.iter().map(|level| level.to_bits()).collect();
            assert!(distinct.len() > levels.len() / 2, "{levels:?}");
            let spread = levels.iter().copied().fold(f32::MIN, f32::max)
                - levels.iter().copied().fold(f32::MAX, f32::min);
            assert!(
                spread < 0.6,
                "one show keeps one volume setting: {levels:?}"
            );
        }
    }
    #[test]
    fn e1a_contains_twenty_asides_and_twenty_in_window_other_speech() {
        for seed in [0, 1, 7, 42] {
            let tape = e1a(seed).unwrap();
            let mut asides = Vec::new();
            let mut in_window = Vec::new();
            for r in tape.records() {
                if let Annotation::Speech {
                    episode: Some(ep),
                    source,
                    ..
                } = r.annotation
                {
                    if source == Stimulus::Aside {
                        assert!(
                            r.annotation.turn().is_none(),
                            "aside cannot belong to a turn"
                        );
                        asides.push((ep, r));
                    }
                    if ep.0 >= 100 && (ep.0 - 100) % 7 == 6 {
                        assert_eq!(source, Stimulus::OtherSpeech);
                        assert!(
                            r.annotation.turn().is_none(),
                            "in-window speech cannot belong to a turn"
                        );
                        in_window.push((ep, r));
                    }
                }
            }
            assert_eq!(asides.len(), 20, "seed {seed} must have exactly 20 asides");
            assert_eq!(
                in_window.len(),
                20,
                "seed {seed} must have exactly 20 in-window segments"
            );
        }
    }
    #[test]
    fn asides_speaker_sim_follows_caller_distribution() {
        let mut aside_sims = Vec::new();
        for seed in 0..=31 {
            let tape = e1a(seed).unwrap();
            for r in tape.records() {
                if r.annotation.source() == Some(Stimulus::Aside) {
                    let Event::Speech { cue, .. } = &r.event else {
                        continue;
                    };
                    let sim = cue.speaker_sim.expect("aside must have speaker_sim");
                    aside_sims.push(sim);
                }
            }
        }
        assert_eq!(aside_sims.len(), 640);
        let mean = aside_sims.iter().copied().sum::<f32>() / aside_sims.len() as f32;
        assert!(
            (0.45..=0.65).contains(&mean),
            "asides mean speaker_sim was {mean}, expected in 0.45..=0.65"
        );
    }
    fn verify_split_block(tape: &Tape, block: u32) {
        let distractor_base = 100 + block * 7;
        let split_episode = EpisodeId(block * 5 + 1);
        let turn = tape
            .turns()
            .iter()
            .find(|t| t.episode == split_episode)
            .expect("split turn must exist");
        assert_eq!(
            turn.segments.len(),
            2,
            "split turn must have 2 user segments"
        );

        let first_seg = turn.segments[0];
        let second_seg = turn.segments[1];
        let mut first_rec = None;
        let mut second_rec = None;
        let mut distractor_rec = None;

        for r in tape.records() {
            if let Annotation::Speech {
                segment, episode, ..
            } = r.annotation
            {
                if segment == first_seg {
                    first_rec = Some(r);
                } else if segment == second_seg {
                    second_rec = Some(r);
                }
                if episode == Some(EpisodeId(distractor_base + 2)) {
                    distractor_rec = Some(r);
                }
            }
        }

        let first_rec = first_rec.unwrap();
        let second_rec = second_rec.unwrap();
        let distractor_rec = distractor_rec.unwrap();

        let name_end = first_rec.event.now().0;
        let content_end = second_rec.event.now().0;
        let content_duration = match &second_rec.event {
            Event::Speech { cue, .. } => u64::from(cue.duration_ms),
            _ => unreachable!(),
        };
        let content_start = content_end - content_duration;
        assert_eq!(turn.available_at.0, content_end);

        if block.is_multiple_of(2) {
            assert_eq!(
                distractor_rec.annotation.source(),
                Some(Stimulus::FalseKeyword)
            );
            assert_eq!(distractor_rec.event.now().0, name_end + 400);
            let pause = content_start - name_end;
            assert!((150..=700).contains(&pause));
        } else {
            assert_eq!(
                distractor_rec.annotation.source(),
                Some(Stimulus::OtherSpeech)
            );
            let distractor_end = distractor_rec.event.now().0;
            let distractor_duration = match &distractor_rec.event {
                Event::Speech { cue, .. } => u64::from(cue.duration_ms),
                _ => unreachable!(),
            };
            let distractor_start = distractor_end - distractor_duration;
            let delay = distractor_start - name_end;
            assert!((50..=300).contains(&delay));
            assert!((300..=600).contains(&distractor_duration));
            let pause = content_start - distractor_end;
            assert!((150..=700).contains(&pause));
        }
    }
    #[test]
    fn odd_blocks_have_adjacent_distractors_and_valid_split_turns() {
        for seed in [0, 1, 7, 42] {
            let tape = e1a(seed).unwrap();
            for block in 0u32..20 {
                verify_split_block(&tape, block);
            }
        }
    }
}
