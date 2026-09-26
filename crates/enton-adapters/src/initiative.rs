//! Enton's own initiative, as the adapters shape it: when the owner silences it (quiet
//! commands), when the house sleeps (quiet hours), and what a drive's deferred intent adds
//! to an answer it rides.
//!
//! The core decides; it never sees text or a wall clock. A quiet command is recognized
//! here, in what the owner typed (or what a transcriber heard), and reaches the core as
//! [`enton_core::Event::Quiet`] in place of the speech cue, so the command itself buys no
//! thought. The quiet hours are a band of local time read here, and only the flag crosses
//! over ([`enton_core::Event::QuietHours`]), at startup and at each edge, so replay stays
//! exact. A drive whose intent rides a thought (`Action::Think::rider`) gets one line in
//! that thought's prompt: the owner's checklist, flattened.

use std::fmt;

use enton_core::Millis;

use crate::checklist;

/// How long quiet lasts when the owner does not say: an hour, long enough for a call or a
/// nap, short enough that a forgotten "Enton, silêncio" does not mute Enton for the day.
pub const DEFAULT_QUIET_MS: u64 = 3_600_000;

/// The longest quiet a command may ask for: twelve hours. A longer one is capped.
pub const MAX_QUIET_MS: u64 = 12 * 3_600_000;

/// A quiet command recognized in what the owner said to Enton.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuietCommand {
    /// Keep Enton's own initiative quiet: for `for_ms` milliseconds when the owner said how
    /// long ("por meia hora"), for [`DEFAULT_QUIET_MS`] when not.
    Hush {
        /// How long the owner asked for, capped at [`MAX_QUIET_MS`].
        for_ms: Option<u64>,
    },
    /// Talk again ("Enton, pode falar").
    Release,
}

impl QuietCommand {
    /// Until when quiet holds, for a command heard at `now`: `now` itself for a release.
    #[must_use]
    pub fn until(self, now: Millis) -> Millis {
        match self {
            Self::Hush { for_ms } => {
                Millis(now.0.saturating_add(for_ms.unwrap_or(DEFAULT_QUIET_MS)))
            }
            Self::Release => now,
        }
    }
}

/// What the owner is told when the command is heard: a fixed acknowledgement, no cortex.
impl fmt::Display for QuietCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hush { for_ms } => {
                let minutes = for_ms.unwrap_or(DEFAULT_QUIET_MS).div_ceil(60_000);
                let span = if minutes % 60 == 0 {
                    format!("{} h", minutes / 60)
                } else {
                    format!("{minutes} min")
                };
                write!(
                    f,
                    "quiet for {span}: Enton keeps its own thoughts to itself, and still answers when called by name"
                )
            }
            Self::Release => write!(f, "quiet released: Enton may speak up on its own again"),
        }
    }
}

/// Recognize a quiet command in a typed or transcribed utterance, in Brazilian Portuguese.
///
/// The utterance must call Enton by name and be nothing but the command: its name, the
/// command, polite filler ("ei", "por favor", "agora") and, to keep quiet, how long. So a
/// question that merely mentions silence ("Enton, o que é silêncio?") is not a command.
/// Accents, case and punctuation do not matter. Releases: "pode falar", "pode voltar a
/// falar", "chega de silêncio" and the like. Commands to keep quiet: "silêncio", "fica
/// quieto", "cala a boca", "fica calado", "para de falar", "modo silêncio" and the like,
/// optionally led by "pode" or "você pode" and followed by a duration ("por uma hora",
/// "por meia hora", "durante 10 minutos", "até eu chamar").
#[must_use]
pub fn quiet_command(text: &str) -> Option<QuietCommand> {
    let folded = fold(text);
    let words: Vec<&str> = folded.split_whitespace().collect();
    if !words.contains(&"enton") {
        return None;
    }
    let words = without_filler(&words);
    if RELEASES.iter().any(|phrase| is_phrase(&words, phrase)) {
        return Some(QuietCommand::Release);
    }
    (1..=words.len()).rev().find_map(|split| {
        let (command, rest) = words.split_at(split);
        let command = PREFIXES
            .iter()
            .find_map(|prefix| strip_phrase(command, prefix))
            .unwrap_or(command);
        if HUSHES.iter().any(|phrase| is_phrase(command, phrase)) {
            hush_for(rest)
        } else {
            None
        }
    })
}

/// Phrases that release quiet, as folded words.
const RELEASES: &[&str] = &[
    "pode falar",
    "pode falar de novo",
    "pode voltar a falar",
    "volta a falar",
    "voltar a falar",
    "pode voltar",
    "pode conversar",
    "fim do silencio",
    "chega de silencio",
    "acabou o silencio",
    "sai do silencio",
    "sair do silencio",
    "sai do modo silencio",
    "sair do modo silencio",
    "desliga o silencio",
    "desliga o modo silencio",
];

/// Phrases that ask for quiet, as folded words.
const HUSHES: &[&str] = &[
    "silencio",
    "modo silencio",
    "em silencio",
    "fica em silencio",
    "fique em silencio",
    "ficar em silencio",
    "quieto",
    "quieta",
    "fica quieto",
    "fique quieto",
    "ficar quieto",
    "fica quieta",
    "fique quieta",
    "ficar quieta",
    "calado",
    "calada",
    "fica calado",
    "fique calado",
    "ficar calado",
    "fica calada",
    "fique calada",
    "ficar calada",
    "cala a boca",
    "cale a boca",
    "cala boca",
    "cale se",
    "para de falar",
    "pare de falar",
    "parar de falar",
    "nao fala nada",
    "nao fale nada",
    "psiu",
    "shh",
    "shhh",
];

/// What may lead a command to keep quiet: "você pode ficar quieto?".
const PREFIXES: &[&str] = &[
    "voce pode",
    "voce poderia",
    "pode",
    "poderia",
    "da pra",
    "quer",
];

/// Polite filler dropped anywhere, as folded words; "por favor" goes as a pair first, so
/// "por" still leads a duration.
const FILLER: &[&str] = &[
    "ei", "oi", "hey", "ok", "okay", "ta", "agora", "ai", "so", "pfv", "pf",
];

/// Lowercase, without accents, with anything but letters and digits as spaces.
fn fold(text: &str) -> String {
    text.chars()
        .flat_map(char::to_lowercase)
        .map(|c| match c {
            'á' | 'à' | 'â' | 'ã' | 'ä' => 'a',
            'é' | 'è' | 'ê' | 'ë' => 'e',
            'í' | 'ì' | 'î' | 'ï' => 'i',
            'ó' | 'ò' | 'ô' | 'õ' | 'ö' => 'o',
            'ú' | 'ù' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            'ñ' => 'n',
            c if c.is_alphanumeric() => c,
            _ => ' ',
        })
        .collect()
}

/// The words without Enton's name and polite filler.
fn without_filler<'a>(words: &[&'a str]) -> Vec<&'a str> {
    let mut kept = Vec::with_capacity(words.len());
    let mut rest = words.iter().copied().peekable();
    while let Some(word) = rest.next() {
        if word == "por" && rest.peek() == Some(&"favor") {
            rest.next();
        } else if word != "enton" && word != "favor" && !FILLER.contains(&word) {
            kept.push(word);
        }
    }
    kept
}

/// Whether `words` are exactly `phrase`.
fn is_phrase(words: &[&str], phrase: &str) -> bool {
    words.iter().copied().eq(phrase.split(' '))
}

/// `words` after a leading `phrase`, if they start with it and go on.
fn strip_phrase<'w, 'a>(words: &'w [&'a str], phrase: &str) -> Option<&'w [&'a str]> {
    let length = phrase.split(' ').count();
    let (head, rest) = words.split_at_checked(length)?;
    (is_phrase(head, phrase) && !rest.is_empty()).then_some(rest)
}

/// The command to keep quiet for as long as the words after it ask: the default when they
/// name no length (nothing, "um pouco", "até eu chamar"), and `None` when they are not a
/// length at all, so the utterance is no command.
fn hush_for(words: &[&str]) -> Option<QuietCommand> {
    let unsaid = QuietCommand::Hush { for_ms: None };
    let words = match words {
        ["por" | "durante", rest @ ..] => rest,
        _ => words,
    };
    match words {
        []
        | ["um", "pouco" | "tempo" | "tempinho" | "instante"]
        | ["uns", "minutos" | "minutinhos"]
        | ["ate", "segunda", "ordem"]
        | [
            "ate",
            "eu",
            "falar" | "mandar" | "chamar" | "pedir" | "avisar" | "voltar",
        ]
        | ["ate", "eu", "te", "chamar" | "avisar"] => Some(unsaid),
        ["meia", "hora"] => Some(QuietCommand::Hush {
            for_ms: Some(30 * 60_000),
        }),
        _ => length(words)
            .filter(|ms| *ms > 0)
            .map(|ms| QuietCommand::Hush {
                for_ms: Some(ms.min(MAX_QUIET_MS)),
            }),
    }
}

/// A spoken length: a number and a unit, and for hours perhaps "e meia" ("uma hora e
/// meia"), in milliseconds.
fn length(words: &[&str]) -> Option<u64> {
    let (count, rest) = number(words)?;
    let (unit, rest) = rest.split_first()?;
    let unit_ms = match *unit {
        "segundo" | "segundos" | "seg" | "s" => 1_000,
        "minuto" | "minutos" | "minutinho" | "minutinhos" | "min" => 60_000,
        "hora" | "horas" | "h" => 3_600_000,
        _ => return None,
    };
    let half = match rest {
        [] => 0,
        ["e", "meia"] if unit_ms == 3_600_000 => 30 * 60_000,
        _ => return None,
    };
    count.checked_mul(unit_ms)?.checked_add(half)
}

/// A number in digits or in words ("dez", "vinte e cinco"), and the words after it.
fn number<'w, 'a>(words: &'w [&'a str]) -> Option<(u64, &'w [&'a str])> {
    let (first, rest) = words.split_first()?;
    if let Ok(digits) = first.parse::<u64>() {
        return Some((digits, rest));
    }
    let value = spelled(first)?;
    // Tens and units: "vinte e cinco".
    if value >= 20
        && value % 10 == 0
        && let ["e", units, after @ ..] = rest
        && let Some(units) = spelled(units).filter(|units| (1..10).contains(units))
    {
        return Some((value + units, after));
    }
    Some((value, rest))
}

/// A number spelled out, from one to ninety.
fn spelled(word: &str) -> Option<u64> {
    Some(match word {
        "um" | "uma" => 1,
        "dois" | "duas" => 2,
        "tres" => 3,
        "quatro" => 4,
        "cinco" => 5,
        "seis" => 6,
        "sete" => 7,
        "oito" => 8,
        "nove" => 9,
        "dez" => 10,
        "onze" => 11,
        "doze" => 12,
        "treze" => 13,
        "catorze" | "quatorze" => 14,
        "quinze" => 15,
        "dezesseis" => 16,
        "dezessete" => 17,
        "dezoito" => 18,
        "dezenove" => 19,
        "vinte" => 20,
        "trinta" => 30,
        "quarenta" => 40,
        "cinquenta" => 50,
        "sessenta" => 60,
        "noventa" => 90,
        _ => return None,
    })
}

/// The environment variable that sets the quiet hours: `HH:MM-HH:MM` in local time, or
/// `off`. Unset, the band is [`QuietHours::DEFAULT`].
pub const QUIET_HOURS_VAR: &str = "ENTON_QUIET_HOURS";

/// Minutes in a day.
const DAY_MINUTES: u16 = 24 * 60;

/// A daily band of local time during which Enton keeps its own initiative quiet, from
/// `start` (inclusive) to `end` (exclusive), in minutes since midnight. A band whose start
/// comes after its end runs across midnight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuietHours {
    start: u16,
    end: u16,
}

impl QuietHours {
    /// 23:00 to 07:00: the night, when a thought of Enton's own would wake someone.
    pub const DEFAULT: Self = Self {
        start: 23 * 60,
        end: 7 * 60,
    };

    /// The band from `start` to `end`, in minutes since midnight; `None` unless both lie
    /// within a day and they differ (an empty or all-day band is not a band).
    #[must_use]
    pub fn new(start: u16, end: u16) -> Option<Self> {
        (start < DAY_MINUTES && end < DAY_MINUTES && start != end).then_some(Self { start, end })
    }

    /// Whether the band holds at `minute`, in minutes since local midnight.
    #[must_use]
    pub fn contains(self, minute: u16) -> bool {
        if self.start < self.end {
            (self.start..self.end).contains(&minute)
        } else {
            minute >= self.start || minute < self.end
        }
    }

    /// Read a band written `HH:MM-HH:MM`, or `off` for none.
    ///
    /// # Errors
    ///
    /// Returns what is wrong with `spec` when it is neither.
    pub fn parse(spec: &str) -> Result<Option<Self>, String> {
        let spec = spec.trim();
        if spec.eq_ignore_ascii_case("off") {
            return Ok(None);
        }
        let invalid = || format!("expected HH:MM-HH:MM or off, got {spec:?}");
        let (start, end) = spec.split_once('-').ok_or_else(invalid)?;
        let minute = |clock: &str| {
            let (hours, minutes) = clock.trim().split_once(':')?;
            let hours = hours.parse::<u16>().ok().filter(|hours| *hours < 24)?;
            let minutes = minutes
                .parse::<u16>()
                .ok()
                .filter(|minutes| *minutes < 60)?;
            Some(hours * 60 + minutes)
        };
        let (start, end) = minute(start).zip(minute(end)).ok_or_else(invalid)?;
        Self::new(start, end)
            .map(Some)
            .ok_or_else(|| format!("the band {spec:?} is empty"))
    }

    /// The band [`QUIET_HOURS_VAR`] sets: [`QuietHours::DEFAULT`] when it is unset.
    ///
    /// # Errors
    ///
    /// Returns what is wrong with the variable's value.
    pub fn configured() -> Result<Option<Self>, String> {
        match std::env::var(QUIET_HOURS_VAR) {
            Ok(spec) => Self::parse(&spec).map_err(|error| format!("{QUIET_HOURS_VAR}: {error}")),
            Err(std::env::VarError::NotPresent) => Ok(Some(Self::DEFAULT)),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{QUIET_HOURS_VAR}: not valid Unicode"))
            }
        }
    }
}

impl fmt::Display for QuietHours {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:02}:{:02}-{:02}:{:02}",
            self.start / 60,
            self.start % 60,
            self.end / 60,
            self.end % 60
        )
    }
}

/// Watches the quiet hours: each observation of the local time says whether the band
/// changed, so the core hears only its edges.
#[derive(Debug, Clone)]
pub struct QuietHoursWatcher {
    hours: Option<QuietHours>,
    /// Whether the band held at the last observation; `None` before the first.
    active: Option<bool>,
}

impl QuietHoursWatcher {
    /// A watcher of `hours` (`None`: no quiet hours).
    #[must_use]
    pub fn new(hours: Option<QuietHours>) -> Self {
        Self {
            hours,
            active: None,
        }
    }

    /// The band watched, if any.
    #[must_use]
    pub fn hours(&self) -> Option<QuietHours> {
        self.hours
    }

    /// Whether the band holds at `minute` (since local midnight), when that is news: at
    /// the first observation, so a restarted Enton learns where it stands, then at each
    /// edge. `None` when nothing changed. Without a band, it never holds.
    pub fn observe(&mut self, minute: u16) -> Option<bool> {
        let active = self.hours.is_some_and(|hours| hours.contains(minute));
        (self.active != Some(active)).then(|| {
            self.active = Some(active);
            active
        })
    }

    /// A line for the owner about the band, now that it is `active` or not.
    #[must_use]
    pub fn describe(&self, active: bool) -> String {
        match self.hours {
            None => format!("no quiet hours ({QUIET_HOURS_VAR}=off)"),
            Some(hours) if active => format!(
                "quiet hours ({hours}): Enton keeps its own thoughts to itself until they end"
            ),
            Some(hours) => format!("outside the quiet hours ({hours})"),
        }
    }

    /// [`Self::observe`] at the current local time.
    #[cfg(feature = "local-time")]
    pub fn poll(&mut self) -> Option<bool> {
        self.observe(local_minute())
    }
}

/// Minutes since local midnight, in the system time zone.
#[cfg(feature = "local-time")]
#[must_use]
pub fn local_minute() -> u16 {
    let now = jiff::Zoned::now();
    u16::try_from(i32::from(now.hour()) * 60 + i32::from(now.minute())).unwrap_or(0)
}

/// How a ride line begins: what [`without_ride`] looks for.
const RIDE_MARK: &str = "[Riding along: ";

/// The one line a drive's deferred intent adds to the answer it rides: the owner's
/// checklist, flattened, and leave to bring one item up in a sentence, or none. `None`
/// when the checklist holds nothing to bring up.
#[must_use]
pub fn ride_line(drive: &str, checklist: &str) -> Option<String> {
    let items: Vec<&str> = checklist::items(checklist).collect();
    (!items.is_empty()).then(|| {
        format!(
            "{RIDE_MARK}your own {drive} rides with this answer. The owner's checklist (CHECKLIST.md): {}. \
             If one item is worth bringing up now, add one short sentence about it after your answer; \
             otherwise say nothing about it.]",
            items.join("; ")
        )
    })
}

/// The prompt of a thought the owner asked for, with the line of the drive whose intent
/// rides it, if one does and the checklist has something on it: the runtime's only hook
/// for rides.
#[must_use]
pub fn with_ride(
    transcript: Option<String>,
    rider: Option<&str>,
    checklist: Option<&str>,
) -> Option<String> {
    let Some(line) = rider
        .zip(checklist)
        .and_then(|(drive, text)| ride_line(drive, text))
    else {
        return transcript;
    };
    Some(match transcript {
        Some(heard) if !heard.trim().is_empty() => format!("{heard}\n\n{line}"),
        _ => line,
    })
}

/// What the owner said, in a prompt a ride line may follow (see [`with_ride`]): the part
/// before it, for the conversation history, which should not keep the checklist as if the
/// owner had said it. `None` when nothing else is left.
#[must_use]
pub fn without_ride(prompt: &str) -> Option<&str> {
    let heard = prompt
        .find(RIDE_MARK)
        .map_or(prompt, |at| prompt.get(..at).unwrap_or(prompt))
        .trim_end();
    (!heard.is_empty()).then_some(heard)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINUTE: u64 = 60_000;
    const HOUR: u64 = 60 * MINUTE;

    fn hush(for_ms: Option<u64>) -> QuietCommand {
        QuietCommand::Hush { for_ms }
    }

    #[test]
    fn commands_to_keep_quiet_are_recognized_with_and_without_a_length() {
        for (said, expected) in [
            ("Enton, silêncio", Some(hush(None))),
            ("Enton, silêncio!", Some(hush(None))),
            ("SILÊNCIO, ENTON.", Some(hush(None))),
            ("Enton silencio", Some(hush(None))),
            ("Enton, fica quieto", Some(hush(None))),
            ("Enton, fique quieta.", Some(hush(None))),
            ("Enton, cala a boca", Some(hush(None))),
            ("Enton, cale-se", Some(hush(None))),
            ("Enton, fica calado", Some(hush(None))),
            ("Enton, para de falar", Some(hush(None))),
            ("Enton, modo silêncio", Some(hush(None))),
            ("Enton, fica em silêncio, por favor", Some(hush(None))),
            ("Ei Enton, fica quieto agora", Some(hush(None))),
            ("Enton, você pode ficar quieto?", Some(hush(None))),
            ("Enton, não fala nada", Some(hush(None))),
            ("Enton, psiu", Some(hush(None))),
            ("Enton, fica quieto um pouco", Some(hush(None))),
            ("Enton, silêncio até eu chamar", Some(hush(None))),
            ("Enton, silêncio até segunda ordem", Some(hush(None))),
            // Cut off before the length: still quiet, for the default hour.
            ("Enton, silêncio por...", Some(hush(None))),
            ("Enton, silêncio por uma hora", Some(hush(Some(HOUR)))),
            (
                "Enton, silêncio por meia hora",
                Some(hush(Some(30 * MINUTE))),
            ),
            (
                "Enton, silêncio por uma hora e meia",
                Some(hush(Some(90 * MINUTE))),
            ),
            ("Enton, silêncio por duas horas", Some(hush(Some(2 * HOUR)))),
            (
                "Enton, fica quieto por 10 minutos",
                Some(hush(Some(10 * MINUTE))),
            ),
            (
                "Enton, fica quieto durante dez minutos",
                Some(hush(Some(10 * MINUTE))),
            ),
            (
                "Enton, cala a boca por vinte e cinco minutos",
                Some(hush(Some(25 * MINUTE))),
            ),
            ("Enton, silêncio por 45 min", Some(hush(Some(45 * MINUTE)))),
            ("Enton, silêncio por 1 h", Some(hush(Some(HOUR)))),
            ("Enton, silêncio por 30 segundos", Some(hush(Some(30_000)))),
            // A day asks too much: it is capped at twelve hours.
            (
                "Enton, silêncio por 24 horas",
                Some(hush(Some(MAX_QUIET_MS))),
            ),
            ("Enton, silêncio por 99999999999999999999 horas", None),
        ] {
            assert_eq!(quiet_command(said), expected, "{said:?}");
        }
    }

    #[test]
    fn releases_are_recognized() {
        for said in [
            "Enton, pode falar",
            "Enton, pode falar de novo.",
            "Enton, pode voltar a falar",
            "Enton, volta a falar, por favor",
            "Enton, chega de silêncio",
            "Enton, acabou o silêncio",
            "Ok Enton, pode conversar",
            "Enton, sai do modo silêncio",
        ] {
            assert_eq!(quiet_command(said), Some(QuietCommand::Release), "{said:?}");
        }
    }

    #[test]
    fn anything_else_is_not_a_command() {
        for said in [
            // Not addressed to Enton.
            "silêncio",
            "fica quieto",
            "pode falar",
            "Benton, silêncio",
            // Questions and requests that merely mention quiet.
            "Enton, o que é silêncio?",
            "Enton, como se diz silêncio em inglês?",
            "Enton, que horas são?",
            "Enton, não fica quieto não",
            "Enton, fala de novo",
            "Enton, pode falar mais alto?",
            "Enton, silêncio por favor quanto é dois mais dois",
            // A length that is no length.
            "Enton, silêncio por zero minutos",
            "Enton, silêncio por 0 minutos",
            "Enton, silêncio por uma banana",
            "Enton, silêncio por meia",
            "Enton, silêncio por 10 minutos e meia",
            "",
            "Enton",
        ] {
            assert_eq!(quiet_command(said), None, "{said:?}");
        }
    }

    #[test]
    fn a_command_sets_when_quiet_ends() {
        let now = Millis(1_000);
        assert_eq!(hush(None).until(now), Millis(1_000 + DEFAULT_QUIET_MS));
        assert_eq!(hush(Some(HOUR / 2)).until(now), Millis(1_000 + HOUR / 2));
        assert_eq!(QuietCommand::Release.until(now), now);
        assert_eq!(
            hush(Some(MAX_QUIET_MS)).until(Millis(u64::MAX)),
            Millis(u64::MAX)
        );
        assert!(hush(None).to_string().starts_with("quiet for 1 h"));
        assert!(
            hush(Some(90 * MINUTE))
                .to_string()
                .starts_with("quiet for 90 min")
        );
        assert!(
            QuietCommand::Release
                .to_string()
                .starts_with("quiet released")
        );
    }

    #[test]
    fn a_band_across_midnight_holds_until_the_morning() {
        let night = QuietHours::DEFAULT;
        assert_eq!(night.to_string(), "23:00-07:00");
        for (minute, inside) in [
            (22 * 60 + 59, false),
            (23 * 60, true),
            (0, true),
            (6 * 60 + 59, true),
            (7 * 60, false),
            (12 * 60, false),
        ] {
            assert_eq!(night.contains(minute), inside, "{minute}");
        }
        let nap = QuietHours::new(13 * 60, 14 * 60 + 30).unwrap();
        assert!(
            !nap.contains(12 * 60 + 59) && nap.contains(13 * 60) && !nap.contains(14 * 60 + 30)
        );
        assert_eq!(QuietHours::new(60, 60), None);
        assert_eq!(QuietHours::new(24 * 60, 60), None);
    }

    #[test]
    fn a_band_reads_from_its_spec() {
        assert_eq!(
            QuietHours::parse("23:00-07:00"),
            Ok(Some(QuietHours::DEFAULT))
        );
        assert_eq!(
            QuietHours::parse(" 22:30 - 6:15 "),
            Ok(QuietHours::new(22 * 60 + 30, 6 * 60 + 15))
        );
        assert_eq!(QuietHours::parse("off"), Ok(None));
        assert_eq!(QuietHours::parse("OFF"), Ok(None));
        for broken in [
            "",
            "23:00",
            "24:00-07:00",
            "23:60-07:00",
            "7-8",
            "23:00-23:00",
            "a:b-c:d",
        ] {
            assert!(QuietHours::parse(broken).is_err(), "{broken:?}");
        }
    }

    #[test]
    fn the_watcher_reports_the_first_reading_then_only_the_edges() {
        let mut watcher = QuietHoursWatcher::new(Some(QuietHours::DEFAULT));
        assert_eq!(watcher.observe(22 * 60), Some(false));
        assert_eq!(watcher.observe(22 * 60 + 30), None);
        assert_eq!(watcher.observe(23 * 60), Some(true));
        assert_eq!(watcher.observe(3 * 60), None);
        assert_eq!(watcher.observe(7 * 60), Some(false));
        // Without a band it says so once, so a restored Enton leaves any old band.
        let mut none = QuietHoursWatcher::new(None);
        assert_eq!(none.observe(0), Some(false));
        assert_eq!(none.observe(23 * 60 + 30), None);
        assert!(
            watcher
                .describe(true)
                .starts_with("quiet hours (23:00-07:00)")
        );
        assert_eq!(
            watcher.describe(false),
            "outside the quiet hours (23:00-07:00)"
        );
        assert!(none.describe(false).starts_with("no quiet hours"));
    }

    #[test]
    fn a_riding_drive_adds_one_line_with_the_checklist() {
        let checklist = "# Hoje\n- [ ] regar as plantas\n- ligar para a mae\n";
        let line = ride_line("curiosity", checklist).unwrap();
        assert!(!line.contains('\n'), "{line}");
        assert!(line.contains("curiosity"), "{line}");
        assert!(
            line.contains("[ ] regar as plantas; ligar para a mae"),
            "{line}"
        );
        assert_eq!(ride_line("curiosity", "# nada\n- [ ]\n"), None);

        let heard = Some("Enton, que horas são?".to_owned());
        assert_eq!(
            with_ride(heard.clone(), Some("curiosity"), Some(checklist)),
            Some(format!("Enton, que horas são?\n\n{line}"))
        );
        assert_eq!(
            with_ride(None, Some("curiosity"), Some(checklist)),
            Some(line.clone())
        );
        // Nothing rides, or nothing to bring up: the prompt is the one heard.
        assert_eq!(with_ride(heard.clone(), None, Some(checklist)), heard);
        assert_eq!(with_ride(heard.clone(), Some("curiosity"), None), heard);

        // The history keeps what the owner said, never the ride line.
        let ridden = with_ride(heard, Some("curiosity"), Some(checklist)).unwrap();
        assert_eq!(without_ride(&ridden), Some("Enton, que horas são?"));
        assert_eq!(
            without_ride("Enton, que horas são?"),
            Some("Enton, que horas são?")
        );
        assert_eq!(without_ride(&line), None);
        assert_eq!(without_ride(""), None);
    }
}
