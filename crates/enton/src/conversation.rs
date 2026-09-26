//! The conversation the cortex is shown with each thought, and what the owner heard of
//! Enton's replies spoken aloud.
//!
//! A spoken reply goes out sentence by sentence. When the owner cuts it off, the
//! conversation keeps only the sentences that finished playing, then a short note, so
//! the cortex never believes it said what nobody heard (the idea behind Open-LLM-VTuber's
//! `handle_interrupt`). A reply that is not spoken, in text mode, is kept whole.

use std::collections::VecDeque;

use enton_core::ports::ConversationTurn;
#[cfg(feature = "voice")]
use enton_core::{ThoughtId, UtteranceId};

/// How many turns the conversation keeps.
const MAX_TURNS: usize = 30;

/// Closes a reply the owner cut off, after the sentences they heard. In English, like the
/// other bracketed notes the cortex reads ("[Trigger reason: ...]"), so it reads as a note
/// about the reply, not as words of it.
#[cfg(feature = "voice")]
pub(crate) const CUT_OFF: &str = "[interrupted: the owner heard only this]";

/// A reply the owner cut off before any of its sentences finished playing.
#[cfg(feature = "voice")]
pub(crate) const CUT_OFF_UNHEARD: &str = "[interrupted before the owner heard anything]";

/// Append `sentence` to a reply a space apart: how a spoken reply's text is built from
/// its sentences, so the part heard is exactly a prefix of the whole.
#[cfg(feature = "voice")]
pub(crate) fn append_sentence(reply: &mut String, sentence: &str) {
    if !reply.is_empty() && !reply.ends_with(' ') {
        reply.push(' ');
    }
    reply.push_str(sentence);
}

/// A turn's place in the conversation: how many turns were written before it.
#[cfg(feature = "voice")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TurnNo(u64);

/// The recent conversation, oldest turn first, at most [`MAX_TURNS`] long.
#[derive(Debug, Default)]
pub(crate) struct Conversation {
    turns: VecDeque<ConversationTurn>,
    /// Turns ever written: the turn numbered `n` sits at `n` minus those dropped.
    written: u64,
}

impl Conversation {
    /// Write `turn` after the others, dropping the oldest beyond [`MAX_TURNS`].
    pub(crate) fn push(&mut self, turn: ConversationTurn) {
        self.written += 1;
        self.turns.push_back(turn);
        while self.turns.len() > MAX_TURNS {
            self.turns.pop_front();
        }
    }

    /// The turns, oldest first, as a request carries them.
    pub(crate) fn turns(&self) -> Vec<ConversationTurn> {
        self.turns.iter().cloned().collect()
    }

    /// How many turns it holds.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.turns.len()
    }

    /// The number of the turn written last, if any.
    #[cfg(feature = "voice")]
    pub(crate) fn newest(&self) -> Option<TurnNo> {
        self.written.checked_sub(1).map(TurnNo)
    }

    /// The turn numbered `no`, while the conversation still holds it.
    #[cfg(feature = "voice")]
    fn turn_mut(&mut self, no: TurnNo) -> Option<&mut ConversationTurn> {
        let dropped = self.written - self.turns.len() as u64;
        let index = no.0.checked_sub(dropped)?;
        self.turns.get_mut(usize::try_from(index).ok()?)
    }
}

/// Replies tracked at once: an older one still playing is forgotten, and kept whole.
#[cfg(feature = "voice")]
const MAX_REPLIES: usize = 4;

/// Sentences tracked per reply: later ones count as never heard.
#[cfg(feature = "voice")]
const MAX_SENTENCES: usize = 256;

/// How far a sentence's utterance got.
#[cfg(feature = "voice")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Playback {
    /// Queued or playing.
    Pending,
    /// Played to its end: heard.
    Played,
    /// Nothing to play (only a stage direction, or refused by the player), or failed.
    Silent,
}

/// A sentence of a spoken reply, as the cortex wrote it.
#[cfg(feature = "voice")]
#[derive(Debug)]
struct Sentence {
    text: String,
    utterance: Option<UtteranceId>,
    playback: Playback,
}

/// A reply being spoken.
#[cfg(feature = "voice")]
#[derive(Debug)]
struct SpokenReply {
    thought: ThoughtId,
    /// What the owner said, when the thought answers them.
    prompt: Option<String>,
    sentences: Vec<Sentence>,
    /// Whether the cortex finished the reply, and so wrote it to the conversation.
    finished: bool,
    /// The reply's own turn in the conversation, once written (never, for silence).
    turn: Option<TurnNo>,
}

#[cfg(feature = "voice")]
impl SpokenReply {
    /// Whether nothing of it is left to play: the owner heard all there was.
    fn settled(&self) -> bool {
        self.finished
            && self
                .sentences
                .iter()
                .all(|sentence| sentence.playback != Playback::Pending)
    }

    /// The reply as the owner heard it before being cut off: the sentences up to the
    /// last one that played (silent ones before it included), then the note.
    fn heard(&self) -> String {
        let heard = self
            .sentences
            .iter()
            .rposition(|sentence| sentence.playback == Playback::Played)
            .map_or(0, |last| last + 1);
        let mut text = String::new();
        for sentence in self.sentences.iter().take(heard) {
            append_sentence(&mut text, &sentence.text);
        }
        if text.trim().is_empty() {
            CUT_OFF_UNHEARD.to_owned()
        } else {
            append_sentence(&mut text, CUT_OFF);
            text
        }
    }
}

/// What the owner heard of the replies Enton speaks: each reply's sentences and the
/// utterances that carry them, until the reply played out or was cut off.
#[cfg(feature = "voice")]
#[derive(Debug, Default)]
pub(crate) struct Heard {
    replies: VecDeque<SpokenReply>,
}

#[cfg(feature = "voice")]
impl Heard {
    /// `thought`'s reply will be spoken; `prompt` is what the owner said to it, if it
    /// answers them.
    pub(crate) fn begin(&mut self, thought: ThoughtId, prompt: Option<String>) {
        if self.replies.len() >= MAX_REPLIES {
            self.replies.pop_front();
        }
        self.replies.push_back(SpokenReply {
            thought,
            prompt,
            sentences: Vec::new(),
            finished: false,
            turn: None,
        });
    }

    /// A sentence of `thought`'s reply went to the player, as `utterance` (`None` when
    /// it had nothing to say out loud or the player refused it).
    pub(crate) fn sentence(
        &mut self,
        thought: ThoughtId,
        text: String,
        utterance: Option<UtteranceId>,
    ) {
        let Some(reply) = self
            .replies
            .iter_mut()
            .find(|reply| reply.thought == thought)
        else {
            // A reply cut off or forgotten while this sentence was on its way.
            return;
        };
        if reply.sentences.len() < MAX_SENTENCES {
            reply.sentences.push(Sentence {
                text,
                playback: if utterance.is_some() {
                    Playback::Pending
                } else {
                    Playback::Silent
                },
                utterance,
            });
        }
    }

    /// `utterance` played to its end (`true`) or failed (`false`).
    pub(crate) fn ended(&mut self, utterance: UtteranceId, played: bool) {
        let sentence = self
            .replies
            .iter_mut()
            .flat_map(|reply| reply.sentences.iter_mut())
            .find(|sentence| sentence.utterance == Some(utterance));
        if let Some(sentence) = sentence {
            sentence.playback = if played {
                Playback::Played
            } else {
                Playback::Silent
            };
        }
        self.replies.retain(|reply| !reply.settled());
    }

    /// The cortex finished `thought`'s reply; `turn` is where the reply was written in
    /// the conversation, `None` when it said nothing.
    pub(crate) fn finished(&mut self, thought: ThoughtId, turn: Option<TurnNo>) {
        if let Some(reply) = self
            .replies
            .iter_mut()
            .find(|reply| reply.thought == thought)
        {
            reply.finished = true;
            reply.turn = turn;
        }
        self.replies.retain(|reply| !reply.settled());
    }

    /// Stop tracking `thought`, abandoned without a cut: what it said stays unwritten.
    pub(crate) fn forget(&mut self, thought: ThoughtId) {
        self.replies.retain(|reply| reply.thought != thought);
    }

    /// The owner cut Enton off and nothing more will play. `played` says whether the
    /// player finished an utterance: it knows before the event reaches the loop. Each
    /// reply not yet heard in full keeps, in `conversation`, only what was heard: a
    /// finished reply's turn is rewritten; one the cortex was still writing, which will
    /// never be finished, is written now with the owner's prompt. One that had not
    /// spoken yet leaves no trace, as before.
    pub(crate) fn cut(
        &mut self,
        conversation: &mut Conversation,
        played: impl Fn(UtteranceId) -> bool,
    ) {
        for mut reply in self.replies.drain(..) {
            for sentence in &mut reply.sentences {
                if sentence.playback == Playback::Pending && sentence.utterance.is_some_and(&played)
                {
                    sentence.playback = Playback::Played;
                }
            }
            if reply.settled() {
                continue;
            }
            let heard = reply.heard();
            match reply.turn {
                Some(turn) => {
                    if let Some(written) = conversation.turn_mut(turn) {
                        written.content = heard;
                    }
                }
                None if reply.finished || reply.sentences.is_empty() => {}
                None => {
                    if let Some(prompt) = reply.prompt {
                        conversation.push(ConversationTurn::user(prompt));
                    }
                    conversation.push(ConversationTurn::assistant(heard));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_conversation_keeps_the_last_thirty_turns() {
        let mut conversation = Conversation::default();
        for n in 0..40 {
            conversation.push(ConversationTurn::user(format!("{n}")));
        }
        let turns = conversation.turns();
        assert_eq!(turns.len(), 30);
        assert_eq!(turns.first().map(|turn| turn.content.as_str()), Some("10"));
        assert_eq!(turns.last().map(|turn| turn.content.as_str()), Some("39"));
    }
}

#[cfg(all(test, feature = "voice"))]
mod heard_tests {
    use super::*;

    const PROMPT: &str = "Enton, conta uma história.";
    const REPLY: [&str; 3] = ["Era uma vez um robô.", "*risos*", "Ele morava num PC."];

    /// A conversation with one earlier exchange, and a reply to [`PROMPT`] whose
    /// sentences went to the player as utterances 11, none (a stage direction) and 12.
    fn speaking(finished: bool) -> (Conversation, Heard) {
        let mut conversation = Conversation::default();
        conversation.push(ConversationTurn::user("Enton, bom dia!"));
        conversation.push(ConversationTurn::assistant("Bom dia!"));
        let mut heard = Heard::default();
        let thought = ThoughtId(2);
        heard.begin(thought, Some(PROMPT.to_owned()));
        for (text, utterance) in REPLY.into_iter().zip([Some(11), None, Some(12)]) {
            heard.sentence(thought, text.to_owned(), utterance.map(UtteranceId));
        }
        if finished {
            let mut whole = String::new();
            for sentence in REPLY {
                append_sentence(&mut whole, sentence);
            }
            conversation.push(ConversationTurn::user(PROMPT));
            conversation.push(ConversationTurn::assistant(whole));
            heard.finished(thought, conversation.newest());
        }
        (conversation, heard)
    }

    #[test]
    fn sentences_join_a_space_apart() {
        let mut reply = String::new();
        for sentence in ["Oi.", "Tudo bem? ", "Bora."] {
            append_sentence(&mut reply, sentence);
        }
        assert_eq!(reply, "Oi. Tudo bem? Bora.");
    }

    fn contents(conversation: &Conversation) -> Vec<String> {
        conversation
            .turns()
            .into_iter()
            .map(|turn| turn.content)
            .collect()
    }

    #[test]
    fn a_reply_cut_off_keeps_only_the_sentences_heard() {
        let (mut conversation, mut heard) = speaking(true);
        heard.ended(UtteranceId(11), true);
        heard.cut(&mut conversation, |_| false);
        assert_eq!(
            contents(&conversation),
            [
                "Enton, bom dia!",
                "Bom dia!",
                PROMPT,
                "Era uma vez um robô. [interrupted: the owner heard only this]",
            ]
        );
        assert!(heard.replies.is_empty());
    }

    #[test]
    fn the_player_knows_what_finished_before_its_event_arrives() {
        let (mut conversation, mut heard) = speaking(true);
        heard.cut(&mut conversation, |id| id == UtteranceId(11));
        assert_eq!(
            contents(&conversation).last().map(String::as_str),
            Some("Era uma vez um robô. [interrupted: the owner heard only this]")
        );
        // Both finished: the whole reply was heard, and stays as it was.
        let (mut conversation, mut heard) = speaking(true);
        heard.cut(&mut conversation, |_| true);
        assert_eq!(
            contents(&conversation).last().map(String::as_str),
            Some("Era uma vez um robô. *risos* Ele morava num PC.")
        );
    }

    #[test]
    fn a_reply_cut_off_before_anything_played_says_so() {
        let (mut conversation, mut heard) = speaking(true);
        heard.cut(&mut conversation, |_| false);
        assert_eq!(
            contents(&conversation).last().map(String::as_str),
            Some(CUT_OFF_UNHEARD)
        );
    }

    #[test]
    fn a_reply_played_to_its_end_is_left_whole_and_forgotten() {
        let (mut conversation, mut heard) = speaking(true);
        heard.ended(UtteranceId(11), true);
        heard.ended(UtteranceId(12), true);
        assert!(heard.replies.is_empty());
        let before = contents(&conversation);
        heard.cut(&mut conversation, |_| false);
        assert_eq!(contents(&conversation), before);
    }

    #[test]
    fn a_failed_sentence_is_not_waited_for() {
        let (conversation, mut heard) = speaking(true);
        heard.ended(UtteranceId(11), true);
        heard.ended(UtteranceId(12), false);
        assert!(heard.replies.is_empty());
        assert_eq!(conversation.len(), 4);
    }

    #[test]
    fn a_reply_cut_off_while_the_cortex_still_wrote_it_is_written_with_its_prompt() {
        let (mut conversation, mut heard) = speaking(false);
        heard.ended(UtteranceId(11), true);
        heard.cut(&mut conversation, |_| false);
        assert_eq!(
            contents(&conversation),
            [
                "Enton, bom dia!",
                "Bom dia!",
                PROMPT,
                "Era uma vez um robô. [interrupted: the owner heard only this]",
            ]
        );
        // A sentence of it still on its way changes nothing.
        heard.sentence(ThoughtId(2), "Fim.".to_owned(), Some(UtteranceId(13)));
        assert!(heard.replies.is_empty());
    }

    #[test]
    fn a_reply_that_had_not_spoken_yet_leaves_no_trace() {
        let mut conversation = Conversation::default();
        let mut heard = Heard::default();
        heard.begin(ThoughtId(1), Some(PROMPT.to_owned()));
        heard.cut(&mut conversation, |_| false);
        assert_eq!(conversation.len(), 0);
        // Silence answering a drive is finished and settled at once.
        heard.begin(ThoughtId(2), None);
        heard.finished(ThoughtId(2), None);
        assert!(heard.replies.is_empty());
    }

    #[test]
    fn a_forgotten_reply_is_not_cut() {
        let (mut conversation, mut heard) = speaking(false);
        heard.forget(ThoughtId(2));
        heard.cut(&mut conversation, |_| false);
        assert_eq!(conversation.len(), 2);
    }

    #[test]
    fn a_turn_dropped_from_the_conversation_is_not_rewritten() {
        let (mut conversation, mut heard) = speaking(true);
        for n in 0..30 {
            conversation.push(ConversationTurn::user(format!("{n}")));
        }
        heard.cut(&mut conversation, |_| false);
        let turns = contents(&conversation);
        assert_eq!(turns.len(), 30);
        assert!(turns.iter().all(|turn| !turn.contains("interrupted")));
    }
}
