use enton_core::ports::{ConversationTurn, TurnRole};

use super::config::CortexConfig;
use super::wire::OutgoingChatMessage;

/// Prunes conversation history from the middle outward to fit within a token budget.
///
/// Adapted from Goose (Apache-2.0):
/// <https://github.com/aaif-goose/goose/blob/main/crates/goose-context-management/src/summarize.rs>
///
/// This keeps the earliest conversational context and the most recent turns while
/// removing stale turns from the middle first.
#[must_use]
pub fn prune_history_middle_out(
    history: &[ConversationTurn],
    max_tokens: usize,
) -> Vec<ConversationTurn> {
    if history.is_empty() {
        return Vec::new();
    }

    // Heuristic estimation: ~4 chars per token plus 4 tokens message framing overhead.
    let turn_tokens = |turn: &ConversationTurn| -> usize { turn.content.len().div_ceil(4) + 4 };

    let mut total_tokens: usize = history.iter().map(turn_tokens).sum();
    if total_tokens <= max_tokens {
        return history.to_vec();
    }

    let count = history.len();
    let mid = count / 2;

    // Sort candidate removal indices by distance to the middle (closest removed first).
    let mut removal_order: Vec<usize> = (0..count).collect();
    removal_order.sort_by_key(|&idx| idx.abs_diff(mid));

    let mut removed = vec![false; count];
    for idx in removal_order {
        if total_tokens <= max_tokens {
            break;
        }
        if let Some(turn) = history.get(idx) {
            if let Some(r) = removed.get_mut(idx) {
                *r = true;
            }
            total_tokens = total_tokens.saturating_sub(turn_tokens(turn));
        }
    }

    history
        .iter()
        .zip(removed.iter())
        .filter_map(
            |(turn, &is_removed)| {
                if is_removed { None } else { Some(turn.clone()) }
            },
        )
        .collect()
}

/// The system prompt with the local date and time appended: the core has no wall
/// clock, so without this the model cannot answer "what time is it?".
pub(super) fn system_prompt_at(config: &CortexConfig, now: &jiff::Zoned) -> String {
    format!("{}\n\n{}", config.system_prompt, clock_line(now))
}

/// One line naming the local date, time and time zone, for example
/// `Current local date and time: Saturday, 2026-09-26 16:42 (America/Sao_Paulo, UTC-03:00).`
pub(super) fn clock_line(now: &jiff::Zoned) -> String {
    let zone = now.time_zone().iana_name().unwrap_or("local time");
    format!(
        "Current local date and time: {} ({zone}, UTC{}).",
        now.strftime("%A, %Y-%m-%d %H:%M"),
        now.strftime("%:z"),
    )
}

pub(super) fn assemble_messages<'a>(
    system_prompt: &'a str,
    user_prompt: &'a str,
    pruned_history: &'a [ConversationTurn],
) -> Vec<OutgoingChatMessage<'a>> {
    let mut messages = Vec::with_capacity(pruned_history.len() + 2);
    messages.push(OutgoingChatMessage {
        role: "system",
        content: system_prompt,
    });

    for turn in pruned_history {
        let role_str = match turn.role {
            TurnRole::User => "user",
            TurnRole::Assistant => "assistant",
        };
        messages.push(OutgoingChatMessage {
            role: role_str,
            content: &turn.content,
        });
    }

    messages.push(OutgoingChatMessage {
        role: "user",
        content: user_prompt,
    });

    messages
}

#[cfg(test)]
mod tests {

    #[test]
    fn the_clock_line_names_date_time_and_zone() {
        let now: jiff::Zoned = "2026-09-26T16:42:00-03:00[America/Sao_Paulo]"
            .parse()
            .unwrap();
        assert_eq!(
            clock_line(&now),
            "Current local date and time: Saturday, 2026-09-26 16:42 (America/Sao_Paulo, UTC-03:00)."
        );
        let config = CortexConfig::default();
        let prompt = system_prompt_at(&config, &now);
        assert!(prompt.starts_with(&config.system_prompt));
        assert!(prompt.ends_with("UTC-03:00)."));
    }

    use super::*;

    #[test]
    fn prune_history_keeps_head_and_tail_when_middle_dropped() {
        let turns: Vec<ConversationTurn> = (0..5)
            .map(|i| ConversationTurn::user(format!("Turn {i}: test turn content")))
            .collect();

        // Budget that fits exactly the 2 boundary turns, dropping the 3 middle turns:
        let pruned = prune_history_middle_out(&turns, 30);
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned[0].content, turns[0].content);
        assert_eq!(pruned[1].content, turns[4].content);
    }

    #[test]
    fn prune_history_within_budget_retains_all() {
        let turns = vec![
            ConversationTurn::user("hi"),
            ConversationTurn::assistant("hello"),
        ];
        let pruned = prune_history_middle_out(&turns, 1000);
        assert_eq!(pruned.len(), 2);
        assert_eq!(pruned[0].content, "hi");
        assert_eq!(pruned[1].content, "hello");
    }
}
