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

pub(super) fn assemble_messages<'a>(
    config: &'a CortexConfig,
    user_prompt: &'a str,
    pruned_history: &'a [ConversationTurn],
) -> Vec<OutgoingChatMessage<'a>> {
    let mut messages = Vec::with_capacity(pruned_history.len() + 2);
    messages.push(OutgoingChatMessage {
        role: "system",
        content: &config.system_prompt,
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
