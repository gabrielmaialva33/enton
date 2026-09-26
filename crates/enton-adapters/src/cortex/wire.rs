use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub(super) struct OutgoingChatMessage<'a> {
    pub(super) role: &'a str,
    pub(super) content: &'a str,
}

#[derive(Debug, Serialize)]
pub(super) struct ChatCompletionRequest<'a> {
    pub(super) model: &'a str,
    pub(super) messages: Vec<OutgoingChatMessage<'a>>,
    pub(super) stream: bool,
    pub(super) reasoning_effort: &'a str,
}

/// A one-token, non-streaming completion that only makes the server load the model.
#[derive(Debug, Serialize)]
pub(super) struct WarmUpRequest<'a> {
    pub(super) model: &'a str,
    pub(super) messages: Vec<OutgoingChatMessage<'a>>,
    pub(super) max_tokens: u32,
    pub(super) stream: bool,
    pub(super) reasoning_effort: &'a str,
}

#[derive(Debug, Deserialize)]
pub(super) struct StreamCompletionChunk {
    pub(super) choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
pub(super) struct StreamChoice {
    pub(super) delta: StreamDelta,
}

#[derive(Debug, Deserialize)]
pub(super) struct StreamDelta {
    #[serde(default)]
    pub(super) content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_completion_request_serializes_reasoning_effort_and_omits_think() {
        let req = ChatCompletionRequest {
            model: "test-model",
            messages: vec![OutgoingChatMessage {
                role: "user",
                content: "hello",
            }],
            stream: true,
            reasoning_effort: "none",
        };
        let json = serde_json::to_string(&req).expect("serialization succeeds");
        assert!(json.contains(r#""reasoning_effort":"none""#));
        assert!(!json.contains("think"));
    }
}
