// Stream parser tests — extracted for governance compliance.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_emit_length() {
        assert_eq!(safe_emit_length("short"), 0);
        assert_eq!(safe_emit_length("a".repeat(30).as_str()), 10);
    }

    #[test]
    fn test_accumulate_tool_call() {
        let mut calls = Vec::new();
        let delta = SseToolCallDelta {
            index: Some(0),
            id: Some("call_1".to_string()),
            function: Some(SseFunctionDelta {
                name: Some("shell".to_string()),
                arguments: Some("{\"cmd\":".to_string()),
            }),
        };
        accumulate_tool_call(&mut calls, &delta);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");

        // Append more arguments
        let delta2 = SseToolCallDelta {
            index: Some(0),
            id: None,
            function: Some(SseFunctionDelta {
                name: None,
                arguments: Some("\"ls\"}".to_string()),
            }),
        };
        accumulate_tool_call(&mut calls, &delta2);
        assert_eq!(calls[0].arguments, "{\"cmd\":\"ls\"}");
    }

    #[test]
    fn test_thought_spiral_detection() {
        assert!(!detect_thought_spiral("short"));

        let spiral = (0..40)
            .map(|_| "Let me think about this again and reconsider the implications")
            .collect::<Vec<_>>()
            .join("\n");
        assert!(detect_thought_spiral(&spiral));
    }

    #[test]
    fn test_no_spiral_in_varied_content() {
        let content = (0..20)
            .map(|i| format!("Unique line {}", i))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!detect_thought_spiral(&content));
    }

    #[tokio::test]
    async fn test_thinking_extraction_gemma4() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut state = ThinkingState::Normal;
        let mut buffer = String::new();

        process_content_delta(
            "Hello <|channel>thought I am reasoning <channel|> world",
            &mut state,
            &mut buffer,
            &tx,
        )
        .await;

        drop(tx);

        let mut events = Vec::new();
        while let Some(e) = rx.recv().await {
            events.push(e);
        }

        assert!(events.iter().any(|e| matches!(e, StreamEvent::ThinkingDelta(t) if t.contains("reasoning"))));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextDelta(t) if t.contains("Hello"))));
    }

    #[tokio::test]
    async fn test_thinking_extraction_legacy() {
        let (tx, mut rx) = mpsc::channel(32);
        let mut state = ThinkingState::Normal;
        let mut buffer = String::new();

        process_content_delta(
            "Before <think>deep thought</think> After",
            &mut state,
            &mut buffer,
            &tx,
        )
        .await;

        drop(tx);

        let mut events = Vec::new();
        while let Some(e) = rx.recv().await {
            events.push(e);
        }

        assert!(events.iter().any(|e| matches!(e, StreamEvent::ThinkingDelta(t) if t.contains("deep thought"))));
    }

    #[tokio::test]
    async fn test_tool_call_emission() {
        let (tx, mut rx) = mpsc::channel(32);
        let calls = vec![ToolCallAccumulator {
            id: "c1".to_string(),
            name: "shell".to_string(),
            arguments: "{\"cmd\":\"ls\"}".to_string(),
        }];

        emit_accumulated_tools(&calls, &tx).await;
        drop(tx);

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, StreamEvent::ToolCall { name, .. } if name == "shell"));
    }
}
