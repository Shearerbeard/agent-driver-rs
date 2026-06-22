//! Integration tests for ClusterGuardian-style configuration.
//!
//! Validates that the same config wiring used in `examples/cluster_guardian.rs`
//! works end-to-end using `MockProvider` — no real Ollama or MCP servers needed.
//!
//! Requires `--features test-support` to compile.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::FutureExt as _;

    use agent_driver_rs::agent::{AgentLoop, AgentLoopConfig, LoopStopReason, MaxToolDepth};
    use agent_driver_rs::provider::mock::*;
    use agent_driver_rs::tool::FnTool;
    use agent_driver_rs::{
        CompletionConfig, DynTool, MaxTokens, ModelId, SessionBuilder, SystemPrompt,
        ToolDefinition, ToolName, ToolResult, ToolSchema,
    };

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn model() -> ModelId {
        ModelId::new("glm-4.7-flash-128k").unwrap()
    }

    const SYSTEM_PROMPT: &str = "You are ClusterGuardian, an expert Kubernetes SRE agent.";

    fn guardian_completion_config() -> CompletionConfig {
        CompletionConfig {
            max_tokens: MaxTokens::new(2000).expect("2000 is non-zero"),
            temperature: None,
            stop_sequences: vec![],
        }
    }

    fn kubectl_tool() -> DynTool {
        let definition = ToolDefinition::new(
            ToolName::new("kubectl_get").expect("hardcoded valid tool name"),
            "Get Kubernetes resources",
            ToolSchema::empty(),
        );
        Arc::new(FnTool::new(definition, |_input, _ctx| {
            async {
            Ok(ToolResult::text(
                "NAME   READY   STATUS    RESTARTS   AGE\nnginx  1/1     Running   0          5d",
            ))
        }
        .boxed()
        }))
    }

    fn memory_tool() -> DynTool {
        let definition = ToolDefinition::new(
            ToolName::new("memory_store").expect("hardcoded valid tool name"),
            "Store findings in memory",
            ToolSchema::empty(),
        );
        Arc::new(FnTool::new(definition, |_input, _ctx| {
            async { Ok(ToolResult::text("Stored successfully")) }.boxed()
        }))
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /// Construct a Session with the same config fields as ClusterGuardian:
    /// custom model, system prompt, max_tokens, and multiple tools.
    #[tokio::test]
    async fn cluster_guardian_session_setup() {
        let session = SessionBuilder::new()
            .with_provider(MockProvider::new(vec![mock_text_response("ready")]))
            .model(model())
            .completion_config(guardian_completion_config())
            .system_prompt(SystemPrompt::new(SYSTEM_PROMPT))
            .tool(kubectl_tool())
            .tool(memory_tool())
            .build()
            .await
            .unwrap();

        // Verify session properties
        assert_eq!(session.model().as_str(), "glm-4.7-flash-128k");
        assert_eq!(session.system_prompt().await.as_str(), SYSTEM_PROMPT);

        // Verify tools are registered
        let tools = session.list_tools().await;
        assert_eq!(tools.len(), 2);
        let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(tool_names.contains(&"kubectl_get"));
        assert!(tool_names.contains(&"memory_store"));

        // Verify the session can send a message
        let response = session.send("ping").await.unwrap();
        assert_eq!(response.text(), "ready");
    }

    /// Full pipeline: MockProvider returns text → tool call → text.
    /// Uses the same AgentLoopConfig as ClusterGuardian.
    #[tokio::test]
    async fn cluster_guardian_agent_loop() {
        let provider = MockProvider::new(vec![
            // First response: call kubectl_get
            mock_tool_call_response("call_1", "kubectl_get", "{}"),
            // Second response: call memory_store
            mock_tool_call_response("call_2", "memory_store", "{}"),
            // Third response: final text
            mock_text_response("Cluster health check complete. All pods running normally."),
        ]);

        let session = SessionBuilder::new()
            .with_provider(provider)
            .model(model())
            .completion_config(guardian_completion_config())
            .system_prompt(SystemPrompt::new(SYSTEM_PROMPT))
            .tool(kubectl_tool())
            .tool(memory_tool())
            .build()
            .await
            .unwrap();

        let config = AgentLoopConfig {
            max_tool_depth: MaxToolDepth::new(100).unwrap(),
            fallback_tool_parsing: true,
            name: Some("ClusterGuardian".into()),
            continue_on_tool_error: true,
        };

        let outcome = AgentLoop::new(&session)
            .with_config(config)
            .run("Perform cluster health check")
            .await
            .unwrap();

        // Verify outcome
        assert_eq!(outcome.iterations, 2);
        assert!(matches!(outcome.stop_reason, LoopStopReason::EndTurn));
        assert_eq!(
            outcome.final_response.text(),
            "Cluster health check complete. All pods running normally."
        );

        // Verify all 3 responses were collected
        assert_eq!(outcome.responses.len(), 3);
    }

    /// MockProvider returns a tool call embedded in text (not native ToolUse blocks).
    /// With `fallback_tool_parsing: true`, the agent loop should extract it and execute.
    #[tokio::test]
    async fn cluster_guardian_fallback_tool_parsing() {
        // First response: text with an embedded tool call (XML-tagged format)
        let embedded_tool_text = r#"Let me check the pods.
<tool_call>{"name": "kubectl_get", "arguments": {}}</tool_call>"#;
        let provider = MockProvider::new(vec![
            mock_text_response(embedded_tool_text),
            // After fallback parsing extracts and executes the tool, model responds with text
            mock_text_response("Pods are healthy."),
        ]);

        let session = SessionBuilder::new()
            .with_provider(provider)
            .model(model())
            .completion_config(guardian_completion_config())
            .system_prompt(SystemPrompt::new(SYSTEM_PROMPT))
            .tool(kubectl_tool())
            .build()
            .await
            .unwrap();

        let config = AgentLoopConfig {
            max_tool_depth: MaxToolDepth::new(100).unwrap(),
            fallback_tool_parsing: true,
            name: Some("ClusterGuardian".into()),
            continue_on_tool_error: true,
        };

        let outcome = AgentLoop::new(&session)
            .with_config(config)
            .run("Check pod status")
            .await
            .unwrap();

        // Fallback parsing should have extracted the tool call and executed it
        assert_eq!(
            outcome.iterations, 1,
            "expected 1 tool iteration from fallback parsing"
        );
        assert_eq!(outcome.final_response.text(), "Pods are healthy.");
        assert!(matches!(outcome.stop_reason, LoopStopReason::EndTurn));
    }
}
