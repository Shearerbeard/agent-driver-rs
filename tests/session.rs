//! Integration tests for the Session.
//!
//! These tests exercise the Session through the public API using
//! `MockProvider` for deterministic, network-free testing.
//!
//! Requires `--features test-support` to compile because `MockProvider` is
//! behind `#[cfg(any(test, feature = "test-support"))]` in the library, and
//! integration tests are a separate crate where `cfg(test)` does not apply
//! to the library.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::FutureExt as _;

    use agent_driver_rs::provider::mock::*;
    use agent_driver_rs::session::SessionBuilder;
    use agent_driver_rs::streaming::StreamEvent;
    use agent_driver_rs::tool::{DynTool, FnTool, ToolDefinition, ToolResult, ToolSchema};
    use agent_driver_rs::types::{Message, ModelId, Role, SystemPrompt, ToolName};

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn model() -> ModelId {
        ModelId::new("mock-model").unwrap()
    }

    fn mock_session(responses: Vec<Vec<StreamEvent>>) -> SessionBuilder {
        SessionBuilder::new()
            .with_provider(MockProvider::new(responses))
            .model(model())
    }

    /// Create a simple echo tool that returns "echoed!" for any input.
    fn echo_tool() -> DynTool {
        let definition = ToolDefinition::new(
            ToolName::new("echo").expect("hardcoded valid tool name"),
            "Echoes back",
            ToolSchema::empty(),
        );
        Arc::new(FnTool::new(definition, |_input, _ctx| {
            async { Ok(ToolResult::text("echoed!")) }.boxed()
        }))
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /// Full send -> tool_execute -> continue cycle.
    ///
    /// 1. Register an "echo" tool.
    /// 2. First `send()` triggers a tool_use response (assistant message added to history).
    /// 3. `process_tool_calls()` executes the tool and adds tool_result to history.
    /// 4. `continue_streaming()` sends current history back to the provider, which
    ///    responds with "all done".
    /// 5. Verify final text and message count.
    #[tokio::test]
    async fn send_then_tool_then_continue() {
        let session = mock_session(vec![
            mock_tool_call_response("call_1", "echo", "{}"),
            mock_text_response("all done"),
        ])
        .tool(echo_tool())
        .build()
        .await
        .unwrap();

        // Step 1: send triggers tool_use response
        let response = session.send("use the echo tool").await.unwrap();
        assert!(
            response.has_tool_use(),
            "expected tool_use in response, got: {:?}",
            response.content
        );

        // Step 2: process tool calls (executes echo, adds tool_result to history)
        let results = session.process_tool_calls(&response).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_success());
        assert_eq!(results[0].content(), "echoed!");

        // Step 3: continue streaming to get the final text response
        let handle = session.continue_streaming().await.unwrap();
        let final_response = handle.collect().await.unwrap();
        assert_eq!(final_response.text(), "all done");

        // History: user + assistant(tool_use) + tool_result = 3
        // (continue_streaming does NOT auto-add the assistant response)
        let msgs = session.messages().await;
        assert_eq!(
            msgs.len(),
            3,
            "expected 3 messages in history, got {}: {:?}",
            msgs.len(),
            msgs.iter().map(|m| &m.role).collect::<Vec<_>>()
        );
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        assert_eq!(msgs[2].role, Role::Tool);
    }

    /// Boundary test with `max_history(2)`.
    ///
    /// After `send()` adds 2 messages (user + assistant), adding one more message
    /// pushes history over the limit and the oldest message is trimmed.
    #[tokio::test]
    async fn history_trimming_max_2() {
        let session = mock_session(vec![mock_text_response("reply")])
            .max_history(2)
            .build()
            .await
            .unwrap();

        // send() adds user + assistant = 2, exactly at limit
        session.send("msg1").await.unwrap();
        assert_eq!(session.message_count().await, 2);

        // Adding one more message pushes to 3, trimmed back to 2
        session.add_message(Message::user("overflow")).await;
        assert_eq!(session.message_count().await, 2);

        // The first message ("msg1") was trimmed
        let msgs = session.messages().await;
        assert_ne!(
            msgs[0].text(),
            "msg1",
            "oldest message should have been trimmed"
        );
        // First message is now the assistant reply
        assert_eq!(msgs[0].role, Role::Assistant);
        assert_eq!(msgs[0].text(), "reply");
        // Second message is the overflow
        assert_eq!(msgs[1].role, Role::User);
        assert_eq!(msgs[1].text(), "overflow");
    }

    /// max_history=4 with 3 sends verifies that trimming preserves recent messages.
    ///
    /// Each `send()` adds 2 messages (user + assistant). After 3 sends there would
    /// be 6 messages, trimmed to 4. The oldest 2 (from the first send) are removed.
    #[tokio::test]
    async fn history_trimming_preserves_recent() {
        let session = mock_session(vec![
            mock_text_response("r1"),
            mock_text_response("r2"),
            mock_text_response("r3"),
        ])
        .max_history(4)
        .build()
        .await
        .unwrap();

        session.send("m1").await.unwrap(); // history: [user(m1), assistant(r1)] = 2
        session.send("m2").await.unwrap(); // history: [user(m1), assistant(r1), user(m2), assistant(r2)] = 4
        session.send("m3").await.unwrap(); // would be 6, trimmed to 4

        let msgs = session.messages().await;
        assert_eq!(msgs.len(), 4);

        // Oldest 2 messages (user m1, assistant r1) were trimmed.
        // msgs[0] should be user "m2" from the second send.
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[0].text(), "m2");
    }

    /// 10 spawned tasks adding messages simultaneously.
    ///
    /// Verifies no writes are lost under concurrent access thanks to the RwLock
    /// protecting the message history.
    #[tokio::test]
    async fn concurrent_add_message() {
        let session = Arc::new(mock_session(vec![]).build().await.unwrap());

        let mut handles = vec![];
        for i in 0..10 {
            let s = Arc::clone(&session);
            handles.push(tokio::spawn(async move {
                s.add_message(Message::user(format!("msg{i}"))).await;
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        assert_eq!(
            session.message_count().await,
            10,
            "expected 10 messages, no writes should be lost"
        );
    }

    /// Tool added mid-conversation appears in the registry.
    ///
    /// Verifies that registering a tool between turns makes it available for
    /// subsequent requests (the session re-reads the registry each turn).
    #[tokio::test]
    async fn register_tool_between_turns() {
        let session = mock_session(vec![mock_text_response("hello back")])
            .build()
            .await
            .unwrap();

        // No tools initially
        assert_eq!(session.list_tools().await.len(), 0);

        // Send a message (works without tools)
        session.send("hello").await.unwrap();

        // Register a tool after the first turn
        session.register_tool(echo_tool()).await;

        // Tool is now visible
        let tools = session.list_tools().await;
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name.as_str(), "echo");
    }

    /// Tool removed mid-conversation is absent from the registry.
    ///
    /// Verifies that `remove_tool` actually removes the tool so it will not
    /// appear in subsequent completion requests.
    #[tokio::test]
    async fn remove_tool_between_turns() {
        let session = mock_session(vec![])
            .tool(echo_tool())
            .build()
            .await
            .unwrap();

        // Tool is registered
        assert_eq!(session.list_tools().await.len(), 1);

        // Remove it
        let removed = session
            .remove_tool(&ToolName::new("echo").expect("hardcoded valid tool name"))
            .await;
        assert!(
            removed.is_some(),
            "remove_tool should return the removed tool"
        );

        // Tool is gone
        assert_eq!(session.list_tools().await.len(), 0);
    }

    /// System prompt swap between turns.
    ///
    /// Verifies that `set_system_prompt` replaces the prompt and the new value
    /// is returned by `system_prompt()`.
    #[tokio::test]
    async fn system_prompt_change_mid_conversation() {
        let session = mock_session(vec![])
            .system_prompt(SystemPrompt::new("initial"))
            .build()
            .await
            .unwrap();

        assert_eq!(session.system_prompt().await.as_str(), "initial");

        session
            .set_system_prompt(SystemPrompt::new("updated"))
            .await;
        assert_eq!(session.system_prompt().await.as_str(), "updated");
    }

    /// `shutdown()` sets the cancelled state.
    ///
    /// Verifies that calling `shutdown()` triggers cancellation so that
    /// `is_cancelled()` returns true.
    #[tokio::test]
    async fn shutdown_cancels_operations() {
        let session = mock_session(vec![]).build().await.unwrap();

        assert!(!session.is_cancelled());

        session.shutdown().await;

        assert!(
            session.is_cancelled(),
            "session should be cancelled after shutdown"
        );
    }
}
