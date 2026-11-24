//! Offline wire-request accounting through the public OpenAI provider.

#[cfg(test)]
#[cfg(feature = "openai")]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "fixture setup and assertions panic on unexpected failures"
)]
mod tests {
    use std::time::Duration;

    use agent_driver::config::{ApiKey, OpenAiConfig, OpenAiModel};
    use agent_driver::provider::{
        CompletionRequest, OpenAiProvider, ProviderContext, ProviderExt as _,
    };
    use agent_driver::{CorrelationId, MaxTokens, Message, ModelId};
    use tokio::io::{AsyncBufReadExt as _, AsyncReadExt as _, AsyncWriteExt as _, BufReader};
    use tokio::net::TcpListener;
    use tokio_util::sync::CancellationToken;
    use tokio_util::task::TaskTracker;

    async fn exchange(status: u16, disable_retries: bool) -> usize {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = OpenAiConfig {
            api_key: ApiKey::new("offline-test-key").unwrap(),
            base_url: Some(format!("http://{}/v1", listener.local_addr().unwrap())),
            model: OpenAiModel::Gpt4o,
            max_tokens: MaxTokens::new(32).unwrap(),
            temperature: None,
            reasoning: None,
            structured_output: false,
        };
        let provider = OpenAiProvider::new(config).unwrap();
        let provider = if disable_retries {
            provider.without_retries().unwrap()
        } else {
            provider
        };
        let done = CancellationToken::new();
        let server = async {
            let mut count = 0;
            loop {
                let (socket, _) = tokio::select! {
                    biased;
                    () = done.cancelled() => break count,
                    accepted = listener.accept() => accepted.unwrap(),
                };
                let mut socket = BufReader::new(socket);
                let mut content_length = 0;
                loop {
                    let mut line = String::new();
                    socket.read_line(&mut line).await.unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            content_length = value.trim().parse().unwrap();
                        }
                    }
                }
                let mut body = vec![0; content_length];
                socket.read_exact(&mut body).await.unwrap();
                count += 1;
                let body = r#"{"error":{"message":"fixture failure","type":"server_error"}}"#;
                socket
                    .write_all(
                        format!(
                            "HTTP/1.1 {status} Fixture\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await
                    .unwrap();
                socket.flush().await.unwrap();
            }
        };
        let request = async {
            let result = provider
                .complete(
                    CompletionRequest::new(
                        ModelId::new("gpt-4o").unwrap(),
                        vec![Message::user("offline fixture")],
                    ),
                    ProviderContext::new(
                        CorrelationId::generate(),
                        CancellationToken::new(),
                        TaskTracker::new(),
                    ),
                )
                .await;
            done.cancel();
            assert!(result.is_err(), "HTTP {status} must reach the caller");
        };
        let ((), count) = tokio::time::timeout(Duration::from_secs(30), async {
            tokio::join!(request, server)
        })
        .await
        .expect("offline exchange must finish");
        count
    }

    #[tokio::test]
    async fn disabled_retries_send_one_request_on_rate_limit() {
        assert_eq!(exchange(429, true).await, 1);
    }

    #[tokio::test]
    async fn disabled_retries_send_one_request_on_server_failure() {
        assert_eq!(exchange(503, true).await, 1);
    }

    #[tokio::test]
    async fn default_provider_retains_sdk_retries() {
        assert_eq!(exchange(503, false).await, 4);
    }
}
