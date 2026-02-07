//! Shared streaming infrastructure for all providers.
//!
//! [`buffered_sse_stream`] encapsulates the common pattern used by
//! Anthropic and OpenRouter providers: `futures::stream::unfold` over an
//! `EventSource` with a `VecDeque` buffer, cancellation-aware `tokio::select!`,
//! and drain-before-fetch ordering.
//!
//! [`buffered_sdk_stream`] provides the equivalent for SDK-based providers
//! (OpenAI, Ollama) that yield items from a `futures::Stream`.

use std::collections::VecDeque;

use futures::{Stream, StreamExt};
use reqwest_eventsource::{Event, EventSource};
use tokio_util::sync::CancellationToken;

use crate::error::StreamError;
use crate::streaming::StreamEvent;

/// Create a cancellation-aware buffered stream from an SSE `EventSource`.
///
/// This handles the common pattern shared by Anthropic and OpenRouter:
/// - `VecDeque` drain on each poll (multi-event SSE messages)
/// - `tokio::select! { biased; }` with cancellation
/// - `Event::Open` is skipped
/// - A `done_message` check for provider-specific termination (e.g. `"[DONE]"`)
///
/// `parse_fn` receives the SSE `data` string and mutable state, returning:
/// - `Some(Ok(events))` — parsed events to emit
/// - `Some(Err(e))` — stream error to emit
/// - `None` — skip this SSE message (continue polling)
///
/// `on_done` is called when the done marker is received. It receives the
/// current state and returns an optional final event to emit before the
/// stream terminates. Return `None` to just end the stream silently.
pub fn buffered_sse_stream<State, F, D>(
    event_source: EventSource,
    cancellation: CancellationToken,
    initial_state: State,
    done_message: Option<&'static str>,
    parse_fn: F,
    on_done: D,
) -> impl Stream<Item = Result<StreamEvent, StreamError>>
where
    State: Send + 'static,
    F: Fn(&str, &mut State) -> Option<Result<Vec<StreamEvent>, StreamError>> + Send + 'static,
    D: Fn(&State) -> Option<StreamEvent> + Send + 'static,
{
    struct SseState<S> {
        event_source: EventSource,
        cancellation: CancellationToken,
        state: S,
        pending: VecDeque<Result<StreamEvent, StreamError>>,
        done_msg: Option<&'static str>,
    }

    let sse = SseState {
        event_source,
        cancellation,
        state: initial_state,
        pending: VecDeque::new(),
        done_msg: done_message,
    };

    futures::stream::unfold((sse, parse_fn, on_done), |(mut s, parse, done)| async move {
        // Drain buffered events first
        if let Some(event) = s.pending.pop_front() {
            return Some((event, (s, parse, done)));
        }

        loop {
            if s.cancellation.is_cancelled() {
                return None;
            }

            tokio::select! {
                biased;

                _ = s.cancellation.cancelled() => {
                    return None;
                }

                event = s.event_source.next() => {
                    match event {
                        Some(Ok(Event::Open)) => continue,
                        Some(Ok(Event::Message(msg))) => {
                            // Check for provider-specific done marker
                            if let Some(done_str) = s.done_msg {
                                if msg.data == done_str {
                                    return done(&s.state)
                                        .map(|evt| (Ok(evt), (s, parse, done)));
                                }
                            }

                            match parse(&msg.data, &mut s.state) {
                                Some(Ok(events)) => {
                                    let mut iter = events.into_iter();
                                    if let Some(first) = iter.next() {
                                        for remaining in iter {
                                            s.pending.push_back(Ok(remaining));
                                        }
                                        return Some((Ok(first), (s, parse, done)));
                                    }
                                    continue;
                                }
                                Some(Err(e)) => {
                                    return Some((Err(e), (s, parse, done)));
                                }
                                None => continue,
                            }
                        }
                        Some(Err(e)) => {
                            let err = StreamError::ConnectionLost(e.to_string());
                            return Some((Err(err), (s, parse, done)));
                        }
                        None => return None,
                    }
                }
            }
        }
    })
}

/// Create a cancellation-aware buffered stream from a `futures::Stream`.
///
/// This handles the common pattern shared by OpenAI and Ollama SDK providers:
/// - `VecDeque` drain on each poll (multi-event chunks)
/// - `tokio::select! { biased; }` with cancellation
/// - Skips chunks where `parse_fn` returns empty vec (continues polling)
/// - `on_end` callback when the inner stream terminates
///
/// `parse_fn` receives an item from the inner stream and mutable state,
/// returning a `Vec` of events to emit (may be empty to skip).
///
/// `on_end` is called when the inner stream yields `None`. Return an optional
/// final event (typically `Completed`) to emit before the stream terminates.
pub fn buffered_sdk_stream<Inner, Item, State, F, E>(
    inner: Inner,
    cancellation: CancellationToken,
    initial_state: State,
    parse_fn: F,
    on_end: E,
) -> impl Stream<Item = Result<StreamEvent, StreamError>>
where
    Inner: Stream<Item = Item> + Send + Unpin + 'static,
    State: Send + 'static,
    F: Fn(Item, &mut State) -> Vec<Result<StreamEvent, StreamError>> + Send + 'static,
    E: Fn(&State) -> Option<Result<StreamEvent, StreamError>> + Send + 'static,
{
    struct SdkState<S, I> {
        inner: I,
        cancellation: CancellationToken,
        state: S,
        pending: VecDeque<Result<StreamEvent, StreamError>>,
        ended: bool,
    }

    let sdk = SdkState {
        inner,
        cancellation,
        state: initial_state,
        pending: VecDeque::new(),
        ended: false,
    };

    futures::stream::unfold((sdk, parse_fn, on_end), |(mut s, parse, end)| async move {
        // Drain buffered events first
        if let Some(event) = s.pending.pop_front() {
            return Some((event, (s, parse, end)));
        }

        loop {
            if s.cancellation.is_cancelled() {
                return None;
            }

            tokio::select! {
                biased;

                _ = s.cancellation.cancelled() => {
                    return None;
                }

                item = s.inner.next() => {
                    match item {
                        Some(raw) => {
                            let events = parse(raw, &mut s.state);
                            if events.is_empty() {
                                continue;
                            }
                            let mut iter = events.into_iter();
                            let first = iter.next().expect("non-empty vec");
                            for remaining in iter {
                                s.pending.push_back(remaining);
                            }
                            return Some((first, (s, parse, end)));
                        }
                        None => {
                            if s.ended {
                                return None;
                            }
                            s.ended = true;
                            return end(&s.state).map(|evt| (evt, (s, parse, end)));
                        }
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{CompletionMetadata, ContentBlockType, StreamDelta};
    use futures::StreamExt;

    #[tokio::test]
    async fn sdk_stream_emits_events_and_on_end() {
        let inner = futures::stream::iter(vec!["hello", "world"]);
        let cancel = CancellationToken::new();

        let stream = buffered_sdk_stream(
            inner,
            cancel,
            0u32,
            |item: &str, state: &mut u32| {
                *state += 1;
                vec![Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: item.to_string(),
                }))]
            },
            |_state| {
                Some(Ok(StreamEvent::Completed {
                    metadata: CompletionMetadata::default(),
                }))
            },
        );

        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 3);
        assert!(matches!(
            &events[0],
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "hello"
        ));
        assert!(matches!(
            &events[1],
            Ok(StreamEvent::Delta(StreamDelta::TextDelta { text })) if text == "world"
        ));
        assert!(matches!(&events[2], Ok(StreamEvent::Completed { .. })));
    }

    #[tokio::test]
    async fn sdk_stream_pre_cancelled_yields_nothing() {
        let inner = futures::stream::iter(vec!["hello"]);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let stream = buffered_sdk_stream(
            inner,
            cancel,
            (),
            |item: &str, _: &mut ()| {
                vec![Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: item.to_string(),
                }))]
            },
            |_| Some(Ok(StreamEvent::Completed { metadata: CompletionMetadata::default() })),
        );

        let events: Vec<_> = stream.collect().await;
        assert!(events.is_empty(), "cancelled stream should yield no events");
    }

    #[tokio::test]
    async fn sdk_stream_multi_event_buffering() {
        let inner = futures::stream::iter(vec!["chunk"]);
        let cancel = CancellationToken::new();

        let stream = buffered_sdk_stream(
            inner,
            cancel,
            (),
            |_item: &str, _: &mut ()| {
                vec![
                    Ok(StreamEvent::ContentBlockStart {
                        index: 0,
                        block_type: ContentBlockType::Text,
                    }),
                    Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                        text: "hello".to_string(),
                    })),
                    Ok(StreamEvent::ContentBlockStop { index: 0 }),
                ]
            },
            |_| None,
        );

        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], Ok(StreamEvent::ContentBlockStart { .. })));
        assert!(matches!(&events[1], Ok(StreamEvent::Delta(..))));
        assert!(matches!(&events[2], Ok(StreamEvent::ContentBlockStop { .. })));
    }

    #[tokio::test]
    async fn sdk_stream_on_end_none_yields_no_final_event() {
        let inner = futures::stream::iter(vec!["only"]);
        let cancel = CancellationToken::new();

        let stream = buffered_sdk_stream(
            inner,
            cancel,
            (),
            |item: &str, _: &mut ()| {
                vec![Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: item.to_string(),
                }))]
            },
            |_| None, // No final event
        );

        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 1); // Just the text delta, no Completed
    }

    #[tokio::test]
    async fn sdk_stream_terminates_after_on_end() {
        // Verifies the on_end infinite-poll bug is fixed:
        // collect() should return without needing .take().
        let inner = futures::stream::iter(vec!["a", "b"]);
        let cancel = CancellationToken::new();

        let stream = buffered_sdk_stream(
            inner,
            cancel,
            0u32,
            |item: &str, state: &mut u32| {
                *state += 1;
                vec![Ok(StreamEvent::Delta(StreamDelta::TextDelta {
                    text: item.to_string(),
                }))]
            },
            |_state| {
                Some(Ok(StreamEvent::Completed {
                    metadata: CompletionMetadata {
                        model: None,
                        stop_reason: Some(crate::streaming::StopReason::EndTurn),
                        usage: None,
                    },
                }))
            },
        );

        // This would hang forever before the fix
        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), 3); // "a", "b", Completed
        assert!(matches!(&events[2], Ok(StreamEvent::Completed { .. })));
    }
}
