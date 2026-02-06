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
    }

    let sdk = SdkState {
        inner,
        cancellation,
        state: initial_state,
        pending: VecDeque::new(),
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
                            return end(&s.state).map(|evt| (evt, (s, parse, end)));
                        }
                    }
                }
            }
        }
    })
}
