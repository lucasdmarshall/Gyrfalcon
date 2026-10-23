//! `answer`: retrieve, fill the prompt, and stream an OpenAI-compatible chat completion.

use std::collections::VecDeque;
use std::io::BufRead;
use std::time::Instant;

use crate::epoch::ReaderGuard;
use crate::error::{Error, Result};
use crate::http::{self, Url};
use crate::index::{Document, Index};
use crate::retrieve::{Hit, RetrieveOptions};

/// The answer prompt, byte for byte from the spec.
pub const ANSWER_PROMPT: &str = include_str!("../../../bench/answer_prompt.txt");
/// The judge prompt, byte for byte from the spec.
pub const JUDGE_PROMPT: &str = include_str!("../../../bench/judge_prompt.txt");
pub const CONTEXT_K: usize = 5;

/// Where `answer` and contextual prefixes send chat completions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatConfig {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerOptions {
    pub rerank: bool,
    /// Bodies placed in the prompt.
    pub context_k: usize,
    /// Overrides the index's chat config for this call.
    pub chat: Option<ChatConfig>,
}

impl Default for AnswerOptions {
    fn default() -> Self {
        AnswerOptions { rerank: true, context_k: CONTEXT_K, chat: None }
    }
}

#[derive(Debug)]
pub enum AnswerEvent {
    /// Milliseconds from query start until the request bytes were written to the socket.
    TimeToRequest { ms: f64 },
    Hit(Hit),
    Token(String),
    Done,
    Error(Error),
}

impl AnswerEvent {
    pub fn name(&self) -> &'static str {
        match self {
            AnswerEvent::TimeToRequest { .. } => "time_to_request",
            AnswerEvent::Hit(_) => "hit",
            AnswerEvent::Token(_) => "token",
            AnswerEvent::Done => "done",
            AnswerEvent::Error(_) => "error",
        }
    }
}

/// Fills `{{chunks}}` with `[doc_key] body` blocks in retrieve order, then `{{question}}`.
pub fn fill_answer_prompt(question: &str, hits: &[Hit]) -> String {
    let chunks = hits.iter().map(|hit| format!("[{}] {}", hit.doc_id, hit.body)).collect::<Vec<_>>().join("\n\n");
    ANSWER_PROMPT.replace("{{chunks}}", &chunks).replace("{{question}}", question)
}

pub fn fill_judge_prompt(question: &str, gold: &str, system: &str) -> String {
    JUDGE_PROMPT.replace("{{question}}", question).replace("{{gold}}", gold).replace("{{system}}", system)
}

fn chat_body(chat: &ChatConfig, prompt: &str, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": chat.model,
        "stream": stream,
        "messages": [{ "role": "user", "content": prompt }],
    })
}

fn headers(chat: &ChatConfig, stream: bool) -> Vec<(&'static str, String)> {
    let mut headers = vec![("Content-Type", "application/json".to_string())];
    if stream {
        headers.push(("Accept", "text/event-stream".to_string()));
    }
    if let Some(key) = &chat.api_key {
        headers.push(("Authorization", format!("Bearer {key}")));
    }
    headers
}

/// One non-streamed chat completion. Returns the message content.
pub fn complete(chat: &ChatConfig, prompt: &str) -> Result<String> {
    let url = Url::parse(&chat.endpoint)?;
    let body = serde_json::to_vec(&chat_body(chat, prompt, false))?;
    let connection = http::send(&url, "POST", &headers(chat, false), &body)?;
    let response = http::read_response(connection)?;
    let status = response.status;
    let bytes = response.read_all()?;
    if !(200..300).contains(&status) {
        return Err(Error::Llm(format!("chat endpoint returned {status}: {}", String::from_utf8_lossy(&bytes))));
    }
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    value["choices"][0]["message"]["content"]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error::Llm("chat response has no message content".into()))
}

/// The events of one `answer` call. Time-to-request and the hits are ready before the model
/// says anything; tokens are read from the SSE body as the caller pulls them.
pub struct AnswerStream {
    pending: VecDeque<AnswerEvent>,
    connection: Option<http::Connection>,
    body: Option<Box<dyn BufRead + Send>>,
    finished: bool,
    _guard: ReaderGuard,
}

impl AnswerStream {
    fn finish(&mut self, event: AnswerEvent) -> Option<AnswerEvent> {
        self.finished = true;
        self.body = None;
        Some(event)
    }

    fn pull(&mut self) -> Option<AnswerEvent> {
        if let Some(connection) = self.connection.take() {
            match http::read_response(connection) {
                Ok(response) if (200..300).contains(&response.status) => self.body = Some(response.body),
                Ok(response) => {
                    let status = response.status;
                    let body = response.read_all().unwrap_or_default();
                    let message = format!("chat endpoint returned {status}: {}", String::from_utf8_lossy(&body));
                    return self.finish(AnswerEvent::Error(Error::Llm(message)));
                }
                Err(err) => return self.finish(AnswerEvent::Error(err)),
            }
        }
        loop {
            let body = self.body.as_mut()?;
            let mut line = String::new();
            match body.read_line(&mut line) {
                Ok(0) => return self.finish(AnswerEvent::Done),
                Ok(_) => {}
                Err(err) => return self.finish(AnswerEvent::Error(Error::Llm(err.to_string()))),
            }
            let Some(data) = line.trim_end().strip_prefix("data:") else { continue };
            let data = data.trim();
            if data == "[DONE]" {
                return self.finish(AnswerEvent::Done);
            }
            let value: serde_json::Value = match serde_json::from_str(data) {
                Ok(value) => value,
                Err(err) => return self.finish(AnswerEvent::Error(Error::Llm(format!("bad SSE data: {err}")))),
            };
            if let Some(token) = value["choices"][0]["delta"]["content"].as_str() {
                if !token.is_empty() {
                    return Some(AnswerEvent::Token(token.to_string()));
                }
            }
        }
    }
}

impl Iterator for AnswerStream {
    type Item = AnswerEvent;

    fn next(&mut self) -> Option<AnswerEvent> {
        if let Some(event) = self.pending.pop_front() {
            return Some(event);
        }
        if self.finished {
            return None;
        }
        self.pull()
    }
}

impl Index {
    /// Sets the chat endpoint used by `answer` and contextual prefixes.
    pub fn set_chat_endpoint(&mut self, endpoint: impl Into<String>) {
        let current = self.chat.take();
        self.chat = Some(ChatConfig {
            endpoint: endpoint.into(),
            model: current.as_ref().map(|c| c.model.clone()).unwrap_or_else(|| "default".into()),
            api_key: current.and_then(|c| c.api_key),
        });
    }

    pub fn set_chat(&mut self, chat: ChatConfig) {
        self.chat = Some(chat);
    }

    /// Retrieves, writes the chat request, and returns once the bytes are on the socket.
    /// The stream yields `TimeToRequest`, the hits, then tokens and `Done` (or `Error`).
    pub fn answer(&self, query: &str, opts: AnswerOptions) -> Result<AnswerStream> {
        let started = Instant::now();
        let guard = self.acquire();
        let chat = opts.chat.clone().or_else(|| self.chat.clone()).ok_or_else(|| Error::Llm("no chat endpoint is configured".into()))?;
        let url = Url::parse(&chat.endpoint)?;
        let hits = self.retrieve(query, RetrieveOptions { rerank: opts.rerank, limit: opts.context_k })?.hits;
        let prompt = fill_answer_prompt(query, &hits);
        let body = serde_json::to_vec(&chat_body(&chat, &prompt, true))?;
        let mut pending = VecDeque::new();
        let connection = match http::send(&url, "POST", &headers(&chat, true), &body) {
            Ok(connection) => {
                pending.push_back(AnswerEvent::TimeToRequest { ms: started.elapsed().as_secs_f64() * 1000.0 });
                Some(connection)
            }
            Err(err) => {
                pending.extend(hits.into_iter().map(AnswerEvent::Hit));
                pending.push_back(AnswerEvent::Error(err));
                return Ok(AnswerStream { pending, connection: None, body: None, finished: true, _guard: guard });
            }
        };
        pending.extend(hits.into_iter().map(AnswerEvent::Hit));
        Ok(AnswerStream { pending, connection, body: None, finished: false, _guard: guard })
    }

    /// Asks the chat model for a chunk's contextual prefix.
    pub(crate) fn contextual_prefix(&self, doc: &Document, body: &str) -> Result<String> {
        let chat = self.chat.as_ref().ok_or_else(|| Error::Llm("no chat endpoint is configured for contextual prefixes".into()))?;
        let prompt = format!(
            "<document>\n{}\n</document>\nHere is a chunk of that document:\n<chunk>\n{}\n</chunk>\nWrite one short sentence that situates this chunk within the document, to improve search retrieval of the chunk. Answer with only that sentence.",
            doc.text, body
        );
        Ok(complete(chat, &prompt)?.trim().to_string())
    }
}
