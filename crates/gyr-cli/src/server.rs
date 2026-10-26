//! The HTTP layer: `std::net::TcpListener`, one thread per connection. Each handler decodes
//! JSON, calls one library function, and writes the result.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use gyrfalcon::{
    AnswerEvent, AnswerOptions, Document, Error, Filter, Hit, Index, IndexConfig, Quantization, RetrieveOptions,
};
use serde_json::{json, Value};

struct App {
    index: RwLock<Index>,
    config: IndexConfig,
    dir: PathBuf,
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

pub fn serve(index: Index, bind: &str, config: IndexConfig, dir: &Path) -> Result<(), Error> {
    let listener = TcpListener::bind(bind)?;
    // The first stdout line is the bound address, so `--bind 127.0.0.1:0` is usable.
    println!("{}", listener.local_addr()?);
    std::io::stdout().flush()?;
    let app = Arc::new(App { index: RwLock::new(index), config, dir: dir.to_path_buf() });
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let app = app.clone();
        std::thread::spawn(move || {
            let _ = handle(&app, stream);
        });
    }
    Ok(())
}

fn read_request(stream: &TcpStream) -> std::io::Result<Option<Request>> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let mut length = 0usize;
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 || line.trim_end().is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body)?;
    Ok(Some(Request { method, path, body }))
}

fn respond(stream: &mut TcpStream, status: u16, body: &Value) -> std::io::Result<()> {
    let text = body.to_string();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        502 => "Bad Gateway",
        _ => "Internal Server Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
        text.len()
    )?;
    stream.flush()
}

pub fn error_json(err: &Error) -> Value {
    match err {
        Error::IncompatibleIndex { field, manifest_value, requested_value } => json!({
            "error": "incompatible_index",
            "field": field.as_str(),
            "manifest": manifest_value,
            "requested": requested_value,
        }),
        other => json!({ "error": other.code(), "message": other.to_string() }),
    }
}

fn hit_json(hit: &Hit, include_prefix: bool) -> Value {
    let mut value = json!({
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "body": hit.body,
        "score": hit.score,
        "byte_start": hit.byte_start,
        "byte_end": hit.byte_end,
        "metadata": metadata_json(&hit.metadata),
    });
    if include_prefix {
        value["prefix"] = json!(hit.prefix);
    }
    value
}

/// Metadata as a JSON object: each key's value, or an array when the key repeats.
fn metadata_json(pairs: &[(String, String)]) -> Value {
    let mut object = serde_json::Map::new();
    for (key, value) in pairs {
        match object.get_mut(key) {
            None => {
                object.insert(key.clone(), json!(value));
            }
            Some(Value::Array(values)) => values.push(json!(value)),
            Some(existing) => *existing = json!([existing.clone(), value]),
        }
    }
    Value::Object(object)
}

/// `{"key": value or [values]}` as `(key, values)` clauses; values are strings, numbers, or
/// booleans.
type Clauses = Vec<(String, Vec<String>)>;

fn clauses(value: Option<&Value>, what: &str) -> Result<Clauses, (u16, Value)> {
    let Some(value) = value.filter(|v| !v.is_null()) else { return Ok(Vec::new()) };
    let object = value.as_object().ok_or_else(|| bad_request(&format!("{what} must be an object")))?;
    let text = |v: &Value| match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        _ => Err(bad_request(&format!("{what} values are strings, numbers, or booleans"))),
    };
    object
        .iter()
        .map(|(key, value)| {
            let values = match value {
                Value::Array(values) => values.iter().map(text).collect::<Result<Vec<_>, _>>()?,
                single => vec![text(single)?],
            };
            Ok((key.clone(), values))
        })
        .collect()
}

fn bad_request(message: &str) -> (u16, Value) {
    (400, json!({ "error": "bad_request", "message": message }))
}

/// The process config with any identity fields the request names. The open gate compares the
/// manifest against it before the handler touches the index.
fn expected_config(app: &App, body: &Value) -> Result<IndexConfig, (u16, Value)> {
    let mut expected = app.config.clone();
    if let Some(value) = body.get("quantization").and_then(Value::as_str) {
        expected.quantization = Quantization::parse(value).ok_or_else(|| bad_request("unknown quantization"))?;
    }
    if let Some(value) = body.get("model_id").and_then(Value::as_str) {
        expected.model_id = value.to_string();
    }
    if let Some(value) = body.get("dimension").and_then(Value::as_u64) {
        expected.dimension = value as u32;
    }
    if let Some(value) = body.get("analyzer_id").and_then(Value::as_str) {
        expected.analyzer_id = value.to_string();
    }
    Ok(expected)
}

fn gate(app: &App, body: &Value) -> Result<(), (u16, Value)> {
    let expected = expected_config(app, body)?;
    let mut opens = 0;
    Index::open_counting(&app.dir, &expected, &mut opens)
        .map(|_| ())
        .map_err(|err| (err.http_status(), error_json(&err)))
}

fn library<T>(result: Result<T, Error>) -> Result<T, (u16, Value)> {
    result.map_err(|err| (err.http_status(), error_json(&err)))
}

fn handle(app: &App, mut stream: TcpStream) -> std::io::Result<()> {
    let Some(request) = read_request(&stream)? else { return Ok(()) };
    let body: Value = if request.body.is_empty() {
        Value::Null
    } else {
        match serde_json::from_slice(&request.body) {
            Ok(body) => body,
            Err(err) => {
                let (status, value) = bad_request(&err.to_string());
                return respond(&mut stream, status, &value);
            }
        }
    };
    if request.method == "POST" && request.path == "/v1/answer" {
        return answer(app, &mut stream, &body);
    }
    let result = route(app, &request, &body);
    let (status, value) = match result {
        Ok(value) => (200, value),
        Err(failure) => failure,
    };
    respond(&mut stream, status, &value)
}

fn route(app: &App, request: &Request, body: &Value) -> Result<Value, (u16, Value)> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => Ok(json!({ "status": "ok" })),
        ("POST", "/v1/documents") => {
            gate(app, body)?;
            let documents = body
                .get("documents")
                .and_then(Value::as_array)
                .ok_or_else(|| bad_request("documents must be an array"))?;
            let mut docs = Vec::new();
            for doc in documents {
                let doc_id =
                    doc.get("doc_id").and_then(Value::as_str).ok_or_else(|| bad_request("doc_id is required"))?;
                let text = doc.get("text").and_then(Value::as_str).ok_or_else(|| bad_request("text is required"))?;
                let metadata = clauses(doc.get("metadata"), "metadata")?
                    .into_iter()
                    .flat_map(|(key, values)| values.into_iter().map(move |value| (key.clone(), value)));
                let mut document = Document::body(doc_id, text).with_metadata(metadata);
                document.contextual_prefix = doc.get("contextual_prefix").and_then(Value::as_bool).unwrap_or(false);
                docs.push(document);
            }
            let mut index = app.index.write().unwrap_or_else(|p| p.into_inner());
            let report = library(index.index_documents(&docs))?;
            Ok(
                json!({ "chunk_ids": report.chunk_ids, "sealed": report.sealed, "epoch": report.epoch, "warning": report.warning }),
            )
        }
        ("DELETE", path) if path.starts_with("/v1/documents/") => {
            let doc_id = percent_decode(&path["/v1/documents/".len()..]);
            let mut index = app.index.write().unwrap_or_else(|p| p.into_inner());
            let report = library(index.delete_document(&doc_id))?;
            Ok(json!({ "deleted_chunks": report.deleted_chunks, "epoch": report.epoch }))
        }
        ("POST", "/v1/retrieve") => {
            gate(app, body)?;
            let query = body.get("query").and_then(Value::as_str).ok_or_else(|| bad_request("query is required"))?;
            let opts = RetrieveOptions {
                rerank: body.get("rerank").and_then(Value::as_bool).unwrap_or(true),
                limit: body.get("limit").and_then(Value::as_u64).unwrap_or(10) as usize,
            };
            let include_prefix = body.get("include_prefix").and_then(Value::as_bool).unwrap_or(false);
            let filter = clauses(body.get("filter"), "filter")?
                .into_iter()
                .fold(Filter::new(), |f, (key, values)| f.any(key, values));
            let index = app.index.read().unwrap_or_else(|p| p.into_inner());
            let hits = library(index.retrieve_filtered(query, opts, Some(&filter)))?;
            Ok(json!({
                "chunk_ids": hits.chunk_ids,
                "hits": hits.hits.iter().map(|hit| hit_json(hit, include_prefix)).collect::<Vec<_>>(),
                "epoch": hits.epoch,
                "fallback": hits.fallback.label(),
                "warning": hits.warning,
            }))
        }
        _ => Err((404, json!({ "error": "not_found" }))),
    }
}

fn answer(app: &App, stream: &mut TcpStream, body: &Value) -> std::io::Result<()> {
    let started = (|| -> Result<_, (u16, Value)> {
        gate(app, body)?;
        let query = body.get("query").and_then(Value::as_str).ok_or_else(|| bad_request("query is required"))?;
        let opts = AnswerOptions {
            rerank: body.get("rerank").and_then(Value::as_bool).unwrap_or(true),
            ..AnswerOptions::default()
        };
        let index = app.index.read().unwrap_or_else(|p| p.into_inner());
        library(index.answer(query, opts))
    })();
    let events = match started {
        Ok(events) => events,
        Err((status, value)) => return respond(stream, status, &value),
    };
    let include_prefix = body.get("include_prefix").and_then(Value::as_bool).unwrap_or(false);
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;
    for event in events {
        let data = match &event {
            AnswerEvent::TimeToRequest { ms } => json!({ "ms": ms }),
            AnswerEvent::Hit(hit) => hit_json(hit, include_prefix),
            AnswerEvent::Token(text) => json!({ "text": text }),
            AnswerEvent::Done => json!({}),
            AnswerEvent::Error(err) => error_json(err),
        };
        write!(stream, "event: {}\ndata: {}\n\n", event.name(), data)?;
        stream.flush()?;
    }
    Ok(())
}

fn percent_decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(byte) = u8::from_str_radix(&value[i + 1..i + 3], 16) {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
