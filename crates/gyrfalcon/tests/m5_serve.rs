mod common;

use std::net::TcpListener;

use common::{chat_server, gyr, read_http_request, sse_response, TestServer};
use gyrfalcon::testing::RecordingEmbedder;
use gyrfalcon::{AnswerEvent, AnswerOptions, Document, Error, Index, IndexConfig, RetrieveOptions};

fn one_doc() -> Index {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("a", "rust safety")]).unwrap();
    index
}

/// The fenced block that follows `marker` in the spec.
fn spec_block(marker: &str) -> String {
    let spec = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/superpowers/specs/2026-10-24-gyrfalcon-design.md"
    ))
    .unwrap();
    let after = &spec[spec.find(marker).unwrap()..];
    let start = after.find("```\n").unwrap() + 4;
    let end = after[start..].find("\n```").unwrap();
    after[start..start + end].to_string()
}

#[test]
fn prompts_match_the_spec_byte_for_byte() {
    assert_eq!(gyrfalcon::ANSWER_PROMPT, spec_block("The answer prompt is this text"));
    assert_eq!(gyrfalcon::JUDGE_PROMPT, spec_block("The judge prompt is this text"));
}

#[test]
fn time_to_request_stops_when_bytes_hit_the_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let (_, body) = read_http_request(&mut stream);
        sender.send(body).unwrap();
        // Never answers: the model is discarded.
        std::thread::sleep(std::time::Duration::from_secs(5));
    });
    let mut index = one_doc();
    index.set_chat_endpoint(format!("http://{addr}/v1/chat/completions"));
    let mut stream = index.answer("rust", AnswerOptions::default()).unwrap();
    let first = stream.next().unwrap();
    assert!(matches!(first, AnswerEvent::TimeToRequest { ms } if ms >= 0.0));
    let hit = stream.next().unwrap();
    assert!(matches!(&hit, AnswerEvent::Hit(hit) if hit.chunk_id == 0));

    // The discard server received the prompt bytes.
    let body: serde_json::Value = serde_json::from_slice(&receiver.recv().unwrap()).unwrap();
    assert_eq!(body["stream"], true);
    let prompt = body["messages"][0]["content"].as_str().unwrap();
    assert!(prompt.contains("Context:\n[a] rust safety\n\nQuestion: rust"), "{prompt}");
}

#[test]
fn answer_streams_tokens_then_done() {
    let (endpoint, _) = chat_server(sse_response(&["Rust ", "is ", "safe."]));
    let mut index = one_doc();
    index.set_chat_endpoint(endpoint);
    let events: Vec<AnswerEvent> = index.answer("rust", AnswerOptions::default()).unwrap().collect();
    let names: Vec<&str> = events.iter().map(AnswerEvent::name).collect();
    assert_eq!(names, vec!["time_to_request", "hit", "token", "token", "token", "done"]);
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            AnswerEvent::Token(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Rust is safe.");
}

#[test]
fn llm_failure_still_emits_the_hits_then_error() {
    let (endpoint, _) = chat_server("HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\n\r\nboom".to_string());
    let mut index = one_doc();
    index.set_chat_endpoint(endpoint);
    let events: Vec<AnswerEvent> = index.answer("rust", AnswerOptions::default()).unwrap().collect();
    assert_eq!(events.iter().map(AnswerEvent::name).collect::<Vec<_>>(), vec!["time_to_request", "hit", "error"]);
    assert!(matches!(events.last().unwrap(), AnswerEvent::Error(Error::Llm(message)) if message.contains("500")));
    assert_eq!(Error::Llm(String::new()).http_status(), 502);
}

#[test]
fn retrieve_does_not_call_the_llm() {
    let (endpoint, requests) = chat_server(sse_response(&["x"]));
    let mut index = one_doc();
    index.set_chat_endpoint(endpoint);
    index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert!(requests.try_recv().is_err());
}

#[test]
fn prefix_off_embeds_the_body_only() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.set_embedder(Box::new(RecordingEmbedder::new(384)));
    index
        .index_documents(&[Document {
            doc_id: "a".into(),
            text: "Body text".into(),
            contextual_prefix: false,
            metadata: vec![],
        }])
        .unwrap();
    assert_eq!(index.embedder_texts(), vec!["Body text".to_string()]);
}

#[test]
fn prefix_on_embeds_prefix_then_body_and_cites_the_body() {
    let completion = serde_json::json!({ "choices": [{ "message": { "content": " About gardening. " } }] }).to_string();
    let (endpoint, requests) = chat_server(format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{completion}",
        completion.len()
    ));
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.set_chat_endpoint(endpoint);
    index.set_embedder(Box::new(RecordingEmbedder::new(384)));
    index
        .index_documents(&[Document {
            doc_id: "a".into(),
            text: "Body text".into(),
            contextual_prefix: true,
            metadata: vec![],
        }])
        .unwrap();
    assert_eq!(index.embedder_texts(), vec!["About gardening.\n\nBody text".to_string()]);
    let request: serde_json::Value = serde_json::from_slice(&requests.recv().unwrap()).unwrap();
    assert_eq!(request["stream"], false);
    let hits = index.retrieve("gardening", RetrieveOptions { rerank: false, limit: 1 }).unwrap();
    assert_eq!(hits.hits[0].body, "Body text");
    assert_eq!(hits.hits[0].prefix.as_deref(), Some("About gardening."));
}

#[test]
fn mismatch_is_http_409() {
    let server = TestServer::spawn();
    server.create_default_index();
    let response = server.post("/v1/retrieve", r#"{"query":"rust","quantization":"f32"}"#);
    assert_eq!(response.status, 409);
    assert_eq!(response.json["error"], "incompatible_index");
    assert_eq!(response.json["field"], "quantization");
    assert_eq!(response.json["manifest"], "i8");
    assert_eq!(response.json["requested"], "f32");
}

#[test]
fn http_routes_call_the_library() {
    let server = TestServer::spawn();
    let indexed = server
        .post("/v1/documents", r#"{"documents":[{"doc_id":"a","text":"rust safety"},{"doc_id":"b","text":"bread"}]}"#);
    assert_eq!(indexed.status, 200, "{}", indexed.body);
    assert_eq!(indexed.json["chunk_ids"], serde_json::json!([0, 1]));
    let retrieved = server.post("/v1/retrieve", r#"{"query":"bread","rerank":false,"limit":1}"#);
    assert_eq!(retrieved.status, 200);
    assert_eq!(retrieved.json["hits"][0]["doc_id"], "b");
    let deleted = server.delete("/v1/documents/b");
    assert_eq!(deleted.json["deleted_chunks"], 1);
    let empty = server.post("/v1/retrieve", r#"{"query":"   "}"#);
    assert_eq!(empty.status, 400);
    assert_eq!(empty.json["error"], "empty_query");
    assert_eq!(server.get("/nope").status, 404);
}

#[test]
fn http_put_replaces_a_document_in_one_write() {
    let server = TestServer::spawn();
    let body = r#"{"documents":[{"doc_id":"guide","text":"install with pip","metadata":{"v":"1"}}]}"#;
    assert_eq!(server.post("/v1/documents", body).status, 200);
    let replaced = server.put("/v1/documents/guide", r#"{"text":"install with cargo","metadata":{"v":"2"}}"#);
    assert_eq!(replaced.status, 200, "{}", replaced.body);
    assert_eq!(replaced.json["chunk_ids"], serde_json::json!([1]));
    let hits = server.post("/v1/retrieve", r#"{"query":"install","limit":10}"#);
    assert_eq!(hits.json["hits"].as_array().unwrap().len(), 1);
    assert_eq!(hits.json["hits"][0]["body"], "install with cargo");
    assert_eq!(hits.json["hits"][0]["metadata"], serde_json::json!({"v":"2"}));
    assert_eq!(server.put("/v1/documents/guide", r#"{"metadata":{}}"#).status, 400);
    let inserted = server.put("/v1/documents/new%20doc", r#"{"text":"fresh"}"#);
    assert_eq!(inserted.status, 200);
    let found = server.post("/v1/retrieve", r#"{"query":"fresh","limit":1}"#);
    assert_eq!(found.json["hits"][0]["doc_id"], "new doc");
}

#[test]
fn http_metadata_and_filters() {
    let server = TestServer::spawn();
    let body = r#"{"documents":[
        {"doc_id":"a","text":"falcon notes","metadata":{"lang":"en","tag":["birds","arctic"],"year":2024}},
        {"doc_id":"b","text":"falcon notes","metadata":{"lang":"fr"}},
        {"doc_id":"c","text":"falcon notes"}]}"#;
    assert_eq!(server.post("/v1/documents", body).status, 200);
    let found = |filter: &str| -> Vec<String> {
        let response = server
            .post("/v1/retrieve", &format!(r#"{{"query":"falcon","rerank":false,"limit":10,"filter":{filter}}}"#));
        assert_eq!(response.status, 200, "{}", response.body);
        let mut ids: Vec<String> = response.json["hits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h["doc_id"].as_str().unwrap().to_string())
            .collect();
        ids.sort();
        ids
    };
    assert_eq!(found("null"), ["a", "b", "c"]);
    assert_eq!(found(r#"{"lang":"en"}"#), ["a"]);
    assert_eq!(found(r#"{"lang":["en","fr"]}"#), ["a", "b"]);
    assert_eq!(found(r#"{"year":2024,"tag":"arctic"}"#), ["a"]);
    let hit = server.post("/v1/retrieve", r#"{"query":"falcon","rerank":false,"limit":1,"filter":{"lang":"en"}}"#);
    assert_eq!(
        hit.json["hits"][0]["metadata"],
        serde_json::json!({"lang":"en","tag":["birds","arctic"],"year":"2024"})
    );
    let bad = server.post("/v1/retrieve", r#"{"query":"falcon","filter":["lang"]}"#);
    assert_eq!(bad.status, 400);
}

#[test]
fn http_answer_streams_sse() {
    let (endpoint, _) = chat_server(sse_response(&["ok"]));
    let dir = tempfile::tempdir().unwrap();
    let collection = dir.path().join("c");
    let mut child = std::process::Command::new(common::gyr_binary())
        .args([
            "serve",
            collection.to_str().unwrap(),
            "--bind",
            "127.0.0.1:0",
            "--embedder",
            "hash",
            "--chat-endpoint",
            &endpoint,
        ])
        .stdout(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut addr = String::new();
    std::io::BufRead::read_line(&mut std::io::BufReader::new(child.stdout.take().unwrap()), &mut addr).unwrap();
    let addr = addr.trim();
    common::http(addr, "POST", "/v1/documents", r#"{"documents":[{"doc_id":"a","text":"rust safety"}]}"#);
    let response = common::http(addr, "POST", "/v1/answer", r#"{"query":"rust","rerank":false}"#);
    let _ = child.kill();
    let _ = child.wait();
    assert_eq!(response.status, 200);
    let events: Vec<&str> = response.body.lines().filter_map(|line| line.strip_prefix("event: ")).collect();
    assert_eq!(events, vec!["time_to_request", "hit", "token", "done"]);
}

#[test]
fn gyr_index_then_query_prints_the_chunk_id() {
    let dir = tempfile::tempdir().unwrap();
    let collection = dir.path().join("collection");
    let fixture = dir.path().join("rust.txt");
    std::fs::write(&fixture, "Memory safety in Rust comes from the borrow checker.").unwrap();
    let other = dir.path().join("bread.txt");
    std::fs::write(&other, "Bread needs flour, water, salt, and time.").unwrap();
    let indexed = gyr(&[
        "index",
        collection.to_str().unwrap(),
        fixture.to_str().unwrap(),
        other.to_str().unwrap(),
        "--embedder",
        "hash",
    ]);
    assert!(indexed.status.success(), "{}", String::from_utf8_lossy(&indexed.stderr));
    let report: serde_json::Value = serde_json::from_slice(&indexed.stdout).unwrap();
    assert_eq!(report["chunks"], 2);
    let queried = gyr(&["query", collection.to_str().unwrap(), "borrow checker", "--embedder", "hash", "--limit", "1"]);
    assert!(queried.status.success(), "{}", String::from_utf8_lossy(&queried.stderr));
    let stdout = String::from_utf8(queried.stdout).unwrap();
    let first: Vec<&str> = stdout.lines().next().unwrap().split('\t').collect();
    assert_eq!(first[0], "0");
    assert_eq!(first[1], fixture.to_str().unwrap());

    // The process config is the open gate: a different quantization is refused before use.
    let refused = gyr(&["query", collection.to_str().unwrap(), "x", "--embedder", "hash", "--quantization", "f32"]);
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("quantization"));
}

#[test]
fn gyr_compact_all_clears_tombstones() {
    let dir = tempfile::tempdir().unwrap();
    let collection = dir.path().join("c");
    let mut index =
        Index::create(&collection, &IndexConfig { model_id: "gyr-hash".into(), ..IndexConfig::test_default() })
            .unwrap();
    index.auto_merge = false;
    index.index_documents(&[Document::body("a", "one")]).unwrap();
    index.index_documents(&[Document::body("b", "two")]).unwrap();
    index.delete_document("a").unwrap();
    drop(index);
    let out = gyr(&["compact", collection.to_str().unwrap(), "--all", "--embedder", "hash"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["segments"], serde_json::json!([1]));
}
