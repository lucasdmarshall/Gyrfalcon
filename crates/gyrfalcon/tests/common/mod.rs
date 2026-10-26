#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::OnceLock;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// Builds the `gyr` binary (once per test process) and returns its path.
pub fn gyr_binary() -> PathBuf {
    static BINARY: OnceLock<PathBuf> = OnceLock::new();
    BINARY
        .get_or_init(|| {
            let root = workspace_root();
            let status = Command::new(env!("CARGO"))
                .args(["build", "--quiet", "-p", "gyr-cli"])
                .current_dir(&root)
                .status()
                .expect("run cargo build");
            assert!(status.success(), "cargo build -p gyr-cli failed");
            let target = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|| root.join("target"));
            target.join("debug").join(format!("gyr{}", std::env::consts::EXE_SUFFIX))
        })
        .clone()
}

/// Builds a cargo example of this package and returns its path.
pub fn example_binary(name: &str) -> PathBuf {
    let root = workspace_root();
    let status = Command::new(env!("CARGO"))
        .args(["build", "--quiet", "-p", "gyrfalcon", "--example", name])
        .current_dir(&root)
        .status()
        .expect("run cargo build");
    assert!(status.success(), "cargo build --example {name} failed");
    let target = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|| root.join("target"));
    target.join("debug").join("examples").join(format!("{name}{}", std::env::consts::EXE_SUFFIX))
}

pub fn gyr(args: &[&str]) -> std::process::Output {
    Command::new(gyr_binary()).args(args).output().expect("run gyr")
}

pub struct HttpResponse {
    pub status: u16,
    pub body: String,
    pub json: serde_json::Value,
}

pub fn http(addr: &str, method: &str, path: &str, body: &str) -> HttpResponse {
    let mut stream = TcpStream::connect(addr).unwrap();
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut raw = String::new();
    stream.read_to_string(&mut raw).unwrap();
    let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((&raw, ""));
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    HttpResponse { status, body: body.to_string(), json: serde_json::from_str(body).unwrap_or(serde_json::Value::Null) }
}

/// `gyr serve` on an ephemeral port with the hash embedder, over a fresh collection.
pub struct TestServer {
    pub addr: String,
    pub dir: tempfile::TempDir,
    child: Child,
}

impl TestServer {
    pub fn spawn() -> TestServer {
        let dir = tempfile::tempdir().unwrap();
        let collection = dir.path().join("collection");
        let mut child = Command::new(gyr_binary())
            .args(["serve", collection.to_str().unwrap(), "--bind", "127.0.0.1:0", "--embedder", "hash"])
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
        TestServer { addr: line.trim().to_string(), dir, child }
    }

    /// The server created the default (process config) collection at start; confirm it is up.
    pub fn create_default_index(&self) {
        assert_eq!(self.get("/health").status, 200);
    }

    pub fn get(&self, path: &str) -> HttpResponse {
        http(&self.addr, "GET", path, "")
    }

    pub fn post(&self, path: &str, body: &str) -> HttpResponse {
        http(&self.addr, "POST", path, body)
    }

    pub fn put(&self, path: &str, body: &str) -> HttpResponse {
        http(&self.addr, "PUT", path, body)
    }

    pub fn delete(&self, path: &str) -> HttpResponse {
        http(&self.addr, "DELETE", path, "")
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Reads one HTTP request: headers until the blank line, then `Content-Length` bytes.
pub fn read_http_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut head = String::new();
    let mut length = 0;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap();
            }
        }
        head.push_str(&line);
        if line == "\r\n" || line.is_empty() {
            break;
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    (head, body)
}

/// A one-shot chat server. It records the request body, then writes `response` verbatim.
pub fn chat_server(response: String) -> (String, mpsc::Receiver<Vec<u8>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { return };
            let (_, body) = read_http_request(&mut stream);
            let _ = sender.send(body);
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (format!("http://{addr}/v1/chat/completions"), receiver)
}

pub fn sse_response(tokens: &[&str]) -> String {
    let mut body = String::new();
    for token in tokens {
        body.push_str(&format!("data: {}\n\n", serde_json::json!({ "choices": [{ "delta": { "content": token } }] })));
    }
    body.push_str("data: [DONE]\n\n");
    format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{body}", body.len())
}
