//! A small blocking HTTP/1.1 client for OpenAI-compatible endpoints: plain TCP for `http`,
//! rustls for `https`, over pooled keep-alive connections. `send` returns once the request bytes are written and flushed, which
//! is where `answer` stops its time-to-request clock.

use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Url {
    pub https: bool,
    pub host: String,
    pub port: u16,
    pub path: String,
}

impl Url {
    pub fn parse(url: &str) -> Result<Url> {
        let bad = || Error::Llm(format!("unsupported url {url}"));
        let (https, rest) = if let Some(rest) = url.strip_prefix("https://") {
            (true, rest)
        } else if let Some(rest) = url.strip_prefix("http://") {
            (false, rest)
        } else {
            return Err(bad());
        };
        let (authority, path) = match rest.find('/') {
            Some(at) => (&rest[..at], &rest[at..]),
            None => (rest, "/"),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((host, port)) => (host, port.parse().map_err(|_| bad())?),
            None => (authority, if https { 443 } else { 80 }),
        };
        if host.is_empty() {
            return Err(bad());
        }
        Ok(Url { https, host: host.to_string(), port, path: path.to_string() })
    }
}

pub enum Connection {
    Plain(TcpStream),
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>),
}

impl Read for Connection {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Connection::Plain(stream) => stream.read(buf),
            Connection::Tls(stream) => stream.read(buf),
        }
    }
}

impl Write for Connection {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Connection::Plain(stream) => stream.write(buf),
            Connection::Tls(stream) => stream.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Connection::Plain(stream) => stream.flush(),
            Connection::Tls(stream) => stream.flush(),
        }
    }
}

fn tls_config() -> Arc<rustls::ClientConfig> {
    static CONFIG: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let roots = rustls::RootCertStore { roots: webpki_roots::TLS_SERVER_ROOTS.to_vec() };
            let provider = Arc::new(rustls::crypto::ring::default_provider());
            Arc::new(
                rustls::ClientConfig::builder_with_provider(provider)
                    .with_safe_default_protocol_versions()
                    .expect("ring supports the default protocol versions")
                    .with_root_certificates(roots)
                    .with_no_client_auth(),
            )
        })
        .clone()
}

fn io_llm(err: impl std::fmt::Display) -> Error {
    Error::Llm(err.to_string())
}

type PoolKey = (bool, String, u16);

/// Idle keep-alive connections per `(https, host, port)`.
#[derive(Default)]
struct Pool {
    idle: Mutex<HashMap<PoolKey, Vec<(Instant, Connection)>>>,
}

const MAX_IDLE_PER_HOST: usize = 8;
const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(Pool::default)
}

impl Pool {
    fn take(&self, key: &PoolKey) -> Option<Connection> {
        let mut idle = self.idle.lock().unwrap_or_else(|p| p.into_inner());
        let list = idle.get_mut(key)?;
        while let Some((since, connection)) = list.pop() {
            if since.elapsed() < IDLE_TIMEOUT {
                return Some(connection);
            }
        }
        None
    }

    fn put(&self, key: PoolKey, connection: Connection) {
        let mut idle = self.idle.lock().unwrap_or_else(|p| p.into_inner());
        let list = idle.entry(key).or_default();
        list.retain(|(since, _)| since.elapsed() < IDLE_TIMEOUT);
        if list.len() < MAX_IDLE_PER_HOST {
            list.push((Instant::now(), connection));
        }
    }
}

fn connect(url: &Url) -> Result<Connection> {
    let stream = TcpStream::connect((url.host.as_str(), url.port)).map_err(io_llm)?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_secs(300))).ok();
    Ok(if url.https {
        let name = rustls::pki_types::ServerName::try_from(url.host.clone()).map_err(io_llm)?;
        let client = rustls::ClientConnection::new(tls_config(), name).map_err(io_llm)?;
        Connection::Tls(Box::new(rustls::StreamOwned::new(client, stream)))
    } else {
        Connection::Plain(stream)
    })
}

fn write_request(connection: &mut Connection, request: &[u8]) -> io::Result<()> {
    connection.write_all(request)?;
    connection.flush()
}

/// A request whose bytes are on the socket. The response has not been read.
pub struct Sent {
    key: PoolKey,
    url: Url,
    connection: Connection,
    /// The connection came from the pool, so the server may already have closed it.
    reused: bool,
    request: Vec<u8>,
}

/// Writes the full request on a pooled keep-alive connection, or a new one, and flushes.
pub fn send(url: &Url, method: &str, headers: &[(&str, String)], body: &[u8]) -> Result<Sent> {
    let mut request =
        format!("{method} {} HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\n", url.path, url.host, body.len());
    for (name, value) in headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    let mut request = request.into_bytes();
    request.extend_from_slice(body);
    let key = (url.https, url.host.clone(), url.port);
    while let Some(mut connection) = pool().take(&key) {
        if write_request(&mut connection, &request).is_ok() {
            return Ok(Sent { key, url: url.clone(), connection, reused: true, request });
        }
    }
    let mut connection = connect(url)?;
    write_request(&mut connection, &request).map_err(io_llm)?;
    Ok(Sent { key, url: url.clone(), connection, reused: false, request })
}

/// A response whose body is read through `body`, de-chunked when needed. Reading the body to
/// its end returns a keep-alive connection to the pool; dropping it early closes it.
pub struct Response {
    pub status: u16,
    pub body: Box<dyn BufRead + Send>,
}

impl Response {
    pub fn read_all(mut self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        self.body.read_to_end(&mut out).map_err(io_llm)?;
        Ok(out)
    }
}

fn read_head(reader: &mut BufReader<Connection>) -> io::Result<(u16, Vec<(String, String)>)> {
    let mut line = String::new();
    if reader.read_line(&mut line)? == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "connection closed before the status line"));
    }
    let status = line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("bad status line {:?}", line.trim())))?;
    let mut headers = Vec::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    Ok((status, headers))
}

/// Reads the status line and headers. A pooled connection the server already closed is
/// replaced by a new one and the request is sent again.
pub fn read_response(sent: Sent) -> Result<Response> {
    let Sent { key, url, connection, reused, request } = sent;
    let mut reader = BufReader::new(connection);
    let (status, headers) = match read_head(&mut reader) {
        Ok(head) => head,
        Err(_) if reused => {
            let mut connection = connect(&url)?;
            write_request(&mut connection, &request).map_err(io_llm)?;
            reader = BufReader::new(connection);
            read_head(&mut reader).map_err(io_llm)?
        }
        Err(err) => return Err(io_llm(err)),
    };
    let header = |name: &str| headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str());
    let close = header("connection").is_some_and(|v| v.eq_ignore_ascii_case("close"));
    let framing = if header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
        Framing::Chunked { remaining: 0, done: false }
    } else if let Some(length) = header("content-length").and_then(|v| v.parse::<u64>().ok()) {
        Framing::Length(length)
    } else {
        Framing::UntilClose
    };
    let keep_alive = !close && !matches!(framing, Framing::UntilClose);
    let body = Body { reader: Some(reader), framing, key, keep_alive };
    Ok(Response { status, body: Box::new(BufReader::new(body)) })
}

enum Framing {
    Length(u64),
    Chunked { remaining: u64, done: bool },
    UntilClose,
}

struct Body {
    reader: Option<BufReader<Connection>>,
    framing: Framing,
    key: PoolKey,
    keep_alive: bool,
}

impl Body {
    /// The body is complete: hand a clean keep-alive connection back to the pool.
    fn finish(&mut self) {
        if let Some(reader) = self.reader.take() {
            if self.keep_alive && reader.buffer().is_empty() {
                pool().put(self.key.clone(), reader.into_inner());
            }
        }
    }
}

impl Read for Body {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let Some(reader) = self.reader.as_mut() else { return Ok(0) };
        let read = match &mut self.framing {
            Framing::UntilClose => reader.read(buf)?,
            Framing::Length(remaining) => {
                if *remaining == 0 {
                    0
                } else {
                    let want = buf.len().min(*remaining as usize);
                    let read = reader.read(&mut buf[..want])?;
                    if read == 0 {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "body shorter than Content-Length"));
                    }
                    *remaining -= read as u64;
                    read
                }
            }
            Framing::Chunked { remaining, done } => read_chunked(reader, remaining, done, buf)?,
        };
        let complete = match &self.framing {
            Framing::Length(remaining) => *remaining == 0,
            Framing::Chunked { done, .. } => *done,
            Framing::UntilClose => read == 0,
        };
        if complete {
            self.finish();
        }
        Ok(read)
    }
}

fn read_chunked(reader: &mut impl BufRead, remaining: &mut u64, done: &mut bool, buf: &mut [u8]) -> io::Result<usize> {
    if *done {
        return Ok(0);
    }
    if *remaining == 0 {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if line.trim().is_empty() {
            // The CRLF that ends the previous chunk.
            line.clear();
            reader.read_line(&mut line)?;
        }
        let size = u64::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad chunk size"))?;
        if size == 0 {
            // Trailer section, ending with an empty line.
            loop {
                line.clear();
                if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
                    break;
                }
            }
            *done = true;
            return Ok(0);
        }
        *remaining = size;
    }
    let want = buf.len().min(*remaining as usize);
    let read = reader.read(&mut buf[..want])?;
    if read == 0 {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "chunked body ended early"));
    }
    *remaining -= read as u64;
    Ok(read)
}

/// POSTs JSON and returns the status and full body.
pub fn post_json(url: &str, bearer: Option<&str>, body: &serde_json::Value) -> Result<(u16, Vec<u8>)> {
    let url = Url::parse(url)?;
    let mut headers = vec![("Content-Type", "application/json".to_string())];
    if let Some(token) = bearer {
        headers.push(("Authorization", format!("Bearer {token}")));
    }
    let sent = send(&url, "POST", &headers, &serde_json::to_vec(body)?)?;
    let response = read_response(sent)?;
    let status = response.status;
    Ok((status, response.read_all()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_urls() {
        let url = Url::parse("http://127.0.0.1:8080/v1/chat/completions").unwrap();
        assert_eq!(
            (url.https, url.host.as_str(), url.port, url.path.as_str()),
            (false, "127.0.0.1", 8080, "/v1/chat/completions")
        );
        let url = Url::parse("https://api.example.com").unwrap();
        assert_eq!((url.https, url.port, url.path.as_str()), (true, 443, "/"));
        assert!(Url::parse("ftp://x").is_err());
    }

    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Serves `responses_per_connection` requests per accepted connection, answering each with
    /// `response`, and counts accepts.
    fn server(response: &'static str, responses_per_connection: usize) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1/x", listener.local_addr().unwrap());
        let accepts = Arc::new(AtomicUsize::new(0));
        let counter = accepts.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let stream = stream.unwrap();
                counter.fetch_add(1, Ordering::SeqCst);
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut writer = stream;
                    for _ in 0..responses_per_connection {
                        let mut length = 0;
                        loop {
                            let mut line = String::new();
                            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                                return;
                            }
                            if let Some((name, value)) = line.split_once(':') {
                                if name.eq_ignore_ascii_case("content-length") {
                                    length = value.trim().parse().unwrap();
                                }
                            }
                            if line == "\r\n" {
                                break;
                            }
                        }
                        let mut body = vec![0; length];
                        reader.read_exact(&mut body).unwrap();
                        writer.write_all(response.as_bytes()).unwrap();
                    }
                });
            }
        });
        (url, accepts)
    }

    #[test]
    fn keep_alive_reuses_one_connection() {
        let (url, accepts) = server("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", 100);
        for _ in 0..3 {
            assert_eq!(post_json(&url, None, &serde_json::json!({})).unwrap(), (200, b"ok".to_vec()));
        }
        assert_eq!(accepts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn chunked_bodies_also_return_the_connection() {
        let (url, accepts) = server("HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\nok\r\n0\r\n\r\n", 100);
        for _ in 0..3 {
            assert_eq!(post_json(&url, None, &serde_json::json!({})).unwrap(), (200, b"ok".to_vec()));
        }
        assert_eq!(accepts.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_closed_pooled_connection_is_replaced() {
        let (url, accepts) = server("HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok", 1);
        for _ in 0..3 {
            assert_eq!(post_json(&url, None, &serde_json::json!({})).unwrap(), (200, b"ok".to_vec()));
        }
        assert_eq!(accepts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn connection_close_is_not_pooled() {
        let (url, accepts) = server("HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: 2\r\n\r\nok", 100);
        for _ in 0..2 {
            assert_eq!(post_json(&url, None, &serde_json::json!({})).unwrap(), (200, b"ok".to_vec()));
        }
        assert_eq!(accepts.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn decodes_chunked_bodies() {
        let mut raw = &b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n"[..];
        let (mut remaining, mut done) = (0, false);
        let mut out = Vec::new();
        let mut buf = [0u8; 3];
        loop {
            let read = read_chunked(&mut raw, &mut remaining, &mut done, &mut buf).unwrap();
            if read == 0 {
                break;
            }
            out.extend_from_slice(&buf[..read]);
        }
        assert_eq!(out, b"Wikipedia");
        assert!(done && raw.is_empty());
    }
}
