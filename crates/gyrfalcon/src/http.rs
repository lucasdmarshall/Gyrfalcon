//! A small blocking HTTP/1.1 client for OpenAI-compatible endpoints: plain TCP for `http`,
//! rustls for `https`. `send` returns once the request bytes are written and flushed, which
//! is where `answer` stops its time-to-request clock.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

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

/// Connects, writes the full request, and flushes. The response is not read.
pub fn send(url: &Url, method: &str, headers: &[(&str, String)], body: &[u8]) -> Result<Connection> {
    let stream = TcpStream::connect((url.host.as_str(), url.port)).map_err(io_llm)?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(Duration::from_secs(300))).ok();
    let mut connection = if url.https {
        let name = rustls::pki_types::ServerName::try_from(url.host.clone()).map_err(io_llm)?;
        let client = rustls::ClientConnection::new(tls_config(), name).map_err(io_llm)?;
        Connection::Tls(Box::new(rustls::StreamOwned::new(client, stream)))
    } else {
        Connection::Plain(stream)
    };
    let mut request = format!(
        "{method} {} HTTP/1.1\r\nHost: {}\r\nContent-Length: {}\r\nConnection: close\r\n",
        url.path,
        url.host,
        body.len()
    );
    for (name, value) in headers {
        request.push_str(&format!("{name}: {value}\r\n"));
    }
    request.push_str("\r\n");
    let mut bytes = request.into_bytes();
    bytes.extend_from_slice(body);
    connection.write_all(&bytes).map_err(io_llm)?;
    connection.flush().map_err(io_llm)?;
    Ok(connection)
}

/// A response whose body is read through `body`, de-chunked when needed.
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

/// Reads the status line and headers.
pub fn read_response(connection: Connection) -> Result<Response> {
    let mut reader = BufReader::new(connection);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(io_llm)?;
    let status = line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse().ok())
        .ok_or_else(|| Error::Llm(format!("bad status line {:?}", line.trim())))?;
    let mut headers = Vec::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).map_err(io_llm)? == 0 {
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
    let header = |name: &str| headers.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_str());
    let body: Box<dyn BufRead + Send> =
        if header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
            Box::new(BufReader::new(Chunked { inner: reader, remaining: 0, done: false }))
        } else if let Some(length) = header("content-length").and_then(|v| v.parse::<u64>().ok()) {
            Box::new(BufReader::new(reader.take(length)))
        } else {
            Box::new(reader)
        };
    Ok(Response { status, body })
}

struct Chunked<R> {
    inner: R,
    remaining: u64,
    done: bool,
}

impl<R: BufRead> Read for Chunked<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.done {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut line = String::new();
            self.inner.read_line(&mut line)?;
            if line.trim().is_empty() {
                // The CRLF that ends the previous chunk.
                line.clear();
                self.inner.read_line(&mut line)?;
            }
            let size = u64::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16)
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad chunk size"))?;
            if size == 0 {
                self.done = true;
                return Ok(0);
            }
            self.remaining = size;
        }
        let want = buf.len().min(self.remaining as usize);
        let read = self.inner.read(&mut buf[..want])?;
        if read == 0 {
            self.done = true;
        }
        self.remaining -= read as u64;
        Ok(read)
    }
}

/// POSTs JSON and returns the status and full body.
pub fn post_json(url: &str, bearer: Option<&str>, body: &serde_json::Value) -> Result<(u16, Vec<u8>)> {
    let url = Url::parse(url)?;
    let mut headers = vec![("Content-Type", "application/json".to_string())];
    if let Some(token) = bearer {
        headers.push(("Authorization", format!("Bearer {token}")));
    }
    let connection = send(&url, "POST", &headers, &serde_json::to_vec(body)?)?;
    let response = read_response(connection)?;
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

    #[test]
    fn decodes_chunked_bodies() {
        let raw = b"4\r\nWiki\r\n5\r\npedia\r\n0\r\n\r\n";
        let mut out = String::new();
        Chunked { inner: &raw[..], remaining: 0, done: false }.read_to_string(&mut out).unwrap();
        assert_eq!(out, "Wikipedia");
    }
}
