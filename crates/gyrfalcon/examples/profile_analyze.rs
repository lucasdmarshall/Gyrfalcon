//! Times `tokenize` and the analyzer steps on a JSONL corpus (`text` field), one thread.
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let texts: Vec<String> = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|v| v["text"].as_str().unwrap_or("").to_string())
        .collect();
    let best = |f: &mut dyn FnMut() -> usize| {
        (0..3)
            .map(|_| {
                let t = Instant::now();
                let n = f();
                (t.elapsed().as_secs_f64(), n)
            })
            .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a })
    };
    let (s, n) = best(&mut || texts.iter().map(|t| gyrfalcon::tokenize(t).len()).sum());
    println!("tokenize            {s:6.2} s  {n} tokens  {:5.0} ns/token", s * 1e9 / n as f64);
    let tokens: Vec<_> = texts.iter().map(|t| gyrfalcon::tokenize(t)).collect();
    let (s, n) = best(&mut || tokens.iter().map(|t| gyrfalcon::analyze(t.clone()).len()).sum());
    println!("clone + analyze     {s:6.2} s  {n} kept    {:5.0} ns/token", s * 1e9 / n as f64);
    let (s, n) = best(&mut || tokens.iter().map(|t| t.clone().len()).sum());
    println!("clone only          {s:6.2} s");
    let (s, n2) = best(&mut || texts.iter().map(|t| gyrfalcon::prepare(t).tokens.len()).sum());
    println!("prepare             {s:6.2} s  {n2} tokens");
    let _ = n;
}
