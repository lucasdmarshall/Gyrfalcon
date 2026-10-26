//! Applies the layout layer to page texts, for `bench/pdf`.
//!
//!     cargo run --release -p gyrfalcon-ingest --example clean_pages -- IN_DIR OUT_DIR
//!
//! Each `IN_DIR/{name}.txt` holds one document's pages, each ended by `bench/pdf`'s page break.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (input, output) = (std::path::Path::new(&args[1]), std::path::Path::new(&args[2]));
    std::fs::create_dir_all(output).unwrap();
    let started = std::time::Instant::now();
    let mut pages_total = 0;
    for entry in std::fs::read_dir(input).unwrap().flatten() {
        let text = std::fs::read_to_string(entry.path()).unwrap();
        let pages: Vec<String> = text.split("\n<<<gyr-page-break>>>\n").map(str::to_string).collect();
        pages_total += pages.len();
        let cleaned = gyrfalcon_ingest::clean(&pages);
        std::fs::write(output.join(entry.file_name()), cleaned.text).unwrap();
    }
    eprintln!("{pages_total} pages in {:.3} s", started.elapsed().as_secs_f64());
}
