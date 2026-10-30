<sub>[Docs](../README.md) / Reference</sub>

# Python API

The complete public surface of the `gyrfalcon` package. The source of truth is
[`python/gyrfalcon/__init__.py`](../../crates/gyrfalcon-py/python/gyrfalcon/__init__.py) and the
type stub [`_native.pyi`](../../crates/gyrfalcon-py/python/gyrfalcon/_native.pyi); the package
ships `py.typed`, so type checkers read the same signatures.

```python
from gyrfalcon import Collection, Hit, read_file, supported_extensions, models_dir, ensure_model
```

| Name | Kind | Summary |
|---|---|---|
| [`Collection`](#collection) | class | A collection on disk: write, search, close |
| [`Hit`](#hit) | class | One search result |
| [`read_file`](#read_file) | function | Extract the text of a PDF, HTML, or text file |
| [`supported_extensions`](#supported_extensions) | function | File extensions a directory walk reads |
| [`models_dir`](#models_dir) | function | Where tier models are stored |
| [`ensure_model`](#ensure_model) | function | Download a tier's model files if missing |
| `__version__` | str | The package version |

---

## `Collection`

```python
Collection(path, tier="lite", *, models=None, create=True, warm="background")
```

Opens the collection in directory `path`, creating it when `create` is true and the directory
holds no collection. The tier's model is downloaded first if it is missing.

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str` or path-like | | The collection directory |
| `tier` | `"lite"` or `"full"` | `"lite"` | The tier to create with, or to expect on open |
| `models` | `str` or path-like | `None` | Root directory of tier models; `None` means `models_dir()` |
| `create` | `bool` | `True` | Create the collection if it does not exist |
| `warm` | `"none"`, `"background"`, or `"eager"` | `"background"` | When to read mapped search files into the page cache. `background` returns immediately and reads on a thread; an early query reads exact rows until those pages are resident. `eager` finishes before open returns. `none` does not read ahead. Same as `gyr --warm`. The default is from the gp3 open measurement in `bench/results/2026-10-29-warm-modes` |

Raises `FileNotFoundError` when `create=False` and there is no collection at `path`, and
`RuntimeError` when the collection was built with another tier or format, or another process
holds it. `tier="full"` also needs ONNX Runtime (`pip install "gyrfalcon-rag[full]"` or
`ORT_DYLIB_PATH`).

The collection holds a lock on the directory until `close()`. Use it as a context manager to
release it reliably.

### Properties

| Property | Type | Meaning |
|---|---|---|
| `path` | `pathlib.Path` | The collection directory |
| `tier` | `str` | `"lite"` or `"full"` |

### `add`

```python
add(doc_id: str, text: str, metadata: Mapping[str, object] | None = None) -> list[int]
```

Indexes one document and returns the ids of its chunks, ascending. Adding a `doc_id` that
already exists keeps the old chunks and adds new ones; use `update` to replace a document.

`metadata` maps keys to a value or a list of values. Values may be `str`, `int`, `float`, or
`bool`; they are stored as text (`True` as `"true"`). A list gives the key several values, and a
filter on any one of them matches.

### `add_many`

```python
add_many(docs: Mapping[str, str] | Iterable[tuple]) -> int
```

Indexes many documents and returns the total number of chunks written. Accepts a
`{doc_id: text}` mapping, or an iterable of `(doc_id, text)` pairs or
`(doc_id, text, metadata)` triples. The iterable is consumed lazily and sent to the engine 512
documents at a time, so a generator over a large corpus stays in constant memory.

### `add_file`

```python
add_file(path, doc_id: str | None = None, metadata: Mapping[str, object] | None = None) -> int
```

Reads a PDF, HTML, or text file and indexes its text as one document. `doc_id` defaults to the
path as given. Returns the number of chunks. PDFs need PDFium; see
[Installation](../getting-started/installation.md).

### `add_path`

```python
add_path(path, metadata: Mapping[str, object] | None = None) -> dict
```

Indexes a file, or every file under a directory whose extension is in `supported_extensions()`.
Directories are walked in sorted order; entries whose name starts with `.` are skipped. Every
document gets `metadata`. A file that cannot be read is skipped and reported, except when
PDFium itself is missing, which raises.

Returns:

```python
{"documents": 12, "chunks": 340, "skipped": [("docs/broken.pdf", "pdf docs/broken.pdf: ...")]}
```

### `update`

```python
update(doc_id: str, text: str, metadata: Mapping[str, object] | None = None) -> int
```

Replaces all chunks and the metadata of `doc_id` in one atomic write and returns the new chunk
count. A search sees either the old document or the new one, never a mix. An unknown `doc_id`
is inserted.

### `delete`

```python
delete(doc_id: str) -> int
```

Removes the document and returns the number of chunks removed; `0` for an unknown id.

### `search`

```python
search(query: str, k: int = 10, filter: Mapping[str, object] | None = None) -> list[Hit]
```

Returns the best `k` chunks for `query`, best first, from both the BM25 and the dense index,
fused with the tier's rule. A blank query returns `[]`.

`filter` keeps chunks whose document matches every key. A key's value may be a single value or
a list; a list matches any of its values:

```python
col.search("pricing", filter={"lang": "en", "product": ["pro", "team"]})
# lang == "en" AND product IN ("pro", "team")
```

Searches see every write that returned before them, including writes not yet sealed.

### `flush`

```python
flush() -> None
```

Seals buffered writes into a segment. Writes are already durable without it; flushing only
makes the next open faster.

### `close`

```python
close() -> None
```

Flushes, waits for background merges to finish, and releases the collection. Calling any other
method afterwards raises `RuntimeError("the collection is closed")`. Closing twice is harmless.

`with Collection(...) as col:` calls `close()` on exit, also when the block raises.

---

## `Hit`

One retrieved chunk. Read-only.

| Attribute | Type | Meaning |
|---|---|---|
| `doc_id` | `str` | The document's id |
| `text` | `str` | The chunk's text |
| `score` | `float` | Fused score; higher is better. Comparable only within one query |
| `start` | `int` | Byte offset of the chunk in the document's UTF-8 text |
| `end` | `int` | Byte offset one past the chunk's end |
| `chunk_id` | `int` | The engine's chunk id; stable across merges, never reused |
| `metadata` | `dict[str, str \| list[str]]` | The document's metadata; a key with several values maps to a list |

Offsets are in bytes, not characters. To slice the original text:

```python
original.encode("utf-8")[hit.start:hit.end].decode("utf-8")
```

---

## `read_file`

```python
read_file(path) -> tuple[str, list[int]]
```

Extracts a file's text the way `add_file` does and returns it with the byte offset where each
page starts (PDFs only; empty for other files). Use it to extract in several processes, or to
keep page numbers:

```python
import bisect
text, page_starts = read_file("paper.pdf")
page = bisect.bisect_right(page_starts, hit.start)     # 1-based page of a hit
```

PDF text goes through a layout layer that removes running headers, footers, and page numbers,
rejoins hyphenated words, and rebuilds paragraphs. HTML keeps the title and block structure and
drops scripts, styles, and navigation.

## `supported_extensions`

```python
supported_extensions() -> list[str]
```

`["pdf", "html", "htm", "xhtml", "txt", "text", "md", "markdown", "rst", "adoc", "org"]`. A file
passed to `add_file` directly is read whatever its extension: PDF and HTML by extension, anything
else as UTF-8 text.

## `models_dir`

```python
models_dir() -> pathlib.Path
```

`$GYR_MODELS` if set, else `$XDG_CACHE_HOME/gyrfalcon/models`, else
`~/.cache/gyrfalcon/models`.

## `ensure_model`

```python
ensure_model(tier: str = "lite", root=None) -> pathlib.Path
```

Returns the tier's model directory under `root` (default `models_dir()`), first downloading any
missing file from Hugging Face. Each file is written to a temporary name and renamed into place,
so an interrupted download never leaves a truncated model. Progress lines go to stderr.

---

## Exceptions

| Exception | Raised when |
|---|---|
| `FileNotFoundError` | `Collection(..., create=False)` finds no collection; a file passed to `add_file` does not exist |
| `ValueError` | an unknown tier name |
| `RuntimeError` | the collection is closed; it was built with another tier, model, or format; another process holds it; a model or library is missing; a file cannot be read (a broken PDF, a text file that is not UTF-8) |

`RuntimeError` messages start with a stable phrase, for example `incompatible index:`,
`another writer (pid N) holds the collection`, or `model unavailable:`.

## Threads and processes

- One `Collection` object may be shared by any number of threads. Every engine call releases the
  GIL. Searches run in parallel with each other; writes take turns and wait for running
  searches.
- One process at a time may open a collection directory. To share one across processes, serve
  it with `gyr serve` and use the [HTTP API](configuration.md#http-api).
- Several collections can be open in one process, each in its own directory.

---

<sub>Previous: [Planner](../concepts/planner.md) · Next: [Rust API](rust-api.md)</sub>
