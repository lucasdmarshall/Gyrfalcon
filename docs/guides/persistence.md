<sub>[Docs](../README.md) / Guides</sub>

# Persistence

A Gyrfalcon collection lives in a directory and is always on disk. There is no separate save
step and no load step beyond opening: every write is durable when the call returns, and opening
a directory maps its files and replays any logged writes.

## Save and load

```python
from gyrfalcon import Collection

# Session 1: build.
with Collection("./kb") as col:
    col.add_path("handbook/")

# Session 2, any later process: open and query. Nothing is rebuilt.
with Collection("./kb", create=False) as col:
    print(col.search("parental leave policy", k=3))
```

`create=False` raises `FileNotFoundError` when the directory holds no collection, which catches
a mistyped path before it silently creates an empty collection.

In Rust, `Tier::open(dir, model_dir, create)` does the same, and `Index::open` opens with an
explicit `IndexConfig`.

Opening a lite collection from Python takes about 60 ms. The cost does not grow with the
collection, because segment files are memory-mapped rather than read.

## What is on disk

```text
kb/
├── manifest.json           the published state: identity, generations, segment list
├── writer.lock             the pid of the process holding the collection (only while open)
├── tombstones/3.bin        ids of deleted chunks
├── wal/3.log               small writes since the last seal
└── segments/
    └── 00000000/           one sealed, immutable segment
        ├── segment.json    identity and counts
        ├── ids.bin         chunk ids, ascending
        ├── keys.bin        chunk ids sorted by document id, for update and delete
        ├── lexical/        BM25: terms.dict, postings.bin, norms.bin, docs.bin (stored text)
        ├── vectors.bin     int8 vectors
        └── codes.bin       one-bit codes for the dense prefilter
```

`manifest.json` is the only file that changes in place, and it changes by writing a new file and
renaming it over the old one. Everything a manifest names is written and fsynced before that
rename. A reader therefore always sees one complete published state.

## Durability

| Event | What survives |
|---|---|
| A write call returns | the write, always |
| The process crashes or the machine loses power during a write | every earlier write; the write in flight is either fully present or absent |
| A crash during a background merge | everything; the merge simply did not happen |
| A crash in the middle of a WAL append | every complete frame; the torn tail is detected by its CRC32 and truncated on the next open |

Small writes (fewer than 256 chunks) are appended to `wal/` and fsynced before the call returns.
Larger writes seal a segment and publish a new manifest before returning. An update's new chunks
and the deletion of its old chunks are written in the same WAL frame or the same manifest
rename, so a crash never leaves both versions, or neither.

`close()` (or `flush()`) seals the logged writes into a segment. That is not needed for
durability; it only makes the next open skip the log replay.

## One process at a time

Opening a collection takes `writer.lock`. A second process that opens the same directory gets:

```text
RuntimeError: another writer (pid 8269) holds the collection
```

If the holder died without closing, its pid is no longer alive and the next open reclaims the
lock. Within one process, a single open collection serves any number of threads: searches run
in parallel, writes take turns.

To serve one collection to many processes, run `gyr serve` and query it over
[HTTP](../reference/configuration.md#http-api).

## Copying, moving, and backing up

Close the collection, then copy the directory with any tool:

```sh
tar czf kb-2026-10-26.tgz kb/
rsync -a kb/ backup-host:/srv/kb/
```

A closed collection is self-contained and position-independent. It opens on another machine as
long as that machine has the same tier's model. The model files are not inside the collection.

Copying a collection while a process has it open can capture a manifest that does not match the
files copied a moment earlier. There is no online snapshot command yet, so close first or stop
writes for the length of the copy.

## Identity and versions

Every collection records the identity it was built with, and every open compares it:

| Field | Example | Changes when |
|---|---|---|
| `format_version` | `7` | the on-disk format changes |
| `model_id` | `minishlab/potion-retrieval-32M` | you choose another model or tier |
| `dimension` | `512` | the model's vector width differs |
| `quantization` | `i8` | you choose `f16`, `f32`, or `b1` storage |
| `analyzer_id` | `gyr-text-v2` | the tokenizer, stopwords, or stemmer change |

A mismatch fails the open before any file is mapped or model loaded, with the field and both
values in the message. A collection from an older format version must be rebuilt from its
sources; there is no in-place migration. Keep the source documents, or an export of them, for
as long as you keep a collection.

## Deleted data

`delete` and `update` mark old chunks as dead in `tombstones/`; the segment files that hold them
are not rewritten at once. Dead chunks never appear in results. They leave the files when a
background merge rewrites their segment, or when you run `gyr compact --all`. If deleted text
must be physically gone from disk, for example after a removal request, run
`gyr compact --all` and then check that no process still holds the old files open.

---

<sub>Previous: [Custom embedders](custom-embedders.md) · Next: [Migrating from LangChain](migrating-from-langchain.md)</sub>
