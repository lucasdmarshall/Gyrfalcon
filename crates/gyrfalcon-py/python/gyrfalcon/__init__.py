"""Gyrfalcon: fast hybrid retrieval (BM25 + dense) for RAG, in one Rust engine.

    from gyrfalcon import Collection

    with Collection("./my-index") as col:          # lite tier: no runtime, fast on any CPU
        col.add("intro", "Gyrfalcons are the largest falcons.", metadata={"topic": "birds"})
        col.add_path("docs/", metadata={"source": "docs"})   # PDF, HTML, Markdown, text
        for hit in col.search("largest falcon", k=5, filter={"topic": ["birds", "raptors"]}):
            print(hit.score, hit.doc_id, hit.metadata, hit.text[:80])

Models download on first use into ``$GYR_MODELS`` (default ``~/.cache/gyrfalcon/models``).
The full tier needs ONNX Runtime and PDFs need PDFium; both are found automatically when the
``onnxruntime`` and ``pypdfium2`` packages are installed (``pip install "gyrfalcon-rag[full,pdf]"``),
or through ``ORT_DYLIB_PATH`` and ``GYR_PDFIUM``.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

from . import _native
from ._native import Hit, read_file, supported_extensions

__all__ = ["Collection", "Hit", "read_file", "models_dir", "ensure_model", "supported_extensions"]
__version__ = _native.__version__

PathLike = Union[str, "os.PathLike[str]"]
Metadata = Optional[Mapping[str, object]]
_BATCH = 512


def _pairs(metadata: Metadata) -> list:
    """``{"tag": ["a", "b"], "year": 2024}`` as ``[("tag", "a"), ("tag", "b"), ("year", "2024")]``."""
    pairs = []
    for key, value in (metadata or {}).items():
        values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        pairs.extend((str(key), _text(v)) for v in values)
    return pairs


def _clauses(filter: Metadata) -> list:
    """``{"tag": ["a", "b"], "lang": "en"}`` as ``[("tag", ["a", "b"]), ("lang", ["en"])]``."""
    clauses = []
    for key, value in (filter or {}).items():
        values = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        clauses.append((str(key), [_text(v) for v in values]))
    return clauses


def _text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def models_dir() -> Path:
    """Where tier models live: ``$GYR_MODELS``, else ``$XDG_CACHE_HOME/gyrfalcon/models``."""
    if os.environ.get("GYR_MODELS"):
        return Path(os.environ["GYR_MODELS"])
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(cache) / "gyrfalcon" / "models"


def ensure_model(tier: str = "lite", root: Optional[PathLike] = None) -> Path:
    """The tier's model directory under ``root``, downloading missing files from Hugging Face."""
    directory = Path(root) if root is not None else models_dir()
    directory = directory / _native.tier_model_dir(tier)
    for repo, remote, name in _native.tier_files(tier):
        target = directory / name
        if target.exists():
            continue
        directory.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/main/{remote}"
        print(f"gyrfalcon: downloading {repo}/{remote}", file=sys.stderr)
        fd, partial = tempfile.mkstemp(dir=directory, prefix=f".{name}.")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url) as response:
                while True:
                    block = response.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
            os.replace(partial, target)
        except BaseException:
            os.unlink(partial)
            raise
    return directory


def _package_dir(name: str) -> Optional[str]:
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def _find_onnxruntime() -> None:
    """Points ``ORT_DYLIB_PATH`` at the library of the ``onnxruntime`` package if it is unset."""
    if os.environ.get("ORT_DYLIB_PATH"):
        return
    package = _package_dir("onnxruntime")
    if package is None:
        return
    for pattern in ("libonnxruntime.so*", "libonnxruntime*.dylib", "onnxruntime.dll"):
        found = sorted(glob.glob(os.path.join(package, "capi", pattern)))
        if found:
            os.environ["ORT_DYLIB_PATH"] = found[0]
            return


def _find_pdfium() -> None:
    """Points ``GYR_PDFIUM`` at the PDFium library ``pypdfium2`` ships if it is unset."""
    if os.environ.get("GYR_PDFIUM"):
        return
    package = _package_dir("pypdfium2_raw")
    if package is None:
        return
    for pattern in ("libpdfium.so", "libpdfium.dylib", "pdfium.dll"):
        found = glob.glob(os.path.join(package, pattern))
        if found:
            os.environ["GYR_PDFIUM"] = found[0]
            return


_find_pdfium()


class Collection:
    """A Gyrfalcon collection on disk: a directory holding the index of one tier.

    Opening creates the collection unless ``create=False``. ``warm`` is ``"none"``,
    ``"background"`` (the default), or ``"eager"``. ``"background"`` returns immediately and
    reads mapped search files on a thread; a query before that finishes reads exact rows.
    ``"eager"`` finishes that read before open returns. One process at a time may hold the
    collection; close it (or use ``with``) to release it. Writes are durable when a call returns.
    """

    def __init__(
        self,
        path: PathLike,
        tier: str = "lite",
        *,
        models: Optional[PathLike] = None,
        create: bool = True,
        warm: str = "background",
    ) -> None:
        if tier == "full":
            _find_onnxruntime()
        model_dir = ensure_model(tier, models)
        self._native = _native.Collection(os.fspath(path), os.fspath(model_dir), tier, create, warm)

    @property
    def path(self) -> Path:
        return Path(self._native.path)

    @property
    def tier(self) -> str:
        return self._native.tier

    def add(self, doc_id: str, text: str, metadata: Metadata = None) -> list:
        """Indexes one document; returns its chunk ids. Use ``update`` to replace a document.

        ``metadata`` maps keys to a value or a list of values (strings, numbers, booleans);
        ``search`` can filter on them.
        """
        return self._native.add(doc_id, text, _pairs(metadata))

    def add_many(self, docs: Union[Mapping[str, str], Iterable[tuple]]) -> int:
        """Indexes ``{doc_id: text}``, ``(doc_id, text)`` pairs, or ``(doc_id, text, metadata)``
        triples; returns the chunk count."""
        items = docs.items() if isinstance(docs, Mapping) else docs
        chunks = 0
        batch = []
        for item in items:
            doc_id, text = item[0], item[1]
            batch.append((doc_id, text, _pairs(item[2] if len(item) > 2 else None)))
            if len(batch) == _BATCH:
                chunks += self._native.add_many(batch)
                batch = []
        if batch:
            chunks += self._native.add_many(batch)
        return chunks

    def add_file(self, path: PathLike, doc_id: Optional[str] = None, metadata: Metadata = None) -> int:
        """Reads a PDF, HTML, or text file and indexes it; returns the chunk count."""
        return self._native.add_file(os.fspath(path), doc_id, _pairs(metadata))

    def add_path(self, path: PathLike, metadata: Metadata = None) -> dict:
        """Indexes a file, or every supported file under a directory (hidden entries skipped),
        each with ``metadata``.

        Returns ``{"documents": n, "chunks": n, "skipped": [(path, reason), ...]}``.
        """
        report = {"documents": 0, "chunks": 0, "skipped": []}
        for file in _walk(Path(path)):
            try:
                report["chunks"] += self.add_file(file, metadata=metadata)
                report["documents"] += 1
            except (RuntimeError, UnicodeDecodeError, OSError) as err:
                if "PDFium unavailable" in str(err):
                    raise
                report["skipped"].append((str(file), str(err)))
        return report

    def update(self, doc_id: str, text: str, metadata: Metadata = None) -> int:
        """Replaces a document's chunks (and metadata) atomically; returns the new chunk count."""
        return self._native.update(doc_id, text, _pairs(metadata))

    def delete(self, doc_id: str) -> int:
        """Removes a document; returns the number of chunks removed."""
        return self._native.delete(doc_id)

    def search(self, query: str, k: int = 10, filter: Metadata = None) -> Sequence[Hit]:
        """The best ``k`` chunks for ``query``, best first.

        ``filter`` keeps documents whose metadata has, for every key, the value or one of the
        listed values: ``{"lang": "en", "tag": ["a", "b"]}`` is ``lang == en and tag in (a, b)``.
        """
        return self._native.search(query, k, _clauses(filter))

    def flush(self) -> None:
        """Seals buffered writes into a segment (they are already durable in the log)."""
        self._native.flush()

    def close(self) -> None:
        """Flushes, waits for background merges, and releases the collection."""
        self._native.close()

    def __enter__(self) -> "Collection":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Collection({str(self.path)!r}, tier={self.tier!r})"


def _walk(path: Path) -> Iterator[Path]:
    if not path.is_dir():
        yield path
        return
    extensions = {"." + e for e in supported_extensions()}
    for entry in sorted(path.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            yield from _walk(entry)
        elif entry.suffix.lower() in extensions:
            yield entry
