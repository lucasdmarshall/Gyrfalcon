"""The Python package against the lite tier. Models come from $GYR_MODELS, downloaded on first
use when missing.

    maturin develop --release && pytest crates/gyrfalcon-py/tests
"""

import threading

import pytest

import gyrfalcon

DOCS = {
    "falcon": "The gyrfalcon is the largest of the falcon species.",
    "peregrine": "Peregrine falcons dive faster than any other animal.",
    "owl": "Owls hunt at night.",
}


@pytest.fixture
def col(tmp_path):
    with gyrfalcon.Collection(tmp_path / "col") as collection:
        yield collection


def test_add_search_update_delete(col):
    assert col.add_many(DOCS) == 3
    top = col.search("fastest diving animal", k=2)
    assert top[0].doc_id == "peregrine"
    assert top[0].text == DOCS["peregrine"]
    assert (top[0].start, top[0].end) == (0, len(DOCS["peregrine"]))
    assert col.update("owl", "Snowy owls hunt in daylight on the tundra.") == 1
    assert col.search("tundra daylight", k=1)[0].doc_id == "owl"
    assert col.delete("peregrine") == 1
    assert "peregrine" not in {h.doc_id for h in col.search("falcons dive", k=10)}
    assert col.search("   ") == []


def test_writes_survive_reopen_and_the_lock_is_released(tmp_path):
    path = tmp_path / "col"
    col = gyrfalcon.Collection(path)
    chunk_ids = col.add("falcon", DOCS["falcon"])
    col.close()
    with pytest.raises(RuntimeError, match="closed"):
        col.search("falcon")
    with gyrfalcon.Collection(path, create=False) as again:
        hit = again.search("largest falcon", k=1)[0]
        assert (hit.doc_id, hit.chunk_id) == ("falcon", chunk_ids[0])


def test_missing_collection_and_tier_mismatch(tmp_path):
    with pytest.raises(FileNotFoundError):
        gyrfalcon.Collection(tmp_path / "missing", create=False)
    with pytest.raises(ValueError, match="tier"):
        gyrfalcon.Collection(tmp_path / "x", tier="huge")


def test_one_writer_per_collection(tmp_path):
    with gyrfalcon.Collection(tmp_path / "col"):
        with pytest.raises(RuntimeError, match="writer"):
            gyrfalcon.Collection(tmp_path / "col")


def test_files_and_directories(col, tmp_path):
    docs = tmp_path / "docs"
    (docs / "nested").mkdir(parents=True)
    (docs / "page.html").write_text(
        "<html><head><title>Raptors</title><script>x()</script></head>"
        "<body><nav>menu</nav><p>Kestrels hover &amp; hunt voles.</p></body></html>"
    )
    (docs / "nested" / "notes.md").write_text("# Notes\n\nOspreys eat fish.\n")
    (docs / ".hidden.txt").write_text("secret")
    (docs / "image.png").write_bytes(b"\x89PNG")
    report = col.add_path(docs)
    assert (report["documents"], report["skipped"]) == (2, [])
    hit = col.search("kestrel voles", k=1)[0]
    assert hit.doc_id.endswith("page.html")
    assert hit.text == "Raptors\n\nKestrels hover & hunt voles."
    assert all("hidden" not in h.doc_id for h in col.search("secret", k=5))
    text, pages = gyrfalcon.read_file(docs / "nested" / "notes.md")
    assert text.startswith("# Notes") and pages == []


def test_searches_run_alongside_each_other(col):
    col.add_many({f"d{i}": f"note {i} about raptor number {i}" for i in range(200)})
    errors = []

    def search():
        try:
            for i in range(50):
                assert col.search(f"raptor number {i}", k=1)[0].doc_id == f"d{i}"
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=search) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
