# TITLE: Chousorus 1 dataset pipeline — 00_setup
#
# Purpose: install dependencies, create the data directory tree, define the
# global CONFIG, and write the shared utility module (qlm_utils.py) that all
# later notebooks import.
#
# Run this notebook FIRST in every fresh Colab session.
#
# Nothing in this notebook touches the network except the pip cell below.

# --- CELL ---
# TITLE: Install dependencies
# RUN LATER IN COLAB
#
# Minimal, CPU-only dependency set. No PyTorch: dataset preparation needs
# none of it, and model training happens later in separate notebooks.
!pip install -q numpy pandas datasets huggingface_hub jsonschema tqdm

# --- CELL ---
# TITLE: Global config and directory tree
from pathlib import Path
import json
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

CONFIG = {
    "seed": RANDOM_SEED,
    "data_root": "/content/data",
    "qlm_entries_dir": "/content/data/entries",
    "max_external_samples_per_source": 1000,
    "finmme_max_samples": 200,
    "allow_network": True,   # loaders still never download unless their cell is run
    "device": "cpu",         # dataset prep is CPU-only by design
}

# Full directory tree used by the pipeline. "logs" holds optional real stdout
# logs for RLM observations; "entries" holds the six QLM v2 JSON files.
DATA_SUBDIRS = [
    "raw", "external", "interim", "processed",
    "manifests", "errors", "breadth", "trajectories",
    "logs", "entries",
]

data_root = Path(CONFIG["data_root"])
for sub in DATA_SUBDIRS:
    (data_root / sub).mkdir(parents=True, exist_ok=True)

# Persist the config so later notebooks (and the user) can see exactly what
# settings produced each artifact.
(data_root / "manifests" / "config.json").write_text(
    json.dumps(CONFIG, indent=2), encoding="utf-8"
)
print("data tree ready under", data_root)
print(json.dumps(CONFIG, indent=2))

# --- CELL ---
# TITLE: Write the shared utility module (qlm_utils.py)
#
# Why a module instead of copy-pasted helpers: seven notebooks share these
# functions; one definition avoids drift. Later notebooks fail fast with a
# clear message if this cell has not been run in the current session.
from pathlib import Path

UTILS_SRC = r'''
"""Shared utilities for the Chousorus 1 dataset pipeline notebooks.

Written by notebooks/00_setup.ipynb. All helpers are deterministic and
CPU-only. Errors are logged to JSONL files, never swallowed silently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RANDOM_SEED = 42

CONFIG = json.loads(
    Path("/content/data/manifests/config.json").read_text(encoding="utf-8")
)


def data_path(*parts) -> Path:
    """Absolute path under the data root."""
    return Path(CONFIG["data_root"]).joinpath(*parts)


def ensure_dir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_load_json(text):
    """Parse JSON, returning None instead of raising on bad input."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def read_jsonl(path, error_path=None):
    """Tolerant JSONL reader.

    Returns (records, n_bad_lines). Bad lines are appended to error_path
    (if given) rather than killing the whole load — one corrupt row must
    never cost the rest of the file.
    """
    path = Path(path)
    records, n_bad = [], 0
    if not path.exists():
        return records, n_bad
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = safe_load_json(line)
            if obj is None:
                n_bad += 1
                if error_path is not None:
                    append_jsonl(
                        {"file": str(path), "lineno": lineno, "raw": line[:2000],
                         "error": "invalid json"},
                        error_path,
                    )
            else:
                records.append(obj)
    return records, n_bad


def write_jsonl(records, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(record, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_error(error_path, payload) -> None:
    """Uniform error logging: every skipped row leaves a trace on disk."""
    append_jsonl(payload, error_path)


def stable_id(prefix: str, payload) -> str:
    """Deterministic content-derived id: same payload -> same id across runs."""
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def count_by(records, key_fn) -> dict:
    """Sorted frequency table; unhashable/missing keys count under '<none>'."""
    counts = {}
    for rec in records:
        try:
            key = key_fn(rec)
        except (KeyError, TypeError, AttributeError):
            key = None
        key = "<none>" if key is None else str(key)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def save_manifest(manifest: dict, path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path)


def require_file(path, hint: str) -> Path:
    """Fail fast with an actionable message instead of a cryptic traceback."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. {hint}")
    return path
'''

Path("/content/qlm_utils.py").write_text(UTILS_SRC, encoding="utf-8")

# Import-test the module immediately so failures surface here, not in 01-06.
import sys
if "/content" not in sys.path:
    sys.path.insert(0, "/content")
import importlib
import qlm_utils
importlib.reload(qlm_utils)
print("qlm_utils ready:", [n for n in dir(qlm_utils) if not n.startswith("_")])

# --- CELL ---
# TITLE: Local self-test of the utilities (no network, no datasets)
#
# A tiny round-trip check so the user knows the helpers work before running
# any loader notebook. Uses a throwaway file under data/raw.
from qlm_utils import (
    read_jsonl, write_jsonl, append_jsonl, safe_load_json,
    stable_id, count_by, data_path,
)

probe = data_path("raw", "_selftest.jsonl")
write_jsonl([{"a": 1}, {"a": 2}], probe)
append_jsonl({"a": 3}, probe)
records, n_bad = read_jsonl(probe)
assert [r["a"] for r in records] == [1, 2, 3], "jsonl round-trip failed"
assert n_bad == 0
assert safe_load_json("not json") is None
assert stable_id("x", {"k": 1}) == stable_id("x", {"k": 1}), "stable_id not deterministic"
assert count_by(records, lambda r: r["a"] % 2) == {"0": 1, "1": 2}
probe.unlink()
print("utility self-test passed")

# --- CELL ---
# TITLE: Optional Google Drive mount
# OPTIONAL: RUN LATER IN COLAB
#
# Only needed if you want artifacts to survive the Colab session. The
# pipeline itself never assumes Drive is mounted.
# from google.colab import drive
# drive.mount("/content/drive")
# Then, to persist outputs at the end of a session:
# !cp -r /content/data /content/drive/MyDrive/chousorus1_data
print("Drive mount is optional; uncomment the lines above if you want persistence.")
