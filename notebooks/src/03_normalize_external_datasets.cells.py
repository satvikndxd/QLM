# TITLE: Chousorus 1 dataset pipeline — 03_normalize_external_datasets
#
# Purpose: convert raw external JSONL (written by notebook 02) into the
# unified intermediate schema. Robust field detection, no invented answers,
# code preserved verbatim, every skipped row logged to an errors file.

# --- CELL ---
# TITLE: Bootstrap
import sys
from pathlib import Path

if not Path("/content/qlm_utils.py").exists():
    raise RuntimeError("Run notebooks/00_setup.ipynb first in this Colab session.")
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

import json
import pandas as pd
from qlm_utils import (
    data_path, read_jsonl, write_jsonl, append_jsonl, save_manifest,
    stable_id, count_by,
)

ERRORS_PATH = data_path("errors", "external_normalization_errors.jsonl")
UNPAIRED_PATH = data_path("interim", "external_unpaired.jsonl")

# --- CELL ---
# TITLE: Message extraction with robust field detection
#
# External datasets disagree on field names. We check an ordered list of
# (question, answer) candidate pairs, plus a native `messages` field. We do
# NOT invent an assistant answer when none exists — unpaired rows go to
# data/interim/external_unpaired.jsonl for later template-based handling.
PAIR_CANDIDATES = [
    ("question", "answer"),
    ("prompt", "response"),
    ("input", "output"),
    ("problem", "solution"),
    ("puzzle", "solution"),
    ("puzzle", "answer"),
    ("query", "response"),
    ("instruction", "output"),
    ("question", "solution"),
    ("text", "answer"),
]


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_messages(row: dict):
    """Return (messages, mode) where mode is 'messages' | 'pair' | None.

    Never truncates content: full puzzle text and full code are preserved
    (whole-artifact principle).
    """
    msgs = row.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 2:
        ok = all(isinstance(m, dict) and m.get("role") and _clean_text(m.get("content"))
                 for m in msgs)
        if ok:
            return ([{"role": m["role"], "content": _clean_text(m["content"])}
                     for m in msgs], "messages")
    for q_key, a_key in PAIR_CANDIDATES:
        q, a = _clean_text(row.get(q_key)), _clean_text(row.get(a_key))
        if q and a:
            return ([{"role": "user", "content": q},
                     {"role": "assistant", "content": a}], "pair")
    return (None, None)


def detect_user_only(row: dict) -> str:
    """Find a question with no answer (for the unpaired file)."""
    for q_key, _ in PAIR_CANDIDATES:
        q = _clean_text(row.get(q_key))
        if q:
            return q
    return ""

# --- CELL ---
# TITLE: Per-source normalization config
#
# Defaults per source; row-level fields override where present. Difficulty
# is a coarse prior (1-6 scale used across the corpus), refined later only
# if the row itself carries a difficulty/level field.
SOURCES = {
    "jane_street_puzzles": {
        "file": "jane_street_puzzles.jsonl",
        "domain": "general_reasoning",
        "task_type": "qa",
        "difficulty": 5,
        "tags": ["puzzle", "jane-street"],
    },
    "quantqa": {
        "file": "quantqa.jsonl",
        "domain": "quant_interview",
        "task_type": "qa",
        "difficulty": 3,
        "tags": ["interview", "quant"],
    },
    "quantcodeeval": {
        "file": "quantcodeeval.jsonl",
        "domain": "quant_coding",
        "task_type": "code_generation",
        "difficulty": 5,
        "tags": ["code", "strategy-reproduction"],
    },
    "bizbench": {
        "file": "bizbench.jsonl",
        "domain": "financial_qa",
        "task_type": "qa",
        "difficulty": 4,
        "tags": ["finance", "program-synthesis"],
    },
    "finmme_subset": {
        "file": "finmme_subset.jsonl",
        "domain": "financial_research",
        "task_type": "qa",
        "difficulty": 4,
        "tags": ["finance", "research"],
    },
}

DOMAIN_HINTS = {
    # refine domain from row-level category fields when present
    "probability": "probability", "statistics": "statistics",
    "finance": "finance", "derivatives": "derivatives",
    "brainteaser": "brainteasers", "brainteasers": "brainteasers",
    "coding": "quant_coding", "math": "mathematics",
}


def refine_domain(row: dict, default: str) -> str:
    for key in ("domain", "category", "topic", "subject", "type"):
        val = _clean_text(row.get(key)).lower()
        if val in DOMAIN_HINTS:
            return DOMAIN_HINTS[val]
    return default


def refine_difficulty(row: dict, default: int) -> int:
    for key in ("difficulty", "level", "hardness"):
        val = row.get(key)
        if isinstance(val, (int, float)) and 1 <= int(val) <= 10:
            return min(6, int(val))  # clamp to corpus scale
    return default

# --- CELL ---
# TITLE: Normalize every available raw file
normalized, unpaired, n_errors = [], [], 0
per_source_counts = {}

for source, cfg in SOURCES.items():
    raw_path = data_path("external", cfg["file"])
    rows, n_bad = read_jsonl(raw_path, ERRORS_PATH)
    if not rows:
        per_source_counts[source] = {"raw": 0, "normalized": 0, "unpaired": 0,
                                     "status": "missing — run notebook 02 first"}
        print(f"[skip] {source}: no raw file at {raw_path}")
        continue
    n_ok = n_unpaired = 0
    for i, row in enumerate(rows):
        messages, mode = extract_messages(row)
        if messages is None:
            question = detect_user_only(row)
            if question:
                unpaired.append({
                    "id": stable_id(f"{source}-unpaired", row),
                    "source": source, "question": question, "raw": row,
                })
                n_unpaired += 1
            else:
                n_errors += 1
                append_jsonl({"source": source, "row_index": i,
                              "error": "no recognizable question/answer fields",
                              "keys": sorted(row.keys())}, ERRORS_PATH)
            continue
        rec = {
            "id": _clean_text(row.get("id")) or stable_id(source, row),
            "source": source,
            "entry_kind": "external",
            "training_mode": "sft",
            "domain": refine_domain(row, cfg["domain"]),
            "task_type": _clean_text(row.get("task_type")) or cfg["task_type"],
            "difficulty": refine_difficulty(row, cfg["difficulty"]),
            "tags": list(cfg["tags"]),
            "messages": messages,
            # provenance: keep original keys (not values) + split, so any
            # record can be traced back to its raw row without bloating size
            "metadata": {"raw_keys": sorted(row.keys()),
                         "split": row.get("_split", ""),
                         "message_mode": mode},
        }
        normalized.append(rec)
        n_ok += 1
    per_source_counts[source] = {"raw": len(rows), "normalized": n_ok,
                                 "unpaired": n_unpaired, "bad_lines": n_bad,
                                 "status": "normalized"}
    print(f"[ok] {source}: {n_ok} normalized, {n_unpaired} unpaired, {n_bad} bad lines")

# --- CELL ---
# TITLE: Deduplicate and write outputs
#
# Dedup by id (content-derived ids make re-runs idempotent), deterministic
# order by (source, id) so the output is stable across runs.
seen, deduped = set(), []
for rec in normalized:
    if rec["id"] in seen:
        continue
    seen.add(rec["id"])
    deduped.append(rec)
deduped.sort(key=lambda r: (r["source"], r["id"]))

out_path = data_path("interim", "external_normalized.jsonl")
write_jsonl(deduped, out_path)
write_jsonl(unpaired, UNPAIRED_PATH)
print(f"wrote {len(deduped)} normalized -> {out_path}")
print(f"wrote {len(unpaired)} unpaired  -> {UNPAIRED_PATH}")

manifest = {
    "total_normalized": len(deduped),
    "total_unpaired": len(unpaired),
    "total_errors": n_errors,
    "per_source": per_source_counts,
    "by_domain": count_by(deduped, lambda r: r["domain"]),
    "by_task_type": count_by(deduped, lambda r: r["task_type"]),
    "by_difficulty": count_by(deduped, lambda r: r["difficulty"]),
}
save_manifest(manifest, data_path("manifests", "external_manifest.json"))

if deduped:
    pd.DataFrame([{k: r[k] for k in ("id", "source", "domain", "task_type", "difficulty")}
                  for r in deduped]).groupby(["source", "domain"]).size()
