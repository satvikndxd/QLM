# TITLE: Chousorus 1 dataset pipeline — 05_build_unified_corpus_and_curriculum
#
# Purpose: combine all processed sources into one unified corpus with full
# provenance, assign deterministic curriculum stages, and write statistics.
#
# Nothing is mixed blindly: every record keeps its source, entry_kind, and
# training_mode, and QLM deep entries keep their complete artifact.

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
from qlm_utils import data_path, read_jsonl, write_jsonl, save_manifest, count_by

SOURCE_FILES = {
    "qlm_v2": data_path("processed", "qlm_v2_entries.jsonl"),
    "qwen38_breadth": data_path("processed", "qwen_breadth.jsonl"),
    "external": data_path("interim", "external_normalized.jsonl"),
    "quantnet_style": data_path("external", "quantnet_style_examples.jsonl"),
}

# --- CELL ---
# TITLE: Load all available sources (missing ones warn, never fail)
pools = {}
for name, path in SOURCE_FILES.items():
    records, n_bad = read_jsonl(path)
    pools[name] = records
    status = f"{len(records)} records" if records else "MISSING — run its notebook first"
    print(f"{name:16s} {status}" + (f" ({n_bad} bad lines)" if n_bad else ""))

if not any(pools.values()):
    raise RuntimeError("No processed sources found. Run notebooks 01-04 first.")

# --- CELL ---
# TITLE: Unify records
#
# Minimal common envelope; source-specific payloads are preserved intact
# (messages for SFT records, the full entry for QLM deep records). The
# correct/adversarial/rlm_environment/breadth/external distinction is
# load-bearing for training and is never collapsed.
def unify(record: dict, pool_name: str) -> dict:
    rec = dict(record)  # shallow copy; payload fields pass through untouched
    rec.setdefault("id", f"{pool_name}-unknown")
    rec.setdefault("source", pool_name)
    rec.setdefault("entry_kind", "external")
    rec.setdefault("training_mode", "sft")
    rec.setdefault("domain", "general")
    rec.setdefault("tags", [])
    # difficulty scale: QLM complexity (1-10) is kept under "complexity";
    # SFT difficulty (1-6) under "difficulty". We expose one comparable
    # field for staging without destroying either original.
    if "complexity" in rec:
        rec["difficulty_unified"] = int(rec["complexity"])
    else:
        rec["difficulty_unified"] = int(rec.get("difficulty", 1))
    return rec


corpus = []
for pool_name, records in pools.items():
    for record in records:
        corpus.append(unify(record, pool_name))

# Dedup by id, deterministic order for stable builds.
seen, deduped = set(), []
for rec in sorted(corpus, key=lambda r: (r["source"], r["id"])):
    if rec["id"] in seen:
        continue
    seen.add(rec["id"])
    deduped.append(rec)
corpus = deduped
print(f"unified corpus: {len(corpus)} records")

# --- CELL ---
# TITLE: Curriculum stage assignment (deterministic rules)
#
# Stage 1 foundation:    easy breadth/external (difficulty <= 2), QLM
#                        complexity <= 3.
# Stage 2 intermediate:  mid breadth/external (difficulty 3-4), QLM
#                        complexity 4-7 (adversarial entries with declared
#                        flaws live here by design — flaw detection is an
#                        intermediate skill).
# Stage 3 advanced:      QLM complexity >= 8, ALL rlm_environment entries,
#                        code reproduction, difficulty >= 5.
def assign_stage(rec: dict) -> int:
    if rec["entry_kind"] == "rlm_environment":
        return 3  # environment interaction is always advanced
    if rec["source"] == "qlm_v2":
        cx = rec["difficulty_unified"]
        return 1 if cx <= 3 else (2 if cx <= 7 else 3)
    if rec.get("task_type") in ("code_generation",):
        return 3  # complex code reproduction
    d = rec["difficulty_unified"]
    return 1 if d <= 2 else (2 if d <= 4 else 3)


for rec in corpus:
    rec["stage"] = assign_stage(rec)

# Within-stage deterministic ordering: easier first, stable tie-break by id.
corpus.sort(key=lambda r: (r["stage"], r["difficulty_unified"], r["id"]))

# --- CELL ---
# TITLE: Write corpus, stage files, and manifest
write_jsonl(corpus, data_path("processed", "unified_corpus.jsonl"))
for stage in (1, 2, 3):
    stage_records = [r for r in corpus if r["stage"] == stage]
    write_jsonl(stage_records, data_path("processed", f"curriculum_stage_{stage}.jsonl"))
    print(f"stage {stage}: {len(stage_records)} records")

manifest = {
    "total_records": len(corpus),
    "by_source": count_by(corpus, lambda r: r["source"]),
    "by_entry_kind": count_by(corpus, lambda r: r["entry_kind"]),
    "by_training_mode": count_by(corpus, lambda r: r["training_mode"]),
    "by_stage": count_by(corpus, lambda r: r["stage"]),
    "by_domain": count_by(corpus, lambda r: r["domain"]),
    "by_difficulty": count_by(corpus, lambda r: r["difficulty_unified"]),
}
save_manifest(manifest, data_path("manifests", "corpus_manifest.json"))

# --- CELL ---
# TITLE: Human-readable summary tables
df = pd.DataFrame([{
    "id": r["id"], "source": r["source"], "entry_kind": r["entry_kind"],
    "training_mode": r["training_mode"], "domain": r["domain"],
    "difficulty": r["difficulty_unified"], "stage": r["stage"],
} for r in corpus])

print("=== records by stage x source ===")
print(df.pivot_table(index="stage", columns="source", values="id",
                     aggfunc="count", fill_value=0))
print()
print("=== records by entry_kind ===")
print(df["entry_kind"].value_counts().to_string())
print()
print("=== difficulty distribution by stage ===")
print(df.groupby("stage")["difficulty"].describe()[["count", "mean", "min", "max"]])
df.head(20)

# --- CELL ---
# TITLE: Sanity checks (assert core invariants, cheap and local)
#
# These protect the architecture-alignment requirements: kinds preserved,
# QLM artifacts whole, adversarial flaws intact.
qlm_recs = [r for r in corpus if r["source"] == "qlm_v2"]
for r in qlm_recs:
    assert "entry" in r and isinstance(r["entry"], dict), f"{r['id']}: QLM artifact lost"
    assert r["entry"]["research_corpus"]["code_implementation"], f"{r['id']}: code missing"
    if r["entry_kind"] == "adversarial":
        assert r["entry"].get("flaws"), f"{r['id']}: adversarial entry lost its flaws"
    if r["entry_kind"] == "rlm_environment":
        assert r["entry"].get("rlm", {}).get("environment_class"), f"{r['id']}: rlm meta lost"

kinds = {r["entry_kind"] for r in corpus}
print("entry kinds present:", sorted(kinds))
assert all(r.get("source") for r in corpus), "provenance missing on some records"
assert [r["stage"] for r in corpus] == sorted(r["stage"] for r in corpus), "stage order broken"
print("sanity checks passed")
