# TITLE: Chousorus 1 dataset pipeline — 01_load_qlm_v2_entries
#
# Purpose: load the six QLM v2 JSON entries, validate them against the
# schema-v2 rules, summarize them, and save normalized records + manifest.
#
# IMPORTANT: this notebook only loads and validates. It never executes the
# Python inside research_corpus.code_implementation, and it never modifies
# the six entry files.

# --- CELL ---
# TITLE: Bootstrap (requires 00_setup to have run in this session)
import sys
from pathlib import Path

if not Path("/content/qlm_utils.py").exists():
    raise RuntimeError("Run notebooks/00_setup.ipynb first in this Colab session.")
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

import json
import pandas as pd
from qlm_utils import (
    CONFIG, data_path, ensure_dir, write_jsonl, save_manifest, log_error,
)

ENTRIES_DIR = Path(CONFIG["qlm_entries_dir"])
EXPECTED_FILES = [
    "001_sma_crossover.json",
    "002_pairs_engle_granger.json",
    "003_garch_qlike_dm.json",
    "004_adversarial_lookahead_sma.json",
    "005_adversarial_p_hacking_sweep.json",
    "006_rlm_environment_garch.json",
]
print("expecting entries in:", ENTRIES_DIR)

# --- CELL ---
# TITLE: Get the entry files into Colab
# RUN LATER IN COLAB
#
# Option A — upload the six JSON files manually into /content/data/entries
# via the Colab file browser, or with the picker below.
#
# Option B — clone the QLM repository and copy its data/entries directory.
# Fill in REPO_URL if you have one; left empty, this cell only prints status.
REPO_URL = ""  # e.g. "https://github.com/<user>/<qlm-repo>.git"

missing = [f for f in EXPECTED_FILES if not (ENTRIES_DIR / f).exists()]
if missing and REPO_URL:
    !git clone {REPO_URL} /content/qlm_repo
    !cp /content/qlm_repo/data/entries/*.json /content/data/entries/
    missing = [f for f in EXPECTED_FILES if not (ENTRIES_DIR / f).exists()]
if missing:
    print("Missing entry files:", missing)
    print("Upload them to", ENTRIES_DIR, "then re-run this cell.")
    # Uncomment for an upload picker (files land in /content, move them after):
    # from google.colab import files
    # uploaded = files.upload()
else:
    print("All six QLM v2 entry files present.")

# --- CELL ---
# TITLE: Schema v2 validation rules
#
# Hand-rolled checks mirroring the QLM v2 gating contract. jsonschema is a
# dependency, but the six explicit rules below are the load-bearing ones and
# hand-rolling keeps the error messages precise for a student audience.
REQUIRED_TOP_LEVEL = [
    "schema_version", "entry_kind", "expected_verdict", "metadata",
    "agent_thought_process", "research_corpus", "adversarial_critique",
    "agent_instructions", "verification",
]
VALID_ENTRY_KINDS = {"correct", "adversarial", "rlm_environment"}


def validate_qlm_v2_entry(entry: dict) -> list:
    """Return a list of violation strings (empty list == valid)."""
    errors = []
    for field in REQUIRED_TOP_LEVEL:
        # expected_verdict is optional for rlm_environment entries, but all
        # six seed entries carry one, so we require it uniformly here.
        if field not in entry:
            errors.append(f"missing top-level field: {field}")
    if errors:
        return errors  # later checks would just cascade confusingly

    if entry["schema_version"] != 2:
        errors.append(f"schema_version must be 2, got {entry['schema_version']!r}")

    kind = entry["entry_kind"]
    if kind not in VALID_ENTRY_KINDS:
        errors.append(f"invalid entry_kind: {kind!r}")

    flaws = entry.get("flaws", [])
    if kind == "adversarial" and (not isinstance(flaws, list) or len(flaws) == 0):
        errors.append("adversarial entry must declare a non-empty flaws list")
    if kind == "correct" and flaws:
        errors.append("correct entry must not declare flaws")
    if kind == "rlm_environment":
        rlm = entry.get("rlm", {})
        if not rlm.get("environment_class"):
            errors.append("rlm_environment entry needs non-null rlm.environment_class")
        if not rlm.get("actions"):
            errors.append("rlm_environment entry needs non-empty rlm.actions")

    if not entry["metadata"].get("id"):
        errors.append("metadata.id (stable id) is required")
    if not entry["verification"].get("must_print"):
        errors.append("verification.must_print is required")
    return errors

# --- CELL ---
# TITLE: Load and validate all entries
error_path = data_path("errors", "qlm_v2_validation_errors.jsonl")
valid_entries, invalid = [], []

for filename in EXPECTED_FILES:
    path = ENTRIES_DIR / filename
    if not path.exists():
        invalid.append((filename, ["file not found — run the previous cell"]))
        log_error(error_path, {"file": filename, "errors": ["file not found"]})
        continue
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        invalid.append((filename, [f"invalid JSON: {exc}"]))
        log_error(error_path, {"file": filename, "errors": [f"invalid JSON: {exc}"]})
        continue
    violations = validate_qlm_v2_entry(entry)
    if violations:
        invalid.append((filename, violations))
        log_error(error_path, {"file": filename, "errors": violations})
    else:
        valid_entries.append((filename, entry))

print(f"valid: {len(valid_entries)}  invalid: {len(invalid)}")
for filename, violations in invalid:
    print(f"[INVALID] {filename}")
    for v in violations:
        print("   -", v)

# --- CELL ---
# TITLE: Summarize entries
rows = []
for filename, entry in valid_entries:
    meta = entry["metadata"]
    rows.append({
        "file": filename,
        "id": meta["id"],
        "entry_kind": entry["entry_kind"],
        "expected_verdict": entry["expected_verdict"],
        "complexity": meta["complexity"],
        "domain": meta["domain"],
        "n_flaws": len(entry.get("flaws", [])),
        "code_chars": len(entry["research_corpus"]["code_implementation"]),
        "rlm_actions": ",".join(entry.get("rlm", {}).get("actions", []) or []),
    })
summary_df = pd.DataFrame(rows)
summary_df

# --- CELL ---
# TITLE: Save normalized records and manifest
#
# Whole-artifact principle: the full entry is preserved verbatim under
# "entry" — no chunking, no field dropping. The wrapper only adds routing
# fields the downstream corpus builder needs.
normalized = []
for filename, entry in valid_entries:
    meta = entry["metadata"]
    normalized.append({
        "id": meta["id"],
        "source": "qlm_v2",
        "entry_kind": entry["entry_kind"],
        "training_mode": "deep_research",
        "domain": meta["domain"],
        "complexity": meta["complexity"],
        "expected_verdict": entry["expected_verdict"],
        "tags": meta.get("tags", []),
        "source_file": filename,
        "entry": entry,  # full, unmodified artifact
    })

out_path = data_path("processed", "qlm_v2_entries.jsonl")
write_jsonl(normalized, out_path)
print("wrote", out_path, f"({len(normalized)} records)")

manifest = {
    "total_entries": len(normalized),
    "invalid_entries": len(invalid),
    "by_entry_kind": dict(pd.Series([r["entry_kind"] for r in normalized]).value_counts().sort_index()) if normalized else {},
    "by_expected_verdict": dict(pd.Series([r["expected_verdict"] for r in normalized]).value_counts().sort_index()) if normalized else {},
    "by_complexity": {str(k): int(v) for k, v in sorted(pd.Series([r["complexity"] for r in normalized]).value_counts().items())} if normalized else {},
    "by_domain": dict(pd.Series([r["domain"] for r in normalized]).value_counts().sort_index()) if normalized else {},
}
save_manifest(manifest, data_path("manifests", "qlm_v2_manifest.json"))
print(json.dumps(manifest, indent=2))
