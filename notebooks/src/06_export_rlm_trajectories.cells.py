# TITLE: Chousorus 1 dataset pipeline — 06_export_rlm_trajectories
#
# Purpose: convert QLM v2 entries into RLM-style training trajectories
# (user -> thought -> tool_call -> observation -> thought -> final).
#
# IMPORTANT CONSTRAINTS honored here:
#  - The Python inside each entry is NOT executed by this notebook.
#  - Observations come from real stdout logs under data/logs/<entry_id>.txt
#    when present; otherwise a clearly marked TEMPLATE observation is used
#    and tagged observation_source=template_not_executed in metadata.
#  - RLM trajectories are built ONLY from QLM deep entries — simple QA
#    sources stay SFT and are never given fake tool-call traces.
#  - No numeric results are fabricated beyond what the entry itself states.

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
from qlm_utils import data_path, read_jsonl, write_jsonl, save_manifest, count_by, require_file

qlm_path = require_file(
    data_path("processed", "qlm_v2_entries.jsonl"),
    "Run notebooks/01_load_qlm_v2_entries.ipynb first.",
)
qlm_records, _ = read_jsonl(qlm_path)
print(f"loaded {len(qlm_records)} QLM records")

LOGS_DIR = data_path("logs")
DEFAULT_LOSS = {  # used only if an entry lacks training_sequence.loss_masking
    "user": 0.0, "environment_observation": 0.0,
    "assistant_thought": 1.0, "tool_call": 1.0, "final_answer": 1.0,
}

# --- CELL ---
# TITLE: (OPTIONAL) Generate REAL observation logs by executing entry code
# OPTIONAL: RUN LATER IN COLAB
#
# The QLM entries are execution-verified upstream, and their code is
# self-contained, seeded, and network-free. If you flip RUN_ENTRY_CODE to
# True and run this cell IN COLAB, each entry's code runs in a subprocess
# and its real stdout is saved to data/logs/<entry_id>.txt — the trajectory
# builder below will then use real observations instead of templates.
# Leave False to build template-observation trajectories without executing
# anything.
RUN_ENTRY_CODE = False

if RUN_ENTRY_CODE:
    import subprocess
    import tempfile
    for rec in qlm_records:
        entry_id = rec["id"]
        code = rec["entry"]["research_corpus"]["code_implementation"]
        timeout_s = int(rec["entry"].get("verification", {}).get("timeout_seconds", 120))
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            script = fh.name
        try:
            proc = subprocess.run([sys.executable, script], capture_output=True,
                                  text=True, timeout=timeout_s)
            if proc.returncode == 0:
                (LOGS_DIR / f"{entry_id}.txt").write_text(proc.stdout, encoding="utf-8")
                print(f"[ok]   {entry_id}: real log saved")
            else:
                print(f"[fail] {entry_id}: exit {proc.returncode}; no log written")
        except subprocess.TimeoutExpired:
            print(f"[fail] {entry_id}: timeout after {timeout_s}s; no log written")
else:
    print("RUN_ENTRY_CODE=False — trajectories will use template observations "
          "where no real log exists under", LOGS_DIR)

# --- CELL ---
# TITLE: Observation builder (real log preferred, marked template otherwise)
def build_observation(rec: dict) -> tuple:
    """Return (observation_text, observation_source)."""
    entry = rec["entry"]
    entry_id = rec["id"]
    log_path = LOGS_DIR / f"{entry_id}.txt"
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8").strip()
        if text:
            return text, "real_stdout_log"

    # Template path: use ONLY facts the entry itself declares (must_print
    # strings, expected verdict, flaw types, rlm metadata). Nothing numeric
    # is invented.
    lines = ["[OBSERVATION TEMPLATE — NOT EXECUTED IN THIS NOTEBOOK]", "RESULTS"]
    if rec["entry_kind"] == "adversarial":
        for flaw in entry.get("flaws", []):
            lines.append(f"flaw_type={flaw['type'].upper()}")
        lines.append(f"audit_verdict={rec['expected_verdict']}")
    if rec["entry_kind"] == "rlm_environment":
        rlm = entry.get("rlm", {})
        lines.append(f"environment_class={rlm.get('environment_class', '')}")
        lines.append(f"actions_executed={','.join(rlm.get('actions', []))}")
    lines.append(f"verdict={rec['expected_verdict']}")
    return "\n".join(lines), "template_not_executed"

# --- CELL ---
# TITLE: Trajectory builder
def summarize(text: str, max_chars: int = 1200) -> str:
    """Trim a long critique for the second thought turn WITHOUT rewriting it.

    We cut at a sentence boundary and note the trim — never silently."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(". ")
    if last_period > 200:
        cut = cut[: last_period + 1]
    return cut + " [trimmed for trajectory; full critique retained in source entry]"


def build_trajectory(rec: dict) -> dict:
    entry = rec["entry"]
    meta = entry["metadata"]
    loss = dict(DEFAULT_LOSS)
    loss.update(entry.get("training_sequence", {}).get("loss_masking", {}) or {})

    user_turn = (
        f"Investigate the following quantitative research task: {meta['domain']}. "
        f"Tags: {', '.join(meta.get('tags', []))}. "
        f"Entry kind: {rec['entry_kind']}. "
        "Produce an execution-verifiable analysis and a calibrated conclusion."
    )

    tool_call_payload = json.dumps(
        {"tool": "python_repl", "code": entry["research_corpus"]["code_implementation"]},
        ensure_ascii=False,  # json.dumps handles newline escaping correctly
    )

    observation, obs_source = build_observation(rec)

    final_parts = [
        f"Calibrated conclusion: {rec['expected_verdict']}.",
        "Operational protocol followed: " + entry["agent_instructions"],
        "Limitations: " + entry["adversarial_critique"]["limitations"],
    ]

    turns = [
        {"role": "user", "content": user_turn, "loss_weight": loss["user"]},
        {"role": "assistant_thought",
         "content": entry["agent_thought_process"]["initial_analysis"],
         "loss_weight": loss["assistant_thought"]},
        {"role": "tool_call", "content": tool_call_payload,
         "loss_weight": loss["tool_call"]},
        {"role": "environment_observation", "content": observation,
         "loss_weight": loss["environment_observation"]},
        {"role": "assistant_thought",
         "content": summarize(entry["adversarial_critique"]["potential_pitfalls"]),
         "loss_weight": loss["assistant_thought"]},
        {"role": "assistant_final", "content": "\n\n".join(final_parts),
         "loss_weight": loss["final_answer"]},
    ]

    return {
        "id": f"{rec['id']}-traj",
        "source_entry_id": rec["id"],
        "source": "qlm_v2",
        "entry_kind": rec["entry_kind"],
        "training_mode": "rlm_trajectory",
        "turns": turns,
        "metadata": {
            "domain": meta["domain"],
            "complexity": meta["complexity"],
            "expected_verdict": rec["expected_verdict"],
            "observation_source": obs_source,
            "flaw_types": [f["type"] for f in entry.get("flaws", [])],
            "rlm_environment_class": entry.get("rlm", {}).get("environment_class"),
            "trajectory_style": entry.get("training_sequence", {}).get("style", "tool_use"),
        },
    }

# --- CELL ---
# TITLE: Build and validate all trajectories
trajectories = [build_trajectory(rec) for rec in qlm_records]

VALID_ROLES = {"user", "assistant_thought", "tool_call",
               "environment_observation", "assistant_final"}
for traj in trajectories:
    assert traj["turns"][0]["role"] == "user"
    assert traj["turns"][-1]["role"] == "assistant_final"
    for turn in traj["turns"]:
        assert turn["role"] in VALID_ROLES, f"bad role {turn['role']}"
        assert isinstance(turn["loss_weight"], (int, float))
        assert turn["content"].strip(), f"{traj['id']}: empty turn content"
    # the tool_call payload must be valid JSON with intact code
    payload = json.loads([t for t in traj["turns"] if t["role"] == "tool_call"][0]["content"])
    assert payload["tool"] == "python_repl" and len(payload["code"]) > 200
print(f"built and validated {len(trajectories)} trajectories")

# --- CELL ---
# TITLE: Write trajectories and manifest
out_path = data_path("trajectories", "qlm_rlm_trajectories.jsonl")
write_jsonl(trajectories, out_path)
print("wrote", out_path)

manifest = {
    "total_trajectories": len(trajectories),
    "by_entry_kind": count_by(trajectories, lambda t: t["entry_kind"]),
    "by_observation_source": count_by(
        trajectories, lambda t: t["metadata"]["observation_source"]),
    "by_trajectory_style": count_by(
        trajectories, lambda t: t["metadata"]["trajectory_style"]),
    "turns_per_trajectory": count_by(trajectories, lambda t: len(t["turns"])),
    "total_chars": sum(len(turn["content"]) for t in trajectories for turn in t["turns"]),
    "note": ("template observations are placeholders; run the optional "
             "execution cell in Colab to replace them with real stdout logs"),
}
save_manifest(manifest, data_path("manifests", "rlm_trajectory_manifest.json"))

pd.DataFrame([{
    "id": t["id"], "entry_kind": t["entry_kind"],
    "observation_source": t["metadata"]["observation_source"],
    "n_turns": len(t["turns"]),
    "total_chars": sum(len(turn["content"]) for turn in t["turns"]),
} for t in trajectories])
