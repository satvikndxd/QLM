"""Smoke tests for the QLM 1 curation pipeline.

Run with:  python -m pytest tests/ -q   (or plain: python tests/test_pipeline.py)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qlm import curriculum, schema, verify  # noqa: E402

ENTRIES_DIR = REPO_ROOT / "data" / "entries"


def _minimal_valid_entry() -> dict:
    pad = "x" * 200
    return {
        "schema_version": 2,
        "entry_kind": "correct",
        "expected_verdict": "FAIL_TO_REJECT_H0",
        "flaws": [],
        "static_checks": {"require_vectorized": True, "forbid_pandas_row_loops": True},
        "rlm": {
            "environment_class": None,
            "actions": [],
            "max_recursion_depth": 1,
            "tool_timeout_seconds": 120,
        },
        "training_sequence": {
            "style": "tool_use",
            "include_adversarial_critique": True,
            "include_final_calibrated_answer": True,
            "loss_masking": {
                "user": 0.0,
                "environment_observation": 0.0,
                "assistant_thought": 1.0,
                "tool_call": 1.0,
                "final_answer": 1.0,
            },
        },
        "metadata": {"id": "qlm1-test", "domain": "Testing Domain", "complexity": 1, "tags": ["test"]},
        "agent_thought_process": {
            "initial_analysis": pad,
            "tool_selection": pad,
            "recursive_delegation": pad,
        },
        "research_corpus": {
            "hypothesis_formulation": pad,
            "data_engineering": pad,
            "methodology_justification": pad,
            "code_implementation": "print('RESULTS')\nprint('value=1.0')\n" + "# pad\n" * 80,
            "statistical_validation": pad,
            "risk_and_backtest_audit": pad,
        },
        "adversarial_critique": {
            "potential_pitfalls": pad,
            "falsification_strategy": pad,
            "limitations": pad,
        },
        "agent_instructions": pad,
        "verification": {"must_print": ["RESULTS"], "forbid_nan_in_stdout": True},
    }


def test_schema_accepts_valid_entry():
    assert schema.validate_entry(_minimal_valid_entry()) == []


def test_schema_rejects_bad_complexity():
    entry = _minimal_valid_entry()
    entry["metadata"]["complexity"] = 11
    errors = schema.validate_entry(entry)
    assert any("complexity" in e for e in errors)


def test_schema_rejects_missing_section():
    entry = _minimal_valid_entry()
    del entry["adversarial_critique"]
    assert schema.validate_entry(entry)


def _example_flaw() -> dict:
    return {
        "type": "lookahead_bias",
        "severity": "fatal",
        "location": "code_implementation:backtest",
        "description": "Signal computed on bar t close is traded on bar t (missing .shift(1)).",
        "detection": "Re-run with a one-bar execution lag; Sharpe collapse exposes the leak.",
        "corrective_action": "Apply position = signal.shift(1) before computing strategy returns.",
    }


def test_schema_v2_accepts_adversarial_with_flaws():
    entry = _minimal_valid_entry()
    entry["entry_kind"] = "adversarial"
    entry["expected_verdict"] = "REJECTED_LOOKAHEAD_BIAS"
    entry["flaws"] = [_example_flaw()]
    assert schema.validate_entry(entry) == []


def test_schema_v2_rejects_adversarial_missing_flaws():
    entry = _minimal_valid_entry()
    entry["entry_kind"] = "adversarial"
    entry["expected_verdict"] = "REJECTED_LOOKAHEAD_BIAS"
    del entry["flaws"]
    assert schema.validate_entry(entry)


def test_schema_v2_rejects_adversarial_empty_flaws():
    entry = _minimal_valid_entry()
    entry["entry_kind"] = "adversarial"
    entry["expected_verdict"] = "REJECTED_LOOKAHEAD_BIAS"
    entry["flaws"] = []
    assert schema.validate_entry(entry)


def test_schema_v2_rejects_invalid_entry_kind():
    entry = _minimal_valid_entry()
    entry["entry_kind"] = "speculative"
    assert schema.validate_entry(entry)


def test_schema_v2_rejects_rlm_environment_null_class():
    entry = _minimal_valid_entry()
    entry["entry_kind"] = "rlm_environment"
    entry["rlm"] = {
        "environment_class": None,
        "actions": ["reset"],
        "max_recursion_depth": 3,
        "tool_timeout_seconds": 120,
    }
    assert schema.validate_entry(entry)


def test_schema_v2_rejects_rlm_environment_empty_actions():
    entry = _minimal_valid_entry()
    entry["entry_kind"] = "rlm_environment"
    entry["rlm"] = {
        "environment_class": "QuantEnvironment",
        "actions": [],
        "max_recursion_depth": 3,
        "tool_timeout_seconds": 120,
    }
    assert schema.validate_entry(entry)


def test_schema_v2_accepts_rlm_environment_entry():
    entry = _minimal_valid_entry()
    entry["entry_kind"] = "rlm_environment"
    entry["rlm"] = {
        "environment_class": "QuantEnvironment",
        "actions": ["reset", "fit_train", "forecast_oos"],
        "max_recursion_depth": 3,
        "tool_timeout_seconds": 120,
    }
    del entry["expected_verdict"]  # optional for rlm_environment
    assert schema.validate_entry(entry) == []


def test_schema_v2_rejects_correct_with_declared_flaws():
    entry = _minimal_valid_entry()
    entry["flaws"] = [_example_flaw()]
    assert schema.validate_entry(entry)


def test_verifier_passes_clean_code():
    entry = _minimal_valid_entry()
    report = verify.verify_entry(entry, Path("mem://test"))
    assert report.ok, report.failures


def test_verifier_rejects_nan_output():
    entry = _minimal_valid_entry()
    entry["research_corpus"]["code_implementation"] = "print('RESULTS')\nprint(float('nan'))\n"
    report = verify.verify_entry(entry, Path("mem://test"))
    assert not report.ok
    assert any("NaN" in f for f in report.failures)


def test_verifier_rejects_crash():
    entry = _minimal_valid_entry()
    entry["research_corpus"]["code_implementation"] = "raise RuntimeError('boom')\n"
    report = verify.verify_entry(entry, Path("mem://test"))
    assert not report.ok
    assert any("exit code" in f for f in report.failures)


def test_verifier_nan_regex_no_false_positive_on_words():
    # 'finance' contains 'nan'; 'information' contains 'inf' — neither may trip.
    entry = _minimal_valid_entry()
    entry["research_corpus"]["code_implementation"] = (
        "print('RESULTS')\nprint('finance information infrastructure')\n"
    )
    report = verify.verify_entry(entry, Path("mem://test"))
    assert report.ok, report.failures


def test_curriculum_ordering_and_shards():
    entries = []
    for cx in (8, 2, 5):
        e = _minimal_valid_entry()
        e["metadata"]["complexity"] = cx
        e["metadata"]["id"] = f"qlm1-test-{cx}"
        entries.append(e)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        report = curriculum.build_curriculum(entries, out)
        assert report.total == 3
        ordered = [
            json.loads(line)["metadata"]["complexity"]
            for line in (out / "curriculum.jsonl").read_text().splitlines()
        ]
        assert ordered == sorted(ordered)
        assert report.per_stage == {
            "stage1_foundational": 1,
            "stage2_intermediate": 1,
            "stage3_research": 1,
        }


def test_seed_entries_validate():
    files = schema.iter_entry_files(ENTRIES_DIR)
    assert len(files) >= 3
    loaded = schema.load_schema()
    for path in files:
        report = schema.validate_file(path, loaded)
        assert report.ok, f"{path.name}: {report.errors}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    raise SystemExit(1 if failures else 0)
