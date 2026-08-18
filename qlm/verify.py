"""Execution-based verification of dataset entries.

Gate #2 of the curation pipeline (the most important one): every entry's
``research_corpus.code_implementation`` is executed in an isolated
subprocess. An entry is only admitted to the training corpus if its code:

  1. exits with code 0 within the timeout,
  2. prints every substring listed in ``verification.must_print``,
  3. produces no NaN/Inf tokens in stdout (unless explicitly allowed),
  4. imports only packages available in the sandbox.

Rationale: for a 100-150M parameter student model, a single training
example with silently broken code (e.g. ``np.log(0)`` NaNs propagating
through a Sharpe ratio) teaches a *plausible-looking but wrong* pattern.
Execution is the only ground truth we accept.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Word-boundary match so "nan" inside e.g. "finance" never false-positives.
_NAN_RE = re.compile(r"(?<![A-Za-z0-9_])(nan|NaN|NAN|inf|Inf|INF|-inf)(?![A-Za-z0-9_])")

DEFAULT_TIMEOUT_S = 120


@dataclass
class VerifyReport:
    """Outcome of executing one entry's code_implementation."""

    path: Path
    ok: bool
    exit_code: int | None = None
    duration_s: float | None = None
    failures: list[str] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""


def _tail(text: str, n_lines: int = 20) -> str:
    lines = text.rstrip().splitlines()
    return "\n".join(lines[-n_lines:])


def verify_entry(entry: dict[str, Any], path: Path, python: str = sys.executable) -> VerifyReport:
    """Run the entry's code in a subprocess and enforce the verification contract."""
    import time

    code = entry["research_corpus"]["code_implementation"]
    contract = entry.get("verification", {})
    timeout_s = int(contract.get("timeout_seconds", DEFAULT_TIMEOUT_S))
    must_print: list[str] = list(contract.get("must_print", []))
    forbid_nan = bool(contract.get("forbid_nan_in_stdout", True))

    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="qlm_verify_") as tmp:
        script = Path(tmp) / "entry_code.py"
        script.write_text(code, encoding="utf-8")
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [python, str(script)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,  # entries must not depend on repo-relative paths
            )
        except subprocess.TimeoutExpired:
            return VerifyReport(
                path=path,
                ok=False,
                exit_code=None,
                duration_s=time.monotonic() - t0,
                failures=[f"timeout after {timeout_s}s"],
            )
        duration = time.monotonic() - t0

    if proc.returncode != 0:
        failures.append(f"non-zero exit code {proc.returncode}")

    for needle in must_print:
        if needle not in proc.stdout:
            failures.append(f"missing required stdout substring: {needle!r}")

    if forbid_nan:
        hit = _NAN_RE.search(proc.stdout)
        if hit:
            failures.append(f"NaN/Inf token in stdout: {hit.group(0)!r}")

    return VerifyReport(
        path=path,
        ok=not failures,
        exit_code=proc.returncode,
        duration_s=duration,
        failures=failures,
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
    )
