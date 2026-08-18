"""Schema validation for QLM 1 dataset entries.

Structural gate #1 of the curation pipeline: an entry that does not
validate against ``schema/entry.schema.json`` never reaches the
execution verifier or the curriculum builder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "schema" / "entry.schema.json"


@dataclass
class ValidationReport:
    """Outcome of validating one entry file."""

    path: Path
    ok: bool
    errors: list[str] = field(default_factory=list)


def load_schema(schema_path: Path = SCHEMA_PATH) -> dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_entry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_entry(entry: dict[str, Any], schema: dict[str, Any] | None = None) -> list[str]:
    """Return a list of human-readable schema violations (empty == valid)."""
    schema = schema or load_schema()
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(entry), key=lambda e: list(e.absolute_path)):
        loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{loc}: {err.message}")
    return errors


def validate_file(path: Path, schema: dict[str, Any] | None = None) -> ValidationReport:
    try:
        entry = load_entry(path)
    except json.JSONDecodeError as exc:
        return ValidationReport(path=path, ok=False, errors=[f"invalid JSON: {exc}"])
    errors = validate_entry(entry, schema)
    return ValidationReport(path=path, ok=not errors, errors=errors)


def iter_entry_files(entries_dir: Path) -> list[Path]:
    return sorted(p for p in entries_dir.glob("*.json") if p.is_file())
