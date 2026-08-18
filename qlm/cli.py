"""QLM 1 pipeline CLI.

Usage:
    python -m qlm.cli validate [--entries DIR]
    python -m qlm.cli verify   [--entries DIR]
    python -m qlm.cli build    [--entries DIR] [--out DIR]   # validate + verify + curriculum

``build`` is the full curation gate: only entries that pass BOTH schema
validation and execution verification are admitted to the curriculum.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qlm import curriculum, schema, verify

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENTRIES = REPO_ROOT / "data" / "entries"
DEFAULT_OUT = REPO_ROOT / "dist"


def cmd_validate(entries_dir: Path) -> int:
    files = schema.iter_entry_files(entries_dir)
    if not files:
        print(f"no entry files found in {entries_dir}", file=sys.stderr)
        return 1
    loaded_schema = schema.load_schema()
    failed = 0
    for path in files:
        report = schema.validate_file(path, loaded_schema)
        status = "PASS" if report.ok else "FAIL"
        print(f"[{status}] {path.name}")
        for err in report.errors:
            print(f"    - {err}")
        failed += 0 if report.ok else 1
    print(f"\nvalidate: {len(files) - failed}/{len(files)} entries passed schema validation")
    return 1 if failed else 0


def cmd_verify(entries_dir: Path) -> int:
    files = schema.iter_entry_files(entries_dir)
    if not files:
        print(f"no entry files found in {entries_dir}", file=sys.stderr)
        return 1
    failed = 0
    for path in files:
        entry = schema.load_entry(path)
        report = verify.verify_entry(entry, path)
        status = "PASS" if report.ok else "FAIL"
        dur = f"{report.duration_s:.1f}s" if report.duration_s is not None else "?"
        print(f"[{status}] {path.name} (exit={report.exit_code}, {dur})")
        for failure in report.failures:
            print(f"    - {failure}")
        if not report.ok and report.stderr_tail:
            print("    stderr tail:")
            for line in report.stderr_tail.splitlines():
                print(f"    | {line}")
        failed += 0 if report.ok else 1
    print(f"\nverify: {len(files) - failed}/{len(files)} entries passed execution verification")
    return 1 if failed else 0


def cmd_build(entries_dir: Path, out_dir: Path) -> int:
    files = schema.iter_entry_files(entries_dir)
    if not files:
        print(f"no entry files found in {entries_dir}", file=sys.stderr)
        return 1
    loaded_schema = schema.load_schema()
    admitted = []
    rejected = 0
    for path in files:
        vreport = schema.validate_file(path, loaded_schema)
        if not vreport.ok:
            rejected += 1
            print(f"[REJECT schema] {path.name}: {'; '.join(vreport.errors)}")
            continue
        entry = schema.load_entry(path)
        xreport = verify.verify_entry(entry, path)
        if not xreport.ok:
            rejected += 1
            print(f"[REJECT exec]   {path.name}: {'; '.join(xreport.failures)}")
            continue
        print(f"[ADMIT]         {path.name} (complexity {entry['metadata']['complexity']})")
        admitted.append(entry)

    if not admitted:
        print("build: no entries admitted; curriculum not written", file=sys.stderr)
        return 1

    creport = curriculum.build_curriculum(admitted, out_dir)
    print(f"\nbuild: {creport.total} admitted, {rejected} rejected")
    for stage, count in creport.per_stage.items():
        print(f"  {stage}: {count} entries")
    print(f"  wrote curriculum to {creport.output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qlm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "verify", "build"):
        p = sub.add_parser(name)
        p.add_argument("--entries", type=Path, default=DEFAULT_ENTRIES)
        if name == "build":
            p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.command == "validate":
        return cmd_validate(args.entries)
    if args.command == "verify":
        return cmd_verify(args.entries)
    return cmd_build(args.entries, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
