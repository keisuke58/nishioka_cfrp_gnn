#!/usr/bin/env python3
"""
Weekly-ish auto commit helper.

- Commits only if there are changes (tracked or untracked).
- Respects .gitignore (because it uses `git add -A`).
- Adds a simple secret-pattern scan on staged content and aborts if suspected.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from typing import Sequence


SECRET_RE = re.compile(
    r"(BEGIN( RSA)? PRIVATE KEY|AWS_SECRET|AWS_ACCESS|api[_-]?key|secret[_-]?key|"
    r"token\s*=|password\s*=)",
    re.IGNORECASE,
)


def sh(cmd: Sequence[str], *, cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def repo_root() -> str:
    p = sh(["git", "rev-parse", "--show-toplevel"])
    return p.stdout.strip()


def has_changes(root: str) -> bool:
    p = sh(["git", "status", "--porcelain=v1"], cwd=root)
    return bool(p.stdout.strip())


def last_autocommit_age_days(root: str, marker: str) -> int | None:
    # Find most recent commit containing marker in subject.
    p = sh(["git", "log", "-n", "50", "--pretty=%ct %s"], cwd=root)
    now = int(dt.datetime.now(dt.timezone.utc).timestamp())
    for line in p.stdout.splitlines():
        try:
            ts_s, subject = line.split(" ", 1)
            ts = int(ts_s)
        except ValueError:
            continue
        if marker in subject:
            return (now - ts) // (24 * 3600)
    return None


def stage_all(root: str) -> None:
    sh(["git", "add", "-A"], cwd=root)


def staged_files(root: str) -> list[str]:
    p = sh(["git", "diff", "--cached", "--name-only"], cwd=root)
    return [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]


def secret_scan(root: str, files: Sequence[str], max_bytes: int = 200_000) -> list[str]:
    flagged: list[str] = []
    for rel in files:
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as f:
                data = f.read(max_bytes)
        except OSError:
            continue
        text = data.decode("utf-8", "ignore")
        if SECRET_RE.search(text):
            flagged.append(rel)
    return flagged


def commit(root: str, message: str) -> None:
    sh(["git", "commit", "-m", message], cwd=root)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval-days", type=int, default=7)
    ap.add_argument("--marker", default="[auto-weekly]")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = repo_root()

    if not has_changes(root):
        print("No changes; skip.")
        return 0

    age = last_autocommit_age_days(root, args.marker)
    if age is not None and age < args.interval_days:
        print(f"Last auto commit is {age} days ago (< {args.interval_days}); skip.")
        return 0

    today = dt.datetime.now().strftime("%Y-%m-%d")
    msg = f"chore: weekly snapshot {today} {args.marker}"

    if args.dry_run:
        print("DRY RUN: would stage and commit with message:")
        print(msg)
        return 0

    stage_all(root)

    files = staged_files(root)
    if not files:
        print("Nothing staged after add; skip.")
        return 0

    flagged = secret_scan(root, files)
    if flagged:
        print("ABORT: suspected secrets in staged files:")
        for f in flagged[:50]:
            print(" -", f)
        print("Fix/ignore them, then retry.")
        return 2

    commit(root, msg)
    print("Committed:", msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

