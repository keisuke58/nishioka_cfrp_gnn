#!/usr/bin/env bash
set -euo pipefail

# Installs/updates a weekly cron job for tools/autocommit_weekly.py (idempotent).
#
# - Uses flock to avoid concurrent runs.
# - Logs to ~/.cache/gnn-autocommit-weekly.log

ROOT="${HOME}/GNN"
PY="/usr/bin/env python3"
SCRIPT="${ROOT}/tools/autocommit_weekly.py"
LOG="${HOME}/.cache/gnn-autocommit-weekly.log"
LOCK="${HOME}/.cache/gnn-autocommit-weekly.lock"

mkdir -p "$(dirname "$LOG")"

# Run every Sunday at 03:15.
SCHEDULE="15 3 * * 0"
CMD="cd \"$ROOT\" && /usr/bin/flock -n \"$LOCK\" $PY \"$SCRIPT\" --interval-days 7 >> \"$LOG\" 2>&1"
LINE="$SCHEDULE $CMD # gnn-autocommit-weekly"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

crontab -l 2>/dev/null | grep -v 'gnn-autocommit-weekly' > "$tmp" || true
echo "$LINE" >> "$tmp"
crontab "$tmp"

echo "Installed cron entry:"
echo "$LINE"
echo "Log: $LOG"

