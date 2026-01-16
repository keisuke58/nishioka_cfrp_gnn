## Weekly auto-commit (local)

This adds a "weekly snapshot" commit **only when there are changes**, and only if the previous auto-commit is at least 7 days old.

### Enable with systemd (recommended on Linux)

```bash
mkdir -p ~/.config/systemd/user
cp tools/autocommit_weekly.systemd/gnn-autocommit-weekly.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gnn-autocommit-weekly.timer
systemctl --user list-timers | grep gnn-autocommit-weekly || true
```

### Dry run

```bash
python3 tools/autocommit_weekly.py --dry-run
```

### Notes

- This runs locally on your machine (not GitHub Actions).
- It will **not push**. Push manually when you want.
- If it detects secret-like patterns, it aborts and prints the files.

