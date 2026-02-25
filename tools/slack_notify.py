#!/usr/bin/env python3
"""
Slack通知ユーティリティ

Usage:
    python tools/slack_notify.py --webhook_url <URL> --message "Training completed"
    python tools/slack_notify.py --webhook_url <URL> --run_dir runs/20250115_abc123
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import urllib.request
import urllib.parse


def send_slack_message(webhook_url: str, message: str, username: str = "GNN Training Bot", 
                      icon_emoji: str = ":robot_face:") -> bool:
    """Slackにメッセージを送信"""
    payload = {
        "text": message,
        "username": username,
        "icon_emoji": icon_emoji,
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(webhook_url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        print(f"[ERR] Failed to send Slack notification: {e}", file=sys.stderr)
        return False


def format_training_summary(run_dir: Path) -> str:
    """トレーニング結果のサマリーをフォーマット"""
    summary_path = run_dir / "meta" / "summary.json"
    metrics_path = run_dir / "metrics.json"
    
    lines = []
    lines.append(f"*Training Completed: {run_dir.name}*")
    lines.append("")
    
    # Summary情報
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        best_f1 = summary.get('best_macro_f1')
        exit_code = summary.get('exit_code', 'N/A')
        start_time = summary.get('start_time', 'N/A')
        end_time = summary.get('end_time', 'N/A')
        
        lines.append(f"*Status:* {'✅ Success' if exit_code == 0 else '❌ Failed (exit_code: ' + str(exit_code) + ')'}")
        if best_f1 is not None:
            lines.append(f"*Best Macro F1:* {best_f1:.6f}")
        lines.append(f"*Start Time:* {start_time}")
        lines.append(f"*End Time:* {end_time}")
    
    # Metrics情報
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        test_metrics = metrics.get('test', {})
        if test_metrics:
            lines.append("")
            lines.append("*Test Metrics:*")
            if test_metrics.get('accuracy') is not None:
                lines.append(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            if test_metrics.get('macro_f1') is not None:
                lines.append(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            if test_metrics.get('weighted_f1') is not None:
                lines.append(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
    
    lines.append("")
    lines.append(f"*Run Directory:* `{run_dir}`")
    
    return "\n".join(lines)


def format_sweep_summary(sweep_csv: Path) -> str:
    """スイープ結果のサマリーをフォーマット"""
    import pandas as pd
    
    try:
        df = pd.read_csv(sweep_csv)
        successful = df[df['exit_code'] == 0] if 'exit_code' in df.columns else df
        
        lines = []
        lines.append(f"*Hyperparameter Sweep Completed*")
        lines.append("")
        lines.append(f"*Total Runs:* {len(df)}")
        lines.append(f"*Successful:* {len(successful)}")
        lines.append(f"*Failed:* {len(df) - len(successful)}")
        
        if not successful.empty and 'best_macro_f1' in successful.columns:
            best_idx = successful['best_macro_f1'].idxmax()
            best = successful.loc[best_idx]
            lines.append("")
            lines.append("*Best Model:*")
            lines.append(f"  Run ID: {best.get('run_id', 'N/A')}")
            lines.append(f"  Learning Rate: {best.get('lr', 'N/A')}")
            lines.append(f"  Best Macro F1: {best.get('best_macro_f1', 'N/A'):.6f}")
        
        lines.append("")
        lines.append(f"*Sweep CSV:* `{sweep_csv}`")
        
        return "\n".join(lines)
    except Exception as e:
        return f"*Sweep Completed*\n\nError reading sweep results: {e}\n\nCSV: `{sweep_csv}`"


def main() -> int:
    parser = argparse.ArgumentParser(description="Send Slack notification")
    parser.add_argument('--webhook_url', type=str, 
                       default=os.environ.get('SLACK_WEBHOOK_URL'),
                       help='Slack webhook URL (or set SLACK_WEBHOOK_URL env var)')
    parser.add_argument('--message', type=str, help='Simple text message to send')
    parser.add_argument('--run_dir', type=str, help='Run directory to summarize')
    parser.add_argument('--sweep_csv', type=str, help='Sweep CSV file to summarize')
    parser.add_argument('--username', type=str, default='GNN Training Bot',
                       help='Slack username')
    parser.add_argument('--icon', type=str, default=':robot_face:',
                       help='Slack icon emoji')
    
    args = parser.parse_args()
    
    if not args.webhook_url:
        print("[ERR] Slack webhook URL is required. Set SLACK_WEBHOOK_URL env var or use --webhook_url", file=sys.stderr)
        return 1
    
    # メッセージの準備
    if args.message:
        message = args.message
    elif args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.exists():
            print(f"[ERR] Run directory not found: {run_dir}", file=sys.stderr)
            return 1
        message = format_training_summary(run_dir)
    elif args.sweep_csv:
        sweep_csv = Path(args.sweep_csv).expanduser().resolve()
        if not sweep_csv.exists():
            print(f"[ERR] Sweep CSV not found: {sweep_csv}", file=sys.stderr)
            return 1
        message = format_sweep_summary(sweep_csv)
    else:
        parser.error("Either --message, --run_dir, or --sweep_csv must be provided")
    
    # 送信
    success = send_slack_message(args.webhook_url, message, args.username, args.icon)
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
