#!/usr/bin/env python3
"""
Gmail通知ユーティリティ

Usage:
    python tools/email_notify.py --to user@example.com --subject "Training completed" --message "Training finished"
    python tools/email_notify.py --to user@example.com --run_dir runs/20250115_abc123
"""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


def send_email(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    to_email: str,
    subject: str,
    body: str,
    is_html: bool = False
) -> bool:
    """メールを送信"""
    try:
        # メッセージの作成
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # 本文の追加
        if is_html:
            msg.attach(MIMEText(body, 'html'))
        else:
            msg.attach(MIMEText(body, 'plain'))
        
        # SMTPサーバーに接続して送信
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"[ERR] Failed to send email: {e}", file=sys.stderr)
        return False


def format_training_summary_html(run_dir: Path) -> str:
    """トレーニング結果のサマリーをHTML形式でフォーマット"""
    summary_path = run_dir / "meta" / "summary.json"
    metrics_path = run_dir / "metrics.json"
    
    html = []
    html.append("<html><body>")
    html.append(f"<h2>Training Completed: {run_dir.name}</h2>")
    html.append("<hr>")
    
    # Summary情報
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        best_f1 = summary.get('best_macro_f1')
        exit_code = summary.get('exit_code', 'N/A')
        start_time = summary.get('start_time', 'N/A')
        end_time = summary.get('end_time', 'N/A')
        
        status_color = "green" if exit_code == 0 else "red"
        status_text = "✅ Success" if exit_code == 0 else f"❌ Failed (exit_code: {exit_code})"
        
        html.append("<h3>Status</h3>")
        html.append(f"<p style='color: {status_color}; font-weight: bold;'>{status_text}</p>")
        
        html.append("<h3>Training Information</h3>")
        html.append("<table border='1' cellpadding='5' cellspacing='0'>")
        if best_f1 is not None:
            html.append(f"<tr><td><strong>Best Macro F1</strong></td><td>{best_f1:.6f}</td></tr>")
        html.append(f"<tr><td><strong>Start Time</strong></td><td>{start_time}</td></tr>")
        html.append(f"<tr><td><strong>End Time</strong></td><td>{end_time}</td></tr>")
        html.append("</table>")
    
    # Metrics情報
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        test_metrics = metrics.get('test', {})
        if test_metrics:
            html.append("<h3>Test Metrics</h3>")
            html.append("<table border='1' cellpadding='5' cellspacing='0'>")
            if test_metrics.get('accuracy') is not None:
                html.append(f"<tr><td><strong>Accuracy</strong></td><td>{test_metrics['accuracy']:.4f}</td></tr>")
            if test_metrics.get('macro_f1') is not None:
                html.append(f"<tr><td><strong>Macro F1</strong></td><td>{test_metrics['macro_f1']:.4f}</td></tr>")
            if test_metrics.get('weighted_f1') is not None:
                html.append(f"<tr><td><strong>Weighted F1</strong></td><td>{test_metrics['weighted_f1']:.4f}</td></tr>")
            if test_metrics.get('balanced_accuracy') is not None:
                html.append(f"<tr><td><strong>Balanced Accuracy</strong></td><td>{test_metrics['balanced_accuracy']:.4f}</td></tr>")
            html.append("</table>")
    
    html.append("<h3>Run Directory</h3>")
    html.append(f"<p><code>{run_dir}</code></p>")
    html.append("</body></html>")
    
    return "\n".join(html)


def format_training_summary_text(run_dir: Path) -> str:
    """トレーニング結果のサマリーをテキスト形式でフォーマット"""
    summary_path = run_dir / "meta" / "summary.json"
    metrics_path = run_dir / "metrics.json"
    
    lines = []
    lines.append(f"Training Completed: {run_dir.name}")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary情報
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        best_f1 = summary.get('best_macro_f1')
        exit_code = summary.get('exit_code', 'N/A')
        start_time = summary.get('start_time', 'N/A')
        end_time = summary.get('end_time', 'N/A')
        
        status_text = "✅ Success" if exit_code == 0 else f"❌ Failed (exit_code: {exit_code})"
        lines.append(f"Status: {status_text}")
        lines.append("")
        lines.append("Training Information:")
        if best_f1 is not None:
            lines.append(f"  Best Macro F1: {best_f1:.6f}")
        lines.append(f"  Start Time: {start_time}")
        lines.append(f"  End Time: {end_time}")
        lines.append("")
    
    # Metrics情報
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        test_metrics = metrics.get('test', {})
        if test_metrics:
            lines.append("Test Metrics:")
            if test_metrics.get('accuracy') is not None:
                lines.append(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            if test_metrics.get('macro_f1') is not None:
                lines.append(f"  Macro F1: {test_metrics['macro_f1']:.4f}")
            if test_metrics.get('weighted_f1') is not None:
                lines.append(f"  Weighted F1: {test_metrics['weighted_f1']:.4f}")
            if test_metrics.get('balanced_accuracy') is not None:
                lines.append(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
            lines.append("")
    
    lines.append(f"Run Directory: {run_dir}")
    
    return "\n".join(lines)


def format_sweep_summary_text(sweep_csv: Path) -> str:
    """スイープ結果のサマリーをフォーマット"""
    try:
        import pandas as pd
        
        df = pd.read_csv(sweep_csv)
        successful = df[df['exit_code'] == 0] if 'exit_code' in df.columns else df
        
        lines = []
        lines.append("Hyperparameter Sweep Completed")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total Runs: {len(df)}")
        lines.append(f"Successful: {len(successful)}")
        lines.append(f"Failed: {len(df) - len(successful)}")
        lines.append("")
        
        if not successful.empty and 'best_macro_f1' in successful.columns:
            best_idx = successful['best_macro_f1'].idxmax()
            best = successful.loc[best_idx]
            lines.append("Best Model:")
            lines.append(f"  Run ID: {best.get('run_id', 'N/A')}")
            lines.append(f"  Learning Rate: {best.get('lr', 'N/A')}")
            lines.append(f"  Best Macro F1: {best.get('best_macro_f1', 'N/A'):.6f}")
            lines.append("")
        
        lines.append(f"Sweep CSV: {sweep_csv}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"Sweep Completed\n\nError reading sweep results: {e}\n\nCSV: {sweep_csv}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Send email notification")
    parser.add_argument('--to', type=str, 
                       default=os.environ.get('GMAIL_TO'),
                       help='Recipient email address (or set GMAIL_TO env var)')
    parser.add_argument('--from_email', type=str,
                       default=os.environ.get('GMAIL_FROM'),
                       help='Sender email address (or set GMAIL_FROM env var)')
    parser.add_argument('--password', type=str,
                       default=os.environ.get('GMAIL_PASSWORD'),
                       help='Gmail app password (or set GMAIL_PASSWORD env var)')
    parser.add_argument('--smtp_server', type=str, default='smtp.gmail.com',
                       help='SMTP server (default: smtp.gmail.com)')
    parser.add_argument('--smtp_port', type=int, default=587,
                       help='SMTP port (default: 587)')
    parser.add_argument('--subject', type=str, help='Email subject')
    parser.add_argument('--message', type=str, help='Simple text message to send')
    parser.add_argument('--run_dir', type=str, help='Run directory to summarize')
    parser.add_argument('--sweep_csv', type=str, help='Sweep CSV file to summarize')
    parser.add_argument('--html', action='store_true', help='Send as HTML email')
    
    args = parser.parse_args()
    
    # 必須パラメータのチェック
    if not args.to:
        print("[ERR] Recipient email is required. Set GMAIL_TO env var or use --to", file=sys.stderr)
        return 1
    
    if not args.from_email:
        print("[ERR] Sender email is required. Set GMAIL_FROM env var or use --from_email", file=sys.stderr)
        return 1
    
    if not args.password:
        print("[ERR] Gmail password is required. Set GMAIL_PASSWORD env var or use --password", file=sys.stderr)
        print("[INFO] Note: Use Gmail App Password, not your regular password", file=sys.stderr)
        return 1
    
    # メッセージの準備
    if args.message:
        body = args.message
        subject = args.subject or "GNN Training Notification"
    elif args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.exists():
            print(f"[ERR] Run directory not found: {run_dir}", file=sys.stderr)
            return 1
        if args.html:
            body = format_training_summary_html(run_dir)
        else:
            body = format_training_summary_text(run_dir)
        subject = args.subject or f"GNN Training Completed: {run_dir.name}"
    elif args.sweep_csv:
        sweep_csv = Path(args.sweep_csv).expanduser().resolve()
        if not sweep_csv.exists():
            print(f"[ERR] Sweep CSV not found: {sweep_csv}", file=sys.stderr)
            return 1
        body = format_sweep_summary_text(sweep_csv)
        subject = args.subject or "GNN Hyperparameter Sweep Completed"
    else:
        parser.error("Either --message, --run_dir, or --sweep_csv must be provided")
    
    # 送信
    success = send_email(
        args.smtp_server,
        args.smtp_port,
        args.from_email,
        args.password,
        args.to,
        subject,
        body,
        is_html=args.html
    )
    
    if success:
        print(f"[INFO] Email sent successfully to {args.to}")
    else:
        print(f"[ERR] Failed to send email to {args.to}", file=sys.stderr)
    
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
