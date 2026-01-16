#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as: python3 tools/autoreport/render_run.py ...
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.autoreport.run_loader import LoadedRun, load_run, to_report_dict  # noqa: E402


def _repo_root() -> Path:
    # tools/autoreport/render_run.py -> repo root is 2 parents up
    return Path(__file__).resolve().parents[2]


def _safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", ".", "+"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "run"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _copy_asset(src: Optional[str], assets_dir: Path, prefix: str) -> Optional[str]:
    if not src:
        return None
    sp = Path(src)
    if not sp.exists():
        return None
    assets_dir.mkdir(parents=True, exist_ok=True)
    dst_name = f"{prefix}_{sp.name}"
    dp = assets_dir / _safe_name(dst_name)
    shutil.copy2(sp, dp)
    return str(dp.relative_to(assets_dir.parent))


def _md_img(rel_path: Optional[str], alt: str) -> str:
    if not rel_path:
        return ""
    return f"![{alt}]({rel_path})"


def _format_args(args: List[str]) -> str:
    if not args:
        return ""
    return " ".join(args)


def _table_kv(rows: List[List[str]]) -> str:
    # rows: [[k,v], ...]
    out = ["| 項目 | 値 |", "|---|---|"]
    for k, v in rows:
        out.append(f"| {k} | {v} |")
    return "\n".join(out)


def _table_worst_accuracy(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    out = ["| filename | accuracy |", "|---|---:|"]
    for it in items:
        out.append(f"| {it.get('filename','')} | {it.get('accuracy','')} |")
    return "\n".join(out)


def _table_top_ratio(items: List[Dict[str, Any]]) -> str:
    if not items:
        return ""
    out = ["| filename | pred_defect_ratio |", "|---|---:|"]
    for it in items:
        out.append(f"| {it.get('filename','')} | {it.get('pred_defect_ratio','')} |")
    return "\n".join(out)


def _render_report(d: Dict[str, Any], assets: Dict[str, Any]) -> str:
    res = d.get("results", {})
    best = res.get("best", {})
    test = res.get("test", {})
    ds = d.get("dataset", {}) or {}

    stats = (d.get("artifacts", {}) or {}).get("file_statistics_summary") or {}
    worst = stats.get("worst_accuracy", []) or []
    top_ratio = stats.get("top_pred_defect_ratio", []) or []

    lines: List[str] = []
    lines.append(f"# 学習結果レポート: {d.get('run_id')}")
    lines.append("")
    lines.append(f"- generated_at: `{d.get('generated_at')}`")
    lines.append(f"- run_dir: `{d.get('run_dir')}`")
    if d.get("git_short_sha"):
        lines.append(f"- git: `{d.get('git_short_sha')}`")
    lines.append("")

    lines.append("## サマリー")
    lines.append(_table_kv([
        ["profile", str(d.get("profile", ""))],
        ["best macro_f1 (val)", str(best.get("macro_f1"))],
        ["best epoch", str(best.get("epoch"))],
        ["best val_loss", str(best.get("val_loss"))],
        ["test macro_f1", str(test.get("macro_f1"))],
        ["test weighted_f1", str(test.get("weighted_f1"))],
        ["test accuracy", str(test.get("accuracy"))],
        ["test balanced_accuracy", str(test.get("balanced_accuracy"))],
        ["test mcc", str(test.get("mcc"))],
    ]))
    lines.append("")

    lines.append("## データセット")
    if ds:
        lines.append(_table_kv([
            ["dataset_type", str(ds.get("dataset_type", ""))],
            ["num_classes", str(ds.get("number_of_classes", ""))],
            ["nodes_per_sample", str(ds.get("nodes_per_sample", ""))],
            ["train_pairs", str(ds.get("train_pairs", ""))],
            ["val_pairs", str(ds.get("val_pairs", ""))],
            ["test_pairs", str(ds.get("test_pairs", ""))],
            ["total_pairs", str(ds.get("total_pairs", ""))],
        ]))
    else:
        lines.append("- dataset_info.txt が見つからなかったため省略")
    lines.append("")

    lines.append("## 設定（主要）")
    lines.append(_table_kv([
        ["script", f"`{d.get('script')}`" if d.get("script") else ""],
        ["nproc_per_node", str((d.get("training", {}) or {}).get("nproc_per_node"))],
        ["args", f"`{_format_args((d.get('training', {}) or {}).get('args', []))}`"],
    ]))
    lines.append("")

    lines.append("## 学習曲線（自動生成図）")
    for k in ["combined_plot", "loss_plot", "macro_f1_plot"]:
        p = assets.get(k)
        if p:
            lines.append(_md_img(p, k))
            lines.append("")
    if not any(assets.get(k) for k in ["combined_plot", "loss_plot", "macro_f1_plot"]):
        lines.append("- loss plot 画像が見つからなかったため省略")
        lines.append("")

    lines.append("## Confusion matrix / クラス別指標（図）")
    for k in ["confusion_counts", "confusion_normalized", "per_class_f1"]:
        p = assets.get(k)
        if p:
            lines.append(_md_img(p, k))
            lines.append("")
    if not any(assets.get(k) for k in ["confusion_counts", "confusion_normalized", "per_class_f1"]):
        lines.append("- confusion/per-class 図が見つからなかったため省略")
        lines.append("")

    lines.append("## 予測の確信度/不確実性（probsから自動生成）")
    for k in ["confidence_hist", "entropy_hist"]:
        p = assets.get(k)
        if p:
            lines.append(_md_img(p, k))
            lines.append("")
    if not any(assets.get(k) for k in ["confidence_hist", "entropy_hist"]):
        lines.append("- all_probs.npy が無い/生成に失敗したため省略")
        lines.append("")

    if assets.get("top_confusions_csv"):
        lines.append("## Top confusion pairs（CSV）")
        lines.append(f"- `{assets.get('top_confusions_csv')}`")
        lines.append("")

    lines.append("## データセット性質（自動集計）")
    for k in [
        "dataset_defect_ndf_counts",
        "dataset_preddefectratio_hist",
        "dataset_layer_counts",
        "dataset_hw_counts",
        "dataset_class_counts_log",
        "dataset_class_weights",
    ]:
        p = assets.get(k)
        if p:
            lines.append(_md_img(p, k))
            lines.append("")

    lines.append("## 予測結果（ファイル単位の統計）")
    if stats and stats.get("count", 0) > 0:
        lines.append(_table_kv([
            ["csv", f"`{stats.get('path','')}`"],
            ["count", str(stats.get("count"))],
            ["mean_accuracy", str(stats.get("mean_accuracy"))],
        ]))
        lines.append("")
        if worst:
            lines.append("### accuracy が低いサンプル（worst）")
            lines.append(_table_worst_accuracy(worst))
            lines.append("")
        if top_ratio:
            lines.append("### PredDefectRatio が高いサンプル（top）")
            lines.append(_table_top_ratio(top_ratio))
            lines.append("")
    else:
        lines.append("- `file_statistics*.csv` が見つからなかったため省略")
        lines.append("")

    lines.append("## 空間可視化（抜粋）")
    spatial = assets.get("spatial_examples", []) or []
    if spatial:
        for p in spatial:
            lines.append(_md_img(p, "spatial"))
        lines.append("")
    else:
        lines.append("- spatial_visualizations が見つからなかったため省略")
        lines.append("")

    lines.append("## 重要サンプル（必ず可視化: 良い/悪い例）")
    worst_imgs = assets.get("key_worst_spatial", []) or []
    best_imgs = assets.get("key_best_spatial", []) or []
    if worst_imgs:
        lines.append("### worst（accuracy低）")
        for p in worst_imgs:
            lines.append(_md_img(p, "worst"))
        lines.append("")
    if best_imgs:
        lines.append("### best（Defectでaccuracy高）")
        for p in best_imgs:
            lines.append(_md_img(p, "best"))
        lines.append("")
    if not worst_imgs and not best_imgs:
        lines.append("- 対象ファイルの spatial_visualization が見つからず省略")
        lines.append("")

    lines.append("## 主要成果物パス")
    art = d.get("artifacts", {}) or {}
    lines.append(_table_kv([
        ["final_model_path", f"`{art.get('final_model_path')}`" if art.get("final_model_path") else ""],
        ["predict_dir", f"`{art.get('predict_dir')}`" if art.get("predict_dir") else ""],
        ["loss_plots_dir", f"`{art.get('loss_plots_dir')}`" if art.get("loss_plots_dir") else ""],
        ["train_log", f"`{art.get('train_log')}`" if art.get("train_log") else ""],
    ]))
    lines.append("")

    lines.append("## 次の打ち手（自動提案）")
    lines.append("- **クラス不均衡が極端**: macro-F1 を上げるには minority class の recall 改善が支配的。`FocalLoss/LogitAdjust/MinoritySampler` 系の比較を同一プロトコルで回す。")
    lines.append("- **誤分類の偏り**: confusion matrix（もし出力があれば）から A→B が多いペアを抽出し、可視化例を増やして原因分類（境界が曖昧/ラベル揺れ/特徴不足）を切る。")
    lines.append("- **閾値/後処理**: NDF/Defect 判定の運用があるなら、`file_decisions_thresholds*.csv` を基に運用指標（FN重視など）へ最適化する。")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_paper_skeleton(d: Dict[str, Any], assets: Dict[str, Any]) -> str:
    # Minimal “paper-like” draft in Markdown (English), easy to migrate to LaTeX later.
    res = d.get("results", {})
    best = res.get("best", {})
    test = res.get("test", {})
    ds = d.get("dataset", {}) or {}

    lines: List[str] = []
    lines.append(f"# Paper Draft: {d.get('run_id')}")
    lines.append("")
    lines.append("## Abstract")
    lines.append(
        "We study defect classification on graph-structured data and evaluate a GNN-based model under a highly imbalanced 19-class setting. "
        f"The best validation macro-F1 reached {best.get('macro_f1')}, and the test macro-F1 reached {test.get('macro_f1')}."
    )
    lines.append("")

    lines.append("## Dataset")
    if ds:
        lines.append(
            f"- Dataset type: {ds.get('dataset_type')}\n"
            f"- #classes: {ds.get('number_of_classes')}\n"
            f"- Train/Val/Test: {ds.get('train_pairs')}/{ds.get('val_pairs')}/{ds.get('test_pairs')}\n"
            f"- Nodes per sample: {ds.get('nodes_per_sample')}"
        )
    lines.append("")

    lines.append("## Method")
    lines.append(f"- Training script: `{d.get('script')}`")
    lines.append(f"- Training args: `{_format_args((d.get('training', {}) or {}).get('args', []))}`")
    lines.append("")

    lines.append("## Results")
    lines.append(
        f"- Best (val): epoch={best.get('epoch')}, val_loss={best.get('val_loss')}, macro-F1={best.get('macro_f1')}\n"
        f"- Test: acc={test.get('accuracy')}, weighted-F1={test.get('weighted_f1')}, macro-F1={test.get('macro_f1')}, balanced-acc={test.get('balanced_accuracy')}, MCC={test.get('mcc')}"
    )
    lines.append("")
    for k in ["combined_plot", "loss_plot", "macro_f1_plot"]:
        p = assets.get(k)
        if p:
            lines.append(_md_img(p, k))
            lines.append("")

    lines.append("## Discussion")
    lines.append(
        "Despite near-perfect accuracy driven by the majority class, macro-F1 highlights the remaining challenges on minority classes. "
        "Future work includes improved imbalance-aware training and targeted error analysis."
    )
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def _latex_escape(s: str) -> str:
    # minimal escape for LaTeX text fields
    rep = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(rep.get(ch, ch))
    return "".join(out)


def _render_paper_tex(d: Dict[str, Any], assets: Dict[str, Any]) -> str:
    res = d.get("results", {})
    best = res.get("best", {})
    test = res.get("test", {})
    ds = d.get("dataset", {}) or {}
    art = d.get("artifacts", {}) or {}
    stats = art.get("file_statistics_summary") or {}
    thr = art.get("threshold_decisions_summary") or {}
    train_sum = art.get("train_log_summary") or {}
    imbalance = train_sum.get("imbalance_summary") or {}
    cli_args = (d.get("training", {}) or {}).get("args", []) or []

    def _fmt_num_title(x: Any) -> str:
        if x is None:
            return "N/A"
        try:
            if isinstance(x, int):
                return str(x)
            return f"{float(x):.4f}".rstrip("0").rstrip(".")
        except Exception:
            return str(x)

    run_id_tex = _latex_escape(str(d.get("run_id", "Paper")))
    ds_type_tex = _latex_escape(str(ds.get("dataset_type", "") or "")).strip()
    main_title_tex = _latex_escape("Run Report: Two-Stage GAT for Imbalanced 19-Class Defect Classification")
    score_tex = _latex_escape(
        f"val macro-F1={_fmt_num_title(best.get('macro_f1'))}, test macro-F1={_fmt_num_title(test.get('macro_f1'))}"
    )

    title_lines = [rf"\textbf{{{main_title_tex}}}"]
    if ds_type_tex:
        title_lines.append(rf"\large {ds_type_tex} \quad ({score_tex})")
    else:
        title_lines.append(rf"\large {score_tex}")
    title_lines.append(rf"\normalsize Run: {run_id_tex}")
    title = r"\\ ".join(title_lines)

    subtitle = _latex_escape(f"profile={d.get('profile','')} git={d.get('git_short_sha','')}")

    def fig(rel: Optional[str], caption: str, width: str = r"0.95\linewidth") -> str:
        if not rel:
            return ""
        p = _latex_escape(rel)
        c = _latex_escape(caption)
        return (
            "\\begin{figure}[t]\n"
            "\\centering\n"
            f"\\includegraphics[width={width}]{{{p}}}\n"
            f"\\caption{{{c}}}\n"
            "\\end{figure}\n"
        )

    def _escape_braces_only(s: str) -> str:
        # for \nolinkurl{...}: keep underscores/slashes as-is (breakable),
        # but escape braces to avoid TeX parsing issues.
        return s.replace("{", r"\{").replace("}", r"\}")

    def _fmt_num(x: Any) -> str:
        if x is None:
            return "N/A"
        try:
            if isinstance(x, int):
                return str(x)
            return f"{float(x):.4f}".rstrip("0").rstrip(".")
        except Exception:
            return _latex_escape(str(x))

    def _extract_hp(args_list: List[Any]) -> Dict[str, Any]:
        # Best-effort parse of common flags: --flag value
        out_hp: Dict[str, Any] = {}
        a = [str(x) for x in args_list]
        for i, tok in enumerate(a):
            if not tok.startswith("--"):
                continue
            key = tok.lstrip("-")
            # boolean flags (no value)
            if i + 1 >= len(a) or a[i + 1].startswith("--"):
                out_hp[key] = True
                continue
            out_hp[key] = a[i + 1]
        # expose a canonical subset
        keep = [
            "hidden_channels",
            "learning_rate",
            "weight_decay",
            "batch_size",
            "epochs",
            "patience",
            "dropout",
            "edge_drop_prob",
            "data_usage_ratio",
            "use_onecycle",
            "use_amp",
            "use_class_frequency_sampler",
        ]
        return {k: out_hp.get(k) for k in keep if k in out_hp}

    hp = _extract_hp(cli_args)

    def _topk_class_lines(k: int = 3) -> List[str]:
        rows = train_sum.get("class_distribution") or []
        if not isinstance(rows, list) or not rows:
            return []
        rows2 = [r for r in rows if isinstance(r, dict) and "samples" in r]
        rows2.sort(key=lambda r: r.get("samples", 0), reverse=True)
        out2 = []
        for r in rows2[:k]:
            out2.append(f"class {r.get('class_id')}: {r.get('samples')} ({_fmt_num(r.get('percent'))}%)")
        return out2

    def _bottomk_class_lines(k: int = 3) -> List[str]:
        rows = train_sum.get("class_distribution") or []
        if not isinstance(rows, list) or not rows:
            return []
        rows2 = [r for r in rows if isinstance(r, dict) and "samples" in r]
        rows2.sort(key=lambda r: r.get("samples", 10**18))
        out2 = []
        for r in rows2[:k]:
            out2.append(f"class {r.get('class_id')}: {r.get('samples')} ({_fmt_num(r.get('percent'))}%)")
        return out2

    lines: List[str] = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{graphicx}")
    # hypertexnames=false helps avoid duplicate figure anchors with xdvipdfmx
    lines.append(r"\usepackage[hypertexnames=false]{hyperref}")
    lines.append(r"\hypersetup{hidelinks}")
    lines.append(r"\usepackage{xurl}")  # better line-breaking for \url/\nolinkurl
    lines.append(r"\usepackage{microtype}")  # nicer justification (usually reduces under/overfull)
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{float}")
    lines.append(r"\usepackage{amsmath}")
    lines.append(r"\usepackage{amssymb}")
    lines.append(r"\usepackage{enumitem}")
    lines.append(r"\usepackage{xcolor}")
    # Nicer algorithm formatting than a boxed monospace block.
    lines.append(r"\usepackage[ruled,vlined]{algorithm2e}")
    lines.append(r"\title{" + title + r"}")
    lines.append(r"\author{AutoReport}")
    lines.append(r"\date{" + _latex_escape(str(d.get("generated_at", ""))) + r"}")
    lines.append(r"\begin{document}")
    lines.append(r"\sloppy")  # reduce overfull/underfull warnings for long tokens
    lines.append(r"\setlength{\emergencystretch}{2em}")
    lines.append(r"\maketitle")
    lines.append(r"\noindent " + subtitle + r"\\")
    lines.append("")

    lines.append(r"\begin{abstract}")
    lines.append(
        "We study defect classification on graph-structured data under a highly imbalanced 19-class setting. "
        f"Our best validation macro-F1 is {best.get('macro_f1')}, and the test macro-F1 is {test.get('macro_f1')}. "
        "We provide automated reporting, error analysis, and qualitative visualizations for faster iteration."
    )
    lines.append(r"\end{abstract}")
    lines.append("")

    lines.append(r"\section{Introduction}")
    lines.append(
        "Graph neural networks (GNNs) are a natural choice for defect detection when the underlying structure is relational. "
        "However, strong class imbalance can make accuracy misleading; macro-F1 provides a more faithful view of minority-class performance. "
        "This paper summarizes an experiment run and provides an analysis template that can be reused across runs."
    )
    lines.append("")

    lines.append(r"\section{Related Work}")
    lines.append(
        "GNNs generalize deep learning to relational data by propagating and aggregating information over edges "
        r"\cite{scarselli2009gnn}. For node- and graph-level prediction, common backbones include graph convolutional "
        r"networks (GCN) \cite{kipf2017gcn} and graph attention networks (GAT) \cite{velickovic2018gat}, which are widely "
        r"used in modern libraries such as PyTorch Geometric \cite{fey2019pyg}."
    )
    lines.append(
        "In defect detection and localization, representing measurement points or finite-element nodes as a graph provides "
        "a principled way to encode neighborhood interactions and geometry. Our recent study integrated finite element "
        "method (FEM) simulations with a GNN to estimate three-dimensional defect locations in perforated CFRP structures "
        r"from stress-distribution features \cite{nishioka2025fmats}. Related GNN-based SHM work also reports strong performance "
        r"for damage detection/localization when sensor topology is treated as a graph \cite{wijethunga2025dualgraph,yehia2025whatlieswithin}."
    )
    lines.append(
        "Learning under extreme class imbalance requires objectives and calibration strategies beyond accuracy. "
        r"Focal loss \cite{lin2017focal} and class-balanced reweighting \cite{cui2019classbalanced} improve minority-class learning, "
        r"and logit adjustment \cite{menon2021logit} provides a simple prior-correction approach for long-tailed recognition. "
        "Motivated by these findings, we report macro-F1 and include imbalance-aware training options and diagnostics."
    )
    lines.append("")

    lines.append(r"\section{Dataset}")
    if ds:
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{ll}")
        lines.append(r"\toprule")
        lines.append(r"Item & Value \\")
        lines.append(r"\midrule")
        lines.append(r"Dataset type & " + _latex_escape(str(ds.get("dataset_type", ""))) + r" \\")
        lines.append(r"\#classes & " + _latex_escape(str(ds.get("number_of_classes", ""))) + r" \\")
        lines.append(
            r"Train/Val/Test & "
            + _latex_escape(f"{ds.get('train_pairs','')}/{ds.get('val_pairs','')}/{ds.get('test_pairs','')}")
            + r" \\"
        )
        lines.append(r"Nodes per sample & " + _latex_escape(str(ds.get("nodes_per_sample", ""))) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Dataset configuration for this run.}")
        lines.append(r"\label{tab:dataset}")
        lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\subsection{Class imbalance}")
    if imbalance:
        lines.append(
            "The training set is highly imbalanced. "
            f"The majority class is class {imbalance.get('majority_class')} "
            f"({ _fmt_num(imbalance.get('majority_percent')) }\\%), while the smallest class is class "
            f"{imbalance.get('minority_class')} "
            f"({imbalance.get('minority_samples')} samples, { _fmt_num(imbalance.get('minority_percent')) }\\%)."
        )
        top3 = _topk_class_lines(3)
        bot3 = _bottomk_class_lines(3)
        if top3 or bot3:
            lines.append(r"\begin{itemize}[leftmargin=*]")
            if top3:
                lines.append(r"\item Top-3 classes by sample count: " + _latex_escape("; ".join(top3)))
            if bot3:
                lines.append(r"\item Bottom-3 classes by sample count: " + _latex_escape("; ".join(bot3)))
            lines.append(r"\end{itemize}")
    else:
        lines.append("Class distribution was not available in the log for this run.")
    lines.append("")

    # Dataset property figures (if available)
    if assets.get("dataset_defect_ndf_counts") and assets.get("dataset_preddefectratio_hist"):
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{cc}")
        lines.append(r"\includegraphics[width=0.46\linewidth]{" + _latex_escape(assets.get("dataset_defect_ndf_counts")) + r"} &")
        lines.append(r"\includegraphics[width=0.46\linewidth]{" + _latex_escape(assets.get("dataset_preddefectratio_hist")) + r"} \\")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Dataset properties: file type counts and PredDefectRatio distribution.}")
        lines.append(r"\label{fig:dataset_props}")
        lines.append(r"\end{figure}")
        lines.append("")

    if assets.get("dataset_class_counts_log") and assets.get("dataset_class_weights"):
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{cc}")
        lines.append(r"\includegraphics[width=0.60\linewidth]{" + _latex_escape(assets.get("dataset_class_counts_log")) + r"} &")
        lines.append(r"\includegraphics[width=0.34\linewidth]{" + _latex_escape(assets.get("dataset_class_weights")) + r"} \\")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Class distribution (log counts) and class weights.}")
        lines.append(r"\label{fig:class_dist}")
        lines.append(r"\end{figure}")
        lines.append("")

    lines.append(r"\section{Method}")
    lines.append(
        "We train a graph attention network (GAT) for node-level multi-class prediction under extreme class imbalance. "
        "The implementation uses a two-stage head: (i) defect detection (no-defect vs defect) and (ii) defect type classification "
        "(18 defect classes). During training, physically impossible classes are masked based on layer constraints."
    )
    lines.append("")

    lines.append(r"\subsection{Model architecture (GATModel)}")
    lines.append(
        "The model operates on node features of dimension $4$ and uses three GATConv layers with multi-head attention "
        "(heads $=4$, concatenation enabled). Each layer uses a residual projection, batch normalization, and dropout. "
        "Let $h$ be the hidden width (CLI: \\texttt{--hidden\\_channels}). The layer shapes are:"
    )
    lines.append(r"\begin{itemize}[leftmargin=*]")
    lines.append(r"\item GATConv$_1$: $4 \rightarrow (4h)$, BN$(4h)$, Dropout")
    lines.append(r"\item GATConv$_2$: $(4h) \rightarrow (8h)$ via $(2h)$ per head, BN$(8h)$, Dropout")
    lines.append(r"\item GATConv$_3$: $(8h) \rightarrow (4h)$, BN$(4h)$, Dropout")
    lines.append(r"\end{itemize}")
    lines.append(
        "The two-stage heads are linear layers: detection head $\\mathbb{R}^{4h}\\rightarrow\\mathbb{R}^{2}$ "
        "and classification head $\\mathbb{R}^{4h}\\rightarrow\\mathbb{R}^{18}$. "
        "The final 19-class logits are formed as $\\ell_0 = \\ell^{det}_0$ and "
        "$\\ell_{1:18} = \\ell^{det}_1 + \\ell^{cls}_{1:18}$, followed by a softmax."
    )
    lines.append("")

    lines.append(r"\subsection{Two-stage probability composition}")
    lines.append(
        "Let $p^{det}(y=0\\mid x)$ and $p^{det}(y>0\\mid x)$ be the detection probabilities for a node, and "
        "$p^{cls}(k\\mid x, y>0)$ be the conditional distribution over defect types $k\\in\\{1,\\dots,18\\}$. "
        "The combined 19-class probabilities are computed as:"
    )
    lines.append(r"\begin{align}")
    lines.append(r"p(0\mid x) &= p^{det}(y=0\mid x),\\")
    lines.append(r"p(k\mid x) &= p^{det}(y>0\mid x)\, p^{cls}(k\mid x, y>0), \qquad k=1,\dots,18.")
    lines.append(r"\end{align}")
    lines.append(
        "Optionally, a detection threshold $\\tau$ can be used to force nodes with $p^{det}(y>0\\mid x) < \\tau$ to class $0$."
    )
    lines.append("")

    lines.append(r"\subsection{Layer masking (physical constraints)}")
    lines.append(
        "Some defect classes are physically impossible depending on the node layer. "
        "We encode this with an allowed-class set $\\mathcal{A}(i)\\subseteq\\{0,\\dots,18\\}$ for node $i$, and mask logits as:"
    )
    lines.append(r"\begin{align}")
    lines.append(r"\tilde{\ell}_{i,k} = \begin{cases}")
    lines.append(r"\ell_{i,k} & \text{if } k \in \mathcal{A}(i),\\")
    lines.append(r"-\infty & \text{otherwise.}")
    lines.append(r"\end{cases}")
    lines.append(r"\end{align}")
    lines.append(
        "In practice, $-\\infty$ is implemented as the minimum representable value of the tensor dtype (AMP-safe)."
    )
    lines.append("")

    lines.append(r"\subsection{Training objective (detection + classification)}")
    lines.append(
        "We optimize a two-stage loss. Define binary detection targets $y^{det}_i = \\mathbb{1}[y_i \\neq 0]$ and "
        "classification targets $y^{cls}_i = y_i-1$ for defect nodes ($y_i>0$). "
        "Let $\\mathcal{I}_{def} = \\{i \\mid y_i>0\\}$. The objective is:"
    )
    lines.append(r"\begin{align}")
    lines.append(r"\mathcal{L}_{det} &= \mathrm{CE}(\ell^{det}, y^{det}),\\")
    lines.append(r"\mathcal{L}_{cls} &= \mathrm{CE}(\ell^{cls}_{\mathcal{I}_{def}}, y^{cls}_{\mathcal{I}_{def}}),\\")
    lines.append(r"\mathcal{L} &= \lambda_{det}\,\mathcal{L}_{det} + \lambda_{cls}\,\mathcal{L}_{cls}.")
    lines.append(r"\end{align}")
    lines.append(
        "Imbalance handling can be further strengthened with class-frequency sampling and focal-style losses; "
        "edge dropout is applied during training (CLI: \\texttt{--edge\\_drop\\_prob})."
    )
    lines.append("")

    lines.append(r"\section{Algorithm}")
    lines.append(
        "Algorithm~\\ref{alg:pipeline} summarizes the end-to-end pipeline. "
        "The reporting step is fully automated from the generated artifacts (metrics, plots, CSVs, and qualitative images)."
    )
    lines.append("")
    lines.append(r"\begin{algorithm}[H]")
    lines.append(r"\DontPrintSemicolon")
    lines.append(r"\KwIn{graph samples $G_i=(V_i,E_i,X_i)$, node labels $y_i\in\{0,\dots,18\}$, run config}")
    lines.append(r"\KwOut{best checkpoint; metrics; plots; qualitative figures; slides}")
    lines.append(r"\BlankLine")
    lines.append(r"\textbf{Pairing}: create train/val/test pairs; leakage checks\;")
    lines.append(r"\textbf{Init}: build GATModel($h$, dropout, edge\_drop)\;")
    lines.append(r"\textbf{Imbalance}: enable class-frequency sampler / focal-style loss\;")
    lines.append(r"\For{$t \leftarrow 1$ \KwTo $T$}{")
    lines.append(r"\quad Apply edge dropout (train only)\;")
    lines.append(r"\quad Forward pass $\rightarrow$ detection logits $\ell^{det}$, classification logits $\ell^{cls}$\;")
    lines.append(r"\quad Apply layer mask (set impossible classes to $-\infty$)\;")
    lines.append(r"\quad Compute two-stage loss $\mathcal{L}=\lambda_{det}\,\mathrm{CE}(\ell^{det},y^{det})+\lambda_{cls}\,\mathrm{CE}(\ell^{cls}_{y>0},y^{cls}_{y>0})$\;")
    lines.append(r"\quad Backprop + optimizer step (optionally AMP)\;")
    lines.append(r"\quad Evaluate on validation; update best checkpoint by macro-F1\;")
    lines.append(r"}")
    lines.append(r"\textbf{Test}: evaluate best checkpoint on test split\;")
    lines.append(r"\textbf{Export}: save arrays + CSV summaries + spatial PNGs\;")
    lines.append(r"\textbf{Report}: auto-generate confusion/per-class/uncertainty/dataset plots + best/worst figures + PDF/PPTX\;")
    lines.append(r"\caption{Training, evaluation, and auto-reporting pipeline.}")
    lines.append(r"\label{alg:pipeline}")
    lines.append(r"\end{algorithm}")
    lines.append("")

    lines.append(r"\section{Experimental Setup}")
    lines.append(r"\subsection{Implementation and hardware}")
    gpu_names = train_sum.get("gpu_names") or train_sum.get("using_device") or []
    if isinstance(gpu_names, list) and gpu_names:
        lines.append("GPUs used: " + _latex_escape(", ".join([str(x) for x in gpu_names])) + ".")
    else:
        lines.append("GPU information is not available.")
    lines.append("")

    lines.append(r"\subsection{Training configuration}")
    lines.append(r"\begin{itemize}[leftmargin=*]")
    if d.get("script"):
        lines.append(r"\item Script: \nolinkurl{" + _escape_braces_only(str(d.get("script"))) + r"}")
    args = (d.get("training", {}) or {}).get("args", [])
    if args:
        lines.append(r"\item CLI args: \nolinkurl{" + _escape_braces_only(" ".join([str(a) for a in args])) + r"}")
    lines.append(r"\end{itemize}")
    lines.append("")

    if hp:
        lines.append(r"\subsection{Key hyperparameters}")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{ll}")
        lines.append(r"\toprule")
        lines.append(r"Hyperparameter & Value \\")
        lines.append(r"\midrule")
        for k, v in hp.items():
            lines.append(_latex_escape(k) + " & " + _latex_escape(str(v)) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Extracted key hyperparameters from CLI arguments.}")
        lines.append(r"\label{tab:hparams}")
        lines.append(r"\end{table}")
        lines.append("")

    lines.append(r"\section{Results}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Value \\")
    lines.append(r"\midrule")
    lines.append(rf"Best macro-F1 (val) & {_fmt_num(best.get('macro_f1'))} \\")
    lines.append(rf"Best epoch & {best.get('epoch')} \\")
    lines.append(rf"Best val loss & {_fmt_num(best.get('val_loss'))} \\")
    lines.append(rf"Test macro-F1 & {_fmt_num(test.get('macro_f1'))} \\")
    lines.append(rf"Test weighted-F1 & {_fmt_num(test.get('weighted_f1'))} \\")
    lines.append(rf"Test accuracy & {_fmt_num(test.get('accuracy'))} \\")
    lines.append(rf"Balanced accuracy & {_fmt_num(test.get('balanced_accuracy'))} \\")
    lines.append(rf"MCC & {_fmt_num(test.get('mcc'))} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Overall evaluation metrics.}")
    lines.append(r"\label{tab:metrics}")
    lines.append(r"\end{table}")
    lines.append("")

    if assets.get("combined_plot"):
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.95\linewidth]{" + _latex_escape(assets.get("combined_plot")) + r"}")
        lines.append(r"\caption{Learning curves (combined).}")
        lines.append(r"\label{fig:curves}")
        lines.append(r"\end{figure}")
        lines.append("")
    if not assets.get("combined_plot"):
        # fallback
        lines.append(fig(assets.get("loss_plot"), "Loss curve.", width=r"0.48\linewidth"))
        lines.append(fig(assets.get("macro_f1_plot"), "Macro-F1 curve.", width=r"0.48\linewidth"))

    if assets.get("per_class_f1"):
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.95\linewidth]{" + _latex_escape(assets.get("per_class_f1")) + r"}")
        lines.append(r"\caption{Per-class F1 scores.}")
        lines.append(r"\label{fig:perclass_f1}")
        lines.append(r"\end{figure}")
        lines.append("")

    if assets.get("confidence_hist") and assets.get("entropy_hist"):
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{cc}")
        lines.append(r"\includegraphics[width=0.46\linewidth]{" + _latex_escape(assets.get("confidence_hist")) + r"} &")
        lines.append(r"\includegraphics[width=0.46\linewidth]{" + _latex_escape(assets.get("entropy_hist")) + r"} \\")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Prediction confidence (max probability) and normalized entropy (uncertainty).}")
        lines.append(r"\label{fig:uncertainty}")
        lines.append(r"\end{figure}")
        lines.append("")

    # Optional confusion plot(s)
    if assets.get("confusion_counts"):
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.92\linewidth]{" + _latex_escape(assets.get("confusion_counts")) + r"}")
        lines.append(r"\caption{Confusion matrix (counts).}")
        lines.append(r"\label{fig:confusion_counts}")
        lines.append(r"\end{figure}")
        lines.append("")
    if assets.get("confusion_normalized"):
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\includegraphics[width=0.92\linewidth]{" + _latex_escape(assets.get("confusion_normalized")) + r"}")
        lines.append(r"\caption{Confusion matrix (row-normalized).}")
        lines.append(r"\label{fig:confusion_norm}")
        lines.append(r"\end{figure}")
        lines.append("")

    lines.append(r"\section{Error Analysis}")
    lines.append(
        "We analyze hard cases using file-level statistics. "
        "Table~\\ref{tab:hardcases} lists samples with the lowest per-file accuracy, and "
        "Table~\\ref{tab:highratio} lists samples with the highest predicted defect ratio."
    )
    lines.append("")

    worst = stats.get("worst_accuracy", []) if isinstance(stats, dict) else []
    top_ratio = stats.get("top_pred_defect_ratio", []) if isinstance(stats, dict) else []
    if worst:
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{ll}")
        lines.append(r"\toprule")
        lines.append(r"Filename & Accuracy \\")
        lines.append(r"\midrule")
        for it in worst[:10]:
            lines.append(_latex_escape(str(it.get("filename", ""))) + " & " + _fmt_num(it.get("accuracy")) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Hard cases by per-file accuracy (lower is worse).}")
        lines.append(r"\label{tab:hardcases}")
        lines.append(r"\end{table}")
        lines.append("")

    if top_ratio:
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{ll}")
        lines.append(r"\toprule")
        lines.append(r"Filename & PredDefectRatio \\")
        lines.append(r"\midrule")
        for it in top_ratio[:10]:
            lines.append(
                _latex_escape(str(it.get("filename", ""))) + " & " + _fmt_num(it.get("pred_defect_ratio")) + r" \\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{High predicted defect ratio samples (higher is more defect-like).}")
        lines.append(r"\label{tab:highratio}")
        lines.append(r"\end{table}")
        lines.append("")

    # Threshold decision summary (optional)
    if isinstance(thr, dict) and thr.get("threshold_counts"):
        lines.append(r"\subsection{Threshold-based decisions (file-level)}")
        lines.append("We summarize how many files are classified as defect under different thresholds.")
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{lr}")
        lines.append(r"\toprule")
        lines.append(r"Column & \#Files \\")
        lines.append(r"\midrule")
        for k, v in sorted((thr.get("threshold_counts") or {}).items()):
            lines.append(_latex_escape(str(k)) + " & " + _latex_escape(str(v)) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Counts of files flagged as defect for each threshold column.}")
        lines.append(r"\label{tab:thresholds}")
        lines.append(r"\end{table}")
        lines.append("")

    lines.append(r"\section{Qualitative Results}")
    # show up to 6 examples as a single grid figure to avoid many PDF anchors
    spatial = (assets.get("spatial_examples") or [])[:6]
    if spatial:
        lines.append(r"\begin{figure}[t]")
        lines.append(r"\centering")
        lines.append(r"\begin{tabular}{cc}")
        for i, p in enumerate(spatial):
            inc = r"\includegraphics[width=0.46\linewidth]{" + _latex_escape(p) + r"}"
            if i % 2 == 0:
                lines.append(inc + " & ")
            else:
                lines.append(inc + r" \\")
        # If odd count, close the last row nicely
        if len(spatial) % 2 == 1:
            lines.append(r" \\")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Spatial visualization examples (subset).}")
        lines.append(r"\label{fig:spatial}")
        lines.append(r"\end{figure}")
        lines.append("")

    # MUST visualize good/bad examples
    worst_imgs = (assets.get("key_worst_spatial") or [])[:6]
    best_imgs = (assets.get("key_best_spatial") or [])[:6]
    if worst_imgs:
        # 3 images per page, single column (large)
        lines.append(r"\clearpage")
        for page_i in range(0, len(worst_imgs), 3):
            chunk = worst_imgs[page_i : page_i + 3]
            lines.append(r"\begin{figure}[p]")
            lines.append(r"\centering")
            for j, p in enumerate(chunk, start=1):
                lines.append(r"\includegraphics[width=0.95\linewidth,height=0.29\textheight,keepaspectratio]{" + _latex_escape(p) + r"}\\[0.6em]")
            lines.append(r"\caption{Worst examples (low per-file accuracy), page " + str(page_i // 3 + 1) + r".}")
            lines.append(r"\label{fig:worst_page_" + str(page_i // 3 + 1) + r"}")
            lines.append(r"\end{figure}")
            lines.append(r"\clearpage")
    if best_imgs:
        lines.append(r"\clearpage")
        for page_i in range(0, len(best_imgs), 3):
            chunk = best_imgs[page_i : page_i + 3]
            lines.append(r"\begin{figure}[p]")
            lines.append(r"\centering")
            for j, p in enumerate(chunk, start=1):
                lines.append(r"\includegraphics[width=0.95\linewidth,height=0.29\textheight,keepaspectratio]{" + _latex_escape(p) + r"}\\[0.6em]")
            lines.append(r"\caption{Best defect examples (high per-file accuracy), page " + str(page_i // 3 + 1) + r".}")
            lines.append(r"\label{fig:best_page_" + str(page_i // 3 + 1) + r"}")
            lines.append(r"\end{figure}")
            lines.append(r"\clearpage")

    lines.append(r"\section{Discussion and Future Work}")
    lines.append(
        "The model achieves near-perfect accuracy due to the dominant majority class, while macro-F1 remains the bottleneck. "
        "Future work should prioritize minority-class recall improvements via imbalance-aware losses, sampling strategies, and calibrated thresholds, "
        "validated by consistent error analysis across runs."
    )
    lines.append("")

    lines.append(r"\section{Conclusion}")
    lines.append(
        f"We reported a GNN defect classification run with best validation macro-F1={_fmt_num(best.get('macro_f1'))} "
        f"and test macro-F1={_fmt_num(test.get('macro_f1'))}, together with automated plots and qualitative examples."
    )
    lines.append("")

    lines.append(r"\begin{thebibliography}{99}")
    lines.append("")

    lines.append(r"\bibitem{nishioka2025fmats}")
    lines.append(
        r"K.~Nishioka, Y.~Kojima, T.~Saito, K.~Kawakami, M.~Washiya, and M.~Muramatsu."
    )
    lines.append(
        r"\newblock Development of defect localization method for perforated carbon-fiber-reinforced plastic specimens using finite element method and graph neural network."
    )
    lines.append(r"\newblock \emph{Frontiers in Materials}, 12:1652484, 2025. doi:\,10.3389/fmats.2025.1652484.")
    lines.append("")

    lines.append(r"\bibitem{scarselli2009gnn}")
    lines.append(r"F.~Scarselli, M.~Gori, A.~C. Tsoi, M.~Hagenbuchner, and G.~Monfardini.")
    lines.append(r"\newblock The graph neural network model.")
    lines.append(r"\newblock \emph{IEEE Transactions on Neural Networks}, 20(1):61--80, 2009.")
    lines.append("")

    lines.append(r"\bibitem{kipf2017gcn}")
    lines.append(r"T.~N. Kipf and M.~Welling.")
    lines.append(r"\newblock Semi-supervised classification with graph convolutional networks.")
    lines.append(r"\newblock In \emph{International Conference on Learning Representations (ICLR)}, 2017. arXiv:1609.02907.")
    lines.append("")

    lines.append(r"\bibitem{velickovic2018gat}")
    lines.append(r"P.~Veli{\v{c}}kovi{\'c}, G.~Cucurull, A.~Casanova, A.~Romero, P.~Li{\`o}, and Y.~Bengio.")
    lines.append(r"\newblock Graph attention networks.")
    lines.append(r"\newblock In \emph{International Conference on Learning Representations (ICLR)}, 2018. arXiv:1710.10903.")
    lines.append("")

    lines.append(r"\bibitem{fey2019pyg}")
    lines.append(r"M.~Fey and J.~E. Lenssen.")
    lines.append(r"\newblock Fast graph representation learning with PyTorch Geometric.")
    lines.append(r"\newblock arXiv:1903.02428, 2019.")
    lines.append("")

    lines.append(r"\bibitem{lin2017focal}")
    lines.append(r"T.-Y. Lin, P.~Goyal, R.~Girshick, K.~He, and P.~Doll{\'a}r.")
    lines.append(r"\newblock Focal loss for dense object detection.")
    lines.append(r"\newblock In \emph{Proceedings of the IEEE International Conference on Computer Vision (ICCV)}, 2017.")
    lines.append("")

    lines.append(r"\bibitem{cui2019classbalanced}")
    lines.append(r"Y.~Cui, M.~Jia, T.-Y. Lin, Y.~Song, and S.~Belongie.")
    lines.append(r"\newblock Class-balanced loss based on effective number of samples.")
    lines.append(
        r"\newblock In \emph{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, pages 9268--9277, 2019. doi:\,10.1109/CVPR.2019.00949."
    )
    lines.append("")

    lines.append(r"\bibitem{menon2021logit}")
    lines.append(r"A.~K. Menon, S.~Jayasumana, A.~S. Rawat, H.~Jain, A.~Veit, and S.~Kumar.")
    lines.append(r"\newblock Long-tail learning via logit adjustment.")
    lines.append(r"\newblock In \emph{International Conference on Learning Representations (ICLR)}, 2021. arXiv:2007.07314.")
    lines.append("")

    lines.append(r"\bibitem{wijethunga2025dualgraph}")
    lines.append(r"R.~Wijethunga, J.~Samarabandu, and A.~Sadhu.")
    lines.append(r"\newblock Robust and efficient dual-graph neural networks for structural damage detection and localization.")
    lines.append(r"\newblock \emph{Engineering Structures}, 343:121265, 2025. doi:\,10.1016/j.engstruct.2025.121265.")
    lines.append("")

    lines.append(r"\bibitem{yehia2025whatlieswithin}")
    lines.append(r"A.~S. Yehia, D.~K. Harris, and A.~A. Aljundi.")
    lines.append(r"\newblock What lies within: Utilizing graph neural networks for subsurface detection in finite element simulations.")
    lines.append(r"\newblock \emph{Engineering Structures}, 341:120842, 2025.")
    lines.append("")

    lines.append(r"\end{thebibliography}")
    lines.append("")

    lines.append(r"\end{document}")
    return "\n".join([l for l in lines if l is not None]).strip() + "\n"


def _which(cmd: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["bash", "-lc", f"command -v {cmd}"], text=True).strip()
        return out or None
    except Exception:
        return None


def _run(cmd: List[str], cwd: Path) -> int:
    p = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(p.returncode)

def _convert_pptx_to_pdf(pptx_path: Path, out_dir: Path) -> Path:
    soffice = _which("soffice") or _which("libreoffice")
    if soffice is None:
        raise SystemExit(
            "[FATAL] PPTX→PDF 変換に libreoffice(soffice) が必要です。\n"
            "インストール例:\n"
            "  sudo apt-get update && sudo apt-get install -y libreoffice\n"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    pptx_path = pptx_path.expanduser().resolve()
    expected_pdf = (out_dir / f"{pptx_path.stem}.pdf").resolve()
    if expected_pdf.exists():
        expected_pdf.unlink()  # ensure we don't mistake an old artifact as success

    cmd = [
        soffice,
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to",
        "pdf:impress_pdf_Export",
        "--outdir",
        str(out_dir.resolve()),
        str(pptx_path),
    ]
    p = subprocess.run(cmd, cwd=str(out_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    out = (p.stdout or "").strip()
    if p.returncode != 0 or ("Error:" in out):
        raise SystemExit(
            "[FATAL] PPTX→PDF conversion failed.\n"
            f"- cmd: {' '.join(cmd)}\n"
            f"- exit: {p.returncode}\n"
            f"- output:\n{out}\n"
        )

    if not expected_pdf.exists():
        raise SystemExit(
            "[FATAL] PPTX→PDF conversion finished but output PDF not found.\n"
            f"- expected: {expected_pdf}\n"
            f"- cmd: {' '.join(cmd)}\n"
            f"- output:\n{out}\n"
        )
    return expected_pdf


def _render_slides(d: Dict[str, Any], assets: Dict[str, Any]) -> str:
    # Marp-compatible Markdown
    res = d.get("results", {})
    best = res.get("best", {})
    test = res.get("test", {})
    ds = d.get("dataset", {}) or {}

    lines: List[str] = []
    lines.append("---")
    lines.append("marp: true")
    lines.append("paginate: true")
    lines.append("size: 16:9")
    lines.append("style: |")
    lines.append("  img { max-width: 100%; max-height: 70vh; object-fit: contain; }")
    lines.append("  table { font-size: 0.9em; }")
    lines.append(f"title: \"{d.get('run_id')}\"")
    lines.append("---")
    lines.append("")

    lines.append(f"# {d.get('run_id')}")
    lines.append("")
    lines.append(f"- profile: `{d.get('profile')}`")
    lines.append(f"- git: `{d.get('git_short_sha')}`")
    lines.append("")

    lines.append("---")
    lines.append("## 目的")
    lines.append("- 19クラス（強い不均衡）で欠陥分類の性能を上げる")
    lines.append("- 指標は **macro-F1** を重視（多数派accuracyだけでは不十分）")
    lines.append("")

    lines.append("---")
    lines.append("## データセット")
    if ds:
        lines.append("")
        lines.append("| item | value |")
        lines.append("|---|---|")
        lines.append(f"| type | {ds.get('dataset_type')} |")
        lines.append(f"| train/val/test | {ds.get('train_pairs')}/{ds.get('val_pairs')}/{ds.get('test_pairs')} |")
        lines.append(f"| nodes/sample | {ds.get('nodes_per_sample')} |")
    lines.append("")

    lines.append("---")
    lines.append("## 結果（要点）")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| best macro-F1 (val) | **{best.get('macro_f1')}** (epoch {best.get('epoch')}) |")
    lines.append(f"| test macro-F1 | **{test.get('macro_f1')}** |")
    lines.append(f"| test weighted-F1 / acc | {test.get('weighted_f1')} / {test.get('accuracy')} |")
    lines.append("")

    if assets.get("combined_plot"):
        lines.append("---")
        lines.append("## 学習曲線")
        lines.append(_md_img(assets.get("combined_plot"), "combined_plot"))
        lines.append("")

    lines.append("---")
    lines.append("## 失敗分析（次に見るもの）")
    lines.append("- worst accuracy ファイルと、PredDefectRatioが高いファイルを重点確認")
    lines.append("- confusion matrix がある場合は A→B の偏りを抽出して可視化を紐づけ")
    lines.append("")

    lines.append("---")
    lines.append("## 今後")
    lines.append("- 不均衡対策（sampler / focal / logit adjust）を同一条件で比較")
    lines.append("- minority class の recall を改善（macro-F1主導）")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate report/paper/slides from a run directory.")
    ap.add_argument("--run", required=True, help="Path to runs/<RUN_ID> directory (or a renamed/symlinked run).")
    ap.add_argument("--out", default="", help="Output directory (default: <repo>/reports/<run_folder_name>).")
    ap.add_argument("--max_spatial", type=int, default=12, help="Max number of spatial visualizations to embed.")
    ap.add_argument("--no_copy_assets", action="store_true", help="Do not copy images; link to originals instead.")
    ap.add_argument("--build_pdf", action="store_true", help="Build paper.pdf from paper.tex (uses tectonic).")
    ap.add_argument("--build_pptx", action="store_true", help="Build slides.pptx (requires python-pptx).")
    ap.add_argument("--build_slides_pdf", action="store_true", help="Build slides.pdf from slides.pptx (uses libreoffice).")
    args = ap.parse_args()

    run_dir = Path(args.run).expanduser()
    loaded: LoadedRun = load_run(run_dir, max_spatial=int(args.max_spatial))

    out_dir = Path(args.out) if args.out else (_repo_root() / "reports" / _safe_name(run_dir.name))
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"

    report_data = to_report_dict(loaded, out_dir=out_dir)
    _write_json(out_dir / "report_data.json", report_data)

    # Resolve assets (copy selected images for portability by default)
    assets: Dict[str, Any] = {}
    plots = ((report_data.get("artifacts", {}) or {}).get("plots", {}) or {})
    for k, prefix in [("combined_plot", "plot"), ("loss_plot", "plot"), ("macro_f1_plot", "plot")]:
        src = plots.get(k)
        assets[k] = _copy_asset(src, assets_dir, prefix) if not args.no_copy_assets else src

    # Copy existing evaluation plots if present (Predict_truth)
    for k, prefix in [
        ("confusion_truth_png", "truth"),
        ("f1score_truth_png", "truth"),
        ("detailed_metrics_truth_png", "truth"),
        ("pred_vs_true_truth_png", "truth"),
    ]:
        src = plots.get(k)
        rel = _copy_asset(src, assets_dir, prefix) if (src and not args.no_copy_assets) else src
        if rel:
            assets[k] = rel

    art = (report_data.get("artifacts", {}) or {})

    # Prefer presentation-friendly spatial PNGs if available.
    # By design, these live next to spatial_visualizations as:
    #   <predict_dir>/spatial_visualizations_presentation
    spatial_srcs = art.get("spatial_examples", []) or []
    spatial_vis_dir = art.get("spatial_visualizations_dir")
    pres_candidates: List[Path] = []
    if isinstance(spatial_vis_dir, str) and spatial_vis_dir:
        pres_dir = Path(spatial_vis_dir).parent / "spatial_visualizations_presentation"
        if pres_dir.exists() and pres_dir.is_dir():
            pres_candidates = sorted(pres_dir.glob("*_spatial_visualization.png"))

    if pres_candidates:
        # Randomly sample from presentation folder. Seed changes per render.
        seed = str(report_data.get("generated_at") or "")
        rng = random.Random(seed)
        k = min(int(args.max_spatial), len(pres_candidates))
        chosen = rng.sample(pres_candidates, k=k)
        spatial_srcs = [str(p) for p in chosen]

    spatial_out: List[str] = []
    if not args.no_copy_assets:
        # Clean old spatial_* to avoid mixing different random samples across re-runs.
        for old in assets_dir.glob("spatial_*.png"):
            try:
                old.unlink()
            except Exception:
                pass
    for i, src in enumerate(spatial_srcs):
        if args.no_copy_assets:
            spatial_out.append(src)
        else:
            rel = _copy_asset(src, assets_dir, f"spatial_{i:03d}")
            if rel:
                spatial_out.append(rel)
    assets["spatial_examples"] = spatial_out

    # Generate confusion/per-class plots from npy (preferred, reproducible)
    pred_arrays = (report_data.get("artifacts", {}) or {}).get("pred_arrays", {}) or {}
    labels_npy = pred_arrays.get("all_labels_npy")
    preds_npy = pred_arrays.get("all_preds_npy")
    num_classes = int((report_data.get("dataset", {}) or {}).get("number_of_classes") or 19)
    if labels_npy and preds_npy and not args.no_copy_assets:
        env_python = (report_data.get("env", {}) or {}).get("python") or sys.executable
        cmd = [
            str(env_python),
            str(_repo_root() / "tools" / "autoreport" / "plot_from_npy.py"),
            "--labels",
            str(labels_npy),
            "--preds",
            str(preds_npy),
            "--probs",
            str((pred_arrays.get("all_probs_npy") or "")),
            "--outdir",
            str(assets_dir),
            "--num_classes",
            str(num_classes),
            "--prefix",
            "auto",
            "--topk",
            "50",
        ]
        subprocess.run(cmd, check=False)

        # If generated, attach to assets for report/paper
        for fname, key in [
            ("auto_confusion_counts.png", "confusion_counts"),
            ("auto_confusion_normalized.png", "confusion_normalized"),
            ("auto_per_class_f1.png", "per_class_f1"),
            ("auto_confidence_hist.png", "confidence_hist"),
            ("auto_entropy_hist.png", "entropy_hist"),
        ]:
            p = assets_dir / fname
            if p.exists():
                assets[key] = str(p.relative_to(out_dir))

        top_csv = assets_dir / "auto_top_confusions.csv"
        if top_csv.exists():
            assets["top_confusions_csv"] = str(top_csv.relative_to(out_dir))

    # Dataset property plots (csv/log derived)
    if not args.no_copy_assets:
        env_python = (report_data.get("env", {}) or {}).get("python") or sys.executable
        cmd = [
            str(env_python),
            str(_repo_root() / "tools" / "autoreport" / "plot_dataset_stats.py"),
            "--report_data",
            str(out_dir / "report_data.json"),
            "--outdir",
            str(assets_dir),
        ]
        subprocess.run(cmd, check=False)
        for fname, key in [
            ("dataset_class_counts_log.png", "dataset_class_counts_log"),
            ("dataset_class_counts.png", "dataset_class_counts"),
            ("dataset_class_weights.png", "dataset_class_weights"),
            ("dataset_defect_ndf_counts.png", "dataset_defect_ndf_counts"),
            ("dataset_preddefectratio_hist.png", "dataset_preddefectratio_hist"),
            ("dataset_layer_counts.png", "dataset_layer_counts"),
            ("dataset_hw_counts.png", "dataset_hw_counts"),
        ]:
            p = assets_dir / fname
            if p.exists():
                assets[key] = str(p.relative_to(out_dir))

    # Ensure "good/bad" files are visualized (copy their spatial pngs if present)
    key_examples = ((report_data.get("artifacts") or {}).get("key_file_examples") or {})
    def _copy_key_examples(list_key: str, prefix: str, max_n: int = 6) -> List[str]:
        items = key_examples.get(list_key) or []
        out_paths: List[str] = []
        if not isinstance(items, list):
            return out_paths
        if not args.no_copy_assets:
            # Clean old artifacts to avoid mixing runs/re-runs
            for old in assets_dir.glob(f"{prefix}_*.png"):
                try:
                    old.unlink()
                except Exception:
                    pass
        for i, it in enumerate(items):
            if i >= max_n:
                break
            if not isinstance(it, dict):
                continue
            sp = it.get("spatial_png")
            rel = _copy_asset(sp, assets_dir, f"{prefix}_{i:02d}") if (sp and not args.no_copy_assets) else sp
            if rel:
                out_paths.append(rel)
        return out_paths

    assets["key_worst_spatial"] = _copy_key_examples("worst_accuracy", "worst", max_n=6)
    assets["key_best_spatial"] = _copy_key_examples("best_accuracy_defect", "best", max_n=6)

    # Write documents
    (out_dir / "report.md").write_text(_render_report(report_data, assets), encoding="utf-8")
    (out_dir / "paper.md").write_text(_render_paper_skeleton(report_data, assets), encoding="utf-8")
    (out_dir / "paper.tex").write_text(_render_paper_tex(report_data, assets), encoding="utf-8")
    (out_dir / "slides.md").write_text(_render_slides(report_data, assets), encoding="utf-8")

    # Convenience pointer
    (out_dir / "GENERATED_AT.txt").write_text(datetime.now().isoformat() + "\n", encoding="utf-8")

    # Optional builds
    if args.build_pdf:
        # Prefer tectonic in the same env as torchrun, if provided.
        env_python = (report_data.get("env", {}) or {}).get("python")
        tectonic_path = None
        if isinstance(env_python, str) and env_python:
            cand = str(Path(env_python).parent / "tectonic")
            if Path(cand).exists():
                tectonic_path = cand
        if tectonic_path is None:
            tectonic_path = _which("tectonic")

        if tectonic_path is None:
            raise SystemExit(
                "[FATAL] tectonic が見つかりません。conda env に入れる例:\n"
                "  /home/nishioka/miniconda3/bin/conda install -n gnn_final_env -c conda-forge tectonic -y\n"
            )
        code = _run([tectonic_path, "paper.tex"], cwd=out_dir)
        if code != 0:
            raise SystemExit(f"[FATAL] PDF build failed (tectonic exit={code}). out_dir={out_dir}")

    if args.build_pptx:
        # Build with python-pptx. If this interpreter doesn't have it, use env python if available.
        try:
            import pptx  # noqa: F401
            code = _run([sys.executable, str(_repo_root() / "tools" / "autoreport" / "build_pptx.py"), "--out", str(out_dir)], cwd=_repo_root())
            if code != 0:
                raise SystemExit(f"[FATAL] PPTX build failed (exit={code}). out_dir={out_dir}")
        except Exception:
            env_python = (report_data.get("env", {}) or {}).get("python")
            if not env_python:
                raise SystemExit(
                    "[FATAL] python-pptx がこの python に入っていません。gnn_final_env の python を使ってください:\n"
                    "  /home/nishioka/miniconda3/envs/gnn_final_env/bin/python tools/autoreport/render_run.py --run <RUN> --build_pptx\n"
                )
            code = _run([str(env_python), str(_repo_root() / "tools" / "autoreport" / "build_pptx.py"), "--out", str(out_dir)], cwd=_repo_root())
            if code != 0:
                raise SystemExit(f"[FATAL] PPTX build failed (exit={code}). out_dir={out_dir}")

    if args.build_slides_pdf:
        pptx_path = out_dir / "slides.pptx"
        if not pptx_path.exists():
            # build pptx first
            env_python = (report_data.get("env", {}) or {}).get("python")
            if env_python:
                code = _run(
                    [str(env_python), str(_repo_root() / "tools" / "autoreport" / "build_pptx.py"), "--out", str(out_dir)],
                    cwd=_repo_root(),
                )
            else:
                code = _run(
                    [sys.executable, str(_repo_root() / "tools" / "autoreport" / "build_pptx.py"), "--out", str(out_dir)],
                    cwd=_repo_root(),
                )
            if code != 0 or not pptx_path.exists():
                raise SystemExit("[FATAL] slides.pptx が作れないため slides.pdf 変換を中止しました。")
        _convert_pptx_to_pdf(pptx_path, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

