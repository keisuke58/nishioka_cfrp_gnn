import os
import sys
import re
import random
import numpy as np

import builtins as _builtins
import logging

# -----------------------------------------------------------------------------
# Logging / print control (DDP-safe)
# -----------------------------------------------------------------------------
# - All logs/prints are gated to rank0 to avoid log spam / slowdown under DDP.
# - Set VERBOSE_PRINT=1 or LOG_LEVEL=DEBUG to enable debug prints (still rank0 only).
VERBOSE_PRINT = int(os.environ.get("VERBOSE_PRINT", "0") or "0")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "").strip().upper() or ("DEBUG" if VERBOSE_PRINT else "INFO")

_CACHED_RANK = None


def _infer_rank() -> int:
    """Best-effort rank detection without hard dependency at import-time."""
    global _CACHED_RANK
    if _CACHED_RANK is not None:
        return int(_CACHED_RANK)

    # 1) Torch distributed (when available & initialized)
    try:
        import torch.distributed as _dist  # lazy import

        if _dist.is_available() and _dist.is_initialized():
            _CACHED_RANK = int(_dist.get_rank())
            return int(_CACHED_RANK)
    except Exception:
        pass

    # 2) Common environment variables (torchrun / slurm)
    for k in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                _CACHED_RANK = int(v)
                return int(_CACHED_RANK)
            except Exception:
                continue

    # 3) Single process fallback
    _CACHED_RANK = 0
    return 0


def _is_rank0(explicit_rank=None) -> bool:
    if explicit_rank is not None:
        try:
            return int(explicit_rank) == 0
        except Exception:
            # fall back to inference below
            pass
    return _infer_rank() == 0


class _Rank0Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Always gate to rank0, regardless of logger usage.
        return _is_rank0()


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("GNN")
    # Ensure idempotent setup (avoid duplicate handlers on re-import)
    if not getattr(logger, "_rank0_configured", False):
        logger.setLevel(logging.DEBUG)  # handler controls effective level
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        handler.addFilter(_Rank0Filter())
        handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
        logger._rank0_configured = True
    return logger


logger = _get_logger()


def print(*args, **kwargs):
    """rank0-only print redirected to logger.

    Existing code uses many print()s; redirecting avoids massive refactors while
    providing structured logging and level control.

    Accepts optional kwarg: rank=...
    """
    explicit_rank = kwargs.pop("rank", None)
    if not _is_rank0(explicit_rank):
        return None

    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    # Keep behavior predictable: ignore unknown kwargs rather than crashing
    kwargs.pop("flush", None)
    kwargs.pop("file", None)

    msg = sep.join("" if a is None else str(a) for a in args)
    if end and end != "\n":
        msg = msg + end

    # Map common prefixes to log levels
    s = msg.lstrip()
    if s.startswith("[ERROR]") or s.startswith("ERROR"):
        logger.error(msg)
    elif s.startswith("[WARNING]") or s.startswith("WARNING") or s.startswith("警告"):
        logger.warning(msg)
    elif s.startswith("[DEBUG]") or s.startswith("DEBUG"):
        logger.debug(msg)
    else:
        logger.info(msg)
    return None

_RE_DEFECT = re.compile(r"Defect_L(?P<L>\d+)_B(?P<B>\d+)_el(?P<el>\d+)")
_RE_NDF = re.compile(r"NoiseDefectFree_(?P<id>\d+)")


def group_id_from_filename(filename: str, group_key: str = "LBel") -> str:
    """
    Derive a group id string from a sample filename for leakage checks / group-purged eval.

    Supported:
    - Defect_L{L}_B{B}_el{el}_... -> keys: L, B, el, LB, LBel (default)
    - NoiseDefectFree_000123...    -> NDF:{id}
    """
    fn = str(filename)
    if fn.startswith("NoiseDefectFree_"):
        m = _RE_NDF.search(fn)
        return f"NDF:{m.group('id')}" if m else f"NDF:{fn}"

    m = _RE_DEFECT.search(fn)
    if not m:
        return fn

    L = m.group("L")
    B = m.group("B")
    el = m.group("el")
    key = str(group_key)
    if key == "L":
        return f"L{L}"
    if key == "B":
        return f"B{B}"
    if key == "el":
        return f"el{el}"
    if key == "LB":
        return f"L{L}_B{B}"
    # default: Layer+Block+element
    return f"L{L}_B{B}_el{el}"


def group_disjoint_split(
    all_pairs_with_dir,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    *,
    group_key: str = "LBel",
    seed: int = 42,
):
    """
    Split samples into train/val/test ensuring **no group id overlaps** across splits.

    Args:
        all_pairs_with_dir: list of tuples (data_file, label_file, data_dir)
        train_ratio/val_ratio/test_ratio: target ratios (by sample count)
        group_key: grouping key for leakage prevention (e.g., LBel)
        seed: RNG seed for reproducible group shuffle

    Returns:
        train_pairs, val_pairs, test_pairs, train_map, val_map, test_map
    """
    pairs = list(all_pairs_with_dir)
    if len(pairs) == 0:
        return [], [], [], {}, {}, {}

    # Normalize ratios
    tr = float(train_ratio)
    vr = float(val_ratio)
    ter = float(test_ratio)
    s = tr + vr + ter
    if s <= 0:
        raise ValueError("split ratios must sum to > 0")
    tr, vr, ter = tr / s, vr / s, ter / s

    # Group items
    group_to_items = {}
    for data_file, label_file, data_dir in pairs:
        gid = group_id_from_filename(data_file, group_key=group_key)
        group_to_items.setdefault(gid, []).append((data_file, label_file, data_dir))

    # Shuffle groups deterministically
    rng = np.random.RandomState(seed)
    group_ids = list(group_to_items.keys())
    rng.shuffle(group_ids)

    total = len(pairs)
    target_train = int(total * tr)
    target_val = int(total * vr)
    target_test = total - target_train - target_val

    train_items = []
    val_items = []
    test_items = []

    # Greedy fill by group while keeping disjointness by construction
    for gid in group_ids:
        items = group_to_items[gid]
        # Choose bucket based on remaining capacity (by sample count)
        rem_train = target_train - len(train_items)
        rem_val = target_val - len(val_items)
        rem_test = target_test - len(test_items)

        # Prefer buckets with remaining room; fallback to smallest bucket
        choices = []
        if rem_train > 0:
            choices.append(("train", rem_train))
        if rem_val > 0:
            choices.append(("val", rem_val))
        if rem_test > 0:
            choices.append(("test", rem_test))
        if choices:
            # Put into the bucket with the most remaining room
            bucket = max(choices, key=lambda x: x[1])[0]
        else:
            # All full (due to integer rounding / group sizes) -> keep splits balanced
            sizes = [("train", len(train_items)), ("val", len(val_items)), ("test", len(test_items))]
            bucket = min(sizes, key=lambda x: x[1])[0]

        if bucket == "train":
            train_items.extend(items)
        elif bucket == "val":
            val_items.extend(items)
        else:
            test_items.extend(items)

    # Post-adjust: ensure exact counts when possible by moving whole groups
    # (This keeps the guarantee: groups remain disjoint.)
    def _bucketize(items_):
        m = {}
        for a, b, c in items_:
            gid = group_id_from_filename(a, group_key=group_key)
            m.setdefault(gid, []).append((a, b, c))
        return m

    train_g = _bucketize(train_items)
    val_g = _bucketize(val_items)
    test_g = _bucketize(test_items)

    def _flatten(gmap):
        out = []
        for gid in gmap:
            out.extend(gmap[gid])
        return out

    def _rebalance(source_map, dest_map, dest_target):
        # Move largest groups first until dest reaches target (or cannot move)
        moved = 0
        if len(_flatten(dest_map)) >= dest_target:
            return moved
        # sort groups by size descending
        gids = sorted(source_map.keys(), key=lambda k: len(source_map[k]), reverse=True)
        for gid in gids:
            if len(_flatten(dest_map)) >= dest_target:
                break
            # move this group
            dest_map[gid] = source_map.pop(gid)
            moved += 1
        return moved

    # If any bucket is underfilled, try to fill it from the most overfilled bucket
    def _size(gmap):
        return len(_flatten(gmap))

    # Iterate a few times to improve counts while preserving disjointness
    for _ in range(3):
        sizes = {"train": _size(train_g), "val": _size(val_g), "test": _size(test_g)}
        # Fill val if needed
        if sizes["val"] < target_val:
            src = "train" if sizes["train"] > sizes["test"] else "test"
            _rebalance(train_g if src == "train" else test_g, val_g, target_val)
        # Fill test if needed
        sizes = {"train": _size(train_g), "val": _size(val_g), "test": _size(test_g)}
        if sizes["test"] < target_test:
            src = "train" if sizes["train"] > sizes["val"] else "val"
            _rebalance(train_g if src == "train" else val_g, test_g, target_test)
        # Fill train if needed
        sizes = {"train": _size(train_g), "val": _size(val_g), "test": _size(test_g)}
        if sizes["train"] < target_train:
            src = "val" if sizes["val"] > sizes["test"] else "test"
            _rebalance(val_g if src == "val" else test_g, train_g, target_train)

    # Final flatten
    train_items = _flatten(train_g)
    val_items = _flatten(val_g)
    test_items = _flatten(test_g)

    train_pairs = [(a, b) for a, b, _ in train_items]
    val_pairs = [(a, b) for a, b, _ in val_items]
    test_pairs = [(a, b) for a, b, _ in test_items]

    train_map = {a: d for a, _, d in train_items}
    val_map = {a: d for a, _, d in val_items}
    test_map = {a: d for a, _, d in test_items}

    return train_pairs, val_pairs, test_pairs, train_map, val_map, test_map


def _select_macro_f1(cm_metrics: dict, macro_mode: str) -> float:
    mode = str(macro_mode).lower()
    if mode in ("support_only", "support", "true_support"):
        return float(cm_metrics.get("macro_f1_support_only", 0.0))
    if mode in ("all", "all_classes"):
        return float(cm_metrics.get("macro_f1_all", 0.0))
    # default: sklearn-like (union of y_true/y_pred labels)
    return float(cm_metrics.get("macro_f1_sklearn_like", cm_metrics.get("macro_f1", 0.0)))


def _best_update_summary_from_cm(
    cm: np.ndarray,
    f1_per_class: np.ndarray,
    support_per_class: np.ndarray,
    topk_classes: int = 5,
    topk_absorb: int = 3,
):
    """
    Low-cost summary used ONLY when best metric updates:
    - list classes with support>0 but F1==0
    - list classes with support==0 (absent in eval set)
    - show "absorb" (top predicted classes) for worst-k classes by F1 (support>0)
    """
    cm = np.asarray(cm)
    f1 = np.asarray(f1_per_class, dtype=float)
    sup = np.asarray(support_per_class, dtype=float)
    num_classes = int(cm.shape[0])

    zero_support = np.where(sup <= 0)[0].tolist()
    zero_f1_with_support = np.where((sup > 0) & (f1 <= 0))[0].tolist()

    # worst-k by F1 among classes that appear in y_true
    valid = np.where(sup > 0)[0]
    worst = []
    if valid.size > 0:
        valid_f1 = f1[valid]
        order = np.argsort(valid_f1)  # ascending
        worst = valid[order[: max(1, int(topk_classes))]].tolist()

    absorb_lines = []
    for c in worst:
        row = cm[c, :].astype(np.int64)
        if row.sum() <= 0:
            continue
        # top predicted labels excluding self (absorb target)
        preds = np.argsort(row)[::-1]
        top = []
        for p in preds:
            if int(p) == int(c):
                continue
            cnt = int(row[p])
            if cnt <= 0:
                continue
            top.append((int(p), cnt))
            if len(top) >= int(topk_absorb):
                break
        if top:
            top_str = ", ".join([f"{p}({cnt})" for p, cnt in top])
            absorb_lines.append(f"  - true {int(c)} (support={int(sup[c])}, f1={float(f1[c]):.4f}) -> {top_str}")

    return {
        "zero_support_classes": zero_support,
        "zero_f1_with_support_classes": zero_f1_with_support,
        "worst_classes": worst,
        "absorb_lines": absorb_lines,
        "num_classes": num_classes,
    }


def metrics_from_confusion_matrix(cm: np.ndarray, eps: float = 1e-12):
    """
    Compute precision/recall/F1 directly from a confusion matrix.

    Notes:
    - Macro-F1 here matches sklearn's default label set behavior by averaging only over
      classes that appear in y_true OR y_pred (i.e., non-zero row or column in cm).
    - Weighted-* averages are weighted by true support (row sums), like sklearn.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square (KxK). Got shape={cm.shape}")

    tp = np.diag(cm)
    support = cm.sum(axis=1)          # true counts per class
    pred_count = cm.sum(axis=0)       # predicted counts per class
    fp = pred_count - tp
    fn = support - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)

    total = support.sum()
    accuracy = tp.sum() / (total + eps) if total > 0 else 0.0

    # Macro definitions (make the "support=0" handling explicit):
    # - sklearn-like: labels in union(y_true, y_pred) => non-zero row OR column
    macro_mask_sklearn_like = (support + pred_count) > 0
    macro_f1_sklearn_like = float(f1[macro_mask_sklearn_like].mean()) if np.any(macro_mask_sklearn_like) else 0.0
    # - support-only: labels that appear in y_true (non-zero row). Recommended for "valに存在しないクラスでMacroが潰れる"回避。
    macro_mask_support_only = support > 0
    macro_f1_support_only = float(f1[macro_mask_support_only].mean()) if np.any(macro_mask_support_only) else 0.0
    # - all-classes: always average over all classes (includes support=0 as 0). Useful only for debugging.
    macro_f1_all = float(f1.mean()) if f1.size > 0 else 0.0

    weighted_precision = float((precision * support).sum() / (total + eps)) if total > 0 else 0.0
    weighted_recall = float((recall * support).sum() / (total + eps)) if total > 0 else 0.0
    weighted_f1 = float((f1 * support).sum() / (total + eps)) if total > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision_per_class": precision,
        "recall_per_class": recall,
        "f1_per_class": f1,
        "support_per_class": support,
        # Backward compatible key (historically used by this script)
        "macro_f1": macro_f1_sklearn_like,
        # Explicit variants (Step0: macro定義の固定)
        "macro_f1_sklearn_like": macro_f1_sklearn_like,
        "macro_f1_support_only": macro_f1_support_only,
        "macro_f1_all": macro_f1_all,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, KFold
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import datetime
import time
from torch.utils.data import DistributedSampler, Sampler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, matthews_corrcoef, balanced_accuracy_score, roc_auc_score, roc_curve, auc
import optuna
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from Loss.focal_loss import FocalLoss

# Global variables for data folder mapping
train_data_folder_map_global = {}
val_data_folder_map_global = {}
test_data_folder_map_global = {}

# Global variables for data folder mapping
train_data_folder_map_global = {}
val_data_folder_map_global = {}
test_data_folder_map_global = {}

# Layer-aware loss wrapper: 層ごとのラベル制約を追加
class LayerAwareLoss(nn.Module):
    """
    層ごとのラベル制約を追加する損失関数ラッパー
    Layer 1 (前半): 0, 1~9 のみ
    Layer 2 (後半): 0, 10~18 のみ
    
    データリークではない理由: これはデータの構造的制約であり、物理的な意味に基づく
    """
    def __init__(self, base_loss_fn, vertices_per_layer=6971, constraint_weight=1.0):
        """
        Args:
            base_loss_fn: ベースとなる損失関数（LogitAdjustLoss, FocalLossLogSoftmax, CrossEntropyLossなど）
            vertices_per_layer: 1層あたりのノード数（デフォルト: 6971）
            constraint_weight: 制約違反のペナルティ重み（デフォルト: 1.0）
        """
        super(LayerAwareLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.vertices_per_layer = vertices_per_layer
        self.constraint_weight = constraint_weight
    
    def forward(self, logits, target, batch=None):
        """
        Args:
            logits: [N, num_classes] のロジット
            target: [N] のクラスインデックス
            batch: [N] のバッチインデックス（各グラフのノードがどのグラフに属するか）
                   バッチサイズが1の場合はNoneでも可
        """
        # ベース損失を計算
        base_loss = self.base_loss_fn(logits, target)
        
        # 層ごとの制約違反ペナルティを計算
        constraint_loss = self._compute_constraint_penalty(logits, batch)
        
        return base_loss + self.constraint_weight * constraint_loss
    
    def _compute_constraint_penalty(self, logits, batch):
        """
        層ごとのラベル制約違反に対するペナルティを計算
        """
        # バッチが指定されていない場合、単一グラフと仮定
        if batch is None:
            # 単一グラフの場合、最初のvertices_per_layerがLayer 1、残りがLayer 2
            N = logits.size(0)
            if N <= self.vertices_per_layer:
                # ノード数が少ない場合は制約を適用しない
                return torch.tensor(0.0, device=logits.device)
            
            layer1_mask = torch.zeros(N, dtype=torch.bool, device=logits.device)
            layer1_mask[:self.vertices_per_layer] = True
            layer2_mask = ~layer1_mask
        else:
            # 複数グラフの場合、各グラフ内でLayer 1とLayer 2を分離
            # batch内の各グラフについて、最初のvertices_per_layerがLayer 1
            unique_batches = torch.unique(batch)
            layer1_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
            layer2_mask = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
            
            for b in unique_batches:
                batch_mask = (batch == b)
                batch_indices = torch.where(batch_mask)[0]
                if len(batch_indices) > self.vertices_per_layer:
                    layer1_indices = batch_indices[:self.vertices_per_layer]
                    layer2_indices = batch_indices[self.vertices_per_layer:]
                    layer1_mask[layer1_indices] = True
                    layer2_mask[layer2_indices] = True
        
        # 予測確率を計算
        probs = F.softmax(logits, dim=1)
        
        # Layer 1の制約違反: 10以上のクラスの予測確率にペナルティ
        layer1_logits = logits[layer1_mask]
        layer1_probs = probs[layer1_mask]
        if layer1_logits.size(0) > 0:
            # クラス10~18の予測確率の合計にペナルティ
            invalid_classes_layer1 = torch.arange(10, 19, device=logits.device)
            layer1_penalty = layer1_probs[:, invalid_classes_layer1].sum(dim=1).mean()
        else:
            layer1_penalty = torch.tensor(0.0, device=logits.device)
        
        # Layer 2の制約違反: 1~9のクラスの予測確率にペナルティ
        layer2_logits = logits[layer2_mask]
        layer2_probs = probs[layer2_mask]
        if layer2_logits.size(0) > 0:
            # クラス1~9の予測確率の合計にペナルティ
            invalid_classes_layer2 = torch.arange(1, 10, device=logits.device)
            layer2_penalty = layer2_probs[:, invalid_classes_layer2].sum(dim=1).mean()
        else:
            layer2_penalty = torch.tensor(0.0, device=logits.device)
        
        return layer1_penalty + layer2_penalty


# LogitAdjust Loss (優先度1の施策: Focal卒業候補)
class LogitAdjustLoss(nn.Module):
    """
    LogitAdjust Loss: logitsにtau*log(pi)を加算してクラス不均衡を補正
    Focal Lossより「全部0」崩壊を防ぎつつminorityを押し上げる効果が高い
    
    Reference: "Long-tail learning via logit adjustment" (Menon et al., 2020)
    
    Note: class_priorは必須（バッチ推定はDDP+不均衡で揺れて危険なため禁止）
    """
    def __init__(self, class_prior: torch.Tensor, tau=1.0, reduction='mean'):
        """
        Args:
            class_prior: 各クラスの事前確率（必須、train全体から計算した固定値）
            tau: 調整強度（デフォルト: 1.0, 推奨範囲: 1.0~2.0）
            reduction: 'mean' or 'sum'
        """
        super(LogitAdjustLoss, self).__init__()
        self.tau = float(tau)
        self.reduction = reduction
        # 固定priorを登録（DDP安全）
        self.register_buffer('log_pi', torch.log(class_prior + 1e-8))
    
    def forward(self, logits, target):
        """
        Args:
            logits: [N, num_classes] のロジット
            target: [N] のクラスインデックス
        """
        # Logit adjustment: logits + tau * log(pi)
        adjusted_logits = logits + self.tau * self.log_pi.unsqueeze(0)
        
        # Cross-entropy loss
        return F.cross_entropy(adjusted_logits, target, reduction=self.reduction)


# Log-softmax version of Focal Loss for numerical stability
class FocalLossLogSoftmax(nn.Module):
    """Focal Loss using log_softmax for numerical stability"""
    def __init__(self, weights=None, gamma=3.0, reduction='mean'):
        super(FocalLossLogSoftmax, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weights is not None:
            self.register_buffer('weights', weights)
        else:
            self.weights = None
    
    def forward(self, logits, target):
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        # Get log probability of the correct class
        target_one_hot = F.one_hot(target, num_classes=logits.size(1))
        log_probs_target = (log_probs * target_one_hot).sum(dim=1)
        probs_target = (probs * target_one_hot).sum(dim=1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs_target) ** self.gamma
        
        # Apply class weights if provided
        if self.weights is not None:
            class_weights = self.weights[target]
            loss = -class_weights * focal_weight * log_probs_target
        else:
            loss = -focal_weight * log_probs_target
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
import matplotlib.cm as cm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# フォントの設定
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 12
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

# カラーマップの設定
dpsss_cmap = 'jet'         # DSPSSデータ用：カラフルな 'jet'
defect_cmap = 'coolwarm'   # Defect Label & Predict Data用：赤青の 'coolwarm'

# グリッド設定
num_cols = 57
num_rows = 125
vertices_per_layer = 6971
total_elements = num_cols * num_rows

# 穴領域の設定
hole_col_start = 26
hole_row_start1 = 51
hole_row_start2 = 77
hole_size_cols = 7
hole_size_rows1 = 7
hole_size_rows2 = 15

# 穴の位置計算
hole_elements = []
for r in range(hole_row_start1, hole_row_start1 + hole_size_rows1):
    for c in range(hole_col_start, hole_col_start + hole_size_cols):
        elem = r * num_cols + c
        hole_elements.append(elem)

for r in range(hole_row_start2, hole_row_start2 + hole_size_rows2):
    for c in range(hole_col_start, hole_col_start + hole_size_cols):
        elem = r * num_cols + c
        hole_elements.append(elem)

hole_elements = [node - 1 for node in hole_elements]
# set化して高速化（線形探索→O(1)検索）
hole_set = set(hole_elements)

# クラス0を黒で表示するカスタムカラーマップ
def get_defect_cmap_with_white_zero():
    """クラス0を黒で表示するカスタムカラーマップ"""
    # coolwarmカラーマップを取得
    coolwarm = cm.get_cmap('coolwarm', 19)
    colors = coolwarm(np.linspace(0, 1, 19))
    # クラス0を黒に設定
    colors[0] = [0.0, 0.0, 0.0, 1.0]  # 黒
    return ListedColormap(colors)

def process_file(file_path):
    """ファイルを読み込んで2層に分けて処理"""
    data = np.load(file_path)
    if len(data.shape) == 2:
        decoded_data = np.argmax(data, axis=1)
    elif len(data.shape) == 1:
        decoded_data = data
    else:
        raise ValueError(f"Unexpected data shape: {data.shape}")

    layer1_data = decoded_data[:vertices_per_layer]
    layer2_data = decoded_data[vertices_per_layer:]

    layer1_full = np.empty(total_elements)
    layer1_full[:] = np.nan
    layer2_full = np.empty(total_elements)
    layer2_full[:] = np.nan

    data_index = 0
    for i in range(total_elements):
        if i not in hole_set and data_index < vertices_per_layer:
            layer1_full[i] = layer1_data[data_index]
            data_index += 1

    data_index = 0
    for i in range(total_elements):
        if i not in hole_set and data_index < vertices_per_layer:
            layer2_full[i] = layer2_data[data_index]
            data_index += 1

    layer1_reshaped = layer1_full.reshape((num_rows, num_cols))
    layer2_reshaped = layer2_full.reshape((num_rows, num_cols))

    return np.flipud(np.rot90(layer1_reshaped, k=1)), np.flipud(np.rot90(layer2_reshaped, k=1))

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# モデル定義 (GAT)
# ----------------------------

def initialize_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)

def edge_dropout(edge_index, drop_prob=0.0002):
    """Edge dropout with device-aware mask generation (DDP安全)"""
    num_edges = edge_index.size(1)
    # device mismatch修正: edge_indexと同じdeviceでmaskを生成
    mask = torch.rand(num_edges, device=edge_index.device) > drop_prob
    edge_index_dropped = edge_index[:, mask]  # ドロップアウト後のエッジを保持
    
    return edge_index_dropped


class GATModel(torch.nn.Module):
    def __init__(self, hidden_channels=8, num_classes=19, dropout=0.1, edge_drop_prob=0.01):
        """
        Args:
            hidden_channels: 隠れ層のチャネル数
            num_classes: クラス数（19クラス: 0=no-defect, 1-18=defect types）
            dropout: Dropout確率（デフォルト: 0.1, 推奨範囲: 0.0~0.3）
            edge_drop_prob: Edge dropout確率（デフォルト: 0.01, 推奨範囲: 0.0~0.05）
        
        Two-stage model:
        - Stage 1: Defect detection (no-defect vs defect) - 2 classes
        - Stage 2: Defect classification (defect types) - 18 classes (1-18)
        """
        super(GATModel, self).__init__()
        self.edge_drop_prob = edge_drop_prob
        
        self.conv1 = GATConv(4, hidden_channels, heads=4, concat=True)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels * 4)
        
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels * 2, heads=4, concat=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels * 8)
        
        self.conv3 = GATConv(hidden_channels * 8, hidden_channels, heads=4, concat=True)
        self.batch_norm3 = nn.BatchNorm1d(hidden_channels * 4)
        
        # Two-stage output layers
        # Stage 1: Defect detection (no-defect=0, defect=1)
        self.fc_detection = nn.Linear(hidden_channels * 4, 2)
        # Stage 2: Defect classification (classes 1-18)
        self.fc_classification = nn.Linear(hidden_channels * 4, 18)
        
        self.dropout = nn.Dropout(p=dropout)

        # プロジェクションレイヤー
        self.proj1 = nn.Linear(4, hidden_channels * 4)
        self.proj2 = nn.Linear(hidden_channels * 4, hidden_channels * 8)
        self.proj3 = nn.Linear(hidden_channels * 8, hidden_channels * 4)

        self.apply(initialize_weights)

    def forward(self, data, return_heads: bool = False):
        x, edge_index = data.x, data.edge_index

        # エッジドロップアウトの適用（トレーニング時のみ）
        if self.training:
            edge_index = edge_dropout(edge_index, drop_prob=self.edge_drop_prob)

        # Layer 1 with residual connection (成功版と同じ方法)
        x_res = self.proj1(x)
        x = F.relu(self.conv1(x, edge_index)) + x_res
        x = self.batch_norm1(x)
        x = self.dropout(x)
        
        # Layer 2 with residual connection
        x_res = self.proj2(x)
        x = F.relu(self.conv2(x, edge_index)) + x_res
        x = self.batch_norm2(x)
        x = self.dropout(x)
        
        # Layer 3 with residual connection
        x_res = self.proj3(x)
        x = F.relu(self.conv3(x, edge_index)) + x_res
        x = self.batch_norm3(x)
        x = self.dropout(x)
        
        # Two-stage heads
        # Stage 1: Defect detection (2 classes: [no-defect, defect])
        detection_logits = self.fc_detection(x)  # [N, 2]

        # Stage 2: Defect classification (18 classes: 1-18)
        classification_logits = self.fc_classification(x)  # [N, 18]

        # Apply layer constraint mask to classification logits (so Stage2 loss is physically consistent)
        # Layer1: only classes {1..9} allowed -> block {10..18} => idx 9..17 in 18-class head
        # Layer2: only classes {10..18} allowed -> block {1..9}  => idx 0..8 in 18-class head
        vertices_per_layer = 6971
        batch = getattr(data, 'batch', None)
        # AMP(fp16) safety: use dtype min instead of -1e9 to avoid overflow
        neg_inf = torch.finfo(classification_logits.dtype).min

        if batch is None:
            N = classification_logits.size(0)
            if N > vertices_per_layer:
                layer1_indices = torch.arange(0, vertices_per_layer, device=classification_logits.device)
                layer2_indices = torch.arange(vertices_per_layer, N, device=classification_logits.device)
                # 18-class indices
                invalid_cls_idx_layer1 = torch.arange(9, 18, device=classification_logits.device)  # classes 10..18
                invalid_cls_idx_layer2 = torch.arange(0, 9, device=classification_logits.device)   # classes 1..9
                if layer1_indices.numel() > 0:
                    classification_logits[layer1_indices.unsqueeze(1), invalid_cls_idx_layer1.unsqueeze(0)] = neg_inf
                if layer2_indices.numel() > 0:
                    classification_logits[layer2_indices.unsqueeze(1), invalid_cls_idx_layer2.unsqueeze(0)] = neg_inf
        else:
            unique_batches = torch.unique(batch)
            invalid_cls_idx_layer1 = torch.arange(9, 18, device=classification_logits.device)
            invalid_cls_idx_layer2 = torch.arange(0, 9, device=classification_logits.device)
            for b in unique_batches:
                batch_mask = (batch == b)
                batch_indices = torch.where(batch_mask)[0]
                if len(batch_indices) > vertices_per_layer:
                    layer1_indices = batch_indices[:vertices_per_layer]
                    layer2_indices = batch_indices[vertices_per_layer:]
                    classification_logits[layer1_indices.unsqueeze(1), invalid_cls_idx_layer1.unsqueeze(0)] = neg_inf
                    classification_logits[layer2_indices.unsqueeze(1), invalid_cls_idx_layer2.unsqueeze(0)] = neg_inf

        # Legacy combined logits (kept for backward compatibility / layer-aware loss wrapper)
        combined_logits = torch.zeros(x.size(0), 19, device=x.device, dtype=x.dtype)
        combined_logits[:, 0] = detection_logits[:, 0]
        combined_logits[:, 1:19] = detection_logits[:, 1:2].expand(-1, 18) + classification_logits
        neg_inf19 = torch.finfo(combined_logits.dtype).min

        # Also apply layer mask to combined logits (same as before)
        if batch is None:
            N = combined_logits.size(0)
            if N > vertices_per_layer:
                layer1_indices = torch.arange(0, vertices_per_layer, device=combined_logits.device)
                layer2_indices = torch.arange(vertices_per_layer, N, device=combined_logits.device)
                invalid_classes_layer1 = torch.arange(10, 19, device=combined_logits.device)
                invalid_classes_layer2 = torch.arange(1, 10, device=combined_logits.device)
                if layer1_indices.numel() > 0:
                    combined_logits[layer1_indices.unsqueeze(1), invalid_classes_layer1.unsqueeze(0)] = neg_inf19
                if layer2_indices.numel() > 0:
                    combined_logits[layer2_indices.unsqueeze(1), invalid_classes_layer2.unsqueeze(0)] = neg_inf19
        else:
            unique_batches = torch.unique(batch)
            invalid_classes_layer1 = torch.arange(10, 19, device=combined_logits.device)
            invalid_classes_layer2 = torch.arange(1, 10, device=combined_logits.device)
            for b in unique_batches:
                batch_mask = (batch == b)
                batch_indices = torch.where(batch_mask)[0]
                if len(batch_indices) > vertices_per_layer:
                    layer1_indices = batch_indices[:vertices_per_layer]
                    layer2_indices = batch_indices[vertices_per_layer:]
                    combined_logits[layer1_indices.unsqueeze(1), invalid_classes_layer1.unsqueeze(0)] = neg_inf19
                    combined_logits[layer2_indices.unsqueeze(1), invalid_classes_layer2.unsqueeze(0)] = neg_inf19

        if return_heads:
            return {
                "combined_logits": combined_logits,
                "detection_logits": detection_logits,
                "classification_logits": classification_logits,
            }

        return combined_logits

# Default gamma for focal loss (can be overridden by args)
# クラス不均衡が極端な場合、gammaを上げることで難易度の高いサンプルに焦点を当てる
default_gamma = 3.0  # Increased from 2.0 to better handle extreme class imbalance
default_class_weight_multiplier = 5.0  # Increased from 1.0 to better handle imbalanced classes


def combine_two_stage_probs(
    detection_logits: torch.Tensor,
    classification_logits: torch.Tensor,
    *,
    det_threshold: float | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Combine two-stage head outputs into 19-class probabilities.

    p0 = p(no-defect)
    p(k) for k=1..18 = p(defect) * p(class=k | defect)

    If det_threshold is set, nodes with p(defect) < threshold are forced to class 0.
    """
    det_probs = torch.softmax(detection_logits, dim=1)  # [N,2]
    p0 = det_probs[:, 0:1]
    pdef = det_probs[:, 1:2]
    cls_probs = torch.softmax(classification_logits, dim=1)  # [N,18]

    probs = torch.cat([p0, pdef * cls_probs], dim=1)  # [N,19]
    # numerical safety (also ensures sum~1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + eps)

    if det_threshold is not None:
        thr = float(det_threshold)
        gate = (pdef.view(-1) >= thr)
        if gate.numel() == probs.size(0):
            forced = ~gate
            if forced.any():
                probs[forced] = 0.0
                probs[forced, 0] = 1.0

    return probs


def two_stage_loss(
    detection_logits: torch.Tensor,
    classification_logits: torch.Tensor,
    y: torch.Tensor,
    det_loss_fn: nn.Module,
    cls_loss_fn: nn.Module,
    *,
    det_weight: float = 1.0,
    cls_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Two-stage training objective:
    - detection: y_det = (y != 0)
    - classification: only for y>0, target = y-1 in [0..17]
    """
    y = y.view(-1).long()
    y_det = (y != 0).long()
    loss_det = det_loss_fn(detection_logits, y_det)

    mask = (y > 0)
    if mask.any():
        y_cls = (y[mask] - 1).long()
        loss_cls = cls_loss_fn(classification_logits[mask], y_cls)
    else:
        loss_cls = torch.zeros((), device=y.device, dtype=loss_det.dtype)

    loss_total = (float(det_weight) * loss_det) + (float(cls_weight) * loss_cls)
    return loss_total, loss_det, loss_cls

def compute_class_weights(labels, multiplier=None, fix_class0_weight=True, class0_weight=1.0):
    """
    クラス重みを計算
    
    Args:
        labels: ラベル配列
        multiplier: 未使用（後方互換性のため）
        fix_class0_weight: Trueの場合、Class 0の重みを固定値にする（推奨）
        class0_weight: Class 0の固定重み値（fix_class0_weight=Trueの場合）
    
    Returns:
        クラス重みテンソル
    """
    class_counts = np.bincount(labels, minlength=19)  # Ensure all 19 classes are counted
    
    if fix_class0_weight:
        # Class 0の重みを固定値にして、他のクラスだけ重み付け（崩壊防止に最も効果的）
        # これにより、Class 0が99.8%を占めていても、重みが極端に小さくなるのを防ぐ
        weights = np.ones(19) * class0_weight  # Class 0を含む全てを初期化
        
        # Class 1-18に対してのみ逆頻度重みを計算
        for i in range(1, 19):
            if class_counts[i] > 0:
                weights[i] = class0_weight * (class_counts[0] / class_counts[i])
            else:
                weights[i] = class0_weight
        
        # 重みの上限を設定（極端に大きくなるのを防ぐ）
        max_weight = class0_weight * 100  # Class 0の100倍まで
        weights = np.clip(weights, class0_weight, max_weight)
    else:
        # 従来の方法（クリップ付き）
        weights = 1.0 / (class_counts + 1e-6)
        # クラス重みのクリップ: 極端に小さくなるのを防ぐ（崩壊防止）
        # 下限を0.1に設定（0.01では正規化後に再び小さくなるため）
        weights = np.clip(weights, 0.1, None)
    
    # 平均が1になるようにスケール（相対比は維持、損失寄与が弱くなるのを防ぐ）
    # 従来の weights.sum() による正規化は全ての重みを小さくしてしまうため、平均基準に変更
    class_weights = weights / np.mean(weights)
    
    return torch.tensor(class_weights, dtype=torch.float)

def save_predictions_to_csv(
    test_pairs,
    all_preds,
    all_labels,
    all_probs,
    output_dir_csv,
    num_nodes_per_sample=13942,
    file_defect_ratio_thresholds=(0.22, 0.23),
):
    """
    予測結果をノード単位およびファイル単位でCSVに保存
    分散学習環境では、実際に処理されたデータ数に基づいて処理する
    """
    os.makedirs(output_dir_csv, exist_ok=True)  # CSV保存先ディレクトリ

    # 実際に処理されたファイル数を計算（分散学習では全ファイルが処理されない可能性がある）
    actual_num_samples = len(all_preds)
    expected_num_files = len(test_pairs)
    actual_num_files = actual_num_samples // num_nodes_per_sample
    
    # データ数の整合性チェック
    if actual_num_samples % num_nodes_per_sample != 0:
        print(f"[WARNING] Total predictions ({actual_num_samples}) is not divisible by num_nodes_per_sample ({num_nodes_per_sample})")
        print(f"         This may indicate incomplete batches. Processing {actual_num_files} complete files.")
    
    # 実際に処理されたファイル数に基づいて処理
    num_files_to_process = min(actual_num_files, expected_num_files)
    
    if num_files_to_process < expected_num_files:
        print(f"[INFO] Processing {num_files_to_process} files out of {expected_num_files} total test files")
        print(f"       (This is normal in distributed training where each rank processes a subset)")
    
    # --- ファイル単位の統計情報を保存 ---
    file_results = []

    # Extra file-level signals (運用向け)
    # - max prob (non0): high-confidence defect node exists
    maxprob_thresholds = (0.90, 0.95, 0.99)
    # - top-k mean (node-wise non0 max prob): moderately confident cluster
    topk_mean_thresholds = (0.01, 0.02, 0.05)
    # - connected component areas on fixed grid (1 layer = 125x57 = 6971 nodes)
    cc_area_thresholds = (1, 10, 25, 50)

    def _topk_mean(x: np.ndarray, k: int) -> float:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return 0.0
        kk = int(k)
        if kk <= 0:
            return float(x.mean())
        kk = min(kk, x.size)
        part = np.partition(x, -kk)[-kk:]
        return float(part.mean())

    def _thr_col_ratio(t: float) -> str:
        return f"PredDefect_t{t:.2f}".replace(".", "_")

    def _thr_col(prefix: str, t: float, fmt: str = ".2f") -> str:
        s = f"{t:{fmt}}"
        return f"PredDefect_{prefix}_t{s}".replace(".", "_")

    def _thr_col_int(prefix: str, t: int) -> str:
        return f"PredDefect_{prefix}_t{int(t)}"

    def _layer_grid(mask_1d: np.ndarray, *, num_rows: int = 125, num_cols: int = 57) -> np.ndarray:
        # Match visualization layout: reshape then rot90+flipud -> (num_cols, num_rows)
        m = np.asarray(mask_1d, dtype=bool).reshape((num_rows, num_cols))
        return np.flipud(np.rot90(m, k=1))

    def _connected_components_4(mask2d: np.ndarray):
        """4-neighborhood connected components. Returns (count, max_area, total_area)."""
        m = np.asarray(mask2d, dtype=bool)
        if m.ndim != 2:
            return 0, 0, 0
        h, w = m.shape
        total_area = int(m.sum())
        if total_area == 0:
            return 0, 0, 0
        visited = np.zeros((h, w), dtype=bool)
        count = 0
        max_area = 0
        for r in range(h):
            for c in range(w):
                if not m[r, c] or visited[r, c]:
                    continue
                count += 1
                stack = [(r, c)]
                visited[r, c] = True
                area = 0
                while stack:
                    rr, cc = stack.pop()
                    area += 1
                    if rr > 0 and m[rr - 1, cc] and not visited[rr - 1, cc]:
                        visited[rr - 1, cc] = True
                        stack.append((rr - 1, cc))
                    if rr + 1 < h and m[rr + 1, cc] and not visited[rr + 1, cc]:
                        visited[rr + 1, cc] = True
                        stack.append((rr + 1, cc))
                    if cc > 0 and m[rr, cc - 1] and not visited[rr, cc - 1]:
                        visited[rr, cc - 1] = True
                        stack.append((rr, cc - 1))
                    if cc + 1 < w and m[rr, cc + 1] and not visited[rr, cc + 1]:
                        visited[rr, cc + 1] = True
                        stack.append((rr, cc + 1))
                if area > max_area:
                    max_area = area
        return int(count), int(max_area), int(total_area)
    for i in range(num_files_to_process):
        filename_tuple = test_pairs[i] if i < len(test_pairs) else f"unknown_file_{i}"
        filename = filename_tuple[0] if isinstance(filename_tuple, tuple) else filename_tuple
        start_idx = i * num_nodes_per_sample
        end_idx = start_idx + num_nodes_per_sample
    
        # 範囲チェック
        if end_idx > len(all_preds) or end_idx > len(all_labels):
            print(f"[WARNING] Skipping file {i}: Index out of range (start_idx={start_idx}, end_idx={end_idx}, total_samples={len(all_preds)})")
            continue
    
        # 各ファイルの予測情報を取得
        sample_preds = np.array(all_preds[start_idx:end_idx])  # 各ファイルの予測
        sample_labels = np.array(all_labels[start_idx:end_idx])  # 実際のラベル
        sample_probs = None
        try:
            if all_probs is not None and len(all_probs) >= end_idx:
                sample_probs = np.array(all_probs[start_idx:end_idx])  # [N,19] expected
        except Exception:
            sample_probs = None
    
        # クラス分布の計算（負の値や範囲外の値がないか確認）
        if np.any(sample_preds < 0) or np.any(sample_preds >= 19):
            print(f"[WARNING] Invalid prediction values found in file: {filename}, skipping...")
            continue
        class_counts = np.bincount(sample_preds.astype(int), minlength=19)  # 19クラス分の予測カウント
    
        # 精度の計算
        accuracy = (sample_preds == sample_labels).sum() / len(sample_labels)

        # --- しきい値ベースのファイル判定（運用用） ---
        # defect_ratio = 1 - count(class0) / total_nodes
        total_nodes = int(class_counts.sum())
        defect_ratio = 0.0
        if total_nodes > 0:
            defect_ratio = 1.0 - (float(class_counts[0]) / float(total_nodes))

        # --- prob-based signals ---
        max_non0_prob = 0.0
        topk50_mean_non0 = 0.0
        topk10_mean_non0 = 0.0
        if sample_probs is not None and getattr(sample_probs, "ndim", 0) == 2 and sample_probs.shape[1] >= 2:
            node_non0 = sample_probs[:, 1:].max(axis=1)  # [N]
            if node_non0.size > 0:
                max_non0_prob = float(node_non0.max())
                topk50_mean_non0 = _topk_mean(node_non0, 50)
                topk10_mean_non0 = _topk_mean(node_non0, 10)

        # --- connected components on fixed grid (per layer) ---
        vertices_per_layer = 6971
        pred_def_mask = (sample_preds.astype(int) != 0)
        cc_l1 = cc_l2 = 0
        cc_l1_max = cc_l2_max = 0
        cc_l1_area = cc_l2_area = 0
        try:
            if pred_def_mask.size >= 2 * vertices_per_layer:
                l1 = _layer_grid(pred_def_mask[:vertices_per_layer])
                l2 = _layer_grid(pred_def_mask[vertices_per_layer:2 * vertices_per_layer])
                cc_l1, cc_l1_max, cc_l1_area = _connected_components_4(l1)
                cc_l2, cc_l2_max, cc_l2_area = _connected_components_4(l2)
        except Exception:
            pass
        cc_total = int(cc_l1 + cc_l2)
        cc_max_area = int(max(cc_l1_max, cc_l2_max))
        cc_total_area = int(cc_l1_area + cc_l2_area)

        # 結果をリストに追加
        row = {
            "Filename": filename,
            "Accuracy": accuracy,
            "Class Distribution": ",".join(map(str, class_counts)),
            "PredDefectRatio": defect_ratio,
            "MaxNon0Prob": max_non0_prob,
            "TopK50MeanNon0Prob": topk50_mean_non0,
            "TopK10MeanNon0Prob": topk10_mean_non0,
            "CC_TotalCount": cc_total,
            "CC_MaxArea": cc_max_area,
            "CC_TotalArea": cc_total_area,
            "CC_L1_Count": int(cc_l1),
            "CC_L1_MaxArea": int(cc_l1_max),
            "CC_L1_TotalArea": int(cc_l1_area),
            "CC_L2_Count": int(cc_l2),
            "CC_L2_MaxArea": int(cc_l2_max),
            "CC_L2_TotalArea": int(cc_l2_area),
        }
        for t in file_defect_ratio_thresholds:
            try:
                tt = float(t)
            except Exception:
                continue
            row[_thr_col_ratio(tt)] = int(defect_ratio >= tt)

        # decisions: max prob / top-k / connected components
        for t in maxprob_thresholds:
            try:
                tt = float(t)
            except Exception:
                continue
            row[_thr_col("MaxProb", tt)] = int(max_non0_prob >= tt)
        for t in topk_mean_thresholds:
            try:
                tt = float(t)
            except Exception:
                continue
            row[_thr_col("TopK50Mean", tt)] = int(topk50_mean_non0 >= tt)
        for t in cc_area_thresholds:
            try:
                ti = int(t)
            except Exception:
                continue
            row[_thr_col_int("CCMaxArea", ti)] = int(cc_max_area >= ti)
            row[_thr_col_int("CCTotalArea", ti)] = int(cc_total_area >= ti)
        file_results.append(row)
    
    # ファイル単位の結果をDataFrameに変換
    if len(file_results) > 0:
        file_results_df = pd.DataFrame(file_results)
        
        # CSV保存
        file_results_csv_path = os.path.join(output_dir_csv, f"file_statistics_{timestamp}.csv")
        try:
            file_results_df.to_csv(file_results_csv_path, index=False)
            print(f"File-level statistics saved to {file_results_csv_path} ({len(file_results)} files)")
        except Exception as e:
            print(f"Error saving file-level statistics: {e}")

        # --- 運用向け: しきい値判定のみのCSVも別途保存 ---
        # 既存のfile_statisticsは維持しつつ、運用用CSVを追加で生成する
        try:
            decision_cols = ["Filename"]
            if "Accuracy" in file_results_df.columns:
                decision_cols.append("Accuracy")
            decision_cols.extend([
                "PredDefectRatio",
                "MaxNon0Prob",
                "TopK50MeanNon0Prob",
                "TopK10MeanNon0Prob",
                "CC_TotalCount",
                "CC_MaxArea",
                "CC_TotalArea",
            ])
            for t in file_defect_ratio_thresholds:
                decision_cols.append(_thr_col_ratio(float(t)))
            for t in maxprob_thresholds:
                decision_cols.append(_thr_col("MaxProb", float(t)))
            for t in topk_mean_thresholds:
                decision_cols.append(_thr_col("TopK50Mean", float(t)))
            for t in cc_area_thresholds:
                decision_cols.append(_thr_col_int("CCMaxArea", int(t)))
                decision_cols.append(_thr_col_int("CCTotalArea", int(t)))
            decision_df = file_results_df[[c for c in decision_cols if c in file_results_df.columns]].copy()
            decision_csv_path = os.path.join(output_dir_csv, f"file_decisions_thresholds_{timestamp}.csv")
            decision_df.to_csv(decision_csv_path, index=False)
            print(f"[INFO] Threshold-based file decisions saved to: {decision_csv_path}")

            # ------------------------------------------------------------
            # Optional: auto threshold optimization (CSVから最適閾値探索)
            # Objective: minimize NDF FP (absolute) then maximize TP (defect recall)
            # ------------------------------------------------------------
            try:
                args_global = globals().get("args", None)
                do_opt = bool(getattr(args_global, "optimize_file_thresholds", False)) if args_global is not None else False
                if do_opt:
                    def _candidate_thresholds(vals: np.ndarray, max_n: int = 2000):
                        v = np.asarray(vals, dtype=np.float64).reshape(-1)
                        if v.size == 0:
                            return np.array([0.0], dtype=np.float64)
                        u = np.unique(v)
                        if u.size <= max_n:
                            return np.sort(u)
                        qs = np.linspace(0.0, 1.0, max_n)
                        return np.unique(np.quantile(v, qs))

                    def _optimize_one(signal_name: str, series: pd.Series, y_true_defect: np.ndarray):
                        x = series.to_numpy()
                        thr_list = _candidate_thresholds(x)
                        ndf = (y_true_defect == 0)
                        defect = ~ndf
                        n_def = int(defect.sum())
                        best = None  # (fp, -tp, thr, tp, tn, fn)
                        for thr in thr_list:
                            pred_def = (x >= thr)
                            fp = int(np.sum(pred_def & ndf))
                            tp = int(np.sum(pred_def & defect))
                            if best is None or (fp < best[0]) or (fp == best[0] and tp > best[2]):
                                tn = int(np.sum((~pred_def) & ndf))
                                fn = int(np.sum((~pred_def) & defect))
                                best = (fp, thr, tp, tn, fn, n_def)
                        fp, thr, tp, tn, fn, n_def = best
                        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                        return {
                            "signal": signal_name,
                            "best_threshold": float(thr),
                            "tp": int(tp),
                            "fp": int(fp),
                            "tn": int(tn),
                            "fn": int(fn),
                            "fp_rate": float(fp_rate),
                            "recall": float(recall),
                            "precision": float(precision),
                            "f1": float(f1),
                            "defect_files": int(n_def),
                        }

                    # GT file label from filename convention
                    filenames = file_results_df["Filename"].astype(str)
                    y_true_defect = (~filenames.str.contains("NoiseDefectFree", regex=False)).astype(int).to_numpy()

                    candidates = [
                        ("PredDefectRatio", "PredDefectRatio"),
                        ("MaxNon0Prob", "MaxNon0Prob"),
                        ("TopK50MeanNon0Prob", "TopK50MeanNon0Prob"),
                        ("TopK10MeanNon0Prob", "TopK10MeanNon0Prob"),
                        ("CC_MaxArea", "CC_MaxArea"),
                        ("CC_TotalArea", "CC_TotalArea"),
                    ]
                    rows = []
                    for disp, col in candidates:
                        if col in file_results_df.columns:
                            rows.append(_optimize_one(disp, file_results_df[col], y_true_defect))

                    if rows:
                        opt_df = pd.DataFrame(rows).sort_values(["fp", "tp"], ascending=[True, False])
                        opt_path = os.path.join(output_dir_csv, f"threshold_optimization_{timestamp}.csv")
                        opt_df.to_csv(opt_path, index=False)
                        print(f"[INFO] Threshold optimization saved to: {opt_path}")
                        # show top-3
                        topn = opt_df.head(3)
                        print("[INFO] Best thresholds (FP-min then TP-max):")
                        for _, r in topn.iterrows():
                            print(
                                f"  - {r['signal']}: thr={r['best_threshold']:.6g}  "
                                f"FP={int(r['fp'])}  TP={int(r['tp'])}  "
                                f"recall={r['recall']:.4f}  fp_rate={r['fp_rate']:.4f}"
                            )
            except Exception as e:
                print(f"[WARNING] Failed to run threshold optimization: {e}")

            # Quick summary for NoiseDefectFree vs others (if filename convention is used)
            if "Filename" in decision_df.columns:
                filenames = decision_df["Filename"].astype(str)
                is_ndf = filenames.str.contains("NoiseDefectFree", regex=False)
                ndf_n = int(is_ndf.sum())
                def_n = int((~is_ndf).sum())
                for t in file_defect_ratio_thresholds:
                    col = _thr_col_ratio(float(t))
                    if col not in decision_df.columns:
                        continue
                    preds = decision_df[col].astype(int)
                    if ndf_n > 0:
                        fp = int(preds[is_ndf].sum())
                        print(f"[INFO] t={float(t):.2f}  NDF_FP_rate={fp/ndf_n:.4f} ({fp}/{ndf_n})")
                    if def_n > 0:
                        tp = int(preds[~is_ndf].sum())
                        print(f"[INFO] t={float(t):.2f}  Defect_recall={tp/def_n:.4f} ({tp}/{def_n})")
        except Exception as e:
            print(f"[WARNING] Failed to save threshold-based file decisions CSV: {e}")
    else:
        print(f"[WARNING] No file results to save. This may indicate a data mismatch issue.")

def evaluate_and_visualize(final_test_loader, train_loader, ddp_model, device, class_weights, test_pairs, train_pairs, num_nodes_per_sample=13942, gamma=2.0, rank=0, world_size=1, test_dataset=None, x_coords=None, y_coords=None, z_coords=None, output_root=None, run_id=None, macro_mode="support_only"):
    """
    推論と可視化を実行
    注意: 分散学習の場合、rank 0のみで実行して順序を保証する
    
    Args:
        x_coords, y_coords, z_coords: 座標データ（可視化の軸ラベル用）
    """
    ddp_model.eval()# モデルを評価モードに切り替え
    
    # rank 0のみで実行（順序を保証するため）
    if rank != 0:
        print(f"[INFO] Rank {rank}: Skipping evaluation (only rank 0 evaluates)")
        return
    
    # rank 0でtest_loaderを再作成（順序を保証するため、samplerなしで全データを順番に処理）
    if test_dataset is not None:
        print(f"[INFO] Recreating test_loader on rank 0 without DistributedSampler to ensure correct file-prediction mapping")
        from torch_geometric.loader import DataLoader
        final_test_loader = DataLoader(test_dataset, batch_size=final_test_loader.batch_size, shuffle=False)
    
    test_preds = []
    test_labels = []
    test_probs = []
    processed_files = []  # list of dicts: {filename, preds, labels, probs}
    total_loss = 0.0
    correct = 0
    total_samples = 0

    # Loss function for evaluation - model outputs logits, use same approach as training
    # Default to log_softmax version for numerical stability
    use_log_softmax = True  # Default to True for numerical stability
    if use_log_softmax:
        loss_fn = FocalLossLogSoftmax(weights=class_weights.to(device), gamma=gamma).to(device)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
    
    print(f"[INFO] Starting evaluation on rank {rank} (ensuring correct file-prediction mapping)")
    
    with torch.no_grad():  # 評価時は勾配計算を無効化
        for batch_idx, batch in enumerate(final_test_loader):
            batch = batch.to(device)
            out = ddp_model(batch)  # Model outputs logits
            y = batch.y

            # Calculate loss (out is logits, loss function handles it)
            loss = loss_fn(out, y)
            total_loss += loss.item()

            # Model outputs logits, apply softmax to get probabilities
            probs = torch.softmax(out, dim=1)
            test_probs.extend(probs.cpu().numpy())

            # 実ラベル
            test_labels.extend(y.cpu().numpy())

            # 予測ラベル
            pred = out.argmax(dim=1)
            test_preds.extend(pred.cpu().numpy())

            correct += (pred == y).sum().item()
            total_samples += y.size(0)
            
            # ---- File name ↔ prediction mapping (DDP-safe, order-safe) ----
            # PyG Batch has:
            # - batch.ptr: node-offset pointer per graph (len=num_graphs+1)
            # - batch.filename: list[str] (collated from each Data.filename)
            if hasattr(batch, 'ptr') and hasattr(batch, 'filename'):
                ptr = batch.ptr
                filenames = batch.filename
                if isinstance(filenames, str):
                    filenames = [filenames]
                # ptr is 1D tensor on device; iterate graph slices
                num_graphs = len(filenames)
                for g in range(num_graphs):
                    s = int(ptr[g].item())
                    e = int(ptr[g + 1].item())
                    processed_files.append({
                        "filename": filenames[g],
                        "preds": pred[s:e].cpu().numpy(),
                        "labels": y[s:e].cpu().numpy(),
                        "probs": probs[s:e].cpu().numpy(),
                    })
            else:
                # Fallback (should not happen because prepare_data sets data.filename)
                processed_files = None

    test_accuracy = correct / total_samples if total_samples > 0 else 0.0
    avg_test_loss = total_loss / len(final_test_loader) if len(final_test_loader) > 0 else 0.0

    # --- 出力ディレクトリの作成（一時的な名前、後でMacro F1スコアを含む名前に変更） ---
    # データセット情報を取得（関数の引数から、またはグローバル変数から）
    import sys
    # グローバル変数から取得を試みる
    globals_dict = globals()
    train_pairs_count = len(train_pairs) if train_pairs is not None else 0
    test_pairs_count = len(test_pairs) if test_pairs is not None else 0
    # val_pairsはグローバル変数から取得
    val_pairs_global = globals_dict.get('val_pairs', [])
    if val_pairs_global:
        val_pairs_count = len(val_pairs_global)
    else:
        val_pairs_count = 0
    # argsとtimestampはグローバル変数から取得
    args_global = globals_dict.get('args', None)
    timestamp_global = globals_dict.get('timestamp', None)
    if args_global is None:
        # argsがグローバル変数にない場合は、デフォルト値を使用
        data_usage_ratio = 1.0
    else:
        data_usage_ratio = getattr(args_global, 'data_usage_ratio', 1.0)
    if timestamp_global is None:
        # timestampがグローバル変数にない場合は、現在時刻から生成
        from datetime import datetime
        timestamp_global = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = timestamp_global
    
    # データセット情報を簡潔に表記（NoiseDefectFree = NDF）
    dataset_info = f"NDF_T{train_pairs_count}V{val_pairs_count}Te{test_pairs_count}"
    if data_usage_ratio < 1.0:
        dataset_info += f"R{int(data_usage_ratio*100)}"
    
    # Output root (default keeps backward-compatible locations)
    if output_root is None or str(output_root).strip() == "":
        output_root = "/home/nishioka/GNN/GNN_hole_2026"

    output_dir_test_predict_temp = os.path.join(
        output_root,
        "Predict_data",
        f"Predict19ClassZscore_Test{timestamp}_{dataset_info}_temp"
    )
    os.makedirs(output_dir_test_predict_temp, exist_ok=True)
    output_dir_test_predict = output_dir_test_predict_temp  # 後で変更される
    
    # データセット情報をREADMEファイルに保存
    readme_path = os.path.join(output_dir_test_predict_temp, "dataset_info.txt")
    with open(readme_path, 'w') as f:
        f.write("=== Dataset Information ===\n")
        f.write(f"Dataset Type: Noise Defect-Free\n")
        f.write(f"Train pairs: {train_pairs_count}\n")
        f.write(f"Val pairs: {val_pairs_count}\n")
        f.write(f"Test pairs: {test_pairs_count}\n")
        f.write(f"Total pairs: {train_pairs_count + val_pairs_count + test_pairs_count}\n")
        f.write(f"Data usage ratio: {data_usage_ratio}\n")
        f.write(f"Number of classes: 19\n")
        f.write(f"Nodes per sample: {num_nodes_per_sample}\n")
    
    # 予測結果のnpyファイルを保存する専用フォルダを作成
    predictions_npy_dir = os.path.join(output_dir_test_predict, "predictions_npy")
    os.makedirs(predictions_npy_dir, exist_ok=True)
    
    #ノード数スライスの整合性をチェック
    expected_test_total = len(test_pairs) * num_nodes_per_sample
    actual_num_samples = len(test_preds)
    actual_num_files = actual_num_samples // num_nodes_per_sample
    
    if actual_num_samples != expected_test_total:
        print("[INFO] The total number of test_preds does not match test_pairs * num_nodes_per_sample.")
        print(f"       len(test_preds)={actual_num_samples}, expected={expected_test_total}")
        print(f"       This is normal in distributed training. Processing {actual_num_files} files out of {len(test_pairs)} total.")
    
    # Save per-file predictions using exact filename mapping from Batch
    if processed_files is not None and len(processed_files) > 0:
        saved_count = 0
        invalid_count = 0
        invalid_layer1_total = 0
        invalid_layer2_total = 0
        # False-positive count on defect-free data (NoiseDefectFree)
        ndf_total_files = 0
        ndf_pred_defect_files = 0
        ndf_pred_defect_examples = []  # store a few filenames for quick inspection
        ndf_pred_defect_node_ratios = []  # ratio of predicted defect nodes per file
        ndf_fp_rows = []  # rows for CSV (only FP files)
        for i, item in enumerate(processed_files):
            filename = item["filename"]
            sample_preds = np.array(item["preds"])
            # Validate length
            if len(sample_preds) != num_nodes_per_sample:
                print(f"[WARNING] File {i} ({filename}): Expected {num_nodes_per_sample} predictions, got {len(sample_preds)}")
                invalid_count += 1
                continue
            # Range check
            if np.any(sample_preds < 0) or np.any(sample_preds >= 19):
                print(f"[WARNING] File {i} ({filename}): Invalid prediction values detected (min={sample_preds.min()}, max={sample_preds.max()})")
                invalid_count += 1
                sample_preds = np.clip(sample_preds, 0, 18)
            if saved_count < 3:
                unique_preds, counts = np.unique(sample_preds, return_counts=True)
                print(f"[DEBUG] File {i} ({filename}): Prediction distribution: {dict(zip(unique_preds.tolist(), counts.tolist()))}")
            
            # Sanity-check: layer constraint violation count (should be 0 if masking works)
            vp = vertices_per_layer
            l1 = sample_preds[:vp]
            l2 = sample_preds[vp:]
            inv_l1 = int(np.sum(l1 >= 10))           # Layer1 should be 0..9 only
            inv_l2 = int(np.sum((l2 >= 1) & (l2 <= 9)))  # Layer2 should be {0,10..18} only
            invalid_layer1_total += inv_l1
            invalid_layer2_total += inv_l2
            if (inv_l1 > 0 or inv_l2 > 0) and saved_count < 10:
                print(f"[WARNING] Layer constraint violation in {filename}: layer1_invalid(>=10)={inv_l1}, layer2_invalid(1..9)={inv_l2}")

            # Count defect-free (NoiseDefectFree) files predicted as defect (any non-zero)
            if str(filename).startswith("NoiseDefectFree_"):
                ndf_total_files += 1
                defect_nodes = int(np.sum(sample_preds != 0))
                if defect_nodes > 0:
                    ndf_pred_defect_files += 1
                    ratio = defect_nodes / float(len(sample_preds))
                    ndf_pred_defect_node_ratios.append(ratio)
                    ndf_fp_rows.append({
                        "Filename": filename,
                        "PredictedDefectNodes": defect_nodes,
                        "PredictedDefectNodeRatio": ratio,
                    })
                    if len(ndf_pred_defect_examples) < 10:
                        ndf_pred_defect_examples.append((filename, defect_nodes))

            base_filename = os.path.splitext(filename)[0]
            pred_filename = f"{base_filename}_pred.npy"
            np.save(os.path.join(predictions_npy_dir, pred_filename), sample_preds)
            saved_count += 1
        print(f"Test data predictions saved for {saved_count} test samples in: {predictions_npy_dir}")
        if invalid_count > 0:
            print(f"[WARNING] {invalid_count} files had invalid predictions and were skipped or corrected")
        if invalid_layer1_total > 0 or invalid_layer2_total > 0:
            print(f"[WARNING] Layer constraint violations detected (total): layer1_invalid(>=10)={invalid_layer1_total}, layer2_invalid(1..9)={invalid_layer2_total}")
        else:
            print(f"[INFO] Layer constraint check passed: no violations detected in saved predictions")

        # Print defect-free false-positive summary
        if ndf_total_files > 0:
            rate = (ndf_pred_defect_files / float(ndf_total_files)) * 100.0
            print(f"\n=== NoiseDefectFree False-Positive Summary ===")
            print(f"NoiseDefectFree files processed: {ndf_total_files}")
            print(f"Predicted as defect (any non-zero): {ndf_pred_defect_files} files ({rate:.2f}%)")
            if len(ndf_pred_defect_node_ratios) > 0:
                avg_ratio = float(np.mean(ndf_pred_defect_node_ratios))
                max_ratio = float(np.max(ndf_pred_defect_node_ratios))
                print(f"Avg predicted-defect node ratio (among FP files): {avg_ratio:.6f}")
                print(f"Max predicted-defect node ratio (among FP files): {max_ratio:.6f}")
            if len(ndf_pred_defect_examples) > 0:
                print("Examples (filename, predicted defect nodes):")
                for fn, dn in ndf_pred_defect_examples:
                    print(f"  - {fn}: {dn}")

            # Thresholded FP counts by predicted-defect node ratio
            # (Useful to separate tiny noise-like FPs vs strong FPs)
            if len(ndf_fp_rows) > 0:
                thresholds = [0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
                ratios = np.array([r["PredictedDefectNodeRatio"] for r in ndf_fp_rows], dtype=float)
                print("\nNoiseDefectFree FP counts by defect-node-ratio threshold:")
                print("  (counts files where ratio > threshold)")
                for th in thresholds:
                    c = int(np.sum(ratios > th))
                    print(f"  - threshold>{th:.4g}: {c} / {ndf_total_files}")

                # Save CSV of FP files (sorted by ratio desc)
                try:
                    # timestamp is module-level; safe to reference here
                    csv_path = os.path.join(output_dir_test_predict, f"NoiseDefectFree_false_positives_{timestamp}.csv")
                    df = pd.DataFrame(ndf_fp_rows).sort_values("PredictedDefectNodeRatio", ascending=False)
                    df.to_csv(csv_path, index=False)
                    print(f"[INFO] NoiseDefectFree false-positive list saved to: {csv_path} ({len(df)} files)")
                except Exception as e:
                    # Fallback to stdlib csv if pandas is unavailable
                    try:
                        import csv as _csv
                        csv_path = os.path.join(output_dir_test_predict, f"NoiseDefectFree_false_positives_{timestamp}.csv")
                        rows = sorted(ndf_fp_rows, key=lambda x: x["PredictedDefectNodeRatio"], reverse=True)
                        with open(csv_path, "w", newline="") as f:
                            w = _csv.DictWriter(f, fieldnames=["Filename", "PredictedDefectNodes", "PredictedDefectNodeRatio"])
                            w.writeheader()
                            w.writerows(rows)
                        print(f"[INFO] NoiseDefectFree false-positive list saved to: {csv_path} ({len(rows)} files) [csv fallback]")
                    except Exception as e2:
                        print(f"[WARNING] Failed to save NoiseDefectFree false-positive CSV: {e} / fallback error: {e2}")
        else:
            print(f"\n[INFO] No NoiseDefectFree files found in processed outputs (cannot compute false-positive file count).")
        
        # 予測結果の全体統計を出力
        print(f"\n=== Prediction Statistics ===")
        all_preds_for_stats = np.array(test_preds)
        unique_preds, counts = np.unique(all_preds_for_stats, return_counts=True)
        print(f"Total predictions: {len(all_preds_for_stats)}")
        print(f"Prediction class distribution:")
        for cls, count in zip(unique_preds, counts):
            percentage = count / len(all_preds_for_stats) * 100
            print(f"  Class {cls}: {count:8d} predictions ({percentage:6.2f}%)")
        print(f"Prediction value range: {all_preds_for_stats.min()} - {all_preds_for_stats.max()}")
    else:
        print("[WARNING] No files to save (filename mapping missing). This may indicate a data mismatch issue.")
    
    all_preds_np = np.array(test_preds)    # テストデータ用
    all_labels_np = np.array(test_labels)  # テストデータ用
    all_probs_np = np.array(test_probs)    # テストデータ用

    print("Shape of all_preds:", all_preds_np.shape)
    print("Shape of all_labels:", all_labels_np.shape)
    print("Shape of all_probs:", all_probs_np.shape)

    output_dir1 = os.path.join(output_dir_test_predict, f"pred_data_{timestamp}")
    os.makedirs(output_dir1, exist_ok=True)

    np.save(os.path.join(output_dir1, "all_preds.npy"), all_preds_np)
    np.save(os.path.join(output_dir1, "all_labels.npy"), all_labels_np)
    np.save(os.path.join(output_dir1, "all_probs.npy"), all_probs_np)

    
    # --- メトリクス計算 (テストデータのみ)
    if len(all_preds_np) > 0 and len(all_labels_np) > 0:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels_np, all_preds_np, average='weighted', zero_division=0
        )
        class_report = classification_report(all_labels_np, all_preds_np, zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels_np, all_preds_np)
        mcc = matthews_corrcoef(all_labels_np, all_preds_np)

        # ROC-AUC (multi-class)
        try:
            roc_auc = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr')
        except ValueError:
            roc_auc = "ROC AUC calculation failed due to missing classes in prediction."

        # Use a fixed label set for consistent confusion-matrix shape (K=19)
        cm_eval = confusion_matrix(all_labels_np, all_preds_np, labels=list(range(19)))
        cm = cm_eval

        # ここで可視化 (混同行列や散布図など) を行う
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(19), yticklabels=range(19))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (Test Data)")
        plt.show()

        # Step0: Make Macro definition explicit (support=0 handling)
        cm_metrics_eval = metrics_from_confusion_matrix(cm_eval)
        macro_f1 = _select_macro_f1(cm_metrics_eval, macro_mode)
        macro_precision = precision_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        macro_recall = recall_score(all_labels_np, all_preds_np, average='macro', zero_division=0)
        
        # フォルダ名をMacro F1スコアを含む名前に変更（データセット情報も含む）
        output_dir_test_predict_new = os.path.join(
            output_root,
            "Predict_data",
            f"Predict19ClassZscore_Test{timestamp}_{dataset_info}_F1_{macro_f1:.2f}"
        )
        if os.path.exists(output_dir_test_predict_temp):
            try:
                os.rename(output_dir_test_predict_temp, output_dir_test_predict_new)
                output_dir_test_predict = output_dir_test_predict_new
                print(f"\n[INFO] Output directory renamed to include Macro F1 score: {output_dir_test_predict}")
            except OSError as e:
                print(f"[WARNING] Failed to rename directory: {e}")
                output_dir_test_predict = output_dir_test_predict_temp
        
        print(f"\nClassification Report:\n{class_report}")
        print(f"\n=== Overall Metrics ===")
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}, Weighted Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}")
        print(f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1-Score: {macro_f1:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
        print(f"ROC AUC Score: {roc_auc}")
        print(f"\n[INFO] Output directory: {output_dir_test_predict}")
        print(f"[INFO] Macro F1 Score: {macro_f1:.4f} (included in folder name)")
        
        # Verbose per-class debug (can be expensive + noisy)
        if VERBOSE_PRINT:
            # Macro/F1 calculation verification (show multiple macro variants)
            class_report_dict_eval = classification_report(
                all_labels_np, all_preds_np, labels=list(range(19)), output_dict=True, zero_division=0
            )
            
            print(f"\n=== Macro F1 Score Calculation Verification (Evaluation Function) ===")
            print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
            print("-" * 80)
            
            per_class_f1_scores_eval = []
            for i in range(19):
                # 混同行列から各クラスのTP, FP, FNを計算
                tp = cm_eval[i, i] if i < cm_eval.shape[0] and i < cm_eval.shape[1] else 0
                fp = cm_eval[:, i].sum() - tp if i < cm_eval.shape[1] else 0
                fn = cm_eval[i, :].sum() - tp if i < cm_eval.shape[0] else 0
                support = cm_eval[i, :].sum() if i < cm_eval.shape[0] else 0
                
                # Precision, Recall, F1を計算
                if tp + fp > 0:
                    class_precision = tp / (tp + fp)
                else:
                    class_precision = 0.0
                
                if tp + fn > 0:
                    class_recall = tp / (tp + fn)
                else:
                    class_recall = 0.0
                
                if class_precision + class_recall > 0:
                    class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
                else:
                    class_f1 = 0.0
                
                per_class_f1_scores_eval.append(class_f1)
                
                # classification_reportの結果と比較
                if str(i) in class_report_dict_eval:
                    metrics = class_report_dict_eval[str(i)]
                    print(
                        f"{i:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} "
                        f"{int(metrics['support']):<12} {tp:<8} {fp:<8} {fn:<8}"
                    )
                else:
                    print(f"{i:<8} {class_precision:<12.4f} {class_recall:<12.4f} {class_f1:<12.4f} {support:<12} {tp:<8} {fp:<8} {fn:<8}")
            
            # Macro F1 variants (explicit)
            manual_macro_f1_all = float(np.mean(per_class_f1_scores_eval)) if len(per_class_f1_scores_eval) > 0 else 0.0
            zero_support_classes = np.where(np.asarray(cm_metrics_eval["support_per_class"]) <= 0)[0].tolist()
            print(f"\n=== Macro definition (explicit) ===")
            print(f"macro_mode={macro_mode}")
            print(f"  - macro_f1_selected={macro_f1:.4f}")
            print(f"  - macro_f1_support_only={cm_metrics_eval.get('macro_f1_support_only', 0.0):.4f}")
            print(f"  - macro_f1_sklearn_like={cm_metrics_eval.get('macro_f1_sklearn_like', 0.0):.4f}")
            print(f"  - macro_f1_all={cm_metrics_eval.get('macro_f1_all', 0.0):.4f}  (manual_all={manual_macro_f1_all:.4f})")
            print(f"  - zero_support_classes={zero_support_classes}")
            
            # Print class distribution for analysis
            unique_labels, label_counts = np.unique(all_labels_np, return_counts=True)
            unique_preds, pred_counts = np.unique(all_preds_np, return_counts=True)
            print(f"\n=== Class Distribution ===")
            print(f"Actual labels distribution: {dict(zip(unique_labels, label_counts))}")
            print(f"Predicted labels distribution: {dict(zip(unique_preds, pred_counts))}")
            
            # 予測分布とラベル分布を比較
            print(f"\n=== Label vs Prediction Distribution ===")
            print(f"{'Class':<8} {'True Labels':<15} {'Predictions':<15} {'Match':<10}")
            print("-" * 50)
            for i in range(19):
                true_count = label_counts[unique_labels == i][0] if i in unique_labels else 0
                pred_count = pred_counts[unique_preds == i][0] if i in unique_preds else 0
                match = "✓" if true_count > 0 and pred_count > 0 else "✗"
                print(f"{i:<8} {true_count:<15} {pred_count:<15} {match:<10}")

    else:
        print("[WARNING] No test predictions available for evaluation.")


    # Directory for saving images
    output_dir = os.path.join(output_root, "Predict_truth", f"pred_vs_true_images_zscore_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
        
    # 1. 予測値 vs. 正解ラベルの散布図
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(all_labels_np)), all_labels_np, color="blue", label="Actual Labels", alpha=0.6)
    plt.scatter(range(len(all_preds_np)), all_preds_np, color="red", marker="x", label="Predicted Labels", alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("Class Label")
    plt.title("Predicted vs. Actual Class Labels")
    plt.legend()
    plt.savefig(f"{output_dir}/pred_vs_true_class_labels{timestamp}.png")  # Save as PNG
    plt.close()

    # 2. 混同行列のヒートマップ
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels_np, all_preds_np)
    # [FIX S-3] Fix confusion matrix label range from 20 to 19 classes
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(19), yticklabels=range(19))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_dir}/confusion_matrix{timestamp}.png")
    plt.close()

    plt.figure(figsize=(14, 12))
    cm = confusion_matrix(all_labels_np, all_preds_np)
    # Apply log scale to avoid overwhelming contrast
    log_cm = np.log1p(cm)  # Use log1p to handle large values in a more balanced way
    # Set up the figure and heatmap for the PDF
    ax = sns.heatmap(log_cm, annot=cm, fmt="d", cmap="Blues", 
                     cbar_kws={'label': 'Counts (Log Scale)'}, vmin=0, square=True, 
                     annot_kws={"size": 8, "weight": "bold"}, linewidths=0.01, linecolor="gray")
    
    # Highlight the diagonal with red borders for correct classifications
    for i in range(cm.shape[0]):
        ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    # Add labels and title
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix 2", fontsize=15)
    plt.savefig(f"{output_dir}/confusion_matrix2{timestamp}.png")

    # 4. クラスごとのF1スコア、Precision、Recallの棒グラフ
    class_report = classification_report(all_labels_np, all_preds_np, zero_division=0, output_dict=True)
    classes = list(range(19))
    f1_scores = [class_report[str(i)]['f1-score'] if str(i) in class_report else 0.0 for i in classes]
    precisions = [class_report[str(i)]['precision'] if str(i) in class_report else 0.0 for i in classes]
    recalls = [class_report[str(i)]['recall'] if str(i) in class_report else 0.0 for i in classes]
    supports = [class_report[str(i)]['support'] if str(i) in class_report else 0.0 for i in classes]
    
    # Create subplots for better visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1 Score
    axes[0, 0].bar(classes, f1_scores, color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel("Class", fontsize=12)
    axes[0, 0].set_ylabel("F1 Score", fontsize=12)
    axes[0, 0].set_title("F1 Score by Class", fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0, 1.0])
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Precision
    axes[0, 1].bar(classes, precisions, color='lightgreen', alpha=0.7)
    axes[0, 1].set_xlabel("Class", fontsize=12)
    axes[0, 1].set_ylabel("Precision", fontsize=12)
    axes[0, 1].set_title("Precision by Class", fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Recall
    axes[1, 0].bar(classes, recalls, color='lightcoral', alpha=0.7)
    axes[1, 0].set_xlabel("Class", fontsize=12)
    axes[1, 0].set_ylabel("Recall", fontsize=12)
    axes[1, 0].set_title("Recall by Class", fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1.0])
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Support (class distribution)
    axes[1, 1].bar(classes, supports, color='plum', alpha=0.7)
    axes[1, 1].set_xlabel("Class", fontsize=12)
    axes[1, 1].set_ylabel("Support (Number of Samples)", fontsize=12)
    axes[1, 1].set_title("Class Distribution (Support)", fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')  # Log scale for better visualization of imbalanced classes
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/detailed_class_metrics{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save the original F1 score plot for compatibility
    plt.figure(figsize=(12, 6))
    plt.bar(classes, f1_scores, color='skyblue', alpha=0.7)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("F1 Score by Class", fontsize=14, fontweight='bold')
    plt.ylim([0, 1.0])
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{output_dir}/f1score_class{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Images have been saved in {output_dir}")
    
    # --- 予測結果の空間的可視化（DSPSS、Label、Predictionの比較） ---
    print("\n予測結果の空間的可視化を開始...")
    visualization_output_dir = os.path.join(output_dir_test_predict, "spatial_visualizations")
    os.makedirs(visualization_output_dir, exist_ok=True)
    
    # Z-score標準化データのディレクトリ（欠陥ありデータ）
    zscore_data_base = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore"
    test_data_folder = os.path.join(zscore_data_base, "test")
    
    # ノイズあり欠陥なしデータのディレクトリ（GNN_zscore_sub_noise_defect_free.pyの場合）
    noise_data_base = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
    noise_test_folder = os.path.join(noise_data_base, "test")
    
    label_data_folder = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"
    
    # カスタムカラーマップ（クラス0を黒）
    defect_cmap_custom = get_defect_cmap_with_white_zero()
    
    # 座標データの読み込み（軸ラベル用）
    if x_coords is None or y_coords is None or z_coords is None:
        try:
            x_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
            y_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
            z_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")
        except Exception as e:
            print(f"警告: 座標データの読み込みに失敗しました: {e}")
            x_coords = y_coords = z_coords = None
    
    # 座標データを2層に分割してリシェイプ（可視化用）
    if x_coords is not None:
        x_layer1 = x_coords[:vertices_per_layer]
        x_layer2 = x_coords[vertices_per_layer:]
        y_layer1 = y_coords[:vertices_per_layer]
        y_layer2 = y_coords[vertices_per_layer:]
        z_layer1 = z_coords[:vertices_per_layer]
        z_layer2 = z_coords[vertices_per_layer:]
        
        # 座標をグリッドにマッピング
        x_layer1_grid = np.empty(total_elements)
        x_layer1_grid[:] = np.nan
        x_layer2_grid = np.empty(total_elements)
        x_layer2_grid[:] = np.nan
        y_layer1_grid = np.empty(total_elements)
        y_layer1_grid[:] = np.nan
        y_layer2_grid = np.empty(total_elements)
        y_layer2_grid[:] = np.nan
        z_layer1_grid = np.empty(total_elements)
        z_layer1_grid[:] = np.nan
        z_layer2_grid = np.empty(total_elements)
        z_layer2_grid[:] = np.nan
        
        data_index = 0
        for j in range(total_elements):
            if j not in hole_set and data_index < vertices_per_layer:
                x_layer1_grid[j] = x_layer1[data_index]
                y_layer1_grid[j] = y_layer1[data_index]
                z_layer1_grid[j] = z_layer1[data_index]
                data_index += 1
        
        data_index = 0
        for j in range(total_elements):
            if j not in hole_set and data_index < vertices_per_layer:
                x_layer2_grid[j] = x_layer2[data_index]
                y_layer2_grid[j] = y_layer2[data_index]
                z_layer2_grid[j] = z_layer2[data_index]
                data_index += 1
        
        # リシェイプと回転（データと同じ変換を適用）
        x_layer1_reshaped = np.flipud(np.rot90(x_layer1_grid.reshape((num_rows, num_cols)), k=1))
        x_layer2_reshaped = np.flipud(np.rot90(x_layer2_grid.reshape((num_rows, num_cols)), k=1))
        y_layer1_reshaped = np.flipud(np.rot90(y_layer1_grid.reshape((num_rows, num_cols)), k=1))
        y_layer2_reshaped = np.flipud(np.rot90(y_layer2_grid.reshape((num_rows, num_cols)), k=1))
        z_layer1_reshaped = np.flipud(np.rot90(z_layer1_grid.reshape((num_rows, num_cols)), k=1))
        z_layer2_reshaped = np.flipud(np.rot90(z_layer2_grid.reshape((num_rows, num_cols)), k=1))
        
        # 軸の範囲を計算（ラベル表示用）
        x_min_layer1, x_max_layer1 = np.nanmin(x_layer1_reshaped), np.nanmax(x_layer1_reshaped)
        x_min_layer2, x_max_layer2 = np.nanmin(x_layer2_reshaped), np.nanmax(x_layer2_reshaped)
        y_min_layer1, y_max_layer1 = np.nanmin(y_layer1_reshaped), np.nanmax(y_layer1_reshaped)
        y_min_layer2, y_max_layer2 = np.nanmin(y_layer2_reshaped), np.nanmax(y_layer2_reshaped)
        z_min_layer1, z_max_layer1 = np.nanmin(z_layer1_reshaped), np.nanmax(z_layer1_reshaped)
        z_min_layer2, z_max_layer2 = np.nanmin(z_layer2_reshaped), np.nanmax(z_layer2_reshaped)
    else:
        x_layer1_reshaped = x_layer2_reshaped = None
        y_layer1_reshaped = y_layer2_reshaped = None
        z_layer1_reshaped = z_layer2_reshaped = None
        x_min_layer1 = x_max_layer1 = x_min_layer2 = x_max_layer2 = None
        y_min_layer1 = y_max_layer1 = y_min_layer2 = y_max_layer2 = None
        z_min_layer1 = z_max_layer1 = z_min_layer2 = z_max_layer2 = None
    
    # テストデータの可視化（欠陥ありと欠陥なしの両方、最大50ファイル）
    max_visualize = min(50, len(test_pairs))
    visualized_count = 0  # 実際に可視化されたファイル数をカウント
    
    # test_data_folder_mapを使用して、各ファイルがどのフォルダにあるかを確認
    # グローバル変数から取得（モジュールレベルで定義されている）
    import sys
    current_module = sys.modules[__name__]
    test_data_folder_map = getattr(current_module, 'test_data_folder_map_global', {})
    
    for i, (data_file, label_file) in enumerate(test_pairs[:max_visualize]):
        try:
            # ファイルパス（test_data_folder_mapから取得、なければファイル名から判断）
            if data_file in test_data_folder_map:
                actual_data_folder = test_data_folder_map[data_file]
            else:
                # ファイル名から判断（NoiseDefectFree_で始まる場合はnoise_test_folder）
                if data_file.startswith("NoiseDefectFree_"):
                    actual_data_folder = noise_test_folder
                else:
                    actual_data_folder = test_data_folder
            
            data_file_path = os.path.join(actual_data_folder, data_file)
            label_file_path = os.path.join(label_data_folder, label_file)
            # 予測ファイルは predictions_npy フォルダに保存されている
            predictions_npy_dir = os.path.join(output_dir_test_predict, "predictions_npy")
            pred_file_path = os.path.join(predictions_npy_dir, f"{os.path.splitext(data_file)[0]}_pred.npy")
            
            # ファイル存在確認
            if not os.path.exists(data_file_path):
                continue
            if not os.path.exists(pred_file_path):
                continue
            
            # データを読み込み・処理
            # DSPSSデータ（Z-score標準化データ）
            dpsss_data = np.load(data_file_path)[:13942]
            dpsss_layer1_full = np.empty(total_elements)
            dpsss_layer1_full[:] = np.nan
            dpsss_layer2_full = np.empty(total_elements)
            dpsss_layer2_full[:] = np.nan
            
            data_index = 0
            for j in range(total_elements):
                if j not in hole_set and data_index < vertices_per_layer:
                    dpsss_layer1_full[j] = dpsss_data[data_index]
                    data_index += 1
            
            data_index = 0
            for j in range(total_elements):
                if j not in hole_set and data_index < vertices_per_layer:
                    dpsss_layer2_full[j] = dpsss_data[vertices_per_layer + data_index]
                    data_index += 1
            
            dpsss_layer1 = np.flipud(np.rot90(dpsss_layer1_full.reshape((num_rows, num_cols)), k=1))
            dpsss_layer2 = np.flipud(np.rot90(dpsss_layer2_full.reshape((num_rows, num_cols)), k=1))
            
            # 予測データ
            pred_layer1, pred_layer2 = process_file(pred_file_path)
            
            # ラベルデータ（存在する場合）
            label_layer1 = None
            label_layer2 = None
            if os.path.exists(label_file_path):
                label_layer1, label_layer2 = process_file(label_file_path)
            
            # 可視化
            num_cols_vis = 3  # DSPSS, Defect Label, Prediction
            num_rows_vis = 2  # Layer 1, Layer 2
            
            plt.figure(figsize=(18, 12))
            
            # ファイルタイプを判定（欠陥あり or 欠陥なし）
            is_defect_free = data_file.startswith("NoiseDefectFree_")
            data_type_label = "NoiseDefectFree" if is_defect_free else "Defect"
            
            # Layer 1 (Bottom Layer)
            # DSPSS Layer 1
            ax1 = plt.subplot(num_rows_vis, num_cols_vis, 1)
            im1 = plt.imshow(dpsss_layer1, cmap=dpsss_cmap, aspect="equal", interpolation='none')
            plt.title(f"Bottom Layer - DSPSS (Z-score)\n{os.path.basename(data_file)}", fontsize=12, pad=50)
            plt.colorbar(im1, label="Z-score Normalized DSPSS", fraction=0.025, pad=0.04)
            
            # Defect Label Layer 1
            # Note: Layer 1 (first half) contains labels 0, 1~9 only
            ax2 = plt.subplot(num_rows_vis, num_cols_vis, 2)
            if label_layer1 is not None:
                label_layer1_int = np.where(np.isnan(label_layer1), 0, label_layer1).astype(int)
                label_layer1_int_masked = np.ma.masked_where(np.isnan(label_layer1), label_layer1_int)
                im = plt.imshow(label_layer1_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                              interpolation='none', vmin=0, vmax=18)
                plt.title(f"Bottom Layer - Defect Label\n(Labels: 0, 1~9)\n{os.path.basename(label_file)}", fontsize=12, pad=50)
                cbar = plt.colorbar(im, label="Defect Label", fraction=0.025, pad=0.04)
                cbar.set_ticks(range(0, 19))
            else:
                plt.text(0.5, 0.5, "No Label", ha='center', va='center', fontsize=14)
                plt.axis('off')
            
            # Prediction Layer 1
            # Note: Layer 1 (first half) should contain labels 0, 1~9 only
            ax3 = plt.subplot(num_rows_vis, num_cols_vis, 3)
            pred_layer1_int = np.where(np.isnan(pred_layer1), 0, pred_layer1).astype(int)
            pred_layer1_int_masked = np.ma.masked_where(np.isnan(pred_layer1), pred_layer1_int)
            pred_max = int(np.max(pred_layer1_int[~np.isnan(pred_layer1)])) if np.any(~np.isnan(pred_layer1)) else 18
            im = plt.imshow(pred_layer1_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                          interpolation='none', vmin=0, vmax=max(pred_max, 18))
            plt.title(f"Bottom Layer - Prediction\n(Expected Labels: 0, 1~9)\n{os.path.basename(pred_file_path)}", fontsize=12, pad=50)
            cbar = plt.colorbar(im, label="Prediction", fraction=0.025, pad=0.04)
            cbar.set_ticks(range(0, max(pred_max, 18) + 1))
            
            # Layer 2 (Upper Layer)
            # DSPSS Layer 2
            ax4 = plt.subplot(num_rows_vis, num_cols_vis, 4)
            im4 = plt.imshow(dpsss_layer2, cmap=dpsss_cmap, aspect="equal", interpolation='none')
            plt.title(f"Upper Layer - DSPSS (Z-score)\n{os.path.basename(data_file)}", fontsize=12, pad=50)
            plt.colorbar(im4, label="Z-score Normalized DSPSS", fraction=0.025, pad=0.04)
            
            # Defect Label Layer 2
            # Note: Layer 2 (second half) contains labels 0, 10~18 only
            ax5 = plt.subplot(num_rows_vis, num_cols_vis, 5)
            if label_layer2 is not None:
                label_layer2_int = np.where(np.isnan(label_layer2), 0, label_layer2).astype(int)
                label_layer2_int_masked = np.ma.masked_where(np.isnan(label_layer2), label_layer2_int)
                im = plt.imshow(label_layer2_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                              interpolation='none', vmin=0, vmax=18)
                plt.title(f"Upper Layer - Defect Label\n(Labels: 0, 10~18)\n{os.path.basename(label_file)}", fontsize=12, pad=50)
                cbar = plt.colorbar(im, label="Defect Label", fraction=0.025, pad=0.04)
                cbar.set_ticks(range(0, 19))
            else:
                plt.text(0.5, 0.5, "No Label", ha='center', va='center', fontsize=14)
                plt.axis('off')
            
            # Prediction Layer 2
            # Note: Layer 2 (second half) should contain labels 0, 10~18 only
            ax6 = plt.subplot(num_rows_vis, num_cols_vis, 6)
            pred_layer2_int = np.where(np.isnan(pred_layer2), 0, pred_layer2).astype(int)
            pred_layer2_int_masked = np.ma.masked_where(np.isnan(pred_layer2), pred_layer2_int)
            pred_max = int(np.max(pred_layer2_int[~np.isnan(pred_layer2)])) if np.any(~np.isnan(pred_layer2)) else 18
            im = plt.imshow(pred_layer2_int_masked, cmap=defect_cmap_custom, aspect="equal", 
                          interpolation='none', vmin=0, vmax=max(pred_max, 18))
            plt.title(f"Upper Layer - Prediction\n(Expected Labels: 0, 10~18)\n{os.path.basename(pred_file_path)}", fontsize=12, pad=50)
            cbar = plt.colorbar(im, label="Prediction", fraction=0.025, pad=0.04)
            cbar.set_ticks(range(0, max(pred_max, 18) + 1))
            
            plt.tight_layout()
            
            # PNGファイルとして保存（英語のみのファイル名）
            base_filename = os.path.splitext(data_file)[0]
            # ファイル名にデータタイプを含める（Defect or NoiseDefectFree）
            output_filename = f"{base_filename}_spatial_visualization.png"
            output_path = os.path.join(visualization_output_dir, output_filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            visualized_count += 1  # 実際に可視化されたファイル数をカウント
            if visualized_count % 10 == 0:
                print(f"  可視化処理中: {visualized_count} ファイル")
                
        except Exception as e:
            print(f"  警告: {data_file} の可視化中にエラーが発生しました: {e}")
            continue
    
    print(f"空間的可視化完了! {visualized_count} ファイルのPNG画像を {visualization_output_dir} に保存しました")

# ----------------------------
# データペアリング関数
# ----------------------------
def extract_layer_block(file_name):
    """ファイル名から層とブロック番号を抽出（すべてのH_Wサイズに対応）"""
    try:
        # H2_W2, H4_W4, H8_W8, H5_W6, H7_W4, H4_W8, H8_W4など、すべてのサイズに対応
        match = re.search(r'L(\d+)_B(\d+)_el(\d+)_H(\d+)_W(\d+)', file_name)
        if match:
            layer = int(match.group(1))
            block = int(match.group(2))
            return (layer, block)
        else:
            print(f"Invalid file name format: {file_name}")
            return None
    except AttributeError:
        print(f"Invalid file name format: {file_name}")
        return None

def create_data_label_pairs(data_files, label_files):
    """データファイルとラベルファイルのペアを作成"""
    data_label_pairs = {}
    unmatched_labels = set(label_files)  # ラベルファイルの集合
    no_label_counter = 0

    # データファイル名からベース名を抽出（_19label.npyを除いた部分）
    for data_file in data_files:
        # データファイル名からベース名を取得（.npyを除く）
        base_name = os.path.splitext(data_file)[0]
        data_label_pairs[base_name] = {"data": data_file}

    # ラベルファイルとマッチング
    for label_file in label_files:
        if label_file.endswith("_19label.npy"):
            # ラベルファイル名からベース名を取得（_19label.npyを除く）
            base_name = label_file.replace("_19label.npy", "")
            if base_name in data_label_pairs:
                data_label_pairs[base_name]["label"] = label_file
                unmatched_labels.discard(label_file)  # マッチしたラベルを削除
            else:
                # マッチしないラベルを記録（最初の3つだけ）
                if no_label_counter < 3:
                    print(f"No matching data file for label: {label_file}")
                    no_label_counter += 1

    # ペアが作成されなかった場合を出力（3つに制限）
    no_data_counter = 0
    for base_name, v in data_label_pairs.items():
        if "label" not in v:
            if no_data_counter < 3:  # 最大3つまで出力
                print(f"No label found for data file: {v['data']}")
                no_data_counter += 1

    # マッチしなかったラベルを出力（最初の3つだけ）
    unmatched_counter = 0
    for unmatched_label in unmatched_labels:
        if unmatched_counter < 3:
            print(f"Unmatched label file: {unmatched_label}")
            unmatched_counter += 1

    valid_pairs = [(v["data"], v["label"]) for k, v in data_label_pairs.items() if "label" in v]
    
    # マッチしたペアの確認ログを追加
    if len(valid_pairs) > 0:
        print(f"✓ Successfully matched {len(valid_pairs)} pairs")
        # 最初の3つのマッチしたペアの例を表示
        print("  Sample matched pairs:")
        for i, (data_file, label_file) in enumerate(valid_pairs[:3]):
            # ベース名が一致しているか確認
            data_base = os.path.splitext(data_file)[0]
            label_base = label_file.replace("_19label.npy", "")
            match_status = "✓" if data_base == label_base else "✗ MISMATCH"
            print(f"    [{i+1}] {match_status} Data: {data_file} <-> Label: {label_file}")
            if data_base != label_base:
                print(f"         WARNING: Base names don't match! Data base: '{data_base}', Label base: '{label_base}'")
    else:
        print("⚠ WARNING: No valid pairs found!")
    
    return valid_pairs


# ----------------------------
# Class-Balanced Sampler
# ----------------------------
class ClassBalancedSampler(Sampler):
    """Class-balanced sampler that ensures each batch contains samples from all classes"""
    def __init__(self, dataset, num_classes=19, samples_per_class=None, shuffle=True):
        self.dataset = dataset
        self.num_classes = num_classes
        self.shuffle = shuffle
        
        # Calculate dominant class for each graph
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx, data in enumerate(dataset):
            # Get the most frequent class in this graph
            classes, counts = torch.unique(data.y, return_counts=True)
            dominant_class = classes[counts.argmax()].item()
            self.class_indices[dominant_class].append(idx)
        
        # Determine samples per class
        if samples_per_class is None:
            # Use the minimum class size to ensure all classes are represented
            min_class_size = min(len(indices) for indices in self.class_indices.values() if len(indices) > 0)
            self.samples_per_class = min_class_size
        else:
            self.samples_per_class = samples_per_class
        
        # Create balanced indices
        self.indices = []
        for class_idx in range(num_classes):
            if len(self.class_indices[class_idx]) > 0:
                class_samples = self.class_indices[class_idx]
                if self.shuffle:
                    np.random.shuffle(class_samples)
                # Repeat if needed to reach samples_per_class
                if len(class_samples) < self.samples_per_class:
                    class_samples = (class_samples * ((self.samples_per_class // len(class_samples)) + 1))[:self.samples_per_class]
                else:
                    class_samples = class_samples[:self.samples_per_class]
                self.indices.extend(class_samples)
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __iter__(self):
        if self.shuffle:
            # Reshuffle class indices
            for class_idx in range(self.num_classes):
                if len(self.class_indices[class_idx]) > 0:
                    np.random.shuffle(self.class_indices[class_idx])
            # Recreate balanced indices
            indices = []
            for class_idx in range(self.num_classes):
                if len(self.class_indices[class_idx]) > 0:
                    class_samples = self.class_indices[class_idx]
                    if len(class_samples) < self.samples_per_class:
                        class_samples = (class_samples * ((self.samples_per_class // len(class_samples)) + 1))[:self.samples_per_class]
                    else:
                        class_samples = class_samples[:self.samples_per_class]
                    indices.extend(class_samples)
            np.random.shuffle(indices)
            return iter(indices)
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class DistributedClassBalancedSampler(DistributedSampler):
    """Distributed version of ClassBalancedSampler"""
    def __init__(self, dataset, num_classes=19, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.num_classes = num_classes
        
        # Calculate dominant class for each graph (with progress indication)
        if rank == 0:
            print(f"Initializing ClassBalancedSampler: processing {len(dataset)} samples...")
        self.class_indices = {i: [] for i in range(num_classes)}
        for idx, data in enumerate(dataset):
            if idx % 1000 == 0 and rank == 0:
                print(f"  Processing sample {idx}/{len(dataset)}...")
            classes, counts = torch.unique(data.y, return_counts=True)
            dominant_class = classes[counts.argmax()].item()
            self.class_indices[dominant_class].append(idx)
        
        if rank == 0:
            print(f"ClassBalancedSampler initialization complete. Class distribution:")
            for cls_idx in range(num_classes):
                if len(self.class_indices[cls_idx]) > 0:
                    print(f"  Class {cls_idx}: {len(self.class_indices[cls_idx])} samples")
        
        # Create balanced indices per rank
        self._generate_indices()
    
    def _generate_indices(self):
        # Determine samples per class (minimum across all classes)
        min_class_size = min(len(indices) for indices in self.class_indices.values() if len(indices) > 0)
        samples_per_class = min_class_size // self.num_replicas
        if samples_per_class == 0:
            samples_per_class = 1
        
        # Create balanced indices for this rank
        indices = []
        for class_idx in range(self.num_classes):
            if len(self.class_indices[class_idx]) > 0:
                class_samples = self.class_indices[class_idx]
                if self.shuffle:
                    np.random.shuffle(class_samples)
                # Distribute across replicas
                rank_samples = class_samples[self.rank::self.num_replicas]
                if len(rank_samples) < samples_per_class:
                    rank_samples = (rank_samples * ((samples_per_class // len(rank_samples)) + 1))[:samples_per_class]
                else:
                    rank_samples = rank_samples[:samples_per_class]
                indices.extend(rank_samples)
        
        if self.shuffle:
            np.random.shuffle(indices)
        
        self.indices = indices
    
    def __iter__(self):
        if self.shuffle:
            self._generate_indices()
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


class DistributedMinorityRatioSampler(DistributedSampler):
    """
    ファイル単位のminority_ratio基準で重み付けサンプリングを行うDistributedSampler（DDP安全版）
    minority_ratioが高いファイル（minority classが多いファイル）を多くサンプルする
    
    DDP安全: torch.multinomialでtotal_sizeを固定し、全rankのバッチ数が揃う
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
                 weight_power=1.0, min_weight=0.1, seed=42):
        """
        Args:
            dataset: PyG Dataオブジェクトのリスト（各Dataにminority_ratio属性が必要）
            num_replicas: DDPの総プロセス数
            rank: 現在のプロセスrank
            shuffle: シャッフルするか
            weight_power: minority_ratioの重み付けの強さ（1.0=線形、2.0=2乗）
            min_weight: 最小重み（0に近いminority_ratioでもサンプルされるように）
            seed: 乱数シード（DDP安全のため）
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.weight_power = float(weight_power)
        self.min_weight = float(min_weight)
        self._build_weights()
    
    def _build_weights(self):
        """重みを構築（全rankで同じ重みを使用）"""
        if self.rank == 0:
            print(f"Initializing DistributedMinorityRatioSampler: processing {len(self.dataset)} samples...")
        
        ratios = []
        for idx, data in enumerate(self.dataset):
            if idx % 1000 == 0 and self.rank == 0:
                print(f"  Processing sample {idx}/{len(self.dataset)}...")
            # minority_ratioが設定されていない場合は計算
            if not hasattr(data, 'minority_ratio'):
                unique_labels, counts = torch.unique(data.y, return_counts=True)
                total_nodes = len(data.y)
                count0 = counts[unique_labels == 0].item() if 0 in unique_labels else 0
                r = 1.0 - (count0 / total_nodes) if total_nodes > 0 else 0.0
            else:
                r = float(data.minority_ratio)
            ratios.append(r)
        
        # 重みを計算: minority_ratio^weight_power + min_weight
        w = (np.array(ratios) ** self.weight_power) + self.min_weight
        # 正規化（確率分布にする）
        w = w / (w.sum() + 1e-12)
        self.weights = torch.tensor(w, dtype=torch.double)
        
        if self.rank == 0:
            print(f"MinorityRatioSampler initialization complete.")
            print(f"  Minority ratio range: {min(ratios):.4f} - {max(ratios):.4f}")
            print(f"  Weight range: {self.weights.min().item():.4f} - {self.weights.max().item():.4f}")
    
    def __iter__(self):
        """DDP安全な重み付きサンプリング"""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # total_size個を「重み付きで復元抽出」（全rankで同じシード→同じ順序）
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g).tolist()
        
        # rankごとに等分（ここがDDP安全：全rankのバッチ数が揃う）
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


class DistributedClassFrequencySampler(DistributedSampler):
    """
    クラス頻度ベースの1/sqrt(freq)重み付けサンプリングを行うDistributedSampler（DDP安全版）
    各サンプル（ファイル）のクラス分布を計算し、最も頻度の低いクラスを含むサンプルを優先的にサンプル
    
    DDP安全: torch.multinomialでtotal_sizeを固定し、全rankのバッチ数が揃う
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
                 weight_type='sqrt', min_weight=0.1, seed=42):
        """
        Args:
            dataset: PyG Dataオブジェクトのリスト
            num_replicas: DDPの総プロセス数
            rank: 現在のプロセスrank
            shuffle: シャッフルするか
            weight_type: 重み付けタイプ ('sqrt'=1/sqrt(freq), 'linear'=1/freq)
            min_weight: 最小重み（頻度が高いクラスでもサンプルされるように）
            seed: 乱数シード（DDP安全のため）
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.weight_type = weight_type
        self.min_weight = float(min_weight)
        self._build_weights()
    
    def _build_weights(self):
        """重みを構築（全rankで同じ重みを使用）"""
        if self.rank == 0:
            print(f"Initializing DistributedClassFrequencySampler: processing {len(self.dataset)} samples...")
        
        # まず全データセットのクラス頻度を計算
        all_class_counts = torch.zeros(19, dtype=torch.long)  # 19 classes
        for idx, data in enumerate(self.dataset):
            if idx % 1000 == 0 and self.rank == 0:
                print(f"  Computing class frequencies: {idx}/{len(self.dataset)}...")
            unique_labels, counts = torch.unique(data.y, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label < 19:
                    all_class_counts[label] += count.item()
        
        total_samples = all_class_counts.sum().item()
        class_frequencies = all_class_counts.float() / (total_samples + 1e-12)
        
        if self.rank == 0:
            print(f"  Class frequencies computed. Total samples: {total_samples}")
            print(f"  Class 0 frequency: {class_frequencies[0]:.6f}")
            print(f"  Minority class frequencies: {class_frequencies[1:].min().item():.6f} - {class_frequencies[1:].max().item():.6f}")
        
        # 各サンプルの重みを計算（最も頻度の低いクラスを含むサンプルに高い重み）
        sample_weights = []
        for idx, data in enumerate(self.dataset):
            if idx % 1000 == 0 and self.rank == 0:
                print(f"  Computing sample weights: {idx}/{len(self.dataset)}...")
            
            unique_labels, counts = torch.unique(data.y, return_counts=True)
            total_nodes = len(data.y)
            
            # サンプル内の各クラスの頻度
            sample_class_freqs = torch.zeros(19)
            for label, count in zip(unique_labels, counts):
                if label < 19:
                    sample_class_freqs[label] = count.item() / total_nodes
            
            # 最も頻度の低いクラス（minority class）の重みを計算
            # 1/sqrt(freq) または 1/freq で重み付け
            if self.weight_type == 'sqrt':
                # 1/sqrt(freq) ベース（過激すぎない）
                weights_per_class = 1.0 / (torch.sqrt(class_frequencies + 1e-12) + 1e-12)
            else:  # 'linear'
                # 1/freq ベース（より過激）
                weights_per_class = 1.0 / (class_frequencies + 1e-12)
            
            # サンプル内のクラス分布に基づいて重みを計算
            # minority classが多いサンプルほど高い重み
            sample_weight = (sample_class_freqs * weights_per_class).sum().item() + self.min_weight
            sample_weights.append(sample_weight)
        
        # 正規化（確率分布にする）
        w = np.array(sample_weights)
        w = w / (w.sum() + 1e-12)
        self.weights = torch.tensor(w, dtype=torch.double)
        
        if self.rank == 0:
            print(f"ClassFrequencySampler initialization complete.")
            print(f"  Sample weight range: {self.weights.min().item():.6f} - {self.weights.max().item():.6f}")
            print(f"  Weight type: {self.weight_type} (1/{self.weight_type}(freq))")
    
    def __iter__(self):
        """DDP安全な重み付きサンプリング"""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # total_size個を「重み付きで復元抽出」（全rankで同じシード→同じ順序）
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=g).tolist()
        
        # rankごとに等分（ここがDDP安全：全rankのバッチ数が揃う）
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)


# ----------------------------
# データ準備関数
# ----------------------------
def prepare_data(pairs, normalized_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=None, data_folder_map=None):
    """
    Args:
        pairs: (data_file, label_file)のタプルのリスト
        normalized_data_folder: デフォルトのデータフォルダ（data_folder_mapがNoneの場合に使用）
        label_data_folder: ラベルフォルダ
        data_folder_map: データファイル名からデータフォルダへのマッピング（オプション）
    """
    data_list = []
    labels = []
    pair_counter = 0  # 確認用カウンタ
    ndf_fallback_label_count = 0  # NoiseDefectFree_ のラベル自動生成回数（デバッグ用）
    missing_data_count = 0
    missing_label_count = 0
    load_error_count = 0
    feature_error_count = 0

    total_pairs = len(pairs)
    for pair_idx, (data_file, label_file) in enumerate(pairs):
        if pair_idx % 500 == 0 and pair_idx > 0:
            print(f"  Loading data: {pair_idx}/{total_pairs} pairs processed...")
        
        # データファイルのパスを決定（data_folder_mapがあればそれを使用）
        if data_folder_map is not None and data_file in data_folder_map:
            data_file_path = os.path.join(data_folder_map[data_file], data_file)
        else:
            data_file_path = os.path.join(normalized_data_folder, data_file)
        
        label_file_path = os.path.join(label_data_folder, label_file)
        label = None  # IMPORTANT: ループごとに初期化（前ループの値が残らないようにする）

        if not os.path.exists(data_file_path):
            missing_data_count += 1
            if VERBOSE_PRINT:
                print(f"データファイルが存在しません: {data_file_path}")
            continue
        if not os.path.exists(label_file_path):
            # 欠陥なし（NoiseDefectFree_）は単一ラベル（全ノード0）として扱えるようにする
            # ラベルファイルが無い場合でも学習/推論に含める
            if str(data_file).startswith("NoiseDefectFree_"):
                label = np.zeros((13942,), dtype=np.int64)
                ndf_fallback_label_count += 1
            else:
                missing_label_count += 1
                if VERBOSE_PRINT:
                    print(f"ラベルファイルが存在しません: {label_file_path}")
                continue

        # データとラベルをロード
        try:
            values = np.load(data_file_path)[:13942]
            # NoiseDefectFree_ でラベルが欠損していない場合は通常通りロード
            if label is None:
                label = np.load(label_file_path)[:13942]
        except Exception as e:
            load_error_count += 1
            if VERBOSE_PRINT:
                print(f"データ読み込みエラー: {e}")
            continue

        # ノード特徴量を作成
        try:
            node_features = np.vstack((x_coords, y_coords, z_coords, values)).T
            x = torch.tensor(node_features, dtype=torch.float)
            
            # ラベルの形状を確認して適切に処理
            label_tensor = torch.tensor(label, dtype=torch.float)
            if len(label_tensor.shape) == 2 and label_tensor.shape[1] > 1:
                # One-hotエンコードされたラベルの場合
                y = torch.argmax(label_tensor, dim=1).long()
            elif len(label_tensor.shape) == 1 or (len(label_tensor.shape) == 2 and label_tensor.shape[1] == 1):
                # 直接クラスIDの場合
                y = label_tensor.long().squeeze()
            else:
                raise ValueError(f"Unexpected label shape: {label_tensor.shape}")
        except Exception as e:
            feature_error_count += 1
            if VERBOSE_PRINT:
                print(f"特徴量作成エラー: {e}, label shape: {label.shape if 'label' in locals() else 'unknown'}")
            continue

        # データリストとラベルを保存
        data = Data(x=x, edge_index=edge_index, y=y)
        # ファイル名を保存（推論時の対応付けのため）
        data.filename = data_file
        
        # ファイル単位のminority_ratioを計算して保存（sampler用）
        # minority_ratio = 1 - (count0 / N) で、高いほどminorityが多い
        unique_labels, counts = torch.unique(y, return_counts=True)
        total_nodes = len(y)
        count0 = counts[unique_labels == 0].item() if 0 in unique_labels else 0
        minority_ratio = 1.0 - (count0 / total_nodes) if total_nodes > 0 else 0.0
        data.minority_ratio = minority_ratio
        
        data_list.append(data)
        labels.extend(y.tolist())

        # デバッグ用にいくつかのペアを出力（ペアの整合性も確認）
        if VERBOSE_PRINT and pair_counter < 1:
            print(f"Pair {pair_counter + 1}:")
            print(f"Data file: {data_file_path}")
            print(f"Label file: {label_file_path}")
            # ペアの整合性を確認
            data_base = os.path.splitext(data_file)[0]
            label_base = label_file.replace("_19label.npy", "")
            if data_base == label_base:
                print(f"✓ Pair match verified: '{data_base}' <-> '{label_base}'")
            else:
                print(f"✗ Pair mismatch detected: '{data_base}' != '{label_base}'")
            print(f"x shape: {x.shape}")
            print(f"y shape: {y.shape}")
            # ラベルの詳細情報を出力
            print(f"Label file shape: {label.shape}")
            print(f"Label file dtype: {label.dtype}")
            print(f"Label file min/max: {label.min()}/{label.max()}")
            if len(label.shape) == 2:
                print(f"Label file is 2D (one-hot): shape={label.shape}")
            else:
                print(f"Label file is 1D (class IDs): shape={label.shape}")
            # 変換後のラベルの分布を確認
            unique_labels, counts = torch.unique(y, return_counts=True)
            print(f"Converted label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")
            print(f"Converted label min/max: {y.min().item()}/{y.max().item()}")
            pair_counter += 1

    if ndf_fallback_label_count > 0:
        print(f"[INFO] NoiseDefectFree label fallback used: {ndf_fallback_label_count} files (generated all-zero labels)")

    # Avoid noisy per-sample warnings by default; summarize once.
    if (missing_data_count or missing_label_count or load_error_count or feature_error_count) and not VERBOSE_PRINT:
        print(
            "[INFO] prepare_data summary: "
            f"missing_data={missing_data_count}, missing_label={missing_label_count}, "
            f"load_errors={load_error_count}, feature_errors={feature_error_count}"
        )

    # クラス重みを計算
    if len(labels) > 0:
        labels_array = np.array(labels)
        class_weights = compute_class_weights(labels_array, multiplier=class_weight_multiplier)
        
        # クラス分布と重みを出力（デバッグ用）
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        if VERBOSE_PRINT:
            print(f"\n=== Class Distribution and Weights ===")
            print(f"Total samples: {len(labels_array)}")
            print(f"Class distribution:")
            for cls, count in zip(unique_labels, counts):
                weight = class_weights[cls].item()
                percentage = count / len(labels_array) * 100
                print(f"  Class {cls}: {count:8d} samples ({percentage:6.2f}%), weight: {weight:.6f}")
            print(f"Class weights: {class_weights.cpu().numpy()}")
            print(f"Class weights min/max: {class_weights.min().item():.6f}/{class_weights.max().item():.6f}")
        else:
            print(
                f"[INFO] class_weights computed: total_labels={len(labels_array)}, "
                f"min/max={class_weights.min().item():.6f}/{class_weights.max().item():.6f}"
            )
    else:
        print("ラベルが存在しません。クラス重みの計算に失敗しました。")
        return None, None

    return data_list, class_weights

# ----------------------------
# シード設定関数
# ----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# プロセスグループの初期化
# ----------------------------
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

# ----------------------------
# プロセスグループのクリーンアップ
# ----------------------------
def cleanup(device=None, rank=0, do_barrier=True):
    """Safely cleanup distributed process group.

    IMPORTANT:
    - Only call a final `dist.barrier()` if *all ranks* will also call it.
      If rank-0 continues into single-process post-work while other ranks exit,
      those exiting ranks must set `do_barrier=False` to avoid deadlock/timeouts.
    """
    try:
        # Check if process group is initialized
        if not dist.is_initialized():
            return
        
        # Get device if not provided (use default)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # First, ensure all CUDA operations are complete
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                if rank == 0:
                    print(f"[WARNING] CUDA sync failed during cleanup: {e}")
        
        # Optional: synchronize all ranks before destroying.
        # WARNING: calling barrier here is only safe if all ranks participate.
        if do_barrier:
            try:
                if dist.is_initialized():
                    try:
                        # Barrier is more reliable than all_reduce for final sync
                        dist.barrier()
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except (RuntimeError, Exception) as barrier_error:
                        # If barrier fails (e.g., communicator aborted), that's OK
                        if rank == 0 and "aborted" not in str(barrier_error).lower():
                            print(f"[WARNING] Final barrier before cleanup failed (continuing anyway): {barrier_error}")
            except Exception as sync_error:
                if rank == 0:
                    print(f"[WARNING] Final barrier before cleanup failed (continuing anyway): {sync_error}")
        
        # Now destroy the process group
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as destroy_error:
            # If destroy fails, that's OK - process will exit anyway
            if rank == 0:
                print(f"[WARNING] Process group destroy failed (this is OK): {destroy_error}")
        
    except Exception as e:
        # Log error but don't raise - allow process to exit gracefully
        if rank == 0:
            print(f"[WARNING] Error during cleanup: {e}")
        # Force destroy if normal destroy fails
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass  # Ignore errors during forced cleanup

# ----------------------------
# メイン関数
# ----------------------------
def main(args):
    set_seed(42)

    # DDP Debug Environment Variables (optional, for troubleshooting):
    # Set these before running to get detailed NCCL/PyTorch distributed debug info:
    #   export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch distributed debug
    #   export NCCL_DEBUG=INFO                 # NCCL debug info
    #   export NCCL_DEBUG_SUBSYS=ALL           # All NCCL subsystems
    # These are helpful for diagnosing collective timeout issues but not required for normal operation.

    # Initialize process group
    dist.init_process_group(
        backend='nccl',         # 'nccl' for GPUs, 'gloo' for CPUs
        init_method='env://',   # initialization method
        rank=int(os.environ['RANK']),
        world_size=int(os.environ['WORLD_SIZE'])
    )

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '12355')

    # Make args visible to helper functions that currently read from globals()
    globals()['args'] = args

    # Output root for artifacts (models / predictions / plots / csv)
    output_root = getattr(args, "output_root", None)
    if output_root is None or str(output_root).strip() == "":
        output_root = "/home/nishioka/GNN/GNN_hole_2026"
    run_id = getattr(args, "run_id", None)  # optional metadata (used by launcher)

    if rank == 0:
        # Record execution command for logging
        command_str = ' '.join(sys.argv)
        print(f"\n{'='*60}")
        print(f"Execution Command:")
        print(f"  {command_str}")
        print(f"{'='*60}\n")
        
        print(f"Rank: {rank}, World Size: {world_size}, Master Addr: {master_addr}, Master Port: {master_port}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # デバイス設定（利用可能なGPU数を確認）
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if rank >= num_gpus:
            print(f"Warning: Rank {rank} exceeds available GPU count ({num_gpus}). Using GPU {rank % num_gpus} instead.")
            local_rank = rank % num_gpus
        else:
            local_rank = rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        local_rank = None
    
    if rank == 0:
        print(f"Rank {rank} using device: {device}")

    # Data preparation and model definition
    # 新しいデータセット構成: 全部ノイズあり（ノイズあり欠陥なし5000個 + ノイズあり欠陥あり5000個）
    
    # ノイズデータのディレクトリ（全部ノイズありに統一）
    noise_data_base = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
    noise_data_folder = os.path.join(noise_data_base, "train")
    
    # 欠陥ありデータもノイズフォルダから読み込む（全部ノイズありに統一）
    defect_data_base = "/home/nishioka/GNN/GNN_hole_2026/all_sub_hole_defect_zscore_noise"
    defect_train_folder = os.path.join(defect_data_base, "train")
    defect_val_folder = os.path.join(defect_data_base, "val")
    defect_test_folder = os.path.join(defect_data_base, "test")
    
    # クラスラベルのディレクトリ
    label_data_folder = "/home/nishioka/GNN/GNN_hole_2026/all_19class_label"

    x_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy")
    y_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy")
    z_coords = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_z_2layer.npy")

    edges = np.load("/home/nishioka/GNN/GNN_hole/GNN_hole_data/hole_edges_2layer_best.npy")
    # edge_indexはCPUで保持（DataLoader/num_workers/fork周りで安全）
    # 学習時は batch.to(device) でGPUへ移動される
    edge_index = torch.tensor(edges.T, dtype=torch.long)  # CPUのまま

    # ノイズあり欠陥なしデータファイルを読み込む（5000個）
    noise_data_files = sorted([f for f in os.listdir(noise_data_folder) 
                               if f.startswith("NoiseDefectFree_") and f.endswith(".npy")])
    
    # 欠陥ありデータファイルを読み込む（全データから5000個を選択）
    all_defect_files = []
    if os.path.exists(defect_train_folder):
        all_defect_files.extend([(f, defect_train_folder) for f in os.listdir(defect_train_folder) 
                                  if f.startswith("Defect_L") and f.endswith(".npy")])
    if os.path.exists(defect_val_folder):
        all_defect_files.extend([(f, defect_val_folder) for f in os.listdir(defect_val_folder) 
                                  if f.startswith("Defect_L") and f.endswith(".npy")])
    if os.path.exists(defect_test_folder):
        all_defect_files.extend([(f, defect_test_folder) for f in os.listdir(defect_test_folder) 
                                  if f.startswith("Defect_L") and f.endswith(".npy")])
    
    # ランダムに5000個を選択（再現性のためシード固定）
    np.random.seed(42)
    random.seed(42)
    num_defect_samples = min(5000, len(all_defect_files))
    selected_defect_indices = np.random.choice(len(all_defect_files), size=num_defect_samples, replace=False)
    selected_defect_files = [all_defect_files[i] for i in selected_defect_indices]
    
    # ラベルファイルを読み込む
    label_files = [
        f for f in os.listdir(label_data_folder)
        if (f.startswith("Defect_L") or f.startswith("NoiseDefectFree_")) and f.endswith("_19label.npy")
    ]

    if rank == 0:
        print(f"\n=== Dataset Configuration ===")
        print(f"All data with noise (unified):")
        print(f"  - Noise defect-free data files: {len(noise_data_files)}")
        print(f"  - Noise defect data files: {len(selected_defect_files)}")
        print(f"Total data files: {len(noise_data_files) + len(selected_defect_files)}")
        print(f"Label files: {len(label_files)}")

    # データとラベルのペアを作成
    if rank == 0:
        print("\n=== Creating data-label pairs ===")
    
    # ノイズあり欠陥なしデータのペアを作成
    noise_pairs = []
    for noise_file in noise_data_files:
        base_name = os.path.splitext(noise_file)[0]
        label_file = f"{base_name}_19label.npy"
        # 欠陥なしは単一ラベル（全ノード0）として扱えるので、ラベルファイルが無くてもペアに入れる
        noise_pairs.append((noise_file, label_file))
    
    # 欠陥ありデータのペアを作成
    defect_pairs = []
    for defect_file, defect_dir in selected_defect_files:
        base_name = os.path.splitext(defect_file)[0]
        label_file = f"{base_name}_19label.npy"
        label_path = os.path.join(label_data_folder, label_file)
        if os.path.exists(label_path):
            defect_pairs.append((defect_file, label_file, defect_dir))
    
    # 全データを統合してtrain/val/testに分割（70%/15%/15%）
    all_pairs = []
    # ノイズデータのペアを追加（データファイル名とラベルファイル名のみ）
    all_pairs.extend([(pair[0], pair[1], noise_data_folder) for pair in noise_pairs])
    # 欠陥データのペアを追加
    all_pairs.extend([(pair[0], pair[1], pair[2]) for pair in defect_pairs])
    
    # ランダムにシャッフル
    np.random.seed(42)
    random.seed(42)
    np.random.shuffle(all_pairs)
    
    # train/val/testに分割
    # Benchmark: split_manifest が指定されている場合は外部マニフェストを使用
    split_manifest_path = getattr(args, "split_manifest", None)
    if split_manifest_path is not None:
        import json as _json
        with open(split_manifest_path, "r") as _f:
            _manifest = _json.load(_f)
        if rank == 0:
            print(f"[BENCHMARK] Using external split manifest: {split_manifest_path}")
            print(f"[BENCHMARK] Split type: {_manifest.get('split_type', 'unknown')}, seed: {_manifest.get('seed', 'N/A')}")
        train_pairs = [tuple(p[:2]) for p in _manifest["train"]]
        val_pairs = [tuple(p[:2]) for p in _manifest["val"]]
        test_pairs = [tuple(p[:2]) for p in _manifest["test"]]
        train_data_folder_map = {p[0]: p[2] for p in _manifest["train"]}
        val_data_folder_map = {p[0]: p[2] for p in _manifest["val"]}
        test_data_folder_map = {p[0]: p[2] for p in _manifest["test"]}
    # 重要: group単位で分割して leakage (val∩test など) をゼロにする
    elif getattr(args, "enforce_disjoint_groups", True):
        train_pairs, val_pairs, test_pairs, train_data_folder_map, val_data_folder_map, test_data_folder_map = group_disjoint_split(
            all_pairs,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            group_key=group_key,
            seed=42,
        )
    else:
        total_samples = len(all_pairs)
        num_train = int(total_samples * 0.7)
        num_val = int(total_samples * 0.15)
        num_test = total_samples - num_train - num_val
        train_pairs = [(pair[0], pair[1]) for pair in all_pairs[:num_train]]
        val_pairs = [(pair[0], pair[1]) for pair in all_pairs[num_train:num_train+num_val]]
        test_pairs = [(pair[0], pair[1]) for pair in all_pairs[num_train+num_val:]]
        # データフォルダのマッピングを保存（prepare_dataで使用するため）
        train_data_folder_map = {pair[0]: pair[2] for pair in all_pairs[:num_train]}
        val_data_folder_map = {pair[0]: pair[2] for pair in all_pairs[num_train:num_train+num_val]}
        test_data_folder_map = {pair[0]: pair[2] for pair in all_pairs[num_train+num_val:]}
    
    # グローバル変数として保存（prepare_data関数で使用）
    global train_data_folder_map_global, val_data_folder_map_global, test_data_folder_map_global
    train_data_folder_map_global = train_data_folder_map
    val_data_folder_map_global = val_data_folder_map
    test_data_folder_map_global = test_data_folder_map

    if rank == 0:
        print(f"\n=== Pair Summary ===")
        print(f"Train pairs: {len(train_pairs)}")
        print(f"Val pairs: {len(val_pairs)}")
        print(f"Test pairs: {len(test_pairs)}")
        print(f"Total pairs: {len(train_pairs) + len(val_pairs) + len(test_pairs)}")
        
        # ペアの整合性を確認（最初のペアの例を表示）
        if len(train_pairs) > 0:
            print(f"\n=== Verifying pair integrity ===")
            sample_pair = train_pairs[0]
            data_file, label_file = sample_pair
            data_base = os.path.splitext(data_file)[0]
            label_base = label_file.replace("_19label.npy", "")
            if data_base == label_base:
                print(f"✓ Pair integrity check passed: '{data_base}' matches '{label_base}'")
            else:
                print(f"✗ Pair integrity check FAILED: '{data_base}' != '{label_base}'")
    
    # データ使用率の適用（10%刻み）
    data_usage_ratio = getattr(args, 'data_usage_ratio', 1.0)
    if data_usage_ratio < 1.0:
        if rank == 0:
            print(f"\nUsing {data_usage_ratio*100:.0f}% of data (data_usage_ratio={data_usage_ratio})")
        
        # ランダムシードを固定して再現性を確保
        np.random.seed(42)
        random.seed(42)
        
        # 各データセットから指定された割合をサンプリング
        num_train = max(1, int(len(train_pairs) * data_usage_ratio))
        num_val = max(1, int(len(val_pairs) * data_usage_ratio))
        num_test = max(1, int(len(test_pairs) * data_usage_ratio))
        
        train_indices = np.random.choice(len(train_pairs), num_train, replace=False)
        val_indices = np.random.choice(len(val_pairs), num_val, replace=False)
        test_indices = np.random.choice(len(test_pairs), num_test, replace=False)
        
        train_pairs = [train_pairs[i] for i in sorted(train_indices)]
        val_pairs = [val_pairs[i] for i in sorted(val_indices)]
        test_pairs = [test_pairs[i] for i in sorted(test_indices)]
        
        if rank == 0:
            print(f"After sampling: Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}")

    # ---------------------------------------------------------------------
    # Step1 (評価だけ): Group overlap check / group-purged eval split
    # ---------------------------------------------------------------------
    group_purge = getattr(args, "group_purge_eval", False)
    group_purge_target = getattr(args, "group_purge_target", "valtest")

    if rank == 0:
        train_groups = {group_id_from_filename(p[0], group_key=group_key) for p in train_pairs}
        val_groups = {group_id_from_filename(p[0], group_key=group_key) for p in val_pairs}
        test_groups = {group_id_from_filename(p[0], group_key=group_key) for p in test_pairs}
        inter_tv = len(train_groups & val_groups)
        inter_tt = len(train_groups & test_groups)
        inter_vt = len(val_groups & test_groups)
        print("\n=== Group overlap (leakage sanity check) ===")
        print(f"Group key: {group_key}")
        print(f"Unique groups: train={len(train_groups)}, val={len(val_groups)}, test={len(test_groups)}")
        print(f"Overlap groups: train∩val={inter_tv}, train∩test={inter_tt}, val∩test={inter_vt}")
        if group_purge:
            print(f"[INFO] group_purge_eval enabled (target={group_purge_target}) -> will purge eval pairs sharing groups with train")

    if group_purge:
        train_groups_local = {group_id_from_filename(p[0], group_key=group_key) for p in train_pairs}

        def _purge(eval_pairs):
            kept = []
            removed = 0
            for p in eval_pairs:
                gid = group_id_from_filename(p[0], group_key=group_key)
                if gid in train_groups_local:
                    removed += 1
                else:
                    kept.append(p)
            return kept, removed

        if group_purge_target in ("val", "valtest"):
            val_pairs, removed = _purge(val_pairs)
            if rank == 0:
                print(f"[INFO] Purged val pairs by train-group overlap: removed={removed}, kept={len(val_pairs)}")
        if group_purge_target in ("test", "valtest"):
            test_pairs, removed = _purge(test_pairs)
            if rank == 0:
                print(f"[INFO] Purged test pairs by train-group overlap: removed={removed}, kept={len(test_pairs)}")

        # Rebuild folder maps to avoid stale keys when purging
        # (train maps unchanged; val/test maps rebuilt from original all_pairs slice maps)
        val_data_folder_map_global = {p[0]: val_data_folder_map.get(p[0], noise_data_folder) for p in val_pairs}
        test_data_folder_map_global = {p[0]: test_data_folder_map.get(p[0], noise_data_folder) for p in test_pairs}

    # Dropout/EdgeDropのパラメータを取得（デフォルト値を推奨値に変更）
    dropout = getattr(args, 'dropout', 0.1)  # デフォルト: 0.1（推奨範囲: 0.0~0.3）
    edge_drop_prob = getattr(args, 'edge_drop_prob', 0.01)  # デフォルト: 0.01（推奨範囲: 0.0~0.05）
    model = GATModel(
        hidden_channels=args.hidden_channels, 
        num_classes=19,
        dropout=dropout,
        edge_drop_prob=edge_drop_prob
    ).to(device)
    
    # Create the DDP model after initializing the process group
    # device_idsはlocal_rankを使用（利用可能なGPU数に合わせる）
    # Check if we'll freeze backbone (need find_unused_parameters for DDP)
    # Note: We check this before DDP initialization because frozen parameters won't receive gradients
    freeze_backbone = getattr(args, "freeze_backbone", False)
    if torch.cuda.is_available() and local_rank is not None:
        ddp_device_ids = [local_rank]
    else:
        ddp_device_ids = None
    # If freezing backbone, we need find_unused_parameters=True for DDP
    # because some parameters won't receive gradients during training
    # This is required for DDP to work correctly with frozen parameters
    ddp_model = DDP(model, device_ids=ddp_device_ids, find_unused_parameters=freeze_backbone)
    if rank == 0 and freeze_backbone:
        print("[INFO] DDP initialized with find_unused_parameters=True (required for frozen backbone)")

    # データセットの準備
    class_weight_multiplier = getattr(args, 'class_weight_multiplier', default_class_weight_multiplier)
    if rank == 0:
        print(f"\nPreparing train dataset ({len(train_pairs)} pairs)...")
    train_dataset, class_weights = prepare_data(train_pairs, noise_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier, data_folder_map=train_data_folder_map_global)
    if rank == 0:
        print(f"Preparing val dataset ({len(val_pairs)} pairs)...")
    val_dataset, _ = prepare_data(val_pairs, noise_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier, data_folder_map=val_data_folder_map_global)
    if rank == 0:
        print(f"Preparing test dataset ({len(test_pairs)} pairs)...")
    test_dataset, _ = prepare_data(test_pairs, noise_data_folder, label_data_folder, x_coords, y_coords, z_coords, edge_index, class_weight_multiplier=class_weight_multiplier, data_folder_map=test_data_folder_map_global)

    if train_dataset is None or class_weights is None:
        if rank == 0:
            raise ValueError("Failed to prepare train_dataset or class_weights")
        return

    if val_dataset is None:
        if rank == 0:
            raise ValueError("Failed to prepare val_dataset")
        return

    if test_dataset is None:
        if rank == 0:
            raise ValueError("Failed to prepare test_dataset")
        return

    if rank == 0:
        print(f"Train Dataset Size: {len(train_dataset)}")
        print(f"Val Dataset Size: {len(val_dataset)}")
        print(f"Test Dataset Size: {len(test_dataset)}")
    
    if rank == 0:
        print("\nInitializing samplers...")
    
    # Sampler選択: minority_ratio基準のsamplerを使うかどうか
    use_minority_sampler = getattr(args, 'use_minority_sampler', False)
    minority_weight_power = getattr(args, 'minority_weight_power', 1.0)
    use_class_frequency_sampler = getattr(args, 'use_class_frequency_sampler', False)
    
    if use_class_frequency_sampler:
        if rank == 0:
            print("Creating train sampler with class frequency weighting (1/sqrt(freq))...")
        train_sampler = DistributedClassFrequencySampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            weight_type='sqrt'  # 1/sqrt(freq) ベース（過激すぎない）
        )
    elif use_minority_sampler:
        if rank == 0:
            print("Creating train sampler with minority_ratio weighting...")
        train_sampler = DistributedMinorityRatioSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            weight_power=minority_weight_power
        )
    else:
        if rank == 0:
            print("Creating train sampler (standard DistributedSampler)...")
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    if rank == 0:
        print("Creating val sampler...")
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    if rank == 0:
        print("Creating test sampler...")
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    if rank == 0:
        print("Creating DataLoaders...")
    # Use drop_last=True for training to ensure all ranks have same number of batches (prevents NCCL timeout)
    # For validation, use drop_last=False to avoid empty batches when validation data is small
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, drop_last=False)
    
    # DDP Safety: Assert all ranks have non-empty loaders
    assert len(train_loader) > 0, f"Rank {rank}: train_loader is empty!"
    assert len(val_loader) > 0, f"Rank {rank}: val_loader is empty!"
    
    # DDP Safety: Log per-rank step counts at epoch 1 to verify identical batch counts
    if rank == 0:
        if use_class_frequency_sampler:
            sampler_type = "DistributedClassFrequencySampler (1/sqrt(freq))"
        elif use_minority_sampler:
            sampler_type = "DistributedMinorityRatioSampler"
        else:
            sampler_type = "DistributedSampler"
        print(f"Using {sampler_type} for training (ensures identical batch counts across ranks)")
        print(f"Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}, Test loader batches: {len(test_loader)}")
        print(f"[DDP AUDIT] Rank 0: train_batches={len(train_loader)}, val_batches={len(val_loader)}")
    
    # All ranks print their batch counts (will be identical with drop_last=True and DistributedSampler)
    print(f"[DDP AUDIT] Rank {rank}: train_batches={len(train_loader)}, val_batches={len(val_loader)}")

    # Fine-tuning support: load pretrained weights and optionally freeze layers
    fine_tune_from = getattr(args, "fine_tune_from", None)
    freeze_backbone = getattr(args, "freeze_backbone", False)
    backbone_lr = getattr(args, "backbone_lr", None)  # If None, use args.learning_rate
    head_lr = getattr(args, "head_lr", None)  # If None, use args.learning_rate
    
    if fine_tune_from is not None and str(fine_tune_from).strip() != "":
        fine_tune_path = str(fine_tune_from)
        if os.path.exists(fine_tune_path):
            try:
                if rank == 0:
                    print(f"[INFO] Fine-tuning: Loading pretrained weights from {fine_tune_path}")
                ckpt = torch.load(fine_tune_path, map_location="cpu")
                # Extract model state dict (support both dict and plain state_dict)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    pretrained_state = ckpt["model_state_dict"]
                else:
                    pretrained_state = ckpt
                
                # Load pretrained weights (strict=False to allow partial loading)
                missing_keys, unexpected_keys = ddp_model.module.load_state_dict(pretrained_state, strict=False)
                if rank == 0:
                    if missing_keys:
                        print(f"[WARNING] Missing keys when loading pretrained model: {len(missing_keys)} keys")
                        if VERBOSE_PRINT:
                            for key in missing_keys[:10]:  # Show first 10
                                print(f"  - {key}")
                    if unexpected_keys:
                        print(f"[WARNING] Unexpected keys in pretrained model: {len(unexpected_keys)} keys")
                        if VERBOSE_PRINT:
                            for key in unexpected_keys[:10]:  # Show first 10
                                print(f"  - {key}")
            except Exception as e:
                if rank == 0:
                    print(f"[WARNING] Failed to load pretrained weights from {fine_tune_path}: {e}")
                    print("[WARNING] Continuing with random initialization")
        else:
            if rank == 0:
                print(f"[WARNING] fine_tune_from not found: {fine_tune_path} (using random initialization)")
    
    # Freeze backbone layers if requested
    if freeze_backbone:
        if rank == 0:
            print("[INFO] Freezing backbone layers (conv1, conv2, conv3, conv4)")
        for name, param in ddp_model.module.named_parameters():
            if any(layer_name in name for layer_name in ['conv1', 'conv2', 'conv3', 'conv4']):
                param.requires_grad = False
        if rank == 0:
            trainable_params = sum(p.numel() for p in ddp_model.module.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in ddp_model.module.parameters())
            print(f"[INFO] Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Setup optimizer with different learning rates for backbone and head if specified
    if backbone_lr is not None or head_lr is not None:
        # Separate parameter groups for backbone and head
        # Head includes: fc, fc_detection, fc_classification (for two-stage models)
        backbone_params = []
        head_params = []
        for name, param in ddp_model.module.named_parameters():
            if param.requires_grad:
                if any(layer_name in name for layer_name in ['conv1', 'conv2', 'conv3', 'conv4']):
                    backbone_params.append(param)
                elif any(head_name in name for head_name in ['fc', 'fc_detection', 'fc_classification', 'proj1', 'proj2', 'proj3']):
                    # Classification heads: fc, fc_detection, fc_classification, and projection layers
                    head_params.append(param)
                else:
                    # Other parameters (e.g., batch norm) go to backbone group
                    backbone_params.append(param)
        
        param_groups = []
        if backbone_params:
            backbone_lr_val = backbone_lr if backbone_lr is not None else args.learning_rate
            param_groups.append({'params': backbone_params, 'lr': backbone_lr_val, 'weight_decay': args.weight_decay})
            if rank == 0:
                print(f"[INFO] Backbone learning rate: {backbone_lr_val}")
        if head_params:
            head_lr_val = head_lr if head_lr is not None else args.learning_rate
            param_groups.append({'params': head_params, 'lr': head_lr_val, 'weight_decay': args.weight_decay})
            if rank == 0:
                print(f"[INFO] Head learning rate: {head_lr_val}")
        
        optimizer = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler: CosineAnnealingLR or OneCycleLR
    # OneCycleLRをデフォルト推奨（GATはLRに敏感で、収束とminority学習が改善しやすい）
    use_onecycle = getattr(args, 'use_onecycle', True)  # argparseの値を使用（デフォルト: True）
    onecycle_max_lr = getattr(args, 'onecycle_max_lr', None)  # max_lrを独立パラメータ化
    if onecycle_max_lr is None:
        # デフォルトはlearning_rateの1-3倍（GATは高いLRが効きやすい）
        onecycle_max_lr = args.learning_rate * 2.0 if args.learning_rate < 1e-3 else args.learning_rate
    
    if use_onecycle:
        # NOTE:
        # - OneCycleLR overwrites optimizer.param_groups[i]["lr"] at the start.
        # - The optimizer's initial lr (args.learning_rate) is NOT the "base_lr" of OneCycleLR unless you tune div_factor.
        # - With pct_start=0.1 and large epochs (e.g., 2000), lr will keep increasing until ~10% of total steps.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=onecycle_max_lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=getattr(args, "onecycle_pct_start", 0.1),
            anneal_strategy=getattr(args, "onecycle_anneal_strategy", "cos"),
            div_factor=getattr(args, "onecycle_div_factor", 25.0),
            final_div_factor=getattr(args, "onecycle_final_div_factor", 1e4),
        )
        if rank == 0:
            total_steps = getattr(scheduler, "total_steps", args.epochs * len(train_loader))
            init_lr = scheduler.base_lrs[0] if getattr(scheduler, "base_lrs", None) else optimizer.param_groups[0]["lr"]
            pct_start = getattr(scheduler, "pct_start", getattr(args, "onecycle_pct_start", 0.1))
            print(
                "Learning rate scheduler initialized: OneCycleLR "
                f"(max_lr={onecycle_max_lr}, initial_lr≈{init_lr:.6g}, "
                f"steps_per_epoch={len(train_loader)}, total_steps={total_steps}, pct_start={pct_start})"
            )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=1e-6
        )
        if rank == 0:
            print(f"Learning rate scheduler initialized: CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)")

    # Optional resume from checkpoint (DDP-safe: every rank loads the same file)
    resume_path = getattr(args, "resume_from", None)
    resume_epoch = 0
    resume_best_val_loss = None
    resume_best_macro_f1 = None
    if resume_path is not None and str(resume_path).strip() != "":
        resume_path = str(resume_path)
        if os.path.exists(resume_path):
            try:
                ckpt = torch.load(resume_path, map_location="cpu")
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    ddp_model.module.load_state_dict(ckpt["model_state_dict"])
                    if "optimizer_state_dict" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                        try:
                            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                        except Exception as e:
                            if rank == 0:
                                print(f"[WARNING] Failed to load scheduler state (continuing): {e}")
                    resume_epoch = int(ckpt.get("epoch", 0))
                    resume_best_val_loss = ckpt.get("best_val_loss", None)
                    resume_best_macro_f1 = ckpt.get("best_macro_f1", None)
                else:
                    # Also support plain state_dict checkpoints (e.g., *_Latest.pth)
                    ddp_model.module.load_state_dict(ckpt)
                    resume_epoch = 0
                if rank == 0:
                    print(f"[INFO] Resumed from: {resume_path} (epoch={resume_epoch})")
            except Exception as e:
                if rank == 0:
                    print(f"[WARNING] Failed to resume from {resume_path} (starting fresh): {e}")
        else:
            if rank == 0:
                print(f"[WARNING] resume_from not found: {resume_path} (starting fresh)")

    start_epoch = max(1, resume_epoch + 1)

    if rank == 0:
        precision_per_epoch = []
        recall_per_epoch = []
        f1_score_per_epoch = []
        macro_f1_per_epoch = []
        val_accuracy_per_epoch = []
        train_loss_per_epoch = []
        val_loss_per_epoch = []

    # Loss関数の選択（優先度1の施策: LogitAdjust推奨）
    use_logit_adjust = getattr(args, 'use_logit_adjust', False)
    logit_adjust_tau = getattr(args, 'logit_adjust_tau', 1.0)
    use_layer_constraint = getattr(args, 'use_layer_constraint', True)  # デフォルトで有効
    layer_constraint_weight = getattr(args, 'layer_constraint_weight', 1.0)  # 制約違反のペナルティ重み
    two_stage = getattr(args, "two_stage", True)
    det_threshold = None if getattr(args, "no_det_gating", False) else float(getattr(args, "det_threshold", 0.90))
    det_loss_weight = float(getattr(args, "det_loss_weight", 1.0))
    cls_loss_weight = float(getattr(args, "cls_loss_weight", 1.0))
    _dg = getattr(args, "det_gamma", None)
    _cg = getattr(args, "cls_gamma", None)
    det_gamma = float(_dg) if _dg is not None else float(getattr(args, "gamma", default_gamma))
    cls_gamma = float(_cg) if _cg is not None else float(getattr(args, "gamma", default_gamma))
    det_pos_weight_cap = float(getattr(args, "det_pos_weight_cap", 20.0))
    
    # ---- Loss functions ----
    # Two-stage mode: detection loss + classification loss (only for y>0)
    # Single-stage mode: legacy 19-class loss (optionally with LayerAwareLoss penalty)
    use_log_softmax = getattr(args, 'use_log_softmax', False)
    if two_stage:
        # Build priors / weights from train dataset (DDP-safe, deterministic)
        labels_array = np.concatenate([d.y.cpu().numpy() for d in train_dataset])
        labels_array = labels_array.astype(np.int64)
        det_counts = np.array([np.sum(labels_array == 0), np.sum(labels_array != 0)], dtype=np.float64)
        # defect-only labels for Stage2 (1..18) -> 0..17
        defect_only = labels_array[labels_array > 0] - 1
        if defect_only.size > 0:
            cls_counts = np.bincount(defect_only, minlength=18).astype(np.float64)
        else:
            cls_counts = np.ones((18,), dtype=np.float64)

        if use_logit_adjust:
            det_prior = torch.tensor(det_counts / (det_counts.sum() + 1e-8), dtype=torch.float, device=device)
            cls_prior = torch.tensor(cls_counts / (cls_counts.sum() + 1e-8), dtype=torch.float, device=device)
            det_loss_fn = LogitAdjustLoss(class_prior=det_prior, tau=logit_adjust_tau).to(device)
            cls_loss_fn = LogitAdjustLoss(class_prior=cls_prior, tau=logit_adjust_tau).to(device)
            if rank == 0:
                print(f"Using two-stage LogitAdjustLoss (tau={logit_adjust_tau})")
        else:
            # Detection weights: cap to avoid extreme emphasis; keep mean~1
            count0 = float(det_counts[0])
            count1 = float(det_counts[1])
            w1 = min(det_pos_weight_cap, (count0 / (count1 + 1e-8))) if count1 > 0 else det_pos_weight_cap
            det_w = np.array([1.0, w1], dtype=np.float64)
            det_w = det_w / (det_w.mean() + 1e-12)
            det_w_t = torch.tensor(det_w, dtype=torch.float, device=device)

            # Classification weights: reuse computed class_weights for classes 1..18 (already normalized mean~1)
            cls_w_t = class_weights[1:19].to(device).float()

            if use_log_softmax:
                det_loss_fn = FocalLossLogSoftmax(weights=det_w_t, gamma=det_gamma).to(device)
                cls_loss_fn = FocalLossLogSoftmax(weights=cls_w_t, gamma=cls_gamma).to(device)
                if rank == 0:
                    print(f"Using two-stage FocalLossLogSoftmax (det_gamma={det_gamma}, cls_gamma={cls_gamma}, det_thr={det_threshold})")
            else:
                det_loss_fn = torch.nn.CrossEntropyLoss(weight=det_w_t).to(device)
                cls_loss_fn = torch.nn.CrossEntropyLoss(weight=cls_w_t).to(device)
                if rank == 0:
                    print(f"Using two-stage CrossEntropyLoss (det_thr={det_threshold})")

        if rank == 0:
            if use_layer_constraint:
                print(f"Layer constraint: enabled via hard-masking in model heads (no extra penalty)")
            else:
                print(f"Layer constraint: disabled (note: model still hard-masks invalid classes internally)")
    else:
        # Legacy single-stage base loss
        if use_logit_adjust:
            labels_array = np.concatenate([data.y.cpu().numpy() for data in train_dataset])
            class_counts = np.bincount(labels_array, minlength=19)
            class_prior = torch.tensor(class_counts / (class_counts.sum() + 1e-8), dtype=torch.float)
            base_loss_fn = LogitAdjustLoss(class_prior=class_prior.to(device), tau=logit_adjust_tau).to(device)
            if rank == 0:
                print(f"Using LogitAdjustLoss (tau={logit_adjust_tau}) - recommended for imbalanced datasets")
        else:
            gamma = getattr(args, 'gamma', default_gamma)
            if use_log_softmax:
                base_loss_fn = FocalLossLogSoftmax(weights=class_weights.to(device), gamma=gamma).to(device)
                if rank == 0:
                    print(f"Using FocalLossLogSoftmax (gamma={gamma}) - logits input with log_softmax for numerical stability")
            else:
                base_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
                if rank == 0:
                    print(f"Using CrossEntropyLoss (logits input) - standard loss for logits")

        if use_layer_constraint:
            loss_fn = LayerAwareLoss(
                base_loss_fn=base_loss_fn,
                vertices_per_layer=vertices_per_layer,
                constraint_weight=layer_constraint_weight
            ).to(device)
            if rank == 0:
                print(f"Layer-aware constraint enabled (weight={layer_constraint_weight})")
                print(f"  Layer 1: labels 0, 1~9 only")
                print(f"  Layer 2: labels 0, 10~18 only")
        else:
            loss_fn = base_loss_fn
            if rank == 0:
                print(f"Layer-aware constraint disabled")
    if rank == 0:
        print(f"Class weights for training: {class_weights.cpu().numpy()}")
        print(f"Class weights range: min={class_weights.min().item():.6f}, max={class_weights.max().item():.6f}")

    # Time tracking for estimated completion time
    train_start_time = time.time()
    epoch_times = []  # Store time per epoch for estimation
    
    best_val_loss = float(resume_best_val_loss) if resume_best_val_loss is not None else float('inf')
    # Best-so-far metric (macro F1 on validation) used for early stopping / selection
    best_macro_f1 = float(resume_best_macro_f1) if resume_best_macro_f1 is not None else 0.0
    counter = 0
    macro_f1_counter = 0  # Counter for macro F1 based early stopping
    best_model_state = None  # Store best model state dict
    best_epoch = 0  # Track epoch with best model
    last_best_checkpoint_path = None  # Track last best checkpoint file for cleanup
    early_stopped = False  # Flag to track if early stopping occurred
    # Shared early stopping flag using tensor (more reliable than file-based)
    early_stop_tensor = torch.tensor(0, dtype=torch.int32, device=device)  # 0 = continue, 1 = stop
    # AMP (Mixed Precision Training)
    use_amp = getattr(args, 'use_amp', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if rank == 0 and use_amp:
        print(f"Mixed Precision Training (AMP) enabled")
    
    # Checkpoint saving directory
    checkpoint_dir = os.path.join(output_root, "GNN_model", "19classmodel_hole_zscore")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        # Don't check for early stopping at the start - check at the end after synchronization
        # This prevents race conditions with pending NCCL operations
        
        # DDP Safety: Verify DataLoader lengths on epoch 1 (all ranks should have identical counts)
        if epoch == 1:
            train_len = len(train_loader)
            val_len = len(val_loader)
            print(f"[DDP AUDIT Epoch 1] Rank {rank}: train_batches={train_len}, val_batches={val_len}")
            # All ranks should have the same lengths due to drop_last=True and DistributedSampler
            # This will be verified by comparing logs across ranks
        
        epoch_start_time = time.time()
        ddp_model.train()
        # sampler.set_epochのガード（custom samplerの場合に存在しない可能性がある）
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if use_amp:
                with torch.cuda.amp.autocast():
                    y = batch.y
                    if two_stage:
                        heads = ddp_model(batch, return_heads=True)
                        loss, _, _ = two_stage_loss(
                            heads["detection_logits"],
                            heads["classification_logits"],
                            y,
                            det_loss_fn,
                            cls_loss_fn,
                            det_weight=det_loss_weight,
                            cls_weight=cls_loss_weight,
                        )
                    else:
                        out = ddp_model(batch)
                        if use_layer_constraint:
                            loss = loss_fn(out, y, batch=batch.batch)
                        else:
                            loss = loss_fn(out, y)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                y = batch.y
                if two_stage:
                    heads = ddp_model(batch, return_heads=True)
                    loss, _, _ = two_stage_loss(
                        heads["detection_logits"],
                        heads["classification_logits"],
                        y,
                        det_loss_fn,
                        cls_loss_fn,
                        det_weight=det_loss_weight,
                        cls_weight=cls_loss_weight,
                    )
                else:
                    out = ddp_model(batch)
                    if use_layer_constraint:
                        loss = loss_fn(out, y, batch=batch.batch)
                    else:
                        loss = loss_fn(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
            # Update OneCycleLR scheduler per step
            if use_onecycle:
                scheduler.step()

        avg_train_loss = torch.tensor(total_loss / len(train_loader), dtype=torch.float).to(device)
        dist.all_reduce(avg_train_loss, op=dist.ReduceOp.SUM)
        avg_train_loss /= world_size
        
        # Ensure NCCL operation completes before proceeding
        torch.cuda.synchronize()

        ddp_model.eval()
        val_loss = 0.0
        correct = 0
        total_samples = 0
        # Build local confusion matrix (19x19) on each rank - much lighter than all_gather_object
        num_classes = 19
        cm_local = torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)
        # File-level (graph-level) defect presence confusion matrix (2x2)
        # rows = GT has_defect (0/1), cols = Pred has_defect (0/1)
        file_cm_local = torch.zeros((2, 2), dtype=torch.long, device=device)
        
        with torch.no_grad():
            val_batch_count = 0
            local_samples = 0
            for batch in val_loader:
                batch = batch.to(device)
                y = batch.y
                if two_stage:
                    heads = ddp_model(batch, return_heads=True)
                    loss, _, _ = two_stage_loss(
                        heads["detection_logits"],
                        heads["classification_logits"],
                        y,
                        det_loss_fn,
                        cls_loss_fn,
                        det_weight=det_loss_weight,
                        cls_weight=cls_loss_weight,
                    )
                    probs = combine_two_stage_probs(
                        heads["detection_logits"],
                        heads["classification_logits"],
                        det_threshold=det_threshold,
                    )
                    pred = probs.argmax(dim=1)
                else:
                    out = ddp_model(batch)
                    if use_layer_constraint:
                        loss = loss_fn(out, y, batch=batch.batch)
                    else:
                        loss = loss_fn(out, y)
                    pred = out.argmax(dim=1)
                val_loss += loss.item()
                correct += (pred == y).sum().item()
                batch_size = y.size(0)
                total_samples += batch_size
                local_samples += batch_size
                
                # Update local confusion matrix (高速化: torch.bincountで一括更新)
                # k = true_label * num_classes + pred_label で1次元インデックス化
                y_flat = y.view(-1).long()
                pred_flat = pred.view(-1).long()
                k = y_flat * num_classes + pred_flat
                # bincountで各組み合わせの出現回数をカウント
                cm_update = torch.bincount(k, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
                cm_local += cm_update

                # --------------------------
                # File-level metric (DDP-safe): "Does this file have any predicted defect?"
                # --------------------------
                # In PyG batching, batch.batch maps each node -> graph index within the batch.
                graph_id = batch.batch.view(-1).long()
                if graph_id.numel() > 0:
                    num_graphs = int(getattr(batch, "num_graphs", int(graph_id.max().item()) + 1))
                    pred_is_defect = (pred_flat != 0).long()
                    gt_is_defect = (y_flat != 0).long()

                    # Count defect nodes per graph; >0 => file has defect
                    pred_defect_counts = torch.zeros((num_graphs,), dtype=torch.long, device=device)
                    gt_defect_counts = torch.zeros((num_graphs,), dtype=torch.long, device=device)
                    pred_defect_counts.scatter_add_(0, graph_id, pred_is_defect)
                    gt_defect_counts.scatter_add_(0, graph_id, gt_is_defect)
                    pred_has_defect = (pred_defect_counts > 0).long()
                    gt_has_defect = (gt_defect_counts > 0).long()

                    # Update 2x2 via bincount
                    k_file = gt_has_defect * 2 + pred_has_defect  # 0..3
                    file_cm_update = torch.bincount(k_file, minlength=4).reshape(2, 2)
                    file_cm_local += file_cm_update
                
                val_batch_count += 1
            
            # Debug: Log validation processing info
            local_cm_sum = cm_local.sum().item()
            if epoch <= 10 or epoch % 100 == 0:  # Log first 10 epochs and every 100 epochs
                print(f"[DEBUG] Rank {rank} Epoch {epoch}: val_batches={val_batch_count}, local_samples={local_samples}, local_cm_sum={local_cm_sum}")
            
            # Debug: Log if no batches were processed (should not happen with drop_last=False)
            if val_batch_count == 0:
                print(f"[WARNING] Rank {rank}: No validation batches processed at epoch {epoch}")

        avg_val_loss = torch.tensor(val_loss / len(val_loader), dtype=torch.float).to(device)
        dist.all_reduce(avg_val_loss, op=dist.ReduceOp.SUM)
        avg_val_loss /= world_size

        total_correct = torch.tensor(correct, dtype=torch.float, device=device)
        total_samples_t = torch.tensor(total_samples, dtype=torch.float, device=device)
        dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)
        
        # Reduce confusion matrix across all ranks (lightweight operation)
        dist.all_reduce(cm_local, op=dist.ReduceOp.SUM)
        # Reduce file-level confusion matrix across all ranks (very lightweight)
        dist.all_reduce(file_cm_local, op=dist.ReduceOp.SUM)
        
        # Ensure NCCL operations complete before proceeding
        torch.cuda.synchronize()
        
        val_accuracy = total_correct / total_samples_t
        
        # Convert confusion matrix to numpy on rank 0 for metric computation
        if rank == 0:
            cm_np = cm_local.cpu().numpy()
            file_cm_np = file_cm_local.cpu().numpy()
            # Debug: Check confusion matrix sum
            cm_sum = cm_np.sum()
            total_samples_val = total_samples_t.item()
            
            # Always log debug info for first 10 epochs or when there's a problem
            cm_nonzero_count = np.count_nonzero(cm_np)
            if epoch <= 10 or epoch % 100 == 0 or cm_sum == 0:
                print(f"[DEBUG] Rank 0 Epoch {epoch}: total_samples_t={total_samples_val}, cm_sum={cm_sum}, cm_nonzero={cm_nonzero_count}")
                # Debug: Show confusion matrix row/column sums to understand distribution
                row_sums = cm_np.sum(axis=1)  # True labels distribution
                col_sums = cm_np.sum(axis=0)  # Predicted labels distribution
                print(f"[DEBUG] Epoch {epoch} - True labels (row sums): {row_sums}")
                print(f"[DEBUG] Epoch {epoch} - Predicted labels (col sums): {col_sums}")
                # Show which classes have non-zero entries
                nonzero_indices = np.nonzero(cm_np)
                if len(nonzero_indices[0]) > 0:
                    print(f"[DEBUG] Epoch {epoch} - Non-zero entries: {len(nonzero_indices[0])} positions")
                    # Show first 20 non-zero entries
                    for idx in range(min(20, len(nonzero_indices[0]))):
                        i, j = nonzero_indices[0][idx], nonzero_indices[1][idx]
                        print(f"[DEBUG]   CM[{i},{j}] = {cm_np[i,j]}")
            
            if cm_sum == 0:
                print(f"[WARNING] Empty confusion matrix at epoch {epoch} - total_samples_t={total_samples_val}, cm_sum={cm_sum}")
                # Additional debug: check if cm_local was properly reduced
                print(f"[DEBUG] cm_local before reduction check: sum={cm_local.sum().item()}, device={cm_local.device}")
            else:
                # Sanity check (recommended): cm sum should match reduced total samples
                if abs(float(cm_sum) - float(total_samples_val)) > 0.5:
                    print(f"[WARNING] Sanity check: cm_np.sum()={cm_sum} != total_samples_t={total_samples_val}")

            # Compute metrics directly from confusion matrix (avoid reconstructing huge arrays)
            if cm_sum > 0:
                cm_metrics = metrics_from_confusion_matrix(cm_np)
            else:
                cm_metrics = None

            # File-level FP rate for defect-free files (GT has_defect=0)
            # file_cm_np rows: GT(0/1), cols: Pred(0/1)
            tn = int(file_cm_np[0, 0])
            fp = int(file_cm_np[0, 1])
            fn = int(file_cm_np[1, 0])
            tp = int(file_cm_np[1, 1])
            file_total = tn + fp + fn + tp
            file_fp_rate = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            file_recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

            if cm_sum == 0 and epoch % 10 == 0:  # Log warning every 10 epochs to avoid spam
                print(f"[WARNING] Empty confusion matrix at epoch {epoch} - no validation data processed (cm_sum={cm_sum}, total_samples_t={total_samples_t.item()})")
        else:
            cm_np = None
            cm_metrics = None
            file_fp_rate = None
            file_recall = None
            file_total = 0
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        if rank == 0:
            epoch_times.append(epoch_time)
            # Keep only last 10 epoch times for more accurate estimation
            if len(epoch_times) > 10:
                epoch_times.pop(0)

        # ここに100エポックごとの詳細出力を追加
        if epoch % 100 == 0 and rank == 0:
            print(f"\n--- Epoch {epoch} ---")
            print(f"Class Weights: {class_weights.cpu().numpy()}")
            
            for param_group in optimizer.param_groups:
                print(f"Learning Rate: {param_group['lr']}")
    
            print(f"Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}")
    
            if cm_metrics is not None:
                print(f"Per-Class Precision: {cm_metrics['precision_per_class']}")
                print(f"Per-Class Recall: {cm_metrics['recall_per_class']}")
                print(f"Per-Class F1-Score: {cm_metrics['f1_per_class']}")
                print(f"Support per Class: {cm_metrics['support_per_class']}")
                print(
                    f"Weighted Precision: {cm_metrics['weighted_precision']:.4f}, "
                    f"Weighted Recall: {cm_metrics['weighted_recall']:.4f}, "
                    f"Weighted F1-Score: {cm_metrics['weighted_f1']:.4f}"
                )

        if rank == 0:
            if cm_metrics is not None:
                precision = cm_metrics["weighted_precision"]
                recall = cm_metrics["weighted_recall"]
                f1 = cm_metrics["weighted_f1"]
                macro_mode = getattr(args, "macro_mode", "support_only")
                macro_f1 = _select_macro_f1(cm_metrics, macro_mode)
                if epoch % 50 == 0:  # Print every 50 epochs to reduce output
                    print(
                        f'Precision: {precision:.4f}, Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}, '
                        f'Macro F1-Score: {macro_f1:.4f} (macro_mode={macro_mode})'
                    )
                    # Step0: macro定義の確認（support=0の扱いを明示）
                    print(
                        f'  - macro_f1_support_only={cm_metrics.get("macro_f1_support_only", 0.0):.4f}, '
                        f'macro_f1_sklearn_like={cm_metrics.get("macro_f1_sklearn_like", 0.0):.4f}, '
                        f'macro_f1_all={cm_metrics.get("macro_f1_all", 0.0):.4f}'
                    )
                    if file_fp_rate is not None and file_recall is not None:
                        print(f'File-level: FP-rate(defect-free)={file_fp_rate:.4f}, Recall(defect files)={file_recall:.4f}, Files={file_total}')
            else:
                # デフォルト値を設定（混同行列が空の場合）
                precision = 0.0
                recall = 0.0
                f1 = 0.0
                macro_f1 = 0.0
                if epoch % 50 == 0:  # Log warning every 50 epochs to avoid spam
                    print(f'[WARNING] Cannot compute metrics - empty validation data at epoch {epoch}')
                    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}, Macro F1-Score: {macro_f1:.4f}')
                    print('File-level: FP-rate(defect-free)=0.0000, Recall(defect files)=0.0000, Files=0')

        if rank == 0:
            precision_per_epoch.append(precision)
            recall_per_epoch.append(recall)
            f1_score_per_epoch.append(f1)
            macro_f1_per_epoch.append(macro_f1)
            val_accuracy_per_epoch.append(val_accuracy.item())
            train_loss_per_epoch.append(avg_train_loss.item())
            val_loss_per_epoch.append(avg_val_loss.item())

        # Update learning rate scheduler (for CosineAnnealingLR)
        if not use_onecycle:
            scheduler.step()
        
        # CRITICAL: Ensure all NCCL operations from this epoch are complete
        # All collectives (all_reduce) are already synchronous, so we just need CUDA sync
        # Removed unnecessary barrier here - all_reduce operations are already synchronized
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # ========================================================================
        # DDP SAFETY AUDIT: Collective Call Order (ALL ranks execute in SAME order)
        # ========================================================================
        # Every epoch, ALL ranks execute these collectives in EXACTLY this order:
        # 1. dist.all_reduce(avg_train_loss, op=SUM)      [Line ~1556]
        # 2. dist.all_reduce(avg_val_loss, op=SUM)        [Line ~1588]
        # 3. dist.all_reduce(total_correct, op=SUM)        [Line ~1593]
        # 4. dist.all_reduce(total_samples_t, op=SUM)      [Line ~1594]
        # 5. dist.all_reduce(cm_local, op=SUM)             [Line ~1597]
        # 6. dist.all_reduce(file_cm_local, op=SUM)        [File-level metrics]
        # 7. dist.broadcast(stop_tensor, src=0)            [Line ~1756] - ALL ranks call this!
        #
        # IMPORTANT:
        # - NO collectives inside "if rank == 0:" conditionals
        # - NO early returns that skip collectives on some ranks
        # - All collectives are synchronous (no async operations)
        # - drop_last=True ensures identical batch counts across ranks
        # ========================================================================
        barrier_success = True
        
        # Track best metrics and save checkpoints
        if rank == 0:
            improved_loss = avg_val_loss.item() < best_val_loss
            improved_macro_f1 = macro_f1 > best_macro_f1
            
            # Track val_loss for checkpoint naming (but not for early stopping)
            if improved_loss:
                best_val_loss = avg_val_loss.item()
            
            # Early stopping is based on Macro F1 only (priority metric for imbalanced datasets)
            if improved_macro_f1:
                best_macro_f1 = macro_f1
                macro_f1_counter = 0
            else:
                macro_f1_counter += 1
            
            # Update best model state when Macro F1 improves (for early stopping)
            if improved_macro_f1:
                import copy
                best_model_state = copy.deepcopy(ddp_model.module.state_dict())
                best_epoch = epoch

            # Step3: On best-update only, print low-cost per-class/confusion summary
            best_report = getattr(args, "best_report", True)
            if improved_macro_f1 and best_report and cm_np is not None and cm_metrics is not None:
                topk = int(getattr(args, "best_report_topk", 5))
                topk_absorb = int(getattr(args, "best_report_confusion_topk", 3))
                summary = _best_update_summary_from_cm(
                    cm=np.asarray(cm_np),
                    f1_per_class=np.asarray(cm_metrics["f1_per_class"]),
                    support_per_class=np.asarray(cm_metrics["support_per_class"]),
                    topk_classes=topk,
                    topk_absorb=topk_absorb,
                )
                print("\n=== Best-update diagnostics (per-class / confusion) ===")
                print(f"macro_mode={getattr(args, 'macro_mode', 'support_only')}, epoch={epoch}")
                print(f"Zero-support (absent) classes: {summary['zero_support_classes']}")
                print(f"F1=0 with support>0 classes: {summary['zero_f1_with_support_classes']}")
                if len(summary["absorb_lines"]) > 0:
                    print("Absorb targets (worst classes):")
                    for line in summary["absorb_lines"]:
                        print(line)

            # Step2: checkpoint saving aligned to the target metric (macro_f1)
            save_on_best = getattr(args, "save_on_best", True)
            save_every = int(getattr(args, "save_every", 100))
            macro_mode = getattr(args, "macro_mode", "support_only")

            # Save "best" checkpoint immediately when macro improves (default ON)
            if save_on_best and improved_macro_f1:
                try:
                    if last_best_checkpoint_path is not None and os.path.exists(last_best_checkpoint_path):
                        try:
                            os.remove(last_best_checkpoint_path)
                            print(f"[CHECKPOINT] Deleted previous best checkpoint: {last_best_checkpoint_path}")
                        except Exception as e:
                            print(f"[WARNING] Failed to delete previous best checkpoint: {e}")

                    checkpoint_path = (
                        f"{checkpoint_dir}/{type(model).__name__}_{timestamp}_Best_Epoch{epoch}"
                        f"_ValLossBest{best_val_loss:.4f}"
                        f"_MacroF1Best{best_macro_f1:.4f}"
                        f"_MacroF1Cur{macro_f1:.4f}"
                        f"_MacroMode{macro_mode}.pth"
                    )
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": ddp_model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                        "best_val_loss": best_val_loss,
                        "best_macro_f1": best_macro_f1,
                        "macro_f1_cur": macro_f1,
                        "macro_mode": macro_mode,
                        "timestamp": timestamp,
                    }, checkpoint_path)
                    last_best_checkpoint_path = checkpoint_path
                    print(
                        f"[CHECKPOINT] Saved BEST (macro improved) epoch {epoch}: {checkpoint_path} "
                        f"(macro_f1_cur={macro_f1:.4f}, macro_f1_best={best_macro_f1:.4f}, val_loss_best={best_val_loss:.4f})"
                    )
                    # Also keep a simple state_dict "Best" for inference convenience (overwrites)
                    best_state_path = f"{checkpoint_dir}/{type(model).__name__}_{timestamp}_Best.pth"
                    torch.save(ddp_model.module.state_dict(), best_state_path)
                except Exception as e:
                    print(f"[WARNING] Failed to save BEST checkpoint: {e}")
                    import traceback
                    traceback.print_exc()

            # Periodic "latest" checkpoint (for resume / safety)
            if save_every > 0 and (epoch % save_every == 0):
                try:
                    latest_checkpoint_path = f"{checkpoint_dir}/{type(model).__name__}_{timestamp}_Latest.pth"
                    torch.save(ddp_model.module.state_dict(), latest_checkpoint_path)
                except Exception as e:
                    print(f"[WARNING] Failed to save Latest checkpoint: {e}")
            
            # Early stopping: stop if Macro F1 doesn't improve (Macro F1 is the priority metric for imbalanced datasets)
            # IMPORTANT: Only trigger early stopping if barrier succeeded
            # If barrier failed, skip early stopping this epoch to avoid hanging
            # barrier_success is available to all ranks (defined before this block)
            should_stop = (macro_f1_counter >= args.patience and barrier_success)
        else:
            # Non-rank-0: initialize should_stop (will be overwritten by broadcast)
            should_stop = False
        
        # CRITICAL DDP FIX: ALL ranks must call dist.broadcast(), not just rank 0
        # Rank 0 computes should_stop, but ALL ranks participate in the collective
        stop_tensor = torch.tensor(1 if should_stop else 0, dtype=torch.int32, device=device)
        try:
            # ALL ranks call broadcast - rank 0 sends, all ranks receive
            dist.broadcast(stop_tensor, src=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure broadcast completes
            
            # All ranks read the broadcast value
            should_stop_all = stop_tensor.item() == 1
        except Exception as e:
            # If broadcast fails, log and use fallback
            if rank == 0:
                print(f"[WARNING] Broadcast of early stop flag failed: {e}")
                should_stop_all = should_stop
            else:
                print(f"[WARNING] Rank {rank}: Broadcast of early stop flag failed: {e}")
                should_stop_all = False
        
        # [FIX S-2] Early stopping - ensure ALL ranks break at the SAME indentation level
        if should_stop_all:
            early_stopped = True
            if rank == 0:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered at epoch {epoch}")
                print(f"  - Macro F1 hasn't improved for {macro_f1_counter} epochs (monitoring Macro F1 only)")
                print(f"  - Best Val Loss: {best_val_loss:.4f}")
                print(f"  - Best Macro F1: {best_macro_f1:.4f}")
                print(f"  - Best model was at epoch {best_epoch}")
                print(f"{'='*60}\n")
                
                # Load and save best model
                if best_model_state is not None:
                    try:
                        print(f"[INFO] Loading best model from epoch {best_epoch}...")
                        ddp_model.module.load_state_dict(best_model_state)
                        print(f"[INFO] Best model loaded successfully")
                        
                        # Save best model immediately
                        print("[INFO] Saving best model after early stopping...")
                        save_dir = checkpoint_dir
                        os.makedirs(save_dir, exist_ok=True)
                        best_final_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Best_EarlyStop.pth"
                        torch.save(best_model_state, best_final_path)
                        print(f"[FINAL_INFO] Best model (epoch {best_epoch}) saved to {best_final_path}")
                        print(f"[INFO] Skipping post-training evaluation to prevent NCCL timeout")
                        print(f"[INFO] Model is saved and ready for use")
                    except Exception as e:
                        print(f"[WARNING] Failed to save best model after early stopping: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Break immediately - all ranks are synchronized, no pending operations
            # This is safe because:
            # 1. Barrier completed successfully (barrier_success == True)
            # 2. All NCCL operations from this epoch are complete
            # 3. Broadcast completed, so all ranks know to break
            # 4. All ranks will break at the same point
            break

        # 100エポックごとの出力と予想時間の表示
        if rank == 0 and epoch % 100 == 0:
            # Calculate estimated completion time
            if len(epoch_times) > 0:
                avg_epoch_time = np.mean(epoch_times)
                remaining_epochs = args.epochs - epoch
                estimated_remaining_time = avg_epoch_time * remaining_epochs
                
                # Format time
                hours = int(estimated_remaining_time // 3600)
                minutes = int((estimated_remaining_time % 3600) // 60)
                seconds = int(estimated_remaining_time % 60)
                
                elapsed_time = time.time() - train_start_time
                elapsed_hours = int(elapsed_time // 3600)
                elapsed_minutes = int((elapsed_time % 3600) // 60)
                elapsed_seconds = int(elapsed_time % 60)
                
                print(f'\n=== Epoch {epoch}/{args.epochs} ===')
                print(f'Train Loss: {avg_train_loss.item():.4f}, Val Loss: {avg_val_loss.item():.4f}, Val Acc: {val_accuracy:.4f}, Macro F1: {macro_f1:.4f}')
                print(f'Elapsed Time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}')
                print(f'Estimated Remaining Time: {hours:02d}:{minutes:02d}:{seconds:02d} (Avg {avg_epoch_time:.2f}s/epoch)')
                print('=' * 50)

    # Early stopping is now handled by breaking from the loop directly
    # No need to check tensor or file - early_stopped flag is set when breaking
    # This is simpler and more reliable
    
    # CRITICAL: Only synchronize if early stopping did NOT occur
    # If early stopping occurred, we already synchronized before setting the flag
    # Additional barriers here can cause hangs if communicator is aborted
    if not early_stopped:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Lightweight barrier to ensure all ranks complete training loop
            # Note: timeout parameter not supported in this PyTorch version
            dist.barrier()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] Post-training barrier failed: {e}")
    
    # Synchronize all ranks after training loop ONLY if early stopping didn't occur
    # Note: If early stopping occurred, best model was already saved
    # But we still want to run evaluation/visualization on rank 0
    if early_stopped:
        # Early stopping occurred - model already saved
        if rank == 0:
            print(f"[INFO] Early stopping occurred - will still run post-training evaluation/visualization")
            print(f"[INFO] Best model saved at: {checkpoint_dir}/{type(model).__name__}_{timestamp}_Best_EarlyStop.pth")
        
        # Clean up flag file (if it exists from fallback)
        try:
            early_stop_flag_file = os.path.join(checkpoint_dir, f".early_stop_{timestamp}.flag")
            if rank == 0 and os.path.exists(early_stop_flag_file):
                os.remove(early_stop_flag_file)
        except:
            pass
        
        # Early stopping occurred - model already saved
        # Skip additional synchronization to avoid hanging on aborted communicators
        # The synchronization already happened before setting the early stop flag
        # But we still want to run evaluation/visualization on rank 0
        # Other ranks should exit here to avoid NCCL timeout
        if rank != 0:
            # Rank-0 continues into post-training single-process work,
            # so non-rank-0 must NOT enter another barrier during cleanup.
            cleanup(device=device, rank=rank, do_barrier=False)
            return
        # Rank 0 continues to post-training evaluation/visualization section below
    else:
        # Normal completion - synchronize before post-training processing
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Note: timeout parameter not supported in this PyTorch version
            dist.barrier()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] Barrier failed after training loop: {e}")

    if rank == 0:
        train_elapsed_time = time.time() - train_start_time
        train_hours = int(train_elapsed_time // 3600)
        train_minutes = int((train_elapsed_time % 3600) // 60)
        train_seconds = int(train_elapsed_time % 60)
        
        # Reconstruct command for final output
        command_str = ' '.join(sys.argv)
        
        print(f"\n{'='*60}")
        print(f"Training Completed")
        print(f"Best Val Loss: {best_val_loss:.4f}, Best Macro F1: {best_macro_f1:.4f}")
        print(f"Best model was at epoch {best_epoch}")
        print(f"Total Training Time: {train_hours:02d}:{train_minutes:02d}:{train_seconds:02d}")
        print(f"{'='*60}")
        print(f"\nExecution Command (for reference):")
        print(f"  {command_str}")
        print(f"{'='*60}\n")
        
        # Ensure best model is loaded (if not already loaded during early stopping)
        if best_model_state is not None:
            try:
                # Only load if not already loaded (early stopping already loaded it)
                current_state = ddp_model.module.state_dict()
                # Simple check: if current state matches best state, skip loading
                if not all(torch.equal(current_state[k], best_model_state[k]) for k in current_state.keys()):
                    print(f"[INFO] Loading best model from epoch {best_epoch}...")
                    ddp_model.module.load_state_dict(best_model_state)
                    print(f"[INFO] Best model loaded successfully")
                else:
                    print(f"[INFO] Best model already loaded")
            except Exception as e:
                print(f"[WARNING] Failed to load best model: {e}")
                import traceback
                traceback.print_exc()
        
        # Save final model immediately (before other time-consuming operations)
        try:
            print("[INFO] Saving final model...")
            save_dir = checkpoint_dir
            os.makedirs(save_dir, exist_ok=True)
            final_model_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Final.pth"
            torch.save(ddp_model.module.state_dict(), final_model_path)
            print(f"[FINAL_INFO] Final model saved to {final_model_path}")
            
            # Also save best model as final if available (if not already saved during early stopping)
            if best_model_state is not None:
                best_final_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Best_Final.pth"
                # Check if early stop file exists, if not save it
                early_stop_path = f"{save_dir}/{type(model).__name__}_{timestamp}_Best_EarlyStop.pth"
                if not os.path.exists(early_stop_path):
                    torch.save(best_model_state, best_final_path)
                    print(f"[FINAL_INFO] Best model (epoch {best_epoch}) saved to {best_final_path}")
                else:
                    print(f"[INFO] Best model already saved during early stopping")
        except Exception as e:
            print(f"[ERROR] Failed to save final model: {e}")
            import traceback
            traceback.print_exc()
    
    # ======================================================================
    # DDP SAFETY (IMPORTANT):
    # After training, rank 0 does time-consuming work (plots / test eval / I/O).
    # If other ranks enter cleanup (destroy PG) while rank 0 is still running,
    # NCCL can time out or error on shutdown. To avoid this:
    # - All ranks first synchronize once (best-effort).
    # - Then non-rank-0 ranks destroy the process group and exit immediately.
    # - Rank 0 also destroys the process group BEFORE running post-training work,
    #   so no collectives can be triggered during evaluation/visualization.
    # ======================================================================
    # IMPORTANT:
    # - If `early_stopped` is True, non-rank-0 ranks may have already exited above.
    #   In that case, calling ANY collective here (barrier/allreduce/etc.) can hang and
    #   trigger NCCL watchdog timeouts.
    # - Therefore, only attempt this barrier on normal completion.
    if not early_stopped:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dist.barrier()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] Final barrier before post-training work failed (will continue): {e}")

    if rank != 0:
        # Rank-0 continues into post-training single-process work,
        # so non-rank-0 must NOT enter another barrier during cleanup.
        cleanup(device=device, rank=rank, do_barrier=False)
        return

    # Rank 0: destroy process group first, then continue in single-process mode.
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            print("[INFO] Rank 0: destroyed process group for safe post-training evaluation (single-process)")
    except Exception as e:
        print(f"[WARNING] Rank 0: failed to destroy process group (continuing anyway): {e}")

    if rank == 0:
        # Loss plotを保存（アーリーストップ時も含む）
        try:
            if len(train_loss_per_epoch) > 0 and len(val_loss_per_epoch) > 0:
                print("Saving loss plots...")
                # データセット情報を取得
                train_pairs_count = len(train_pairs) if 'train_pairs' in locals() else 0
                val_pairs_count = len(val_pairs) if 'val_pairs' in locals() else 0
                test_pairs_count = len(test_pairs) if 'test_pairs' in locals() else 0
                data_usage_ratio = getattr(args, 'data_usage_ratio', 1.0)
                dataset_info = f"NDF_T{train_pairs_count}V{val_pairs_count}Te{test_pairs_count}"
                if data_usage_ratio < 1.0:
                    dataset_info += f"R{int(data_usage_ratio*100)}"
                
                # Macro F1スコアとデータセット情報を含むフォルダ名を作成
                output_plot_dir = os.path.join(
                    output_root,
                    "Predict_data",
                    f"Loss_plots_{timestamp}_{dataset_info}_F1_{best_macro_f1:.2f}"
                )
                os.makedirs(output_plot_dir, exist_ok=True)
                print(f"[INFO] Loss plots directory: {output_plot_dir} (Macro F1: {best_macro_f1:.4f})")
                
                # データセット情報をREADMEファイルに保存
                readme_path = os.path.join(output_plot_dir, "dataset_info.txt")
                with open(readme_path, 'w') as f:
                    f.write("=== Dataset Information ===\n")
                    f.write(f"Dataset Type: Noise Defect-Free\n")
                    f.write(f"Train pairs: {train_pairs_count}\n")
                    f.write(f"Val pairs: {val_pairs_count}\n")
                    f.write(f"Test pairs: {test_pairs_count}\n")
                    f.write(f"Total pairs: {train_pairs_count + val_pairs_count + test_pairs_count}\n")
                    f.write(f"Data usage ratio: {data_usage_ratio}\n")
                    f.write(f"Number of classes: 19\n")
                    f.write(f"Best Macro F1 Score: {best_macro_f1:.4f}\n")
                    f.write(f"Best Epoch: {best_epoch}\n")
                    f.write(f"Best Val Loss: {best_val_loss:.4f}\n")
                
                epochs = range(1, len(train_loss_per_epoch) + 1)
                
                # Loss plot
                try:
                    plt.figure(figsize=(12, 6))
                    plt.plot(epochs, train_loss_per_epoch, 'b-', label='Train Loss', linewidth=2)
                    plt.plot(epochs, val_loss_per_epoch, 'r-', label='Val Loss', linewidth=2)
                    plt.xlabel('Epoch', fontsize=12)
                    plt.ylabel('Loss', fontsize=12)
                    plt.yscale('log')  # Log scale for y-axis
                    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
                    plt.legend(fontsize=11)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_plot_dir, f"loss_plot_{timestamp}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Loss plot saved to: {output_plot_dir}/loss_plot_{timestamp}.png")
                except Exception as e:
                    print(f"[WARNING] Failed to save loss plot: {e}")
                
                # Macro F1 plot
                if len(macro_f1_per_epoch) > 0:
                    try:
                        plt.figure(figsize=(12, 6))
                        plt.plot(epochs, macro_f1_per_epoch, 'g-', label='Macro F1-Score', linewidth=2)
                        plt.xlabel('Epoch', fontsize=12)
                        plt.ylabel('Macro F1-Score', fontsize=12)
                        plt.title('Macro F1-Score per Epoch', fontsize=14, fontweight='bold')
                        plt.legend(fontsize=11)
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_plot_dir, f"macro_f1_plot_{timestamp}.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"✓ Macro F1 plot saved to: {output_plot_dir}/macro_f1_plot_{timestamp}.png")
                    except Exception as e:
                        print(f"[WARNING] Failed to save macro F1 plot: {e}")
                
                # Combined plot (Loss + Macro F1)
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Loss plot
                    ax1.plot(epochs, train_loss_per_epoch, 'b-', label='Train Loss', linewidth=2)
                    ax1.plot(epochs, val_loss_per_epoch, 'r-', label='Val Loss', linewidth=2)
                    ax1.set_xlabel('Epoch', fontsize=12)
                    ax1.set_ylabel('Loss', fontsize=12)
                    ax1.set_yscale('log')  # Log scale for y-axis
                    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
                    ax1.legend(fontsize=11)
                    ax1.grid(True, alpha=0.3)
                    
                    # Macro F1 plot
                    if len(macro_f1_per_epoch) > 0:
                        ax2.plot(epochs, macro_f1_per_epoch, 'g-', label='Macro F1-Score', linewidth=2)
                        ax2.set_xlabel('Epoch', fontsize=12)
                        ax2.set_ylabel('Macro F1-Score', fontsize=12)
                        ax2.set_title('Macro F1-Score per Epoch', fontsize=14, fontweight='bold')
                        ax2.legend(fontsize=11)
                        ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_plot_dir, f"combined_plot_{timestamp}.png"), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✓ Combined plot saved to: {output_plot_dir}/combined_plot_{timestamp}.png")
                except Exception as e:
                    print(f"[WARNING] Failed to save combined plot: {e}")
            else:
                print("[WARNING] No loss data available for plotting")
        except Exception as e:
            print(f"[ERROR] Failed to save loss plots: {e}")
            import traceback
            traceback.print_exc()

        # Test evaluation and visualization run on rank 0 only (now single-process; no DDP collectives)
        if rank == 0:
            print("Evaluating on Test Data...")
            
            # [FIX S-1] Create a new test DataLoader without DistributedSampler for rank-0 evaluation
            # This ensures full test dataset evaluation with correct order matching test_pairs
            test_loader_eval = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False
            )
            
            # テストデータの評価
            ddp_model.eval()
            
            test_preds = []
            test_labels = []
            test_probs = []  # 確率を格納するリスト
            
            # --- loss function for test must match training choice ---
            use_logit_adjust = getattr(args, 'use_logit_adjust', False)
            logit_adjust_tau = getattr(args, 'logit_adjust_tau', 1.0)
            use_log_softmax = getattr(args, 'use_log_softmax', False)
            gamma = getattr(args, 'gamma', default_gamma)
            use_layer_constraint = getattr(args, 'use_layer_constraint', True)
            layer_constraint_weight = getattr(args, 'layer_constraint_weight', 1.0)
            
            # ベース損失関数を作成（学習時と同じ）
            if use_logit_adjust:
                # trainと同じpriorを作る（train_datasetから固定）
                labels_array = np.concatenate([d.y.cpu().numpy() for d in train_dataset])
                class_counts = np.bincount(labels_array, minlength=19)
                class_prior = torch.tensor(class_counts / (class_counts.sum() + 1e-8), dtype=torch.float, device=device)
                base_loss_fn = LogitAdjustLoss(class_prior=class_prior, tau=logit_adjust_tau).to(device)
            else:
                if use_log_softmax:
                    base_loss_fn = FocalLossLogSoftmax(weights=class_weights.to(device), gamma=gamma).to(device)
                else:
                    base_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device)).to(device)
            
            # 層制約を適用（学習時と同じ設定）
            if use_layer_constraint:
                loss_fn = LayerAwareLoss(
                    base_loss_fn=base_loss_fn,
                    vertices_per_layer=vertices_per_layer,
                    constraint_weight=layer_constraint_weight
                ).to(device)
            else:
                loss_fn = base_loss_fn
            
            correct = 0
            total_loss = 0.0
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader_eval):
                    batch = batch.to(device)
                    out = ddp_model(batch)  # Model outputs logits
                    y = batch.y
            
                    # 層制約を適用する場合、batch情報を渡す
                    if use_layer_constraint:
                        loss = loss_fn(out, y, batch=batch.batch)
                    else:
                        loss = loss_fn(out, y)
                    total_loss += loss.item()
            
                    # 予測ラベル
                    pred = out.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total_samples += y.size(0)
            
                    # 予測ラベル・実ラベル・確率を格納
                    test_preds.extend(pred.cpu().numpy())
                    test_labels.extend(y.cpu().numpy())
                    
                    # Model outputs logits, apply softmax to get probabilities
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    test_probs.extend(probs)
            
            test_loss = total_loss / len(test_loader_eval)  # 修正: test_loader_evalを使う
            test_accuracy = correct / total_samples
            
            # numpy 配列に変換
            test_preds = np.array(test_preds)
            test_labels = np.array(test_labels)
            test_probs = np.array(test_probs)
            
            # Sanity-check: layer constraint violations in node-level predictions
            # Each file has 13942 nodes = 6971 (Layer1) + 6971 (Layer2)
            try:
                pos = np.arange(len(test_preds)) % 13942
                inv_l1 = int(np.sum((pos < 6971) & (test_preds >= 10)))  # Layer1 should be 0..9
                inv_l2 = int(np.sum((pos >= 6971) & (test_preds >= 1) & (test_preds <= 9)))  # Layer2 should be {0,10..18}
                if inv_l1 > 0 or inv_l2 > 0:
                    print(f"[WARNING] Layer constraint violations in TEST preds: layer1_invalid(>=10)={inv_l1}, layer2_invalid(1..9)={inv_l2}")
                else:
                    print(f"[INFO] Layer constraint check passed on TEST preds (no violations)")
            except Exception as e:
                print(f"[WARNING] Layer constraint check failed to run: {e}")
            
            print(f"\n[INFO] Test set result:")
            print(f" - Test Loss: {test_loss:.4f}")
            print(f" - Test Accuracy: {test_accuracy:.4f}")
            precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
            recall = recall_score(test_labels, test_preds, average='weighted')
            f1 = f1_score(test_labels, test_preds, average='weighted')
            macro_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
            macro_precision = precision_score(test_labels, test_preds, average='macro', zero_division=0)
            macro_recall = recall_score(test_labels, test_preds, average='macro', zero_division=0)
            balanced_acc = balanced_accuracy_score(test_labels, test_preds)
            mcc = matthews_corrcoef(test_labels, test_preds)
            
            print(f" - Weighted Precision: {precision:.4f}, Weighted Recall: {recall:.4f}, Weighted F1-Score: {f1:.4f}")
            print(f" - Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1-Score: {macro_f1:.4f}")
            print(f" - Balanced Accuracy: {balanced_acc:.4f}")
            print(f" - Matthews Correlation Coefficient: {mcc:.4f}")
            print(f" - test_preds.shape={test_preds.shape}, test_labels.shape={test_labels.shape}, test_probs.shape={test_probs.shape}")
            print(f"\n[INFO] Macro F1 Score: {macro_f1:.4f} (logged for reference)")
            
            # Print per-class performance summary with detailed debugging (verbose only)
            if VERBOSE_PRINT:
                class_report_dict = classification_report(test_labels, test_preds, output_dict=True, zero_division=0)
                
                # 混同行列を計算して詳細を確認
                cm = confusion_matrix(test_labels, test_preds, labels=list(range(19)))
                
                # 各クラスのF1スコアを手動計算して検証
                print(f"\n=== Macro F1 Score Calculation Verification ===")
                print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
                print("-" * 80)
                
                per_class_f1_scores = []
                for i in range(19):
                    # 混同行列から各クラスのTP, FP, FNを計算
                    tp = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
                    fp = cm[:, i].sum() - tp if i < cm.shape[1] else 0
                    fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
                    support = cm[i, :].sum() if i < cm.shape[0] else 0
                    
                    # Precision, Recall, F1を計算
                    if tp + fp > 0:
                        class_precision = tp / (tp + fp)
                    else:
                        class_precision = 0.0
                    
                    if tp + fn > 0:
                        class_recall = tp / (tp + fn)
                    else:
                        class_recall = 0.0
                    
                    if class_precision + class_recall > 0:
                        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
                    else:
                        class_f1 = 0.0
                    
                    per_class_f1_scores.append(class_f1)
                    
                    # classification_reportの結果と比較
                    if str(i) in class_report_dict:
                        metrics = class_report_dict[str(i)]
                        print(
                            f"{i:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} "
                            f"{int(metrics['support']):<12} {tp:<8} {fp:<8} {fn:<8}"
                        )
                    else:
                        print(f"{i:<8} {class_precision:<12.4f} {class_recall:<12.4f} {class_f1:<12.4f} {support:<12} {tp:<8} {fp:<8} {fn:<8}")
                
                # Macro F1を手動計算して検証
                manual_macro_f1 = np.mean(per_class_f1_scores)
                print(f"\n[VERIFICATION] Manual Macro F1 calculation: {manual_macro_f1:.4f}")
                print(f"[VERIFICATION] sklearn Macro F1: {macro_f1:.4f}")
                print(f"[VERIFICATION] Difference: {abs(manual_macro_f1 - macro_f1):.6f}")
                
                # 予測分布とラベル分布を比較
                unique_labels, label_counts = np.unique(test_labels, return_counts=True)
                unique_preds, pred_counts = np.unique(test_preds, return_counts=True)
                print(f"\n=== Label vs Prediction Distribution ===")
                print(f"{'Class':<8} {'True Labels':<15} {'Predictions':<15} {'Match':<10}")
                print("-" * 50)
                for i in range(19):
                    true_count = label_counts[unique_labels == i][0] if i in unique_labels else 0
                    pred_count = pred_counts[unique_preds == i][0] if i in unique_preds else 0
                    match = "✓" if true_count > 0 and pred_count > 0 else "✗"
                    print(f"{i:<8} {true_count:<15} {pred_count:<15} {match:<10}")
                
                print(f"\n=== Per-Class Performance Summary ===")
                print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
                print("-" * 60)
                for i in range(19):
                    if str(i) in class_report_dict:
                        metrics = class_report_dict[str(i)]
                        print(f"{i:<8} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1-score']:<12.4f} {int(metrics['support']):<12}")
            
            # ---- 予測結果を CSV に保存 ----
            output_dir_csv = os.path.join(output_root, "Predict_csv", f"Predict_csv_zscore_{timestamp}")
            os.makedirs(output_dir_csv, exist_ok=True)
            
            save_predictions_to_csv(
                test_pairs,
                test_preds,
                test_labels,
                test_probs,
                output_dir_csv,
                num_nodes_per_sample=13942
            )
            
            print("[INFO] CSV save completed.")
            
            # ---- 可視化や追加の評価 ----
            # CRITICAL: Visualization runs only on rank 0
            # Other ranks have already exited, so no synchronization needed
            try:
                gamma = getattr(args, 'gamma', default_gamma)
                evaluate_and_visualize(
                    test_loader,
                    train_loader,
                    ddp_model,
                    device,
                    class_weights,
                    test_pairs,
                    train_pairs,
                    num_nodes_per_sample=13942,
                    gamma=gamma,
                    rank=rank,
                    world_size=world_size,
                    test_dataset=test_dataset,
                    x_coords=x_coords,
                    y_coords=y_coords,
                    z_coords=z_coords,
                    output_root=output_root,
                    run_id=run_id,
                    macro_mode=getattr(args, "macro_mode", "support_only")
                )
            except Exception as e:
                print(f"[ERROR] Failed during evaluation/visualization: {e}")
                import traceback
                traceback.print_exc()
                # Continue even if visualization fails - model is already saved
    # Rank 0 single-process cleanup (no distributed PG at this point)
    cleanup(device=device, rank=rank)

# ----------------------------
# エントリーポイント
# ----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distributed Training Script for Z-score Normalized Data')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Number of hidden channels')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate (default: 0.002, reduced from 0.01 to prevent collapse)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=300, help='Early stopping patience')
    parser.add_argument('--gamma', type=float, default=3.0, help='Focal loss gamma parameter (default: 3.0, increased for extreme class imbalance)')
    parser.add_argument('--class_weight_multiplier', type=float, default=5.0, help='Class weight multiplier for minority classes (default: 5.0, increased for better handling of imbalanced classes)')
    parser.add_argument('--use_logit_adjust', action='store_true', default=False, help='Use LogitAdjustLoss instead of FocalLoss (default: False, recommended for imbalanced datasets - priority 1)')
    parser.add_argument('--logit_adjust_tau', type=float, default=1.0, help='Tau parameter for LogitAdjustLoss (default: 1.0, recommended range: 1.0~2.0)')
    # Boolean引数の修正: store_true + default=Trueは常にTrueになる問題を修正
    # default=Falseにして、--no_xxxでOFFにする方式（互換性重視）
    parser.add_argument('--use_log_softmax', action='store_true', default=False, help='Use log_softmax version of FocalLoss for numerical stability (default: False)')
    parser.add_argument('--no_log_softmax', action='store_false', dest='use_log_softmax', help='Disable log_softmax version of FocalLoss')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use mixed precision training (AMP) (default: False)')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='Disable mixed precision training')
    parser.add_argument('--use_onecycle', action='store_true', default=True, help='Use OneCycleLR scheduler instead of CosineAnnealingLR (default: True)')
    parser.add_argument('--no_onecycle', dest='use_onecycle', action='store_false', help='Disable OneCycleLR scheduler (use CosineAnnealingLR)')
    parser.add_argument('--onecycle_max_lr', type=float, default=None, help='Max learning rate for OneCycleLR (default: learning_rate * 2.0, recommended: 1e-3 ~ 3e-3 for GAT)')
    # OneCycleLR fine tuning (optional)
    parser.add_argument('--onecycle_pct_start', type=float, default=0.1, help='OneCycleLR pct_start (fraction of steps to increase lr). With epochs=2000, 0.1 means lr increases for ~200 epochs (default: 0.1)')
    parser.add_argument('--onecycle_anneal_strategy', type=str, default='cos', choices=['cos', 'linear'], help='OneCycleLR anneal_strategy (default: cos)')
    parser.add_argument('--onecycle_div_factor', type=float, default=25.0, help='OneCycleLR div_factor (initial_lr = max_lr/div_factor) (default: 25.0)')
    parser.add_argument('--onecycle_final_div_factor', type=float, default=1e4, help='OneCycleLR final_div_factor (min_lr = initial_lr/final_div_factor) (default: 1e4)')
    parser.add_argument('--use_minority_sampler', action='store_true', default=False, help='Use DistributedMinorityRatioSampler for training (default: False, recommended for imbalanced datasets)')
    parser.add_argument('--minority_weight_power', type=float, default=1.0, help='Power for minority_ratio weighting in sampler (default: 1.0, higher=more emphasis on minority-rich files)')
    parser.add_argument('--use_class_frequency_sampler', action='store_true', default=False, help='Use DistributedClassFrequencySampler with 1/sqrt(freq) weighting (default: False, recommended for highly imbalanced datasets)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability (default: 0.1, recommended range: 0.0~0.3)')
    parser.add_argument('--edge_drop_prob', type=float, default=0.01, help='Edge dropout probability (default: 0.01, recommended range: 0.0~0.05)')
    parser.add_argument('--data_usage_ratio', type=float, default=0.5, choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       help='Ratio of data to use (10%% increments: 0.1, 0.2, ..., 1.0). Default: 1.0 (100%%)')
    # Layer constraint: デフォルトで有効（--no_layer_constraintで無効化可能）
    parser.add_argument('--no_layer_constraint', dest='use_layer_constraint', action='store_false', default=True, help='Disable layer-aware label constraints (default: enabled, Layer 1: 0,1~9 only, Layer 2: 0,10~18 only)')
    parser.add_argument('--layer_constraint_weight', type=float, default=1.0, help='Weight for layer constraint penalty (default: 1.0, higher=stronger constraint)')
    parser.add_argument('--output_root', type=str, default=None, help='Root directory for all outputs (models/predictions/plots/csv). Default keeps legacy paths under /home/nishioka/GNN/GNN_hole_2026')
    parser.add_argument('--run_id', type=str, default=None, help='Optional run identifier (stored in logs/metadata by launcher).')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume training from a checkpoint path (.pth). Supports both dict checkpoints (with optimizer/scheduler) and plain state_dict checkpoints.')
    
    # Fine-tuning options
    parser.add_argument('--fine_tune_from', type=str, default=None, help='Load pretrained model weights for fine-tuning (weights only, optimizer/scheduler reset). Use --resume_from for full resume.')
    parser.add_argument('--freeze_backbone', action='store_true', default=False, help='Freeze backbone layers (conv1-4) during fine-tuning. Only classification head (fc) will be trained.')
    parser.add_argument('--backbone_lr', type=float, default=None, help='Learning rate for backbone layers (conv1-4). If not set, uses --learning_rate.')
    parser.add_argument('--head_lr', type=float, default=None, help='Learning rate for classification head (fc). If not set, uses --learning_rate.')
    
    # Step0: Macro definition (support=0 handling)
    parser.add_argument(
        '--macro_mode',
        type=str,
        default='support_only',
        choices=['support_only', 'sklearn_like', 'all'],
        help='Macro-F1 definition. support_only=average over classes with y_true support>0 (recommended). '
             'sklearn_like=union(y_true,y_pred) labels. all=all 19 classes incl support=0 as 0 (debug only).'
    )
    
    # Step1: Group overlap / group-purged eval (evaluation only; train split unchanged)
    parser.add_argument('--group_key', type=str, default='LBel', help='Group key for leakage checks (e.g., L, B, el, LB, LBel). Default: LBel')
    parser.add_argument('--enforce_disjoint_groups', dest='enforce_disjoint_groups', action='store_true', default=True, help='Enforce group-disjoint train/val/test split by group_key (default: enabled).')
    parser.add_argument('--no_enforce_disjoint_groups', dest='enforce_disjoint_groups', action='store_false', help='Disable group-disjoint split (legacy random slice split).')
    parser.add_argument('--group_purge_eval', action='store_true', default=False, help='Purge val/test samples whose group appears in train (evaluation-only leakage check).')
    parser.add_argument('--group_purge_target', type=str, default='valtest', choices=['val', 'test', 'valtest'], help='Which eval split(s) to purge when --group_purge_eval is enabled.')

    # Step2: True two-stage training/inference (detection + defect-type classification)
    parser.add_argument('--no_two_stage', dest='two_stage', action='store_false', default=True, help='Disable true two-stage loss/inference (legacy single-stage 19-class loss). Default: two-stage enabled.')
    parser.add_argument('--det_threshold', type=float, default=0.90, help='Detection gating threshold on p(defect). Higher -> fewer FP (default: 0.90).')
    parser.add_argument('--no_det_gating', action='store_true', default=False, help='Disable detection gating (do not force class 0 based on p(defect)).')
    parser.add_argument('--det_loss_weight', type=float, default=1.0, help='Weight for detection loss term (default: 1.0).')
    parser.add_argument('--cls_loss_weight', type=float, default=1.0, help='Weight for classification loss term (default: 1.0).')
    parser.add_argument('--det_gamma', type=float, default=None, help='Focal gamma for detection head (default: use --gamma).')
    parser.add_argument('--cls_gamma', type=float, default=None, help='Focal gamma for classification head (default: use --gamma).')
    parser.add_argument('--det_pos_weight_cap', type=float, default=20.0, help='Cap for detection positive-class weight (default: 20.0).')
    
    # Step2: Checkpoint policy aligned to macro_f1
    parser.add_argument('--save_on_best', dest='save_on_best', action='store_true', default=True, help='Save BEST checkpoint immediately when macro_f1 improves (default: enabled).')
    parser.add_argument('--no_save_on_best', dest='save_on_best', action='store_false', help='Disable save-on-best behavior.')
    parser.add_argument('--save_every', type=int, default=100, help='Save Latest checkpoint every N epochs (default: 100). Set 0 to disable.')
    
    # Step3: Best-update diagnostics (best更新時だけ)
    parser.add_argument('--best_report', dest='best_report', action='store_true', default=True, help='Print per-class/confusion summary when macro_f1 improves (default: enabled).')
    parser.add_argument('--no_best_report', dest='best_report', action='store_false', help='Disable best-update diagnostics printing.')
    parser.add_argument('--best_report_topk', type=int, default=5, help='How many worst classes (by F1, support>0) to summarize on best update (default: 5).')
    parser.add_argument('--best_report_confusion_topk', type=int, default=3, help='Top-K absorb targets to show per worst class (default: 3).')
    
    # Step4: File-level threshold optimization from CSV signals
    parser.add_argument('--optimize_file_thresholds', action='store_true', default=False, help='Run file-level threshold optimization from file_statistics CSV signals (default: False). Saves threshold_optimization_*.csv')

    # Benchmark: external split manifest
    parser.add_argument('--split_manifest', type=str, default=None,
                        help='JSON file with pre-computed train/val/test file lists (for benchmark runner). '
                             'When provided, skips internal splitting and uses the manifest lists.')

    args = parser.parse_args()

    main(args)
