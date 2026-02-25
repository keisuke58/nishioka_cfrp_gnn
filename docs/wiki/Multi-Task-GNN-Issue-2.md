# Multi-Task GNN — 位置 + 欠陥サイズ分類 (Issue #2)

> **PR**: [#7](https://github.com/keisuke58/nishioka_cfrp_gnn/pull/7) — merged 2026-02-25

## Overview

既存の **19クラス位置分類（ノード単位）** に加えて、**3クラス欠陥サイズ分類（グラフ単位）** ヘッドを追加した。
共有 GAT バックボーンから 2 つのタスクヘッドに分岐する multi-task 学習アーキテクチャ。

### サイズクラス定義

| Class | パターン | 意味 |
|-------|----------|------|
| 0 | `H2_W2` | Small |
| 1 | `H4_W4` | Medium |
| 2 | `H8_W8` | Large |
| -1 | `NoiseDefectFree` 等 | 除外（size loss 不参加） |

## Architecture

```
Input (x, edge_index)
       │
  ┌────┴────┐
  │  Shared │
  │ GAT x4  │  GATConv → ReLU → Dropout (×4 layers)
  └────┬────┘
       │ shared features [N, hidden*heads]
       │
  ┌────┴──────────────────┐
  │                       │
  ▼                       ▼
head_location          global_mean_pool
Linear → [N, 19]         │
(node-level)              ▼
                     head_size
                     Linear → [G, 3]
                     (graph-level)
```

### Key Design Decisions

1. **共有バックボーン**: 既存 `GATModel` と同一構成の 4 層 GAT。パラメータ共有によりサイズヘッドが位置特徴の学習を補助する。
2. **Graph-level pooling**: `global_mean_pool` でノード特徴をグラフ単位に集約。
3. **後方互換**: `forward(data, multitask=False)` で既存と同じ単一テンソル出力。

## Changed Files

| File | Change |
|------|--------|
| `gnn_common/models.py` | `GATMultiTaskModel` 追加、`global_mean_pool` import |
| `gnn_common/losses.py` | `MultiTaskLoss` 追加（location + weighted size loss） |
| `gnn_common/data_utils.py` | `extract_size_class()` 追加、`prepare_data()` で `data.size_class` 付与 |
| `gnn_common/config.py` | `ModelConfig` に `multitask`, `size_weight` フィールド追加 |

## API Usage

### Model

```python
from gnn_common.models import GATMultiTaskModel

model = GATMultiTaskModel(
    hidden_channels=64,
    num_classes=19,        # location classes
    num_size_classes=3,    # size classes (small/medium/large)
    num_heads=4,
    dropout=0.2,
)

# Multi-task output
outputs = model(batch_data, multitask=True)
# outputs["location"].shape → [N, 19]  (node-level)
# outputs["size"].shape     → [G, 3]   (graph-level)

# Backward-compatible single-task output
logits = model(batch_data, multitask=False)
# logits.shape → [N, 19]
```

### Loss

```python
from gnn_common.losses import MultiTaskLoss, FocalLossLogSoftmax

loss_fn = MultiTaskLoss(
    location_loss_fn=FocalLossLogSoftmax(gamma=3.0),
    size_weight=0.5,  # size loss の重み
)

total_loss, details = loss_fn(outputs, location_target, size_labels)
# details = {"location_loss": ..., "size_loss": ..., "total_loss": ...}
```

### Size Label Extraction

```python
from gnn_common.data_utils import extract_size_class

extract_size_class("L1_B1_el1_H2_W2.npy")           # → 0 (small)
extract_size_class("L2_B3_el5_H4_W4.npy")           # → 1 (medium)
extract_size_class("L1_B1_el1_H8_W8.npy")           # → 2 (large)
extract_size_class("NoiseDefectFree_something.npy")  # → -1 (excluded)
```

`prepare_data()` は自動的に各 `Data` オブジェクトに `data.size_class` を付与する。

### Config

```python
from gnn_common.config import ModelConfig

cfg = ModelConfig(
    model_type="GATMultiTask",
    multitask=True,
    size_weight=0.5,
)
```

## Verification Results

All smoke tests passed:

### extract_size_class

| Input | Expected | Result |
|-------|----------|--------|
| `L1_B1_el1_H2_W2.npy` | 0 | 0 PASS |
| `L2_B3_el5_H4_W4.npy` | 1 | 1 PASS |
| `L1_B1_el1_H8_W8.npy` | 2 | 2 PASS |
| `NoiseDefectFree_something.npy` | -1 | -1 PASS |
| `L1_B1_el1_H5_W6.npy` | -1 | -1 PASS |
| `random_name.npy` | -1 | -1 PASS |

### GATMultiTaskModel Forward Pass

| Test | Expected Shape | Actual Shape | Result |
|------|---------------|--------------|--------|
| Single graph, `multitask=False` | `[100, 19]` | `[100, 19]` | PASS |
| Single graph, location | `[100, 19]` | `[100, 19]` | PASS |
| Single graph, size | `[1, 3]` | `[1, 3]` | PASS |
| Batch (2 graphs), location | `[110, 19]` | `[110, 19]` | PASS |
| Batch (2 graphs), size | `[2, 3]` | `[2, 3]` | PASS |

### MultiTaskLoss

| Test | Result |
|------|--------|
| Normal (2/3 graphs valid) | `total=4.2070, loc=2.9733, size=2.4675` PASS |
| All excluded (`size_labels=[-1,-1,-1]`) | `size_loss=0.0` PASS |

## Next Steps

- [ ] Training script の multi-task 対応（training loop で size_labels を batch から取得）
- [ ] Multi-task 学習の実験実行・精度比較
- [ ] Issue #2 の残タスク: severity head, type classification head の追加検討
