# IR Thermography × FEM-GNN Fusion 設計書 (Issue #3)

> Phase: 設計 (Design-only) — 実装は実 IR データ入手後

## 1. 概要

FEM-DSPSS で学習した GNN に赤外線サーモグラフィ (IR) の表面温度特徴を融合し、
シミュレーション↔実験のドメインギャップを埋めるマルチモーダル欠陥定位を実現する。

```
IR Camera (surface)          FEM Simulation (3D internal)
   │                              │
   ▼                              ▼
 Temporal Feature Extraction   DSPSS values
   │                              │
   ▼                              ▼
 Image→Mesh Projection        Node coordinates (x,y,z)
   │                              │
   └──────────┬───────────────────┘
              ▼
      Feature Concatenation
      [x, y, z, DSPSS, IR_feat1, IR_feat2, ...]
              │
              ▼
         GNN (GAT/GCN)
              │
              ▼
      Defect Localization (19-class)
```

## 2. 前処理パイプライン設計

### 2.1 IR データ形式 (想定)
- Flash thermography: `[T, H, W]` (T=時間フレーム数, H×W=画素)
- 典型的: T≈100-500, H×W≈320×256 or 640×512
- 14-16 bit → float32

### 2.2 時系列特徴抽出
各画素 (h, w) に対して時系列 T(t) から以下を計算:

| 特徴 | 計算 | 意味 |
|------|------|------|
| `max_contrast` | max(T) − T(0) | 最大温度上昇 |
| `slope` | d/dt[log(T−T∞)] の傾き | 熱拡散速度 (欠陥で変化) |
| `peak_time` | argmax(T) | ピーク到達時間 |

→ 各画素につき 3 次元特徴ベクトル

### 2.3 座標アライメント (IR pixel ↔ FEM node)
```
FEM node (x,y) ──[Affine transform]──→ IR pixel (u,v)

Calibration:
  - 試験片4隅の対応点 → 2D Affine (6 param)
  - または fiducial markers
```

### 2.4 画像→メッシュ投影
```python
# 擬似コード
for node_idx, (x, y, z) in enumerate(fem_coords):
    u, v = affine_transform(x, y)  # FEM → IR pixel
    ir_feat = bilinear_interpolate(ir_feature_map, u, v)  # [3]
    # Interior ノード (z が内部層) → ir_feat = 0 or mean
```

**重要**: IR は表面のみ観測 → interior ノードには直接特徴なし
→ 既存の cross-edge (Yehia方式) が surface→interior 情報伝播を担う

## 3. データ API 設計

### 3.1 設定 (config追加案)
```python
@dataclass
class IRConfig:
    ir_data_dir: Optional[str] = None      # IR データフォルダ
    ir_feature_dim: int = 3                 # 特徴次元数
    ir_normalize: bool = True               # Z-score 正規化
    interpolation: str = "bilinear"         # 補間手法
    fusion_mode: str = "concatenate"        # "concatenate" | "add"
    calibration_file: Optional[str] = None  # アフィン変換パラメータ
```

### 3.2 data_utils.py 変更点

現在の特徴構築 (L162):
```python
node_features = np.vstack((x_coords, y_coords, z_coords, values)).T  # [N, 4]
```

IR 融合後:
```python
# ir_features: [N, 3] (surface以外は0埋め)
node_features = np.vstack((x_coords, y_coords, z_coords, values, *ir_features.T)).T  # [N, 7]
```

### 3.3 追加関数 (擬似コード)

```python
def load_ir_features(ir_path, calibration, fem_coords, surface_mask):
    """IR画像を読み込み、FEMノードに投影"""
    ir_raw = load_ir_sequence(ir_path)           # [T, H, W]
    ir_feat_map = extract_temporal_features(ir_raw)  # [3, H, W]

    ir_node_features = np.zeros((len(fem_coords), 3))
    for i, (x, y) in enumerate(fem_coords[:, :2]):
        if surface_mask[i]:
            u, v = apply_affine(calibration, x, y)
            ir_node_features[i] = bilinear_interp(ir_feat_map, u, v)

    if normalize:
        ir_node_features = zscore(ir_node_features, axis=0)
    return ir_node_features
```

## 4. モデル変更 (最小限)

モデルは入力次元をパラメータで受けるため、変更不要:
```python
# 現在: GCNModel(hidden_channels=128) → conv1 = GCNConv(4, hidden_channels)
# IR融合: GCNModel(hidden_channels=128, input_channels=7) → conv1 = GCNConv(7, hidden_channels)
```

**ただし**: 現在のモデルは `input_channels=4` がハードコードされている。
→ `input_channels` を引数化する修正が必要 (小規模)

## 5. 既存インフラの活用

| 機能 | 場所 | 状態 |
|------|------|------|
| Surface ノード識別 | `data_utils.py:917` `identify_surface_nodes()` | 実装済み |
| Cross-edge 作成 | `data_utils.py:952` `create_cross_edges()` | 実装済み |
| Cross-edge 付与 | `data_utils.py:1001` `add_cross_edges_to_data()` | 実装済み |
| OOD split | `data_utils.py:496` | 実装済み |

→ surface→interior 情報伝播の機構は既にある。IR 特徴を surface ノードに載せれば
  cross-edge 経由で interior にも伝わる。

## 6. 検証戦略

### Phase 1: 合成 IR データ (実装不要で設計のみ)
- FEM の温度場から擬似 IR 画像を生成 (ガウスノイズ付加)
- Sanity check: IR+GNN vs GNN-only で macro-F1 比較
- 期待: 独立な信号源があれば +5-10%

### Phase 2: 実 IR データ (データ入手後)
- 実試験片の flash thermography データ
- キャリブレーション・投影パイプライン検証
- 本格的な比較実験

## 7. 主要文献

| 文献 | 手法 | ポイント |
|------|------|---------|
| Yehia et al. 2025 (Eng Struct) | FE+GNN+cross-edge | Surface↔Interior routing |
| Fang et al. 2021 | Pulsed thermo + DL | IR前処理・セグメンテーション |
| Ishikawa et al. 2012 | CFRP + IRT | 熱異方性の影響 |
| Keo et al. 2015 | CO₂ laser + lock-in IRT | SNR/感度比較 |
| Yang et al. 2013 | Ultrasonic × IR hybrid | マルチモーダル励起 |

## 8. 実装ロードマップ

```
[現在] 設計完了 (このドキュメント)
  │
  ├─ 実 IR データ入手
  │    │
  │    ▼
  │  Phase 1: data_utils.py に IR ローダー追加
  │    │      models.py の input_channels 引数化
  │    │      キャリブレーション実装
  │    ▼
  │  Phase 2: 合成データで sanity check
  │    │
  │    ▼
  │  Phase 3: 実データで比較実験
  │    │
  │    ▼
  └─ Issue #3 クローズ
```
