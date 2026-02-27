# FEM 多物性・境界ぼかし 前処理 API 仕様

> M2-2: FEM ソルバーが要素ごとに有効物性を渡す際のインターフェース契約。
> 本リポジトリの `tools/micro_defect_preprocess.py` は、FEM が未対応の場合の代替（合成改変）として使用可能。

---

## 1. 入力仕様（FEM → 前処理）

### 1.1 欠陥パラメータ（defect_params_schema.yaml 準拠）

| パラメータ | 型 | 範囲 | 説明 |
|------------|-----|------|------|
| `phi` | float | [0, 0.3] | 空隙率。Halpin-Tsai で E_eff に変換 |
| `E_ratio` | float | [0.1, 1.0] | E_defect / E_base |
| `delamination_ratio` | float | [0, 1] | 層間せん断剛性の低減率 |
| `blur_sigma` | float | [0, 5] | 境界のぼかし幅（要素数単位） |
| `aspect_ratio` | float | [0.5, 2.0] | 楕円アスペクト比 |

### 1.2 要素ごとの有効物性（FEM が受け取る形式）

```python
# 要素 i に対する有効物性
element_properties[i] = {
    "E_eff": float,      # 有効弾性係数
    "G_eff": float,      # 有効せん断係数（脱層時）
    "weight": float,     # 境界ぼかし用 [0,1]。1=欠陥中心、0=健全
}
```

- **E_eff**: Halpin-Tsai(φ) または E_ratio から計算
- **weight**: 欠陥中心からの距離に基づくガウシアン重み。境界で 0 に近づく

---

## 2. 出力仕様（前処理 → GNN）

- ノード値: `[N]` float32。DSPSS 差分（z-score 前）
- 既存パイプライン: `subtract_hole_no_defect` → `normalize_all_subtracted_zscore` で z-score 化
- 出力形式: `.npy` 配列、形状 `(13942,)` または `(N,)`

---

## 3. FEM 側が実装するインターフェース（推奨）

```python
def set_defect_element_properties(
    mesh,
    defect_center: Tuple[float, float, float],
    defect_params: Dict[str, float],
    defect_region: Optional[ArrayLike] = None,
) -> None:
    """
    欠陥領域の要素に有効物性を付与。

    Args:
        mesh: FEM メッシュオブジェクト
        defect_center: 欠陥中心 (x, y, z)
        defect_params: defect_params_schema に準拠
        defect_region: 要素インデックス配列。None の場合は形状から推定
    """
    pass
```

---

## 4. 本リポジトリでの代替（FEM 未対応時）

`tools/micro_defect_preprocess.py` を使用:

- 既存の DSPSS 差分 .npy とラベルから欠陥マスクを取得
- `apply_defect_params()` でスケール・ぼかしを適用
- 合成データとして M2-3 のパラメータサンプリングに利用

```bash
python tools/micro_defect_preprocess.py \
  --input GNN_hole_2026/all_sub_hole_defect_zscore_noise/train \
  --labels GNN_hole_2026/all_19class_label/train \
  --output GNN_hole_2026/all_micro_defect_zscore/train \
  --phi 0.15 --E_ratio 0.4 --blur_sigma 1.0 \
  --max_files 100
```
