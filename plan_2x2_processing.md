# 2x2フォルダ処理計画

## 処理の流れ（4x4と同様）

### ステップ1: 引き算処理
1. **入力データ**: `/home/nishioka/GNN/GNN_hole_2026/Defect_hole_2x2_Region1_21_npy/`
   - 正規化前の元データ（2,189ファイル）
   
2. **基準データ**: `/home/nishioka/GNN/GNN_hole_2026/DSPSS_8x8/hole_no_defect_original.npy`
   - 各ファイルからこのデータを引き算

3. **処理内容**:
   - 各ファイルのデータから `hole_no_defect_original.npy` を引き算
   - 形状チェック（不一致の場合は警告）
   - 引き算結果を保存

4. **出力先**: `/home/nishioka/GNN/GNN_hole_2026/Defect_hole_2x2_Region1_21_npy_subtracted/`
   - 引き算後の全.npyファイル
   - `subtraction_statistics.csv`（統計情報）
   - `subtracted_data_analysis.png`（可視化グラフ）

### ステップ2: 引き算後の正規化処理
1. **入力データ**: ステップ1で作成した引き算後のフォルダ
   - 全ファイルのデータを結合

2. **処理内容**:
   - 全データから最小値・最大値を計算（統一パラメータ）
   - Min-Max正規化: `(x - min) / (max - min)` → [0, 1]範囲
   - 全ファイルを統一パラメータで正規化

3. **出力先**: `/home/nishioka/GNN/GNN_hole_2026/Defect_hole_2x2_Region1_21_npy_subtracted_normalized/`
   - 正規化後の全.npyファイル
   - `normalization_statistics.csv`（統計情報）
   - `normalized_data_analysis.png`（可視化グラフ）

## 処理フロー図

```
正規化前データ (Defect_hole_2x2_Region1_21_npy)
    ↓
hole_no_defect_original.npy を引き算
    ↓
引き算結果を保存 (Defect_hole_2x2_Region1_21_npy_subtracted)
    ↓
全データを結合して統計計算
    ↓
統一パラメータで正規化
    ↓
正規化結果を保存 (Defect_hole_2x2_Region1_21_npy_subtracted_normalized)
```

## 4x4との違い

- **ファイル数**: 2x2は約2,189ファイル、4x4は約3,942ファイル
- **処理時間**: 2x2の方が少ないファイル数のため、処理時間は短い
- **出力フォルダ名**: `2x2` と `4x4` の違いのみ

## 実行スクリプト

4x4用のスクリプト（`subtract_and_normalize_4x4.py`）をベースに、
2x2用のスクリプト（`subtract_and_normalize_2x2.py`）を作成します。

## 確認事項

- [ ] 元データフォルダの存在確認
- [ ] 基準データファイルの存在確認
- [ ] 出力フォルダの作成
- [ ] 形状の一致確認
- [ ] 処理後の統計確認
