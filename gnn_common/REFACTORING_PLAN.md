# 追加リファクタリング計画

## 実施済み

### 1. 共通モジュールの作成
- ✅ `models.py`: GCN/GATモデル定義
- ✅ `data_utils.py`: データ準備・クラス重み計算（元ファイルの実装を基に）
- ✅ `training_utils.py`: シード設定、DDP初期化（元ファイルの実装を基に）
- ✅ `metrics.py`: 評価メトリクス計算（`metrics_from_confusion_matrix`追加）
- ✅ `config.py`: 設定管理
- ✅ `losses.py`: 損失関数（LayerAwareLoss, LogitAdjustLoss, FocalLossLogSoftmax）

## 推奨される追加リファクタリング

### 2. Samplerクラスの分離（推奨）
`GNN_zscore_sub_noise_defect_free.py`には以下のSamplerクラスがあります：
- `ClassBalancedSampler`
- `DistributedClassBalancedSampler`
- `DistributedMinorityRatioSampler`
- `DistributedClassFrequencySampler`

これらを `gnn_common/samplers.py` に分離することで：
- コードの可読性向上
- 再利用性向上
- テスト容易性向上

### 3. ロギング機能の分離（推奨）
`GNN_zscore_sub_noise_defect_free.py`には以下のロギング機能があります：
- `_infer_rank()`: ランク推論
- `_is_rank0()`: ランク0判定
- `_get_logger()`: ロガー取得
- `print()`: rank0のみのprint

これらを `gnn_common/logging_utils.py` に分離することで：
- DDP安全なロギングを全プロジェクトで統一
- コードの重複削減

### 4. 可視化機能の分離（オプション）
- `get_defect_cmap_with_white_zero()`: カラーマップ取得
- `evaluate_and_visualize()`: 評価と可視化（既に`metrics.py`に一部あり）

これらを `gnn_common/visualization.py` に分離可能

## 実施方法

### オプション1: 段階的リファクタリング（推奨）
1. まず`losses.py`を使用（既に作成済み）
2. 次に`samplers.py`を作成
3. 最後に`logging_utils.py`を作成

### オプション2: 一括リファクタリング
すべてのモジュールを一度に作成し、`GNN_zscore_sub_noise_defect_free.py`からインポートに置き換え

## 注意事項

- `GNN_zscore_sub_noise_defect_free.py`は4605行の大きなファイル
- 段階的な移行を推奨（一度にすべてを変更しない）
- 既存の動作を維持しながらリファクタリング
- テストを追加して動作確認

## 現在の状態

- ✅ 基本的な共通モジュールは完成
- ✅ 元ファイルの実装を基に更新済み
- ⏳ 追加のリファクタリングは任意（必要に応じて実施）