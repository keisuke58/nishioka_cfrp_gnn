# 次のステップ - 実装完了サマリー

## ✅ 実装完了した内容

### 1. 新機能の実装
- ✅ OOD分割機能（`gnn_common/data_utils.py`）
- ✅ Localization指標（`gnn_common/metrics.py`）
- ✅ Cross-edge機能（`gnn_common/data_utils.py`, `gnn_common/models.py`）

### 2. 統合ユーティリティ
- ✅ `gnn_common/integration_utils.py`: 既存コードへの統合用ヘルパー関数

### 3. テスト・実験スクリプト
- ✅ `tools/test_new_features.py`: 動作確認用テストスクリプト
- ✅ `tools/run_ablation_experiment.py`: アブレーション実験スクリプト
- ✅ `tools/run_ablation_sweep.sh`: アブレーション実験のスイープ

### 4. ドキュメント
- ✅ `IMPLEMENTATION_GUIDE.md`: 実装ガイド
- ✅ `USAGE_EXAMPLES.md`: 使用例
- ✅ `INTEGRATION_GUIDE.md`: 統合ガイド
- ✅ `IMPLEMENTATION_SUMMARY.md`: 実装サマリー

---

## 🚀 次のステップ（実行順序）

### Step 1: 動作確認（最優先）

```bash
# テストスクリプトを実行
cd /home/nishioka/GNN
python tools/test_new_features.py
```

**確認事項**:
- すべてのテストがパスするか
- エラーがないか
- 実際のデータで動作するか

**問題がある場合**:
- エラーメッセージを確認
- `INTEGRATION_GUIDE.md`のトラブルシューティングを参照

---

### Step 2: 既存スクリプトへの統合

#### 2.1 OOD分割の統合

既存のトレーニングスクリプト（`GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free.py`）を確認し、データ分割部分を修正：

```python
# 既存コードを探す
# train_test_split や ランダム分割の部分

# 以下に置き換え
from gnn_common.integration_utils import apply_ood_split, load_config_for_splitting

split_config = load_config_for_splitting('config.yaml')
train_pairs, val_pairs, test_pairs = apply_ood_split(
    pairs,
    split_type=split_config['split_type'],
    label_data_folder=label_data_folder,
    **split_config['ood_params']
)
```

#### 2.2 Cross-edgeの統合

データ準備部分を修正：

```python
# 既存の prepare_data 呼び出しを探す

# 以下に置き換え
from gnn_common.integration_utils import (
    prepare_data_with_cross_edges,
    load_config_for_cross_edges
)

cross_edge_config = load_config_for_cross_edges('config.yaml')
data_list, class_weights = prepare_data_with_cross_edges(
    pairs, ...,
    use_cross_edges=cross_edge_config['enabled'],
    cross_edge_k=cross_edge_config['k'],
    cross_edge_method=cross_edge_config['surface_method']
)
```

#### 2.3 Localization指標の統合

評価ループを修正：

```python
# 既存の評価ループを探す

# 以下に置き換え
from gnn_common.integration_utils import evaluate_with_localization_metrics

results = evaluate_with_localization_metrics(
    model, test_loader, device,
    calculate_localization=True
)
```

**詳細**: `INTEGRATION_GUIDE.md`を参照

---

### Step 3: 設定ファイルの準備

`config.yaml`を作成（`config.yaml.example`をコピー）：

```bash
cp config.yaml.example config.yaml
```

必要に応じて設定を調整：

```yaml
# OOD分割の設定
data_splitting:
  split_type: "defect_size"  # または 'iid', 'defect_ratio', 'layer'
  ood_params:
    defect_size:
      size_threshold: 5

# Cross-edgeの設定
cross_edge:
  enabled: false  # 最初はfalseでテスト
  k: 1
  surface_method: "outer_layer"
```

---

### Step 4: アブレーション実験の実行

#### 4.1 個別実験

```bash
# Baseline（Cross-edge OFF）
python tools/run_ablation_experiment.py \
    --experiment baseline \
    --split_type iid \
    --output_dir ./ablation_results

# Cross-edge ON
python tools/run_ablation_experiment.py \
    --experiment cross_edge_on \
    --split_type iid \
    --cross_edges \
    --cross_edge_k 1 \
    --output_dir ./ablation_results
```

#### 4.2 スイープ実行

```bash
bash tools/run_ablation_sweep.sh
```

#### 4.3 結果の比較

実験結果を比較して、最適な設定を決定：

- Cross-edge ON/OFFの性能差
- 異なるOOD分割方法の性能差
- Cross-edge k値の影響

---

## 📋 チェックリスト

### 実装確認
- [ ] `python tools/test_new_features.py` がすべてパスする
- [ ] エラーがない
- [ ] 実際のデータで動作する

### 統合確認
- [ ] OOD分割が既存スクリプトに統合されている
- [ ] Cross-edgeがデータローダーに統合されている
- [ ] Localization指標が評価ループに統合されている
- [ ] 設定ファイルが正しく読み込まれている

### 実験確認
- [ ] Baseline実験が実行できる
- [ ] Cross-edge実験が実行できる
- [ ] OOD分割実験が実行できる
- [ ] 結果が正しく保存されている

---

## 🔍 トラブルシューティング

### テストが失敗する場合

1. **依存関係の確認**
   ```bash
   pip install scikit-learn scipy matplotlib seaborn
   ```

2. **パスの確認**
   - プロジェクトルートが正しく設定されているか
   - データフォルダのパスが正しいか

3. **エラーメッセージの確認**
   - エラーメッセージを読む
   - `INTEGRATION_GUIDE.md`のトラブルシューティングを参照

### 統合がうまくいかない場合

1. **既存コードの確認**
   - データ分割がどこで行われているか
   - データ準備がどこで行われているか
   - 評価ループがどこにあるか

2. **段階的な統合**
   - まずOOD分割だけ統合
   - 動作確認後、Cross-edgeを統合
   - 最後にLocalization指標を統合

3. **設定ファイルの確認**
   - `config.yaml`が正しく作成されているか
   - 設定値が正しいか

---

## 📚 参考ドキュメント

- `IMPLEMENTATION_GUIDE.md`: 実装の詳細
- `USAGE_EXAMPLES.md`: 使用例
- `INTEGRATION_GUIDE.md`: 統合方法
- `NEXT_PLAN.md`: 研究計画

---

## 🎯 成功基準

1. **動作確認**: すべてのテストがパスする
2. **統合**: 既存スクリプトに統合され、正常に動作する
3. **実験**: アブレーション実験が実行でき、結果が得られる
4. **比較**: 異なる設定の性能を比較できる

---

## 💡 ヒント

1. **段階的に進める**: 一度にすべてを統合せず、段階的に進める
2. **バックアップ**: 既存コードを変更する前にバックアップを取る
3. **ログを確認**: 詳細なログを有効にして、問題を特定する
4. **小さく始める**: まず小さなデータセットでテストする

---

## 📞 サポート

問題が発生した場合：

1. エラーメッセージを確認
2. 関連ドキュメントを参照
3. テストスクリプトで個別に確認
4. 設定ファイルを確認

---

**次のアクション**: `python tools/test_new_features.py` を実行して動作確認を開始してください！
