# 実行順序ガイド

## ⚠️ 重要な注意

**現在、`run_ablation_experiment.py`はプレースホルダーです。**
実際の実験を実行するには、既存のトレーニングスクリプトに統合する必要があります。

---

## 📋 正しい実行順序

### Step 1: 動作確認 ✅（完了）

```bash
python tools/test_new_features.py
```

**結果**: 
- ✅ Localization指標: 成功
- ✅ Cross-edge: 成功  
- ⚠️ OOD分割: 実際のデータファイルが必要（スキップ）

---

### Step 2: 既存スクリプトへの統合（必須）

**現在の状況**: 
- 新機能は実装済み
- 既存のトレーニングスクリプトへの統合が必要

**統合方法**:
1. `GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free.py`を確認
2. `INTEGRATION_GUIDE.md`を参照して統合
3. 統合後、通常のトレーニングを実行して動作確認

**統合が必要な箇所**:
- データ分割部分 → `apply_ood_split()`を使用
- データ準備部分 → `prepare_data_with_cross_edges()`を使用  
- 評価ループ → `evaluate_with_localization_metrics()`を使用

---

### Step 3: アブレーション実験の実行

**統合完了後**に実行：

```bash
# 個別実験（統合後）
python tools/run_ablation_experiment.py --experiment baseline --split_type iid

# スイープ実行（統合後）
bash tools/run_ablation_sweep.sh
```

**注意**: 現在はプレースホルダーのため、実際の実験は実行されません。

---

## 🔧 今すぐできること

### オプション1: 統合を先に進める（推奨）

既存のトレーニングスクリプトを確認して統合：

```bash
# 既存スクリプトを確認
cat GNN_hole_2026/GNN_program/GNN_zscore_sub_noise_defect_free.py | grep -A 10 "train_test_split\|prepare_data\|evaluate"
```

その後、`INTEGRATION_GUIDE.md`を参照して統合。

### オプション2: 簡易的な実験スクリプトを作成

実際のトレーニングを実行する簡易版を作成することも可能です。

---

## ❓ 質問

1. **既存のトレーニングスクリプトを確認しますか？**
   - 統合箇所を特定できます

2. **簡易的な実験スクリプトを作成しますか？**
   - すぐに実験を実行できます

3. **統合を手動で進めますか？**
   - `INTEGRATION_GUIDE.md`を参照してください

---

## 📝 まとめ

**現時点で実行すべきこと**:

1. ✅ `python tools/test_new_features.py` - 完了
2. ⏳ 既存スクリプトへの統合 - **次のステップ**
3. ⏳ アブレーション実験 - 統合後

**`run_ablation_experiment.py`をそのまま実行しても、実際の実験は行われません。**
まず統合を完了してください。
