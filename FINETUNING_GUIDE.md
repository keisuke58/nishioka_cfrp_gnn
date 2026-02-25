# ファインチューニング改善ガイド

## 📊 現在の問題

- **事前学習モデル**: Macro F1 = **0.730**
- **ファインチューニング後**: Macro F1 = **0.7249** ❌ (性能が下がっている)
- **主な原因**: 学習率が低すぎる (0.0001 vs 元の0.001)

## 🚀 改善されたスクリプトの使用方法

### 推奨: 戦略1 - バックボーン凍結（最も安全で効果的）

```bash
# バックボーンを凍結して分類ヘッドのみ学習
STRATEGY=freeze_backbone bash run_finetune_improved.sh
```

**設定**:
- バックボーン: 凍結
- 分類ヘッド: LR = 0.001
- Epochs: 500
- Patience: 150

### 戦略2 - 層別学習率

```bash
# バックボーン: 低LR, 分類ヘッド: 高LR
STRATEGY=differential_lr bash run_finetune_improved.sh
```

**設定**:
- バックボーン: LR = 0.0001
- 分類ヘッド: LR = 0.001
- Epochs: 1000
- Patience: 200

### 戦略3 - より高い学習率

```bash
# 全層をより高い学習率で更新
STRATEGY=higher_lr bash run_finetune_improved.sh
```

**設定**:
- 全層: LR = 0.0005 (デフォルトの5倍)
- Epochs: 1000
- Patience: 200

## 📝 カスタマイズ例

### 学習率を調整

```bash
# 戦略1で分類ヘッドの学習率を変更
STRATEGY=freeze_backbone HEAD_LR=0.002 bash run_finetune_improved.sh
```

### Patienceを調整

```bash
# より長く待つ
STRATEGY=freeze_backbone PATIENCE=200 bash run_finetune_improved.sh
```

### OneCycleLRを無効化（CosineAnnealingLRを使用）

```bash
# スクリプト内で USE_ONECYCLE=0 に設定するか、直接修正
STRATEGY=freeze_backbone USE_ONECYCLE=0 bash run_finetune_improved.sh
```

## 🔍 結果の確認

### ログの確認

```bash
# 最新のログを確認
tail -f /home/nishioka/GNN/runs/_logs/finetune_improved_*.log | tail -1

# 実行中のログを確認
ls -lth /home/nishioka/GNN/runs/*/logs/train.log | head -1 | awk '{print $NF}' | xargs tail -f
```

### 性能の確認

```bash
# 最新の実行結果を確認
ls -ltd /home/nishioka/GNN/runs/*/meta/summary.json | head -1 | xargs cat | python3 -m json.tool | grep best_macro_f1
```

## 📈 期待される改善

- **目標**: Macro F1 = 0.730 → **0.735以上** (+0.005以上)
- **テストセット**: Macro F1 = 0.6578 → **0.670以上** (+0.012以上)

## 🎯 実行順序の推奨

1. **まず戦略1（バックボーン凍結）を実行**
   ```bash
   STRATEGY=freeze_backbone bash run_finetune_improved.sh
   ```

2. **結果が良ければ終了、改善が不十分なら戦略2を試す**
   ```bash
   STRATEGY=differential_lr bash run_finetune_improved.sh
   ```

3. **さらに改善が必要なら戦略3を試す**
   ```bash
   STRATEGY=higher_lr bash run_finetune_improved.sh
   ```

## ⚠️ 注意事項

- 各戦略は独立して実行してください
- 実行前にGPUの空きを確認してください
- ログファイルを定期的に確認して、エラーがないかチェックしてください

## 📚 詳細な分析

詳細な分析は `FINETUNING_ANALYSIS.md` を参照してください。
