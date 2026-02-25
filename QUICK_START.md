# クイックスタートガイド

## セットアップ

### 1. 依存関係のインストール

```bash
# 既存のconda環境を使用する場合
conda activate gnn_final_env

# 必要なPythonパッケージをインストール（オプション）
pip install -r requirements.txt
```

### 2. 基本的なトレーニング実行

```bash
# デフォルト設定でトレーニング
bash run_train_recommended.sh
```

### 3. ハイパーパラメータのカスタマイズ

```bash
# 学習率とエポック数を変更
LR=0.001 EPOCHS=1000 bash run_train_recommended.sh
```

### 4. 学習率スイープ

```bash
# デフォルトの学習率でスイープ
bash run_sweep_lr.sh

# カスタム学習率でスイープ
LRS="0.001 0.002 0.005" bash run_sweep_lr.sh

# 自動リトライ付きスイープ
MAX_RETRIES=3 bash run_sweep_lr.sh
```

## 結果の分析

### スイープ結果の分析

```bash
python tools/analyze_sweeps.py --sweep_dir runs/_sweeps --output_dir reports
```

### 学習曲線の可視化

```bash
python tools/visualize_training.py --run_dir runs/20250115_abc123
```

### 実行結果の比較

```bash
python tools/compare_runs.py --run_dirs runs/run1 runs/run2 runs/run3
```

## トラブルシューティング

### 依存関係エラー

```bash
# pandas, matplotlib, seabornがインストールされていない場合
pip install pandas matplotlib seaborn scikit-learn pyyaml
```

### パスエラー

```bash
# スクリプトが実行できない場合
chmod +x run_sweep_lr.sh run_train_recommended.sh
chmod +x tools/*.py
```

## 次のステップ

詳細なドキュメントは `README.md` を参照してください。
