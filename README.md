# GNN Training Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.3389%2Ffmats.2025.1652484-blue.svg)](https://doi.org/10.3389/fmats.2025.1652484)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Graph Neural Network (GNN) による CFRP 欠陥位置推定のトレーニングフレームワーク

---

## 目次

- [概要](#概要)
- [研究背景・課題](#研究背景課題)
- [主な機能](#主な機能)
- [ディレクトリ構造](#ディレクトリ構造)
- [クイックスタート](#クイックスタート)
- [セットアップ](#セットアップ)
- [論文との対応](#frontiers-2025-実験との対応)
- [ドキュメント](#ドキュメントマップ)
- [API リファレンス](#api-ドキュメント)
- [トラブルシューティング](#トラブルシューティング)
- [引用](#関連論文リポジトリ)

---

## 概要

このプロジェクトは，**Frontiers in Materials** に掲載された論文  
**["Development of Defect Localization Method for Perforated Carbon-fiber-reinforced Plastic Specimens Using Finite Element Method and Graph Neural Network"](https://doi.org/10.3389/fmats.2025.1652484) (Nishioka et al., 2025)**  
の実験コードをベースにした，**汎用 GNN トレーニングフレームワーク** です。

| 項目 | 内容 |
|------|------|
| **タスク** | CFRP 穴あき試験片の 3D 欠陥位置推定（19クラス：18層＋欠陥なし） |
| **入力** | FEM 由来の DSPSS（主応力和）をノード特徴量としたメッシュグラフ |
| **モデル** | Graph Attention Network (GAT / GATv2), GCN |
| **用途** | 論文再現・他メッシュへの転用・OOD解析・Localization 評価 |

## 研究背景・課題

CFRP（炭素繊維強化プラスチック）の非破壊検査において，**欠陥の層位置を正確に推定する**ことは重要です。本フレームワークは，

1. **FEM シミュレーション**で得た DSPSS をグラフのノード特徴量として利用
2. **GAT** により隣接ノード間の関係を学習
3. **19クラス分類**（層1〜18 + 欠陥なし）で欠陥位置を推定

というパイプラインを実装し，論文実験の再現や，OOD 分割・Top-k 精度・距離誤差などの追加解析を容易にしています。

## 主な機能

- **実験再現**: 論文と同等の前処理・学習パイプライン
- **OOD 分割**: `iid` / `defect_size` / `defect_ratio` / `layer` による Out-of-Distribution 評価
- **Localization 指標**: Top-k Accuracy, 距離誤差, AUPRC
- **Cross-edge**: Surface-to-interior ルーティング（Yehia 方式）
- **分散学習**: マルチ GPU 対応（torchrun）
- **実験管理**: スイープ・可視化・通知（メール/Slack）

### パイプライン概要

```
FEM シミュレーション (DSPSS)
        ↓
前処理（差分・正規化・z-score）
        ↓
メッシュ → グラフ構築（ノード=要素, エッジ=隣接）
        ↓
GAT / GCN で学習
        ↓
19クラス分類（層1〜18 + 欠陥なし）
        ↓
評価（Accuracy, Macro F1, Top-k, 距離誤差）
```

### 評価指標

| 指標 | 説明 |
|------|------|
| Accuracy / Macro F1 | 分類精度 |
| Top-k Accuracy | 正解が上位 k 件に含まれる割合 |
| 距離誤差 | 予測層と正解層の距離（層単位） |
| AUPRC | 欠陥検出の Precision-Recall 曲線下面積 |
| MCC | Matthews 相関係数（不均衡データ向け） |

## ディレクトリ構造

```
GNN/
├── gnn_common/              # 共通モジュール（データ・モデル・評価）
│   ├── config.py           # 設定管理
│   ├── data_utils.py       # データ読み込み・OOD分割・Cross-edge
│   ├── models.py           # GCN / GAT / GATv2 / Cross-edge 対応モデル
│   ├── training_utils.py   # シード・分散学習・学習ループ
│   ├── metrics.py          # 分類・Top-k・距離誤差・AUPRC
│   └── losses.py           # Ordinal Focal Loss 等
├── tools/                  # 実験管理・解析ツール
│   ├── launch_run.py       # トレーニングランチャー
│   ├── analyze_sweeps.py   # 学習率スイープ解析
│   ├── visualize_training.py  # 学習曲線可視化
│   ├── compare_runs.py     # 実行結果比較
│   └── config_loader.py    # YAML 設定読み込み
├── GNN_hole_2026/          # 論文用 CFRP データセット & 前処理・学習
│   ├── Dataset_original/   # FEM 生データ・正規化結果
│   └── GNN_program/        # 論文ベースの学習スクリプト
├── wiki/                   # GitHub Wiki 用ソース（実運用ロードマップ等）
├── run_train_recommended.sh  # 推奨トレーニング
├── run_sweep_lr.sh         # 学習率スイープ
├── config.yaml.example     # 設定例
├── requirements.txt        # Python 依存関係
└── runs/                   # 実行結果（自動生成）
```

## クイックスタート

```bash
# 1. 環境構築
conda activate gnn_final_env
pip install -r requirements.txt

# 2. 推奨設定でトレーニング
bash run_train_recommended.sh

# 3. 学習率スイープ
bash run_sweep_lr.sh

# 4. 結果分析
python tools/analyze_sweeps.py --sweep_dir runs/_sweeps --output_dir reports
python tools/visualize_training.py --run_dir runs/<run_id>
```

詳細は [QUICK_START.md](QUICK_START.md) を参照してください。

---

## Frontiers 2025 実験との対応

論文中の実験と，本リポジトリ内のコードの対応関係は概ね以下の通りです。

- **データセット & 前処理 (`GNN_hole_2026/`)**
  - `Dataset_original/` : FEM 由来の生データと，各種正規化・差分処理の結果
  - `normalize_all_subtracted*.py` : 欠陥なし試験片との差分・正規化処理
  - `subtract_hole_no_defect.py` : 欠陥なし基準からの差分生成
  - `visualize_normalization_comparison.py`, `visualize_normalized_zscore_spatial.py` : 正規化・ z-score の空間的分布の可視化
  - `GNN_program/` : 論文用の GNN 学習スクリプト群（元々の単一スクリプト実装をベースにしたもの）

- **共通フレームワーク (`gnn_common/`)**
  - `data_utils.py` : データ読み込み・前処理・OOD 分割・Cross-edge 生成
  - `models.py` : GCN / GAT / Cross-edge 対応モデル
  - `metrics.py` : 分類指標に加え，Top-k / 距離誤差 / AUPRC などの Localization 指標
  - `training_utils.py` : 乱数シード設定，学習ループ補助，分散学習ヘルパ
  - `losses.py` : 19 クラス用の損失関数（例：Ordinal Focal Loss など）

- **実験管理 & 解析 (`tools/`, `runs/`, `reports/`)**
  - `tools/launch_run.py`, `run_train_recommended.sh` : 推奨設定での学習ランチャ
  - `tools/analyze_sweeps.py` : 学習率スイープ結果の解析
  - `tools/visualize_training.py` : 学習曲線やメトリクスの可視化

詳細は `IMPLEMENTATION_GUIDE.md`, `INTEGRATION_GUIDE.md`, `USAGE_EXAMPLES.md` を参照してください。

## ドキュメントマップ

フレームワークの使い方や研究計画は、以下のドキュメントに整理されています。

- **Wiki** : [実運用ロードマップ](wiki/実運用ロードマップ（ミクロ欠陥×GNN-MultiTask）.md)（ミクロ欠陥×GNN Multi-Task 計画）
- `literature_notes/plan_micro_defect_gnn_multitask.md` : 詳細計画書（タスク一覧・マイルストーン）
- `QUICK_START.md` : 最小ステップでの実行方法
- `USAGE_EXAMPLES.md` : OOD分割・Localization指標・Cross-edge の具体的なコード例
- `IMPLEMENTATION_GUIDE.md` : 実装の詳細設計と背景
- `IMPLEMENTATION_SUMMARY.md` : 実装した機能のサマリー
- `INTEGRATION_GUIDE.md` : 既存スクリプト（例: `GNN_hole_2026` 系）への統合手順
- `INTEGRATION_SUMMARY.md` : 旧コードベースとの対応付けサマリー
- `FINETUNING_GUIDE.md` : 事前学習済みモデルのファインチューニング手順
- `FINETUNING_ANALYSIS.md` / `FINETUNING_RESULTS_COMPARISON.md` : ファインチューニング結果の整理
- `EXECUTION_ORDER.md` : 推奨する実行順序・ワークフローメモ
- `NEXT_PLAN.md` / `NEXT_STEPS_COMPLETE.md` : 今後の研究・実装計画
- `EMAIL_SETUP.md` / `SLACK_SETUP.md` : 通知設定（オプション）

## セットアップ

### 必要な環境

- Python 3.8+
- PyTorch 2.0+（CUDA 対応推奨）
- conda 環境 `gnn_final_env`（または同等）

### 依存関係のインストール

```bash
# conda環境をアクティベート
conda activate gnn_final_env

# 分析・可視化ツール用の依存関係（オプション）
pip install -r requirements.txt
```

必要なパッケージ:
- `pandas`: データ分析
- `matplotlib`, `seaborn`: 可視化
- `scikit-learn`: メトリクス計算
- `pyyaml`: 設定ファイル読み込み

### 環境変数の設定（推奨）

`.env`ファイルを使用して環境変数を管理することを推奨します。

```bash
# .env.exampleをコピー
cp .env.example .env

# .envファイルを編集して値を設定
vim .env
```

詳細は `ENV_SETUP.md` を参照してください。

### 通知の設定（オプション）

トレーニング完了時に通知を送信できます。メールまたはSlack、または両方を設定可能です。

#### Gmail通知の設定

`.env`ファイルに以下を設定：
```bash
GMAIL_FROM="your-email@gmail.com"
GMAIL_TO="recipient@gmail.com"
GMAIL_PASSWORD="your-app-password"  # Gmailアプリパスワード
```

詳細は `EMAIL_SETUP.md` を参照してください。

#### Slack通知の設定

`.env`ファイルに以下を設定：
```bash
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXXXX/YYYYY/ZZZZZ"
```

詳細は `SLACK_SETUP.md` を参照してください。

## 実行手順

### 1. 基本的なトレーニング実行

```bash
bash run_train_recommended.sh
```

### 2. ハイパーパラメータのカスタマイズ

環境変数で設定を上書き:

```bash
LR=0.001 EPOCHS=1000 BATCH_SIZE=32 bash run_train_recommended.sh
```

### 3. 学習率スイープ

```bash
bash run_sweep_lr.sh
```

カスタム学習率でスイープ:

```bash
LRS="0.001 0.002 0.005 0.01" bash run_sweep_lr.sh
```

自動リトライ機能付き:

```bash
MAX_RETRIES=3 RETRY_DELAY=60 bash run_sweep_lr.sh
```

## 設定管理

### 環境変数による設定

主要な環境変数:

- `LR`: 学習率 (default: 0.002)
- `EPOCHS`: エポック数 (default: 2000)
- `BATCH_SIZE`: バッチサイズ (default: 64)
- `HIDDEN`: 隠れ層のチャネル数 (default: 16)
- `OUTPUT_BASE`: 出力ディレクトリ (default: /home/nishioka/GNN/runs)
- `DATASET_TAG`: データセットタグ (default: NDF)
- `MAX_RETRIES`: 自動リトライ回数 (default: 0 = 無効)
- `RETRY_DELAY`: リトライ前の待機時間（秒） (default: 60)

### YAML設定ファイル

`config.yaml.example` を `config.yaml` にコピーして編集:

```bash
cp config.yaml.example config.yaml
# 編集
vim config.yaml
```

## 分析と可視化

### スイープ結果の分析

```bash
python tools/analyze_sweeps.py --sweep_dir runs/_sweeps --output_dir reports
```

### 学習曲線の可視化

```bash
python tools/visualize_training.py --run_dir runs/20250115_abc123
```

または:

```bash
python tools/visualize_training.py --metrics_json runs/20250115_abc123/metrics.json
```

## 主要機能

### 1. 自動リトライ機能

トレーニングが失敗した場合、自動的にリトライ:

```bash
MAX_RETRIES=3 bash run_sweep_lr.sh
```

### 2. 結果の自動記録

各実行の結果は以下の形式で記録されます:

- `runs/{run_id}/meta/summary.json`: 実行サマリー
- `runs/{run_id}/metrics.json`: 詳細メトリクス
- `runs/{run_id}/logs/train.log`: トレーニングログ
- `runs/_sweeps/sweep_lr_*.csv`: スイープ結果CSV

### 3. ベストモデルの自動特定

スイープ実行後、最良のモデルが自動的に特定され、レポートに記録されます。

## API ドキュメント

### gnn_common モジュール

#### `gnn_common.models`

- `GCNModel`: Graph Convolutional Network モデル
- `GATModel`: Graph Attention Network モデル

#### `gnn_common.metrics`

- `metrics_from_confusion_matrix()`: 混同行列からメトリクスを計算
- `evaluate_and_visualize()`: 評価と可視化

#### `gnn_common.training_utils`

- `set_seed()`: 再現性のためのシード設定
- `setup()`: 分散学習の初期化
- `cleanup()`: 分散学習のクリーンアップ

### tools モジュール

#### `tools.launch_run`

トレーニング実行を管理するランチャー。

主要な引数:
- `--profile`: トレーニングプロファイル名
- `--run_id`: 実行ID（省略時は自動生成）
- `--resume`: リジューム動作 (auto|off)
- `--dry_run`: ドライラン（実行せずにコマンドを表示）

#### `tools.analyze_sweeps`

スイープ結果を分析し、可視化を生成。

主要な引数:
- `--sweep_csv`: スイープCSVファイルのパス
- `--sweep_dir`: スイープCSVが含まれるディレクトリ
- `--output_dir`: 出力ディレクトリ

#### `tools.visualize_training`

学習曲線とメトリクスを可視化。

主要な引数:
- `--run_dir`: 実行ディレクトリ
- `--metrics_json`: メトリクスJSONファイルのパス
- `--output_dir`: 出力ディレクトリ

## トラブルシューティング

### GPUが見つからない

```bash
# GPUの確認
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# GPU数を調整
NPROC_PER_NODE=1 bash run_train_recommended.sh
```

### メモリ不足エラー

バッチサイズを減らす:

```bash
BATCH_SIZE=32 bash run_train_recommended.sh
```

### 分散学習エラー

NCCLデバッグを有効化:

```bash
NCCL_DEBUG=INFO bash run_train_recommended.sh
```

## ベストプラクティス

1. **実験管理**: 各実験に一意の `RUN_ID` を設定
2. **ログ確認**: 実行後は `runs/{run_id}/logs/train.log` を確認
3. **リソース監視**: GPU使用率とメモリ使用量を監視
4. **定期的なバックアップ**: 重要なモデルと結果をバックアップ
5. **バージョン管理**: コード変更時はGitでコミット

## 関連論文・リポジトリ

本フレームワークは，以下の研究・コードベースを発展させたものです。

- 研究論文  
  - Nishioka, K., Kojima, Y., Saito, T., Kawakami, K., Washiya, M., & Muramatsu, M. (2025).  
    *Development of Defect Localization Method for Perforated Carbon-fiber-reinforced Plastic Specimens Using Finite Element Method and Graph Neural Network.*  
    Frontiers in Materials (Computational Materials Science). Manuscript ID: 1652484.  
    DOI: 10.3389/fmats.2025.1652484  
    オンライン版: https://www.frontiersin.org/journals/materials/articles/10.3389/fmats.2025.1652484/full

  **引用用 BibTeX**

  ```text
  @article{Nishioka2025CFRP,
    title   = {Development of Defect Localization Method for Perforated Carbon-fiber-reinforced Plastic Specimens Using Finite Element Method and Graph Neural Network},
    author  = {Nishioka, Keisuke and Kojima, Yuta and Saito, Toshiya and Kawakami, Kosuke and Washiya, Masahito and Muramatsu, Mayu},
    journal = {Frontiers in Materials},
    year    = {2025},
    doi     = {10.3389/fmats.2025.1652484},
    note    = {Section: Computational Materials Science, Manuscript ID: 1652484},
    publisher = {Frontiers Media S.A.}
  }
  ```

- もとの単一スクリプト実装  
  - keisuke58/CFRP_GNN (`code/GNN_noise_each_0913_all.py`)  
    FEM + GAT ベースの単一スクリプト実装から，本リポジトリでは「再利用しやすい共通フレームワーク」として `gnn_common/` 以下に機能を切り出しています。

## ライセンス

本プロジェクトは論文と同様に **CC BY 4.0**（Creative Commons Attribution 4.0 International）で提供しています。  
利用・改変・再配布は自由ですが、適切な引用表記が必要です。

## 貢献

バグ報告・機能提案・プルリクエストを歓迎します。変更を加える場合は、既存のコードスタイルとドキュメント方針に従ってください。
