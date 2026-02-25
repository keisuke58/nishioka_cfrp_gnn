# 環境変数の設定方法

このプロジェクトでは、`.env`ファイルを使用して環境変数を管理することを推奨します。

## クイックスタート

### 1. .envファイルの作成

```bash
# .env.exampleをコピー
cp .env.example .env

# .envファイルを編集
vim .env
# または
nano .env
```

### 2. .envファイルの設定

`.env`ファイルを開いて、必要な値を設定してください：

```bash
# Gmail通知設定
GMAIL_FROM="your-email@gmail.com"
GMAIL_TO="recipient@gmail.com"
GMAIL_PASSWORD="your-16-char-app-password"

# Slack通知設定（オプション）
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXXXX/YYYYY/ZZZZZ"
```

### 3. 環境変数の読み込み

#### 方法1: スクリプト実行前に読み込む

```bash
# .envファイルを読み込む
source tools/load_env.sh
# または
. tools/load_env.sh

# その後、スクリプトを実行
bash run_train_recommended.sh
```

#### 方法2: 自動読み込み（推奨）

`run_train_recommended.sh` と `run_sweep_lr.sh` は自動的に `.env` ファイルを読み込みます。

```bash
# .envファイルがあれば自動的に読み込まれます
bash run_train_recommended.sh
```

## .envファイルの構造

```bash
# Gmail通知設定
GMAIL_FROM="your-email@gmail.com"
GMAIL_TO="recipient@gmail.com"
GMAIL_PASSWORD="your-app-password"

# Slack通知設定（オプション）
SLACK_WEBHOOK_URL=""

# トレーニング設定（オプション）
# LR=0.002
# EPOCHS=2000
# BATCH_SIZE=64
# HIDDEN=16
# OUTPUT_BASE="/home/nishioka/GNN/runs"
# DATASET_TAG="NDF"

# スイープ設定（オプション）
# MAX_RETRIES=0
# RETRY_DELAY=60
```

## セキュリティ

### 重要な注意事項

- **`.env`ファイルはGitにコミットされません**（`.gitignore`に含まれています）
- **`.env`ファイルには機密情報（パスワード、APIキーなど）が含まれます**
- `.env`ファイルの権限を適切に設定してください：
  ```bash
  chmod 600 .env  # 所有者のみ読み書き可能
  ```

### .env.exampleについて

- `.env.example`はテンプレートファイルです
- このファイルはGitにコミットされます
- 実際の値は含まれていません
- チームメンバーが設定を理解するための参考として使用します

## 環境変数の優先順位

環境変数は以下の優先順位で読み込まれます（高い順）：

1. **コマンドラインで指定された環境変数**
   ```bash
   LR=0.001 bash run_train_recommended.sh
   ```

2. **シェルの環境変数**
   ```bash
   export LR=0.001
   bash run_train_recommended.sh
   ```

3. **.envファイル**
   ```bash
   # .envファイルに LR=0.001 が設定されている場合
   bash run_train_recommended.sh
   ```

4. **スクリプト内のデフォルト値**

## トラブルシューティング

### .envファイルが読み込まれない

1. `.env`ファイルがプロジェクトルートに存在するか確認
   ```bash
   ls -la .env
   ```

2. ファイルの権限を確認
   ```bash
   ls -l .env
   # 所有者が読み取り可能であることを確認
   ```

3. 手動で読み込む
   ```bash
   source tools/load_env.sh
   ```

### 環境変数が設定されていない

1. `.env`ファイルの内容を確認
   ```bash
   cat .env
   ```

2. 環境変数が読み込まれているか確認
   ```bash
   source tools/load_env.sh
   echo $GMAIL_FROM
   ```

3. スクリプト実行前に明示的に設定
   ```bash
   export GMAIL_FROM="your-email@gmail.com"
   bash run_train_recommended.sh
   ```

## 複数の環境で使用する場合

異なる環境（開発、本番など）で異なる設定を使用する場合：

```bash
# 開発環境用
cp .env.example .env.development

# 本番環境用
cp .env.example .env.production

# 使用する環境を選択
cp .env.development .env
```

または、環境変数で指定：

```bash
ENV_FILE=.env.production bash run_train_recommended.sh
```

## 参考

- `EMAIL_SETUP.md`: Gmail通知の詳細設定
- `SLACK_SETUP.md`: Slack通知の詳細設定
- `.env.example`: 環境変数のテンプレート
