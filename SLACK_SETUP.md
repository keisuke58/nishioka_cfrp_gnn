# Slack通知の設定方法

トレーニング完了時にSlackに自動通知を送信する機能です。

## セットアップ

### 1. Slack Incoming Webhookの作成

1. Slackワークスペースにログイン
2. [Slack Apps](https://api.slack.com/apps) にアクセス
3. "Create New App" → "From scratch" を選択
4. App名とワークスペースを選択
5. "Incoming Webhooks" を有効化
6. "Add New Webhook to Workspace" をクリック
7. 通知を送信したいチャンネルを選択
8. Webhook URLをコピー（例: `https://hooks.slack.com/services/XXXXX/YYYYY/ZZZZZ`）

### 2. 環境変数の設定

#### 方法1: 環境変数として設定

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXXXX/YYYYY/ZZZZZ"
```

#### 方法2: .bashrc/.zshrcに追加（永続化）

```bash
echo 'export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXXXX/YYYYY/ZZZZZ"' >> ~/.bashrc
source ~/.bashrc
```

#### 方法3: 実行時に指定

```bash
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXXXX/YYYYY/ZZZZZ" bash run_train_recommended.sh
```

## 使用方法

### 基本的なトレーニング実行

```bash
# SLACK_WEBHOOK_URLが設定されていれば自動的に通知が送信されます
bash run_train_recommended.sh
```

### スイープ実行

```bash
# スイープ完了時にサマリーが送信されます
bash run_sweep_lr.sh
```

### 手動で通知を送信

```bash
# シンプルなメッセージ
python tools/slack_notify.py --webhook_url <URL> --message "Training started"

# 実行結果のサマリー
python tools/slack_notify.py --webhook_url <URL> --run_dir runs/20250115_abc123

# スイープ結果のサマリー
python tools/slack_notify.py --webhook_url <URL> --sweep_csv runs/_sweeps/sweep_lr_20250115.csv
```

## 通知内容

### トレーニング完了通知

- 実行ID
- ステータス（成功/失敗）
- Best Macro F1スコア
- 開始/終了時刻
- テストメトリクス（精度、F1スコアなど）
- 実行ディレクトリパス

### スイープ完了通知

- 総実行数
- 成功/失敗数
- 最良モデルの情報
- スイープCSVのパス

## カスタマイズ

### ユーザー名とアイコンの変更

```bash
python tools/slack_notify.py \
  --webhook_url <URL> \
  --message "Custom message" \
  --username "My Bot" \
  --icon ":rocket:"
```

## トラブルシューティング

### 通知が送信されない

1. Webhook URLが正しく設定されているか確認
   ```bash
   echo $SLACK_WEBHOOK_URL
   ```

2. 手動でテスト
   ```bash
   python tools/slack_notify.py --webhook_url <URL> --message "Test message"
   ```

3. エラーログを確認
   - スクリプト実行時の `[WARN] Failed to send Slack notification` メッセージを確認

### Webhook URLのセキュリティ

- Webhook URLは機密情報です。Gitにコミットしないでください
- `.gitignore` に環境変数ファイルを追加することを推奨します
