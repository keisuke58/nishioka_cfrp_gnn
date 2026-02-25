# Gmail通知の設定方法

トレーニング完了時にGmailで自動通知を送信する機能です。

## セットアップ

### 1. Gmailアプリパスワードの作成

Gmailのセキュリティ設定により、通常のパスワードではSMTP接続できません。アプリパスワードを作成する必要があります。

1. Googleアカウントにログイン
2. [Googleアカウント設定](https://myaccount.google.com/) にアクセス
3. 「セキュリティ」タブを開く
4. 「2段階認証プロセス」が有効になっていることを確認（無効の場合は有効化）
5. 「アプリパスワード」を検索または「2段階認証プロセス」の下にある「アプリパスワード」をクリック
6. 「アプリを選択」→「メール」
7. 「デバイスを選択」→「その他（カスタム名）」→「GNN Training」などと入力
8. 「生成」をクリック
9. 表示された16文字のパスワードをコピー（例: `abcd efgh ijkl mnop`）

**重要**: このパスワードは一度しか表示されません。安全な場所に保存してください。

### 2. 環境変数の設定

#### 方法1: .envファイルを使用（推奨）

```bash
# .env.exampleをコピー
cp .env.example .env

# .envファイルを編集して値を設定
vim .env

# .envファイルを読み込む
source tools/load_env.sh
# または
. tools/load_env.sh
```

`.env`ファイルの例:
```bash
GMAIL_FROM="your-email@gmail.com"
GMAIL_TO="recipient@gmail.com"
GMAIL_PASSWORD="abcdefghijklmnop"
```

#### 方法2: 一時的に設定（現在のセッションのみ）

```bash
export GMAIL_FROM="your-email@gmail.com"
export GMAIL_TO="recipient@gmail.com"  # 自分自身でもOK
export GMAIL_PASSWORD="abcd efgh ijkl mnop"  # アプリパスワード（スペースなしでもOK）
```

#### 方法3: .bashrc/.zshrcに追加（永続化）

```bash
echo 'export GMAIL_FROM="your-email@gmail.com"' >> ~/.bashrc
echo 'export GMAIL_TO="recipient@gmail.com"' >> ~/.bashrc
echo 'export GMAIL_PASSWORD="abcdefghijklmnop"' >> ~/.bashrc
source ~/.bashrc
```

#### 方法4: 実行時に指定

```bash
GMAIL_FROM="your-email@gmail.com" \
GMAIL_TO="recipient@gmail.com" \
GMAIL_PASSWORD="abcdefghijklmnop" \
bash run_train_recommended.sh
```

## 使用方法

### 基本的なトレーニング実行

```bash
# 環境変数が設定されていれば自動的にメール通知が送信されます
bash run_train_recommended.sh
```

### スイープ実行

```bash
# スイープ完了時にサマリーがメールで送信されます
bash run_sweep_lr.sh
```

### 手動でメールを送信

```bash
# シンプルなメッセージ
python tools/email_notify.py \
  --to recipient@gmail.com \
  --from_email your-email@gmail.com \
  --password "abcdefghijklmnop" \
  --subject "Training started" \
  --message "Training has started"

# 実行結果のサマリー（HTML形式）
python tools/email_notify.py \
  --to recipient@gmail.com \
  --from_email your-email@gmail.com \
  --password "abcdefghijklmnop" \
  --run_dir runs/20250115_abc123 \
  --html

# スイープ結果のサマリー
python tools/email_notify.py \
  --to recipient@gmail.com \
  --from_email your-email@gmail.com \
  --password "abcdefghijklmnop" \
  --sweep_csv runs/_sweeps/sweep_lr_20250115.csv
```

## 通知内容

### トレーニング完了通知（HTML形式）

- 実行ID
- ステータス（成功/失敗、色分け表示）
- Best Macro F1スコア
- 開始/終了時刻
- テストメトリクス（表形式）
  - Accuracy
  - Macro F1
  - Weighted F1
  - Balanced Accuracy
- 実行ディレクトリパス

### スイープ完了通知

- 総実行数
- 成功/失敗数
- 最良モデルの情報
- スイープCSVのパス

## トラブルシューティング

### 認証エラー

**エラー**: `SMTPAuthenticationError`

**原因**: 
- アプリパスワードが正しくない
- 2段階認証が有効になっていない
- 通常のパスワードを使用している

**解決方法**:
1. 2段階認証が有効か確認
2. 新しいアプリパスワードを生成
3. パスワードにスペースが含まれている場合は削除して試す

### 接続エラー

**エラー**: `SMTPConnectError` またはタイムアウト

**原因**: 
- ファイアウォールでSMTPポート（587）がブロックされている
- ネットワーク接続の問題

**解決方法**:
1. ネットワーク接続を確認
2. ポート587が開いているか確認
3. 別のSMTPサーバーを試す（企業メールなど）

### メールが送信されない

1. 環境変数が正しく設定されているか確認
   ```bash
   echo $GMAIL_FROM
   echo $GMAIL_TO
   echo $GMAIL_PASSWORD
   ```

2. 手動でテスト
   ```bash
   python tools/email_notify.py \
     --to your-email@gmail.com \
     --from_email your-email@gmail.com \
     --password "your-app-password" \
     --subject "Test" \
     --message "This is a test email"
   ```

3. エラーログを確認
   - スクリプト実行時の `[WARN] Failed to send email notification` メッセージを確認

## セキュリティに関する注意事項

- **アプリパスワードは機密情報です**。Gitにコミットしないでください
- `.gitignore` に環境変数ファイルを追加することを推奨します
- アプリパスワードは定期的に再生成することを推奨します
- 不要になったアプリパスワードは削除してください

## メールとSlackの両方を使用する場合

メールとSlackの両方の通知を有効にできます：

```bash
# 両方の環境変数を設定
export GMAIL_FROM="your-email@gmail.com"
export GMAIL_TO="recipient@gmail.com"
export GMAIL_PASSWORD="your-app-password"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXXXX/YYYYY/ZZZZZ"

# 実行すると両方に通知が送信されます
bash run_train_recommended.sh
```
