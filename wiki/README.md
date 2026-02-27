# Wiki コンテンツ

このフォルダは GitHub Wiki 用のソースです。

## GitHub Wiki への反映方法

### 方法1: ブラウザから手動追加

1. リポジトリの **Wiki** タブを開く
2. 「New Page」で新規ページを作成
3. このフォルダの `.md` ファイルの内容をコピー＆ペースト

### 方法2: CLI から push

```bash
# Wiki リポジトリをクローン（初回のみ）
git clone git@github.com:keisuke58/nishioka_cfrp_gnn.wiki.git wiki_repo
cd wiki_repo

# このフォルダからファイルをコピー
cp wiki/Home.md .
cp "wiki/実運用ロードマップ（ミクロ欠陥×GNN-MultiTask）.md" .

# commit（commit.template が設定されている場合は env -i で回避）
env -i HOME=$HOME PATH=/usr/bin:/bin USER=$USER git -c commit.template= add .
env -i HOME=$HOME PATH=/usr/bin:/bin USER=$USER git -c commit.template= commit -m "Add plan: 実運用ロードマップ"

# push（SSH 推奨）
git push origin master
```

### リンクの注意

Wiki 内の `../` リンクは、Wiki リポジトリでは動作しません。必要に応じて `https://github.com/keisuke58/nishioka_cfrp_gnn/blob/main/...` 形式に変更してください。
