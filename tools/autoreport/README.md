# AutoReport (runs → report / paper / slides)

`runs/<RUN_ID>/` の学習結果（`metrics.json`, `meta/*.json`, `outputs`）から、以下を自動生成します。

- `report.md`（日本語の実験レポート）
- `paper.md`（論文風の骨子）
- `paper.tex`（LaTeX）
- `paper.pdf`（LaTeX→PDF、任意）
- `slides.md`（Marp 用スライド）
- `slides.pptx`（PowerPoint、任意）
- `slides.pdf`（PPTX→PDF、任意・Cursorで閲覧しやすい）

## 使い方

リポジトリルートで実行:

```bash
python3 tools/autoreport/render_run.py --run "runs/20260115_224356_3519019c_ndf_recommendedF072"
```

PDF/PowerPoint も作る（依存ツールが必要）:

```bash
python3 tools/autoreport/render_run.py --run "runs/20260115_224356_3519019c_ndf_recommendedF072" --build_pdf --build_pptx
```

PPTX→PDF も作る（Cursorでプレビューしたい場合）:

```bash
python3 tools/autoreport/render_run.py --run "runs/20260115_224356_3519019c_ndf_recommendedF072" --build_slides_pdf
```

出力先（デフォルト）:

- `reports/<run_folder_name>/`

任意の出力先を指定:

```bash
python3 tools/autoreport/render_run.py --run "runs/<RUN_ID>" --out "reports/<ANY_NAME>"
```

画像コピーをせずにオリジナルへリンクする（軽量）:

```bash
python3 tools/autoreport/render_run.py --run "runs/<RUN_ID>" --no_copy_assets
```

## 可変 run 対応の考え方

- `meta/summary.json` に書かれているパスが古い/別名 run を指していても、`runs/<RUN_ID>/outputs/` 配下を探索して補正します。
- 画像探索は Python 側で `glob/rglob` を使うため、run フォルダ名やpngの列挙数が変わっても動くようにしています。

## 依存（PDF/PPTXを作る場合）

- **PDF（LaTeX→PDF）**: `tectonic`（推奨、condaで入れられます）
  - 例: `/home/nishioka/miniconda3/bin/conda install -n gnn_final_env -c conda-forge tectonic -y`
- **PPTX**: `python-pptx`
  - 例: `/home/nishioka/miniconda3/envs/gnn_final_env/bin/python -m pip install -U python-pptx`
- **slides.pdf（PPTX→PDF）**: `libreoffice`（`soffice`）
  - 例: `sudo apt-get update && sudo apt-get install -y libreoffice`

