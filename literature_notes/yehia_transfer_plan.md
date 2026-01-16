# Yehia et al. (Eng. Structures 2025) transfer plan → this repo

対象論文:
- Ayatollah S. Yehia, Devin K. Harris, Amro Alabsi Aljundi, *Engineering Structures* 341 (2025) 120842. doi:10.1016/j.engstruct.2025.120842

目的:
- Yehiaの「subsurface推定に効くグラフ設計（A–B cross-edge）」と「MeshGraphNet的な edge/node encoder」を、現行の **Two-Stage GAT (NDF vs defect → defect type)** に移植・比較できる形にする。

---

## 1) Yehiaの要点（再現に必要な“構造”だけ）

- **タスク**: node-level binary classification（damaged / intact）
- **観測制約**: “表面で得られる量（strain/displacement相当）”から内部損傷を推定
- **キモ**: 内部ノードに表面情報を届けるための **A–B cross edges**

この “surface→interior routing” は、CFRPの層方向推定にも類似の必要性がある。

---

## 2) このリポジトリでの対応付け（概念→実装）

### 2.1 YehiaのA/Bを、ここではどう定義する？
このプロジェクトは穴あきCFRPの層構造があるため、A/Bは複数の定義候補がある。

- **案A（層ベース）**
  - **B（surface-like）**: 外層（最外層/観測しやすい層）ノード
  - **A（interior-like）**: 内層ノード（推定したい層を含む）
  - 直感: 「観測に近い層→内部層へ情報伝搬」

- **案B（“穴周り”ベース）**
  - **B**: 穴周辺・応力集中が観測しやすいリング領域ノード
  - **A**: それ以外（または穴から遠い）ノード
  - 直感: 「信号が強い地点→弱い地点へルーティング」

- **案C（現状維持＋追加）**
  - 既存のノード集合はそのまま（A=全ノード）
  - **Bを“補助ノード集合”として追加**（同一グラフ内に2タイプのノードを導入）
  - 実装コストは高いが、Yehiaに最も近い。

まずは案Aが実装・解釈ともに簡単でおすすめ。

### 2.2 A–B cross-edge の作り方（ミニマム）
1. Bノード集合を決める（例: layer==outer）
2. 各Aノードに対し、Bの **kNN** で \(k\) 個結ぶ（例: k=1 or 3 or 5）
3. cross-edgeに **edge_attr** を持たせる（推奨）
   - \(\Delta x,\Delta y,\Delta z\)、\(\|\Delta\|\)
   - layer差（\(\Delta \text{layer}\)）
   - 穴中心からの相対（\(\Delta r\) など。穴があるなら強い特徴）

### 2.3 既存エッジとの併用（3種類の隣接を持つ）
- **mesh/元の近傍エッジ**
- **kNN（空間）エッジ**
- **cross-edge（A–B）**

GATのままなら、エッジ集合を結合して1グラフとして扱う（まずはこれ）。
より攻めるなら「エッジタイプ埋め込み」を入れて注意重みを分ける。

---

## 3) モデル側の移植オプション

### Option 1: 現行Two-Stage GATを維持（最短）
- 追加: cross-edge + edge_attr（使えるなら） + edgeタイプ埋め込み（任意）
- 比較: cross-edge なし vs あり

### Option 2: MeshGraphNet風の encoder–processor–decoder（中）
- Node encoder: MLP（入力特徴→latent）
- Edge encoder: MLP（edge_attr→latent）
- Processor: message passing block × N（残差）
- Decoder: MLP（latent→logits）

まずは **検出ヘッド（NDF vs defect）だけを二値で** これに置き換えると比較が綺麗。

---

## 4) 評価の合わせ方（Yehiaと比較可能にする）

Yehiaは damaged-class F1 を主指標にしている。
このプロジェクトでは以下を合わせると議論が強くなる:

- **Binary detection**（defect vs NDF）の defect-class F1（= damaged F1相当）
- **欠陥サイズ/強度別のF1**（PredDefectRatio等でbinning）
- 層別F1（layerごと）

---

## 5) 実験の優先順位（最小コスト順）
1. **cross-edge ON/OFF**（GAT維持）
2. cross-edge の **k** を振る（1/3/5）
3. edge_attr（距離）を入れる（できるモデルへ）
4. 2値（検出ヘッドのみ）でMeshGraphNet風へ置換

