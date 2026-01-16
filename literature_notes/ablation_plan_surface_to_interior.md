# Ablation plan: surface→interior routing (Yehia-style) for this repo

目的:
- 「macro-F1がボトルネック」「minority欠陥が難しい」状況で、**グラフ設計（cross-edge）** と **損失/サンプリング** の寄与を分離して評価する。

前提:
- 現行: Two-stage（検出: NDF vs defect、分類: 18class）＋ layer mask
- 指標: macro-F1（19class）に加えて、二値検出の defect-F1（= damaged-F1相当）も併記する

---

## 共通の評価セット（固定）
- train/val/test split は固定
- seed固定
- best checkpoint の選定基準は macro-F1（または detection-F1）
- 追加で出す指標:
  - 19class macro-F1
  - detection head の defect-F1
  - layer別F1
  - PredDefectRatio（あるなら）ビン別F1（小欠陥が落ちるかを見る）

---

## E0: Baseline（現状）
- **Graph**: 現状のエッジのみ
- **Loss/Sampling**: 現状（class_frequency_sampler on/off はプロファイルに従う）
- **期待**: ここを基準値にする

---

## E1: Cross-edge ON/OFF（最重要）
### E1-a: cross-edgeなし（=E0）
### E1-b: cross-edgeあり（k=1）
- **Bノード定義**: まずは「外層（surface-like）」を想定（layerでフィルタ）
- **Cross-edge**: 各Aノード→最近傍Bノードへ1本
- **Edge type**: タイプを識別できるなら type-id を持たせる（最初は無しでもOK）

**狙い**: Yehiaの「A–B edgeが効く」をここでも再現できるか。

---

## E2: Cross-edge の密度（k）の影響
- k ∈ {1, 3, 5}
- **期待**: kが大きすぎるとノイズも伝搬しやすい。小さすぎると情報不足。

---

## E3: Cross-edge の“どこをsurfaceと見なすか”の影響
- B定義の候補:
  - 外層（layer-based）
  - 穴周りリング（hole-centric）
  - “高信号ノード集合”（例: DSPSS上位p%）

**狙い**: 物理的に「情報源」をどう定義すると効くか決める。

---

## E4: Edge attributes（距離など）を入れる
（GATがedge_attrを直接使わない場合は、edge属性を埋め込みとして注意計算に渡すモデルへ）
- edge_attr候補:
  - \|\Delta\|, (\Delta x,\Delta y,\Delta z)
  - layer delta
  - hole center からの半径/角度差

**狙い**: MeshGraphNet的な「幾何をエッジ特徴として入れる」効果を見る。

---

## E5: 2値タスクへ切り出し（Yehiaに寄せた比較）
- 19classではなく、検出ヘッドだけで **binary defect detection** を学習・評価
- その上で cross-edge ON/OFF を比較

**狙い**: multi-classの難しさを排除して、routing設計の純粋な寄与を測る。

---

## 収束判定/停止条件（おすすめ）
- patienceは固定（現状の設定）
- ただし比較の公正性のため、epochs上限とearly-stop条件は統一

---

## 記録テンプレ（結果を表にする）
各実験について次を1行で記録:
- config_id
- B定義（outer/hole/top-p）
- cross-edge k
- edge_attr（none/dist/dist+layer/…）
- detection defect-F1（val/test）
- 19class macro-F1（val/test）
- 最悪クラス（F1最低のclass id）と改善幅

