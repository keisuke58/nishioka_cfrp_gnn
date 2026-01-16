# Related work quick table (living document)

目的: 関連研究を「入力→グラフ→モデル→タスク→指標→示唆」で比較し、今後の設計判断を速くする。

凡例:
- **Input**: 何を観測/シミュレーションして使うか
- **Graph**: ノード/エッジの定義（複数グラフ含む）
- **Model**: 主要アーキテクチャ
- **Task**: 2値/多クラス、node/graphレベルなど
- **Metric**: 代表指標（論文が強調）
- **Takeaway for this repo**: このGNN欠陥推定プロジェクトに直結する学び

| Paper | Domain | Input | Graph | Model | Task | Metric (reported) | Takeaway for this repo |
|---|---|---|---|---|---|---|---|
| **Nishioka et al. 2025** (Frontiers in Materials) doi:10.3389/fmats.2025.1652484 | NDT/SHM, CFRP | FEMで得た応力/特徴（DSPSSなど）＋（実計測IR応力計測への接続を想定） | FEノードをグラフ化（近傍関係でメッセージパッシング） | GNN | 欠陥の3D局在（層＋平面） | （論文内指標） | **FEM×GNN×実計測**を繋ぐ基本形。層制約/穴周り等の物理制約の入れ方が鍵。 |
| **Yehia et al. 2025** (Engineering Structures) doi:10.1016/j.engstruct.2025.120842 | SHM/NDE | FEシミュレーション。表面相当のひずみ/変位から内部損傷（void）を推定 | **2系統ノード＋A–B cross-edge**（内部↔表面の情報ルーティング） | **MeshGraphNet** | node-level 2値（damaged/intact） | damaged F1 ≈ 0.691 | **surface→interior routing**が効く。cross-edge設計とedge_attr（距離等）を真似る価値大。 |
| **Wijethunga et al. 2025** (Engineering Structures) doi:10.1016/j.engstruct.2025.121265 | SHM | 振動/センサ時系列（典型） | **dual-graph**（物理トポロジ + 相関トポロジ） | Dual-graph GNN | 検出＋局在 | （論文内指標） | 複数隣接（mesh/kNN/相関）を併用してロバスト化する発想。 |
| **Hasebe et al. 2020** (Mech. Syst. Signal Process.) doi:10.1016/j.ymssp.2019.106381 | CFRP, 実験/学習 | **表面ひずみ分布**（実験/計測想定） | - | Multi-task learning | 内部損傷の推定（マルチタスク） | - | 「表面場→内部状態」学習の先行。GNNでなくても**表面情報の設計**が重要という根拠。 |
| **Hasebe et al. 2023** (Compos. Sci. Technol.) doi:10.1016/j.compscitech.2022.109820 | CFRP, 実験/学習 | **表面プロファイル**（impact後） | - | Multi-task learning | impact損傷情報の推定 | - | 実験系の“表面観測→損傷情報”タスクの具体例。評価指標（F1/回帰誤差）設計の参考。 |
| **Kikukawa & Ugai 1997** (JJSASS) doi:10.2322/jjsass1969.45.380 | CFRP, 穴周り | - | - | - | **円孔縁の層間はく離成長** | - | 「穴周りでdelaminationが成長する」ことの機械系根拠。**hole-centric edge/feature**の動機づけに使える。 |
| **Nasrin et al. 2023** (Appl. Mech.) doi:10.3390/appmech3040045 | CFRP, ボルト/穴 | - | - | Review | bolted jointの損傷/破壊レビュー | - | 穴・締結周りの損傷モード（bearing/net-tension等）整理に便利。 |
| **Dilonardo et al. 2020** (Compos. Sci. Technol.) doi:10.1016/j.compscitech.2020.108093 | CFRP, 実験 | **μCT/高分解能X線CT** | - | - | void/配向不良などの可視化 | - | IR/応力場からの推定結果の**ground truth**側（欠陥形状/位置の確定）として引用価値。 |
| **Ishikawa et al. 2012** (Adv. Compos. Mater.) doi:10.1163/156855112x629513 | NDT, CFRP | **Pulse Phase Thermography**（位相画像） | - | - | 欠陥検出（異方性影響） | - | CFRPの**熱異方性が検出深さ/コントラストに効く**。IR系入力の限界説明に使える。 |
| **Keo et al. 2015** (Compos. Part B) doi:10.1016/j.compositesb.2014.09.018 | NDT, CFRP | **CO₂レーザ励起IRT** vs **Lock-in IRT** | - | - | 欠陥検出比較 | - | 励起方式（レーザ/lock-in）でSNRや深さ感度が変わる。実験条件の議論に便利。 |
| **Swiderski 2019** (Compos. Struct.) doi:10.1016/j.compstruct.2018.11.013 | NDT, CFRP | **レーザ励起サーモグラフィ** | - | - | 欠陥検出 | - | 大面積・非接触NDTとしての位置づけ強化。 |
| **Yang et al. 2013** (Infrared Phys. & Technol.) doi:10.1016/j.infrared.2013.04.010 | NDT, CFRP | **Ultrasonic IR thermography**（超音波励起+IR） | - | - | 航空CFRPの欠陥検出/評価 | - | **超音波×IR**の複合手法。将来的な“多モダリティ”議論に使える。 |
| **Fang et al. 2021** (BDCC) doi:10.3390/bdcc5010009 | NDT×DL | **Pulsed Thermography**（合成＋実験） | - | Deep learning | 欠陥の**セグメンテーション/同定** | - | 合成データ併用・セグメンテーションの先例。あなたの「FEM→実験」橋渡しに近い。 |
| **Popow & Gurka 2020** (NDT&E Int.) doi:10.1016/j.ndteint.2020.102359 | NDT, CFRP | **Pulse Phase Thermography**（定量化） | - | - | 欠陥の自動定量 | - | “自動定量（深さ/面積）”の文脈。推定結果の**後処理/定量**の比較軸になる。 |
| **Kidangan et al. 2021** (NDT&E Int.) doi:10.1016/j.ndteint.2021.102498 | NDT, CFRP | **Induction thermography** | - | - | 繊維破断方向の同定 | - | 欠陥“タイプ”推定（delam以外）を議論する時の代表例。 |
| **Zalameda & Parker 2014** (NASA NTRS 20140006406) | NDT, サンドイッチ | **Pulsed thermography + PCA**（ハニカム） | - | - | サンドイッチのdisbond/損傷検出 | - | ロケット/航空の**サンドイッチ構造**に直結。深部コア損傷の限界も明確。 |
| **Ura et al. 1998** (MHI Tech. Rev.) | ロケット, 構造 | - | - | - | H-IIA系の複合材interstage改良 | - | “ロケットinterstageにCFRPが実用”の背景引用。 |
| **Shimazaki et al. 2015** (Space Structures & Materials symposium) | ロケット, 構造 | - | - | - | コアロケット複合材構造の課題 | - | “適用上の課題/設計上の論点”の一次資料枠。 |
| **Kang et al. 2007** (Compos. Struct.) doi:10.1016/j.compstruct.2005.11.005 | CFRP, 極低温/接着 | **CFRP–Al double-lap**（RTと-150℃） | - | - | 接着強度/FE解析 | - | interstage/タンク周辺の**極低温接着**の代表例。 |
| **Yoshimura et al. 2012** (J. Adhes. Sci. Technol.) doi:10.1163/156856111X593694 | CFRP, 極低温/破壊 | DCB/ENF（296K/223K/77K） | - | - | Mode I/II 破壊靭性 | - | 77Kでの脆化・界面破壊など、極低温での**破壊モード変化**を説明する根拠。 |
| **MDPI Instruments 2024** (Vol.8, Issue 1, 16) `https://www.mdpi.com/2410-390X/8/1/16` | ロケット, SHM総説 | SRMの検査・センサ（光ファイバ等） | - | Review | - | - | ロケット運用のSHM文脈（破壊的試験→CBM/光学センサ）を入れると、研究の位置づけが強くなる。 |
| **Scarselli et al. 2009** | GNN基礎 | - | - | Graph Neural Network | - | - | 背景（引用用）。 |
| **Kipf & Welling 2017** (GCN) | GNN基礎 | - | 隣接行列 | GCN | 半教師あり分類 | - | シンプルな基準モデル/比較対象。 |
| **Veličković et al. 2018** (GAT) | GNN基礎 | - | 注意機構 | GAT | - | - | 現行実装のバックボーン。 |
| **Lin et al. 2017** (Focal Loss) | 不均衡 | - | - | - | 不均衡分類 | - | minority重視の損失設計。 |
| **Cui et al. 2019** (Class-Balanced) doi:10.1109/CVPR.2019.00949 | 不均衡 | - | - | - | 不均衡分類 | - | effective numberで重み付け。 |
| **Menon et al. 2021** (Logit Adjustment) arXiv:2007.07314 | 不均衡/長尾 | - | - | - | 長尾分類 | - | 事前分布補正（logit調整）。 |
| **Fey & Lenssen 2019** (PyG) arXiv:1903.02428 | 実装 | - | - | PyTorch Geometric | - | - | 実装の引用。 |

## 追記候補（あとで入れたい枠）
- IR thermography × DL の総説/代表例（CFRP delamination向け）
- FEMU（finite element model updating）× DIC の代表（Yehiaが比較した系）

