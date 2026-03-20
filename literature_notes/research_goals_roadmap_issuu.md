# 研究目標・ロードマップ（高目標版）

**Frontiers 2025 採択後 → 次の論文へ**  
最終更新: 2026-02-26  
※ Issuu 等での共有・配布用

---

## 1. ビジョン（5年スパン）

| 項目 | 内容 |
|------|------|
| **最終ゴール** | FEM+GNN による CFRP 欠陥推定を、**実計測・実構造**に適用可能な技術として確立する |
| **学術的ポジション** | 「表面観測のみから内部欠陥を推定する物理整合型GNN」の先行研究グループとして認知される |
| **産業的価値** | NDE/SHM の補助ツールとして、設計段階・検査段階の両方で参照される |

---

## 2. 論文シリーズ戦略（3本構成）

| # | 論文の芯 | 狙うジャーナル | 想定時期 |
|---|----------|----------------|----------|
| **Paper 1** | Frontiers 2025（済）— 基本手法の確立 | Frontiers in Materials | ✅ 採択済 |
| **Paper 2** | OOD汎化 × Surface→Interior × Cross-edge | **Engineering Structures** / **NDT&E International** | 2026–2027 |
| **Paper 3** | 物理正則化 / 階層GNN / 実データ検証 | **Computer-Aided Civil and Infrastructure Engineering** / **Structural Health Monitoring** | 2027–2028 |

---

## 3. 狙う論文誌（優先順・難易度付き）

### 3.1 メインターゲット（Paper 2 向け）

| ジャーナル | 分野 | インパクト | 狙いやすさ | 備考 |
|------------|------|------------|------------|------|
| **Engineering Structures** | 構造工学・SHM | IF ~5.5 | ◎ | Yehia, Wijethunga が掲載。FEM+GNN の文脈で受け入れられやすい |
| **NDT&E International** | 非破壊検査 | IF ~4.5 | ◎ | CFRP・欠陥検出の本流。実用寄りの主張が効く |
| **Composite Structures** | 複合材料 | IF ~5.5 | ○ | CFRP 特化。材料・構造の読者に刺さる |
| **Mechanical Systems and Signal Processing** | 振動・SHM | IF ~7.5 | △ | より高度な信号処理・物理モデルが求められる |

### 3.2 上位ターゲット（Paper 3 向け）

| ジャーナル | 分野 | インパクト | 狙いやすさ | 備考 |
|------------|------|------------|------------|------|
| **Computer-Aided Civil and Infrastructure Engineering** | 計算土木・AI | IF ~9.5 | △ | 手法・理論の新規性が重要 |
| **Structural Health Monitoring** | SHM | IF ~5.5 | ○ | 実構造・実データが強く評価される |
| **IEEE Transactions on Neural Networks and Learning Systems** | 機械学習 | IF ~10 | × | GNN の理論的貢献が必須 |
| **Nature Communications** (Materials) | 総合 | IF ~14 | × | 実データ＋物理的洞察＋汎用性が必要 |

### 3.3 会議・ショート論文（布石・早期発表用）

| 会議/ジャーナル | 用途 |
|-----------------|------|
| **NDE for Structural Health Monitoring** (Workshop) | 業界との接点、早期フィードバック |
| **ECCOMAS / WCCM** (計算力学系) | FEM コミュニティへの露出 |
| **NeurIPS / ICML** (Workshop: ML4Eng, AI4Science) | ML コミュニティへの露出 |
| **Sensors** (MDPI) | オープンアクセス、比較的早い採択 |

---

## 4. 高目標：Paper 2 で達成すべき水準

### 4.1 主張の強さ（必須）

| 軸 | 具体的な示し方 |
|----|----------------|
| **OOD汎化** | defect_size / layer / position で OOD 分割し、IID 比で性能低下を定量化。提案手法でその低下を **20%以上抑制** |
| **Surface→Interior** | Cross-edge のアブレーションで、欠陥クラス F1 が **+10%以上**、距離誤差が **有意に減少** |
| **再現性** | コード・データ公開、または明確な再現手順の記載 |

### 4.2 実験の網羅性（推奨）

| 項目 | 内容 |
|------|------|
| 分割 | IID + OOD（defect_size, layer, position の少なくとも2種） |
| ベースライン | GCN, GAT, MLP + Cross-edge ON/OFF |
| 指標 | Accuracy, Macro-F1, Top-k, 距離誤差, AUPRC, 層別F1 |
| アブレーション | Cross-edge k, surface 定義（outer/hole/top-p）, 損失設計 |

### 4.3 ストーリーの完成形（一文）

> **We propose a graph neural network with surface-to-interior cross-edges that localizes internal defects in layered CFRP from FEM-derived surface fields, and demonstrate that it maintains localization accuracy under out-of-distribution conditions (defect size, layer, position) where standard GNNs degrade.**

---

## 5. 高目標：Paper 3 で狙う水準

### 5.1 追加要素（いずれか必須）

| 要素 | 内容 |
|------|------|
| **物理正則化** | 応力の滑らかさ、層境界との整合性を損失に組み込み、定性・定量で効果を示す |
| **階層GNN** | 層ごとのサブグラフ＋層間エッジで、積層構造を明示的にモデル化 |
| **実データ** | 実験計測（DIC, 超音波等）との比較、または sim-to-real の検証 |
| **不確実性** | 予測の信頼度・不確実性を出力し、実用時の判断材料として評価 |

### 5.2 ストーリーの方向性

> **We extend the surface-to-interior GNN with physics-informed regularization and hierarchical layer modeling, and validate its robustness on experimental data and out-of-distribution scenarios.**

---

## 6. マイルストーン（2026–2028）

| 時期 | マイルストーン | 成果物 |
|------|----------------|--------|
| **2026 Q1** | OOD 分割固定、Cross-edge アブレーション完了 | 比較表、プロット |
| **2026 Q2** | Paper 2 ドラフト完成、第一稿投稿 | Eng. Struct. または NDT&E へ |
| **2026 Q3–Q4** | 査読対応、リビジョン | 採択目標 |
| **2027** | 物理正則化 or 階層GNN の実装・実験 | Paper 3 の布石 |
| **2027–2028** | 実データ取得・検証（可能なら） | Paper 3 投稿 |

---

## 7. 成功の定義（自己評価用）

| レベル | 達成内容 |
|--------|----------|
| **Bronze** | Paper 2 を Eng. Struct. / NDT&E / Composite Struct. のいずれかに採択 |
| **Silver** | Paper 2 採択 + Paper 3 の第一稿投稿 |
| **Gold** | Paper 2 採択 + Paper 3 を CACAIE / Struct. Health Monit. クラスに採択 |
| **Platinum** | 実データでの検証を含む Paper 3 採択 + 他研究グループからの引用・追試 |

---

## 8. 参照

- `research_plan_onepager.md` — 研究計画の要約
- `ablation_plan_surface_to_interior.md` — アブレーション実験の詳細
- `2025_engstruct_gnn_subsurface_and_dualgraph.md` — 関連論文（Yehia, Wijethunga）
