# 次のプラン（Next Plan）

最終更新: 2026-01-16

## 📋 直近2週間の優先タスク

### ★★★ 最優先: OOD分割を固定

**目的**: 評価設計の基盤を確立する

**タスク**:
1. **IID分割とOOD分割の実装**
   - IID分割: ランダム分割（従来通り）
   - OOD分割: 以下の条件で分割
     - 欠陥サイズ（PredDefectRatio等でbinning）
     - 欠陥位置（穴からの距離、層位置）
     - 層（layer）別
     - 境界条件（BC）別

2. **分割ロジックの実装場所**
   - `gnn_common/data_utils.py` に追加
   - 設定ファイル（`config.yaml`）で分割方法を選択可能に

3. **分割結果の記録**
   - 各分割の統計情報を保存
   - train/val/test の分布を可視化

**実装方針**:
```python
def create_ood_split(pairs, split_type='defect_size', ...):
    """
    OOD分割を作成
    
    Args:
        pairs: データペアのリスト
        split_type: 'defect_size', 'defect_position', 'layer', 'bc'
    """
    # 実装
    pass
```

---

### ★★☆ 優先度2: Localization指標で比較表作成

**目的**: 位置推定の性能を定量的に評価する

**タスク**:
1. **Localization指標の実装**
   - **Top-k Accuracy**: 予測確率上位k個のノードに正解が含まれる割合
   - **距離誤差**: 予測位置と正解位置のユークリッド距離
   - **AUPRC**: Precision-Recall曲線下面積（位置推定用）

2. **指標計算の実装場所**
   - `gnn_common/metrics.py` に追加
   - 既存の分類指標と併用

3. **比較表の生成**
   - 複数のモデル/設定での結果を比較
   - CSV形式で出力
   - 可視化（ヒートマップ等）

**実装方針**:
```python
def calculate_localization_metrics(
    predictions,  # 予測確率 [N, num_classes]
    labels,       # 正解ラベル [N]
    coordinates,  # ノード座標 [N, 3]
    top_k_list=[1, 3, 5, 10]
):
    """
    Localization指標を計算
    
    Returns:
        {
            'top_k_accuracy': {1: 0.xx, 3: 0.xx, ...},
            'mean_distance_error': float,
            'auprc': float
        }
    """
    pass
```

---

### ★☆☆ 優先度3: アブレーション3点のみ

**目的**: モデル設計の各要素の寄与を分離評価する

**最小アブレーションセット**:
1. **入力特徴**: どの特徴量が重要か
2. **層間エッジ有無**: cross-edge（Yehia方式）の効果
3. **損失設計**: 異なる損失関数の比較

**実装方針**:
- 既存のアブレーションプラン（`ablation_plan_surface_to_interior.md`）を参照
- E0（Baseline）→ E1（Cross-edge ON/OFF）を最優先
- 各実験の結果を記録テンプレートに従って記録

---

## 🔬 アブレーション実験の詳細

### E0: Baseline（現状）
- **Graph**: 現状のエッジのみ
- **Loss/Sampling**: 現状の設定
- **期待**: 基準値として使用

### E1: Cross-edge ON/OFF（最重要）
- **E1-a**: cross-edgeなし（=E0）
- **E1-b**: cross-edgeあり（k=1）
  - Bノード定義: 外層（surface-like）ノード
  - Cross-edge: 各Aノード→最近傍Bノードへ1本

**実装手順**:
1. `gnn_common/data_utils.py` に cross-edge 生成関数を追加
2. `gnn_common/models.py` で cross-edge を処理できるように修正
3. 設定ファイルで cross-edge の有無を切り替え可能に

---

## 📊 記録テンプレート

各実験について以下を記録:

| 項目 | 内容 |
|------|------|
| config_id | 実験ID |
| B定義 | outer/hole/top-p |
| cross-edge k | 1/3/5 |
| edge_attr | none/dist/dist+layer/... |
| detection defect-F1 | val/test |
| 19class macro-F1 | val/test |
| localization metrics | Top-k, 距離誤差, AUPRC |
| 最悪クラス | F1最低のclass idと改善幅 |

---

## 🛠️ 実装チェックリスト

### Phase 1: OOD分割（最優先）
- [ ] OOD分割関数の実装
- [ ] 設定ファイルでの分割方法選択
- [ ] 分割統計の可視化
- [ ] 既存コードとの統合

### Phase 2: Localization指標
- [ ] Top-k Accuracy の実装
- [ ] 距離誤差の計算
- [ ] AUPRC の計算
- [ ] 比較表の生成ツール

### Phase 3: アブレーション実験
- [ ] Cross-edge 生成の実装
- [ ] モデル側の対応
- [ ] 実験設定の準備
- [ ] 結果記録の自動化

---

## 📝 参考ドキュメント

- `literature_notes/research_plan_onepager.md`: 研究計画の全体像
- `literature_notes/ablation_plan_surface_to_interior.md`: アブレーション実験の詳細
- `literature_notes/yehia_transfer_plan.md`: Yehia et al. の手法移植計画

---

## 🎯 成功基準

1. **OOD分割**: IIDとOODで明確な性能差を確認できる
2. **Localization指標**: 位置推定の性能を定量的に評価できる
3. **アブレーション**: Cross-edge の効果を定量的に示せる

---

## ⏱️ タイムライン（目安）

- **Week 1**: OOD分割の実装と検証
- **Week 2**: Localization指標の実装とアブレーション実験の開始
