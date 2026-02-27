# 性能保証ドキュメント (M1-6)

> 「このモデルはこの条件でこの性能」を保証するための記録

## 条件と目標値

| 条件 | 目標 | 備考 |
|------|------|------|
| IID split, seed=42 | macro_f1 ≥ 0.65 | ベースライン |
| defect_size OOD | macro_f1 低下率 < 20% | 大欠陥への汎化 |
| layer OOD | macro_f1 低下率 < 30% | 未学習層への汎化 |
| property_ood | macro_f1 ≥ 0.50 | 物性OOD（size_class 代理） |
| 推論速度 (GPU) | < 100 ms/サンプル | 実運用目標 |

## ベースライン実績（2026-02時点）

- **IID**: test_macro_f1 0.66〜0.72（複数 run）
- **推論**: GPU 3.4 ms/サンプル（目標達成）
- **層別F1**: layer1_macro_f1, layer2_macro_f1 を metrics に追加済み

## 更新履歴

- 2026-02-27: 初版作成（M0, M1 進捗反映）
