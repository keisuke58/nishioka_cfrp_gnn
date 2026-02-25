# 統合サマリー

## ✅ 完了した作業

### 1. 統合箇所の特定
- ✅ データ分割部分（line ~3000）
- ✅ データ準備部分（line ~3143）
- ✅ モデル作成部分（line ~3124）
- ✅ 評価部分（line ~4277）

### 2. 統合パッチの作成
- ✅ `INTEGRATION_PATCHES.md`: 具体的なパッチ手順
- ✅ `GNN_zscore_sub_noise_defect_free_with_new_features.py`: 統合版テンプレート

---

## 📋 次のステップ

### オプション1: 手動でパッチを適用（推奨）

1. 既存スクリプトをバックアップ
2. `INTEGRATION_PATCHES.md`を参照して各箇所をパッチ
3. 動作確認

### オプション2: 新しいファイルに完全版を作成

既存スクリプトの全内容をコピーして、統合箇所のみを変更した完全版を作成

---

## 🔍 統合箇所の詳細

### 箇所1: データ分割（line 2996-3026）

**変更内容**:
- `apply_ood_split()`を使用してOOD分割をサポート
- 既存の`group_disjoint_split`と併用可能

**影響範囲**: データ分割のみ

---

### 箇所2: データ準備（line 3143-3149）

**変更内容**:
- `prepare_data_with_cross_edges()`を使用
- Cross-edgeを自動的に追加

**影響範囲**: データローダーの作成

---

### 箇所3: モデル作成（line 3124-3129）

**変更内容**:
- `create_model_with_cross_edges()`を使用
- Cross-edge対応モデルを自動選択

**影響範囲**: モデル定義のみ

---

### 箇所4: 評価（line 4277-4355）

**変更内容**:
- `evaluate_with_localization_metrics()`を使用
- Localization指標を自動計算

**影響範囲**: 評価ループのみ

---

## ⚠️ 注意事項

1. **既存スクリプトは変更しない**: バックアップを取ってから作業
2. **設定ファイル**: `config.yaml`が必要（存在しない場合はデフォルト値を使用）
3. **段階的統合**: 一度にすべてを統合せず、段階的に進める

---

## 📝 使用方法

統合後、設定ファイルで新機能を有効化:

```yaml
# config.yaml
data_splitting:
  split_type: "defect_size"  # OOD分割を使用
  ood_params:
    defect_size:
      size_threshold: 5

cross_edge:
  enabled: true  # Cross-edgeを使用
  k: 1
  surface_method: "outer_layer"
```

既存の方法（IID分割、Cross-edgeなし）も引き続き使用可能です。
