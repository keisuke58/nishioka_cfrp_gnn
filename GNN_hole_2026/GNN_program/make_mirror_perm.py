#!/usr/bin/env python3
"""左右ミラー（反転）データ拡張用の置換 permutation を座標から生成する。

スロット固定メッシュ前提:
  各スロット k は固定座標 (x_k, y_k) を持つ。x を層内の中心で反転した鏡像位置
  (xc - x_k, y_k) に最も近いスロット j を mirror_perm[k] = j とする。
  穴があるため 57x125 の完全格子ではない（vertices_per_layer=6971）ので、
  最近傍マッチングで対応を取る。層（最外2層）ごとに独立に計算する。

学習側 (GNN_zscore_sub.py --mirror_augment --mirror_perm_path <out>) で:
  mirrored_DSPSS[k] = DSPSS[mirror_perm[k]],  mirrored_label[k] = label[mirror_perm[k]]
  （座標/幾何特徴列はスロット固有なので不変）

使い方:
  python make_mirror_perm.py \
    --x /home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_x_2layer.npy \
    --y /home/nishioka/GNN/GNN_hole/GNN_hole_data/normalized_y_2layer.npy \
    --out mirror_perm.npy
"""
import argparse
import numpy as np


def nearest_indices(query_xy, ref_xy, chunk=512):
    """query 各点に対する ref 内の最近傍 index を返す（scipy 無しでも動くチャンク版）。"""
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_xy)
        _, idx = tree.query(query_xy, k=1)
        return idx.astype(np.int64)
    except Exception:
        n = query_xy.shape[0]
        out = np.empty(n, dtype=np.int64)
        for s in range(0, n, chunk):
            q = query_xy[s:s + chunk]  # (c, 2)
            d = ((q[:, None, :] - ref_xy[None, :, :]) ** 2).sum(-1)  # (c, R)
            out[s:s + chunk] = d.argmin(1)
        return out


def build_layer_perm(x, y):
    """1層分の (x,y) からミラー置換（層内ローカル index）を返す。"""
    xc = x.min() + x.max()  # 反転中心 *2 （x -> xc - x）
    mirrored = np.column_stack((xc - x, y))
    ref = np.column_stack((x, y))
    perm_local = nearest_indices(mirrored, ref)
    return perm_local


def main():
    ap = argparse.ArgumentParser(description="Generate left-right mirror permutation for CFRP DSPSS mesh.")
    ap.add_argument('--x', required=True, help='Path to x-coordinate npy (>=13942).')
    ap.add_argument('--y', required=True, help='Path to y-coordinate npy (>=13942).')
    ap.add_argument('--out', default='mirror_perm.npy', help='Output permutation npy path.')
    ap.add_argument('--total_nodes', type=int, default=13942)
    ap.add_argument('--vertices_per_layer', type=int, default=6971)
    args = ap.parse_args()

    x = np.load(args.x).astype(np.float64)[:args.total_nodes]
    y = np.load(args.y).astype(np.float64)[:args.total_nodes]
    vpl = args.vertices_per_layer
    assert x.shape[0] == args.total_nodes, f"x has {x.shape[0]} nodes, expected {args.total_nodes}"

    perm = np.empty(args.total_nodes, dtype=np.int64)
    for layer, sl in enumerate([slice(0, vpl), slice(vpl, args.total_nodes)]):
        local = build_layer_perm(x[sl], y[sl])
        perm[sl] = local + sl.start  # グローバル index に変換

    # 健全性チェック
    n = args.total_nodes
    coverage = len(np.unique(perm)) / n
    involution = np.mean(perm[perm] == np.arange(n))  # 鏡像の鏡像が元に戻る割合
    print(f"[make_mirror_perm] nodes={n}, unique-target coverage={coverage:.3f}, "
          f"involution rate={involution:.3f}")
    if coverage < 0.95:
        print("  ⚠ coverage が低い: 座標ファイルやvertices_per_layerを確認してください。")
    if involution < 0.95:
        print("  ⚠ involution が低い: 完全な左右対称でない可能性（拡張効果は限定的）。")

    np.save(args.out, perm.astype(np.int64))
    print(f"[make_mirror_perm] saved -> {args.out}")


if __name__ == '__main__':
    main()
