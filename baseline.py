#!/usr/bin/env python3
"""NSG 基线复现：构建 → 搜索 → Recall-QPS 曲线"""

import os, time, json
import numpy as np
import h5py

DATA_DIR = "/home/hermes/hermes-workspace/nsg/data"
K = 100  # Top-K for evaluation

def load_sift_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test  = np.array(f["test"],  dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    print(f"  train: {train.shape}  test: {test.shape}  gt_neighbors: {neighbors.shape}")
    return train, test, neighbors

def compute_recall(pred_results, gt_neighbors, k):
    correct = sum(
        len(set(map(int, pred_results[i][:k])) & set(map(int, gt_neighbors[i][:k])))
        for i in range(len(pred_results))
    )
    return correct / (len(pred_results) * k)

def main():
    # ── 1. 加载数据 ──
    print("=" * 50)
    print("1. 加载 SIFT1M 数据")
    hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
    train, test, gt = load_sift_hdf5(hdf5_path)
    N, D = train.shape

    # ── 2. 构建 kNN 图 ──
    knn_path = os.path.join(DATA_DIR, "sift_knn.graph")
    if not os.path.exists(knn_path):
        print("\n2. 构建 kNN 图 (faiss HNSW)")
        from pynsg.graph_creator import create_graph_file
        t0 = time.time()
        create_graph_file(knn_path, train, k=200)
        print(f"  耗时: {time.time()-t0:.1f}s")
    else:
        print(f"\n2. kNN 图已存在: {knn_path}")

    # ── 3. 构建 NSG ──
    print("\n3. 构建 NSG 索引")
    from pynsg import NSG, Metric
    nsg = NSG(dimension=D, num_points=N, metric=Metric.L2)
    t0 = time.time()
    nsg.build_index(train, knn_path, L=40, R=50, C=500)
    build_time = time.time() - t0
    print(f"  构图耗时: {build_time:.1f}s")

    # ── 4. 优化图 ──
    print("\n4. 优化图布局（向量+邻居交错存储）")
    t0 = time.time()
    nsg.optimize_graph(train)
    opt_time = time.time() - t0
    print(f"  耗时: {opt_time:.1f}s")

    # ── 5. Recall-QPS 评测 ──
    print(f"\n5. Recall-QPS 曲线 (K={K})")
    search_L_list = [20, 40, 80, 160, 320, 640]
    results_log = []

    for L in search_L_list:
        t0 = time.time()
        pred = nsg.search_opt(test, k=K, search_L=L)
        elapsed = time.time() - t0

        recall = compute_recall(pred, gt, K)
        qps = test.shape[0] / elapsed
        results_log.append({
            "search_L": L, "recall@100": round(recall, 4),
            "qps": round(qps, 1), "time_s": round(elapsed, 2)
        })
        print(f"  L={L:>4d}  Recall@100={recall:.4f}  QPS={qps:>8.1f}  ({elapsed:.1f}s)")

    # ── 6. 汇总 ──
    print("\n" + "=" * 50)
    print(f"{'search_L':>8}  {'Recall@100':>10}  {'QPS':>8}")
    for r in results_log:
        print(f"{r['search_L']:>8d}  {r['recall@100']:>10.4f}  {r['qps']:>8.1f}")

    out_path = os.path.join(DATA_DIR, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\n✅ 结果已保存: {out_path}")

if __name__ == "__main__":
    main()
