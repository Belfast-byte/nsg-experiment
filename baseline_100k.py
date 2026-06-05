#!/usr/bin/env python3
"""NSG 基线复现 — 小内存版（100K 子集）"""

import os, time, json
import numpy as np
import h5py

DATA_DIR = "/home/hermes/hermes-workspace/nsg/data"
SUBSET_SIZE = 100000  # 10万子集，适配 2GB 内存
K = 100

def load_sift_hdf5(hdf5_path, subset_size):
    with h5py.File(hdf5_path, "r") as f:
        train = np.array(f["train"][:subset_size], dtype=np.float32)
        test  = np.array(f["test"],  dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    print(f"  train: {train.shape}  test: {test.shape}")
    return train, test, neighbors

def compute_recall(pred_results, gt_neighbors, k):
    correct = sum(
        len(set(map(int, pred_results[i][:k])) & set(map(int, gt_neighbors[i][:k])))
        for i in range(len(pred_results))
    )
    return correct / (len(pred_results) * k)

def main():
    hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")

    print("=" * 50)
    print(f"1. 加载 SIFT1M 子集 ({SUBSET_SIZE//1000}K)")
    train, test, gt = load_sift_hdf5(hdf5_path, SUBSET_SIZE)
    N, D = train.shape

    # kNN 图
    knn_path = os.path.join(DATA_DIR, f"sift_{SUBSET_SIZE//1000}k_knn.graph")
    if not os.path.exists(knn_path):
        print(f"\n2. 构建 kNN 图 (faiss HNSW, {SUBSET_SIZE//1000}K 节点)")
        from pynsg.graph_creator import create_graph_file
        t0 = time.time()
        create_graph_file(knn_path, train, k=100, hnsw_M=16)  # M=16 省内存
        print(f"  耗时: {time.time()-t0:.1f}s")
    else:
        print(f"\n2. kNN 图已存在")

    # NSG
    print("\n3. 构建 NSG")
    from pynsg import NSG, Metric
    nsg = NSG(dimension=D, num_points=N, metric=Metric.L2)
    t0 = time.time()
    nsg.build_index(train, knn_path, L=40, R=50, C=500)
    print(f"  构图: {time.time()-t0:.1f}s")

    # 优化
    print("\n4. 优化图")
    t0 = time.time()
    nsg.optimize_graph(train)
    print(f"  耗时: {time.time()-t0:.1f}s")

    # 搜索
    print(f"\n5. Recall-QPS 曲线 (K={K})")
    search_L_list = [20, 40, 80, 160, 320, 640]
    results = []

    for L in search_L_list:
        t0 = time.time()
        pred = nsg.search_opt(test, k=K, search_L=L)
        elapsed = time.time() - t0
        recall = compute_recall(pred, gt, K)
        qps = test.shape[0] / elapsed
        results.append({"search_L": L, "recall@100": round(recall, 4), "qps": round(qps, 1)})
        print(f"  L={L:>4d}  Recall@100={recall:.4f}  QPS={qps:>8.1f}  ({elapsed:.1f}s)")

    print("\n" + "=" * 50)
    print(f"{'search_L':>8}  {'Recall@100':>10}  {'QPS':>8}")
    for r in results:
        print(f"{r['search_L']:>8d}  {r['recall@100']:>10.4f}  {r['qps']:>8.1f}")

    with open(os.path.join(DATA_DIR, "baseline_100k.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存")

if __name__ == "__main__":
    main()
