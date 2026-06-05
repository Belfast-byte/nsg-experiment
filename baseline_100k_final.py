#!/usr/bin/env python3
"""NSG 基线复现 — 使用已建好的 100K kNN 图"""

import os, time, json
import numpy as np
import h5py

DATA_DIR = "/home/hermes/hermes-workspace/nsg/data"
SUBSET = 100000
K = 100

def load_sift_hdf5(hdf5_path, subset):
    with h5py.File(hdf5_path, "r") as f:
        train = np.array(f["train"][:subset], dtype=np.float32)
        test  = np.array(f["test"],  dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    print(f"  train: {train.shape}  test: {test.shape}")
    return train, test, neighbors

def compute_recall(pred, gt, k):
    correct = sum(
        len(set(map(int, pred[i][:k])) & set(map(int, gt[i][:k])))
        for i in range(len(pred))
    )
    return correct / (len(pred) * k)

def main():
    hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
    knn_path  = os.path.join(DATA_DIR, "sift_100k_knn.graph")

    print("=" * 60)
    print("NSG 基线复现 — 100K 子集")
    print("=" * 60)

    # 1. 加载
    t0 = time.time()
    print("\n[1/5] 加载数据...")
    train, test, gt = load_sift_hdf5(hdf5_path, SUBSET)
    load_time = time.time() - t0
    N, D = train.shape
    print(f"  耗时: {load_time:.1f}s, 内存: {train.nbytes/1024/1024:.0f}MB")

    # 2. kNN 图
    knn_size = os.path.getsize(knn_path)
    print(f"\n[2/5] kNN 图: {knn_path} ({knn_size/1024/1024:.1f}MB)")

    # 3. NSG
    print("\n[3/5] 构建 NSG (L=40,R=50,C=500)...")
    from pynsg import NSG, Metric
    nsg = NSG(dimension=D, num_points=N, metric=Metric.L2)
    t1 = time.time()
    nsg.build_index(train, knn_path, L=40, R=50, C=500)
    build_time = time.time() - t1
    print(f"  耗时: {build_time:.1f}s")

    # 4. 优化图
    print("\n[4/5] 优化图布局...")
    t2 = time.time()
    nsg.optimize_graph(train)
    opt_time = time.time() - t2
    print(f"  耗时: {opt_time:.1f}s")

    # 5. 搜索
    print(f"\n[5/5] Recall-QPS 评测 (K={K})")
    search_L_list = [20, 40, 80, 160, 320, 640]
    results = []
    total_search_time = 0

    for L in search_L_list:
        t3 = time.time()
        pred = nsg.search_opt(test, k=K, search_L=L)
        elapsed = time.time() - t3
        total_search_time += elapsed
        recall = compute_recall(pred, gt, K)
        qps = test.shape[0] / elapsed
        results.append({"search_L": L, "recall@100": round(recall, 4), "qps": round(qps, 1)})
        print(f"  L={L:>4d}  Recall@100={recall:.4f}  QPS={qps:>8.1f}  ({elapsed:.1f}s)")

    # 汇总
    print("\n" + "=" * 60)
    print("基线汇总")
    print(f"  底库规模:    {N//1000}K × {D}")
    print(f"  查询数:      {test.shape[0]}")
    print(f"  加载耗时:    {load_time:.1f}s")
    print(f"  kNN图大小:   {knn_size/1024/1024:.1f}MB")
    print(f"  NSG构图:     {build_time:.1f}s")
    print(f"  图优化:      {opt_time:.1f}s")
    print(f"  搜索总耗时:  {total_search_time:.1f}s")
    print(f"\n  {'search_L':>8}  {'Recall@100':>10}  {'QPS':>8}")
    for r in results:
        print(f"  {r['search_L']:>8d}  {r['recall@100']:>10.4f}  {r['qps']:>8.1f}")

    # 保存
    summary = {
        "subset": SUBSET, "dim": D, "num_queries": test.shape[0],
        "load_time_s": load_time, "knn_size_mb": round(knn_size/1024/1024, 1),
        "build_time_s": round(build_time, 1), "opt_time_s": round(opt_time, 1),
        "search_time_s": round(total_search_time, 1),
        "results": results
    }
    out = os.path.join(DATA_DIR, "baseline_100k_final.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ 已保存: {out}")

if __name__ == "__main__":
    main()
