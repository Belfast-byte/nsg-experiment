#!/usr/bin/env python3
"""优化版评测：Version Array + Heap Pool + 32B对齐"""

import os, time, json, tracemalloc
import numpy as np
import h5py

DATA_DIR = "/home/hermes/hermes-workspace/nsg/data"
K = 100

def load_sift_hdf5(hdf5_path, subset_size=100000):
    with h5py.File(hdf5_path, "r") as f:
        train = np.array(f["train"][:subset_size], dtype=np.float32)
        test  = np.array(f["test"],  dtype=np.float32)
    return train, test

def compute_gt(train, test):
    """子集内暴力 ground truth"""
    import faiss
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    _, gt = index.search(test, K)
    return gt

def compute_recall(pred, gt, k):
    correct = 0
    for i in range(len(pred)):
        p = pred[i]
        g = gt[i]
        if not isinstance(p, (list, np.ndarray)):
            p = [p]
        p_set = set(int(x) for x in p[:k])
        g_set = set(int(x) for x in g[:k])
        correct += len(p_set & g_set)
    return correct / (len(pred) * k)

def main():
    hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
    knn_path  = os.path.join(DATA_DIR, "sift_100k_knn.graph")

    print("=" * 55)
    print("优化版评测：Version Array + Heap Pool + 32B 对齐")
    print("=" * 55)

    # Load data
    print("\n[1] 加载数据...")
    train, test = load_sift_hdf5(hdf5_path)
    gt = compute_gt(train, test)
    N, D = train.shape

    # Build NSG
    print(f"\n[2] 构建 NSG (N={N})...")
    from pynsg import NSG, Metric
    nsg = NSG(dimension=D, num_points=N, metric=Metric.L2)

    tracemalloc.start()
    t0 = time.time()
    nsg.build_index(train, knn_path, L=40, R=50, C=500)
    build_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  构图: {build_time:.1f}s, 峰值内存: {peak_mem/1024/1024:.1f}MB")

    # Optimize graph
    print("\n[3] 优化图 (32B对齐)...")
    tracemalloc.start()
    t0 = time.time()
    nsg.optimize_graph(train)
    opt_time = time.time() - t0
    _, peak_opt_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  耗时: {opt_time:.1f}s, 图内存: {peak_opt_mem/1024/1024:.1f}MB")

    # Search evaluation
    print(f"\n[4] Recall-QPS (K={K})")
    search_L_list = [20, 40, 80, 160, 320, 640]
    results = []

    for L in search_L_list:
        t0 = time.time()
        pred = nsg.search_opt(test, k=K, search_L=L)
        elapsed = time.time() - t0
        recall = compute_recall(pred, gt, K)
        qps = test.shape[0] / elapsed
        results.append({
            "search_L": L, "recall@100": round(recall, 4),
            "qps": round(qps, 1), "time_s": round(elapsed, 2)
        })
        print(f"  L={L:>4d}  Recall@100={recall:.4f}  QPS={qps:>8.1f}  ({elapsed:.1f}s)")

    # Summary
    print("\n" + "=" * 55)
    print("基线 vs 优化版 对比 (L=160, K=100)")
    print(f"{'指标':<20} {'基线':>10} {'优化版':>10} {'提升':>8}")
    # Baseline data from previous run
    baseline = {
        20: (0.2101, 7726.0), 40: (0.4100, 5078.0), 80: (0.8099, 3123.0),
        160: (0.9951, 1799.0), 320: (0.9994, 773.0), 640: (0.9999, 394.0)
    }
    for r in results:
        L = r["search_L"]
        if L in baseline:
            b_recall, b_qps = baseline[L]
            qps_gain = (r["qps"] - b_qps) / b_qps * 100
            print(f"L={L:>3d}   Recall:  {b_recall:.4f} -> {r['recall@100']:.4f}   "
                  f"QPS: {b_qps:.0f} -> {r['qps']:.0f}   ({qps_gain:+.1f}%)")

    # Save
    out = os.path.join(DATA_DIR, "optimized_v1_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 已保存: {out}")

if __name__ == "__main__":
    main()
