#!/usr/bin/env python3
"""1M 子集基线 — 用 faiss IVFPQ 省内存建 kNN 图"""

import os, time, json, struct
import numpy as np
import h5py
import faiss

DATA_DIR = "/home/hermes/hermes-workspace/nsg/data"
K = 100

def load_sift_hdf5(hdf5_path, subset_size=None):
    with h5py.File(hdf5_path, "r") as f:
        if subset_size:
            train = np.array(f["train"][:subset_size], dtype=np.float32)
        else:
            train = np.array(f["train"], dtype=np.float32)
        test  = np.array(f["test"],  dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    print(f"  train: {train.shape}  test: {test.shape}")
    return train, test, neighbors

def write_knn_graph(filename, knn_indices):
    """写入 NSG 兼容的 kNN 图格式"""
    N, K = knn_indices.shape
    with open(filename, "wb") as f:
        f.write(struct.pack("I", K))  # header: K
        for i in range(N):
            f.write(struct.pack("I", K))  # each row: K
            f.write(knn_indices[i].astype(np.uint32).tobytes())

def build_knn_graph_ivf(train, k=100, nlist=1024, nprobe=64, pq_m=64):
    """用 faiss IVFPQ 建 kNN 图，极致省内存"""
    N, D = train.shape
    print(f"  训练 IVF{nlist},PQ{pq_m}x8 索引...")
    t0 = time.time()

    # 训练 IVF+PQ 索引
    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, pq_m, 8)
    sample = train[np.random.choice(N, min(N, 100000), replace=False)]
    index.train(sample)
    index.add(train)
    index.nprobe = nprobe
    print(f"  索引构建: {time.time()-t0:.1f}s, 内存: ~{index.ntotal * pq_m / 1024 / 1024:.0f}MB")

    # 分批搜索，避免一次分配太大
    batch = 5000
    all_indices = np.zeros((N, k), dtype=np.int32)
    print(f"  搜索 kNN (batch={batch})...")
    t1 = time.time()
    for i in range(0, N, batch):
        end = min(i + batch, N)
        D_b, I_b = index.search(train[i:end], k)
        all_indices[i:end] = I_b.astype(np.int32)
        if i % 50000 == 0 and i > 0:
            print(f"    {i//1000}K/{N//1000}K  ({time.time()-t1:.1f}s)")
    print(f"  搜索完成: {time.time()-t1:.1f}s")
    return all_indices

def compute_recall(pred_results, gt_neighbors, k):
    correct = sum(
        len(set(map(int, pred_results[i][:k])) & set(map(int, gt_neighbors[i][:k])))
        for i in range(len(pred_results))
    )
    return correct / (len(pred_results) * k)

def main():
    hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
    knn_path  = os.path.join(DATA_DIR, "sift_1m_knn_ivf.graph")

    print("=" * 50)
    print("1. 加载 SIFT1M 全量")
    train, test, gt = load_sift_hdf5(hdf5_path)
    N, D = train.shape

    # kNN 图
    if not os.path.exists(knn_path):
        print(f"\n2. 构建 kNN 图 (IVFPQ, {N//1000}K 节点)")
        knn = build_knn_graph_ivf(train, k=100, nlist=1024, nprobe=64, pq_m=64)
        print(f"  写入 {knn_path}...")
        write_knn_graph(knn_path, knn)
        del knn  # 释放内存
    else:
        print(f"\n2. kNN 图已存在")

    # NSG
    print(f"\n3. 构建 NSG (L=40,R=50,C=500)")
    from pynsg import NSG, Metric
    nsg = NSG(dimension=D, num_points=N, metric=Metric.L2)
    t0 = time.time()
    nsg.build_index(train, knn_path, L=40, R=50, C=500)
    print(f"  构图: {time.time()-t0:.1f}s")

    print("\n4. 优化图")
    t0 = time.time()
    nsg.optimize_graph(train)
    print(f"  耗时: {time.time()-t0:.1f}s")

    print(f"\n5. Recall-QPS  (K={K})")
    search_L_list = [20, 40, 80, 160, 320, 640]
    results = []
    for L in search_L_list:
        t0 = time.time()
        pred = nsg.search_opt(test, k=K, search_L=L)
        elapsed = time.time() - t0
        recall = compute_recall(pred, gt, K)
        qps = test.shape[0] / elapsed
        results.append({"search_L": L, "recall@100": round(recall, 4), "qps": round(qps, 1)})
        print(f"  L={L:>4d}  Recall={recall:.4f}  QPS={qps:>8.1f}  ({elapsed:.1f}s)")

    print("\n" + "=" * 50)
    print(f"{'search_L':>8}  {'Recall@100':>10}  {'QPS':>8}")
    for r in results:
        print(f"{r['search_L']:>8d}  {r['recall@100']:>10.4f}  {r['qps']:>8.1f}")

    with open(os.path.join(DATA_DIR, "baseline_1m_ivf.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 结果已保存")

if __name__ == "__main__":
    main()
