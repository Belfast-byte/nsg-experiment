#!/usr/bin/env python3
"""NSG 基线复现 — 含峰值内存追踪，跑两次"""

import os, sys, time, json, resource
import numpy as np
import h5py

DATA_DIR = "/home/hermes/hermes-workspace/nsg/data"
SUBSET   = 100000
K        = 100
RUNS     = 2

# ── 工具函数 ──

def peak_mem_mb():
    """当前进程峰值 RSS (MB) — 仅在 Linux 下有效"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def peak_mem_total_mb():
    """系统空闲内存，用来推算实际占用"""
    with open("/proc/meminfo") as f:
        mem = {}
        for line in f:
            k, v = line.split(":")
            mem[k.strip()] = int(v.strip().split()[0])
    total = mem.get("MemTotal", 0) / 1024
    free  = mem.get("MemAvailable", mem.get("MemFree", 0)) / 1024
    return total - free  # 系统已用内存 ≈ 进程占用

def load_sift_hdf5(hdf5_path, subset):
    with h5py.File(hdf5_path, "r") as f:
        train = np.array(f["train"][:subset], dtype=np.float32)
        test  = np.array(f["test"],  dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)
    return train, test, neighbors

def compute_recall(pred, gt, k):
    correct = sum(
        len(set(map(int, pred[i][:k])) & set(map(int, gt[i][:k])))
        for i in range(len(pred))
    )
    return correct / (len(pred) * k)

def compute_ground_truth(train, test):
    """用 brute-force 在子集内计算 ground truth"""
    import faiss
    print("  计算子集内 ground truth...")
    idx = faiss.IndexFlatL2(train.shape[1])
    idx.add(train)
    _, gt = idx.search(test, K)
    return gt

# ── 单次运行 ──

def single_run(run_id):
    hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
    knn_path  = os.path.join(DATA_DIR, "sift_100k_knn.graph")

    log = {
        "run": run_id,
        "subset": SUBSET,
        "dim": 128,
        "query_count": 10000,
    }

    print(f"\n{'='*60}")
    print(f"第 {run_id} 次实验")
    print(f"{'='*60}")

    # ── 1) 加载 ──
    mem_before = peak_mem_mb()
    t0 = time.time()
    train, test, _ = load_sift_hdf5(hdf5_path, SUBSET)
    log["load_time_s"] = round(time.time() - t0, 2)
    log["peak_mem_load_mb"] = round(peak_mem_mb() - mem_before, 1)
    log["data_mem_mb"] = round(train.nbytes / 1024 / 1024 + test.nbytes / 1024 / 1024, 1)
    print(f"[1] 加载: {log['load_time_s']}s, 峰值+{log['peak_mem_load_mb']}MB")

    # ── 1.5) 子集内 ground truth ──
    print("    计算子集内 ground truth (faiss brute-force)...")
    gt = compute_ground_truth(train, test)

    # ── 2) kNN 图 ──
    log["knn_size_mb"] = round(os.path.getsize(knn_path) / 1024 / 1024, 1)
    print(f"[2] kNN 图: {log['knn_size_mb']}MB (已有)")

    # ── 3) 构建 NSG ──
    mem_before = peak_mem_mb()
    from pynsg import NSG, Metric
    nsg = NSG(dimension=128, num_points=SUBSET, metric=Metric.L2)
    t0 = time.time()
    nsg.build_index(train, knn_path, L=40, R=50, C=500)
    log["build_time_s"] = round(time.time() - t0, 2)
    log["peak_mem_build_mb"] = round(peak_mem_mb() - mem_before, 1)
    print(f"[3] NSG构图: {log['build_time_s']}s, 峰值+{log['peak_mem_build_mb']}MB")

    # ── 4) 优化图 ──
    mem_before = peak_mem_mb()
    t0 = time.time()
    nsg.optimize_graph(train)
    log["opt_time_s"] = round(time.time() - t0, 2)
    log["peak_mem_opt_mb"]   = round(peak_mem_mb() - mem_before, 1)
    log["peak_mem_overall_mb"] = round(peak_mem_mb(), 1)
    print(f"[4] 优化图: {log['opt_time_s']}s, 峰值+{log['peak_mem_opt_mb']}MB")
    print(f"    总峰值RSS: {log['peak_mem_overall_mb']}MB")

    # ── 5) 搜索评测 ──
    t_total = 0
    results = []
    for L in [20, 40, 80, 160, 320, 640]:
        t0 = time.time()
        pred = nsg.search_opt(test, k=K, search_L=L)
        elapsed = time.time() - t0
        t_total += elapsed
        recall = compute_recall(pred, gt, K)
        results.append({"L": L, "recall": round(recall, 4), "qps": round(test.shape[0] / elapsed, 1), "time_s": round(elapsed, 2)})
        print(f"  L={L:>4d}  Recall@100={recall:.4f}  QPS={test.shape[0]/elapsed:>8.1f}")

    log["search_total_s"] = round(t_total, 2)
    log["results"] = results

    return log

# ── 主流程 ──

all_logs = []
for r in range(1, RUNS + 1):
    all_logs.append(single_run(r))

# 汇总 + 平均
print(f"\n{'='*60}")
print("两次实验汇总对比")
print(f"{'='*60}")
for i, log in enumerate(all_logs):
    print(f"\n第{i+1}次:")
    print(f"  加载: {log['load_time_s']}s | NSG构图: {log['build_time_s']}s | 图优化: {log['opt_time_s']}s")
    print(f"  峰值内存(RSS): {log['peak_mem_overall_mb']}MB")
    print(f"  搜索总耗时: {log['search_total_s']}s")

avg = {}
for key in ["load_time_s", "build_time_s", "opt_time_s", "search_total_s", "peak_mem_overall_mb"]:
    vals = [log[key] for log in all_logs]
    avg[key] = round(sum(vals) / len(vals), 2)

print(f"\n平均值:")
print(f"  加载: {avg['load_time_s']}s | NSG构图: {avg['build_time_s']}s | 图优化: {avg['opt_time_s']}s")
print(f"  峰值内存(RSS): {avg['peak_mem_overall_mb']}MB")
print(f"  搜索总耗时: {avg['search_total_s']}s")

# 保存
out_all = os.path.join(DATA_DIR, "baseline_runs.json")
with open(out_all, "w") as f:
    json.dump({"runs": all_logs, "average": avg}, f, indent=2, ensure_ascii=False)
print(f"\n✅ 已保存: {out_all}")
