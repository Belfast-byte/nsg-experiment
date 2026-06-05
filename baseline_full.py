#!/usr/bin/env python3
"""NSG 全流程基线 — 记录耗时/内存/索引大小"""

import os, time, json, struct, subprocess, resource
import numpy as np
import h5py
import faiss

DATA = "/home/hermes/hermes-workspace/nsg/data"
NSG_DIR = "/home/hermes/hermes-workspace/nsg"
EFANNA = "/home/hermes/hermes-workspace/efanna_graph/tests/test_nndescent"
K = 100
LOG = []  # 累积日志

def log(msg):
    t = time.strftime("%H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    LOG.append(line)

def mem_mb():
    """当前进程 RSS (MB)"""
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS:"):
                return int(l.split()[1]) // 1024

def file_mb(path):
    return os.path.getsize(path) / 1024 / 1024 if os.path.exists(path) else 0

# ═══════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════
log("=" * 50)
log("1. 加载 SIFT1M")
hdf5 = os.path.join(DATA, "sift-128-euclidean.hdf5")
with h5py.File(hdf5, "r") as f:
    train = np.array(f["train"], dtype=np.float32)
    test  = np.array(f["test"],  dtype=np.float32)
    gt    = np.array(f["neighbors"], dtype=np.int32)
N, D = train.shape
log(f"   train: {N}x{D}  test: {test.shape}  mem={mem_mb()}MB")

# ═══════════════════════════════════════
# 2. 构建 kNN 图 (efanna_graph NN-Descent)
# ═══════════════════════════════════════
knn_path = os.path.join(DATA, "sift_200nn.graph")
base_fvecs = os.path.join(DATA, "sift_base.fvecs")

if not os.path.exists(knn_path):
    log(f"\n2. 构建 kNN 图 (NN-Descent)")
    log(f"   参数: K=200 L=200 iter=10 S=10 R=100")
    t0 = time.time()

    # 先确保 fvecs 存在
    if not os.path.exists(base_fvecs):
        log("   转换 fvecs...")
        with open(base_fvecs, "wb") as out:
            for i in range(N):
                out.write(struct.pack("I", D))
                out.write(train[i].tobytes())

    cmd = [EFANNA, base_fvecs, knn_path, "200", "200", "10", "10", "100"]
    log(f"   命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    log(f"   stdout: {result.stdout.strip()[-200:]}")
    if result.stderr:
        log(f"   stderr: {result.stderr.strip()[-200:]}")

    knn_time = time.time() - t0
    knn_size = file_mb(knn_path)
    log(f"   ✅ 完成! 耗时={knn_time:.1f}s  大小={knn_size:.1f}MB")
else:
    knn_time = 0
    knn_size = file_mb(knn_path)
    log(f"\n2. kNN 图已存在  大小={knn_size:.1f}MB")

# ═══════════════════════════════════════
# 3. 构建 NSG
# ═══════════════════════════════════════
log(f"\n3. 构建 NSG (L=40,R=50,C=500)  mem={mem_mb()}MB")
from pynsg import NSG, Metric

nsg = NSG(dimension=D, num_points=N, metric=Metric.L2)
t0 = time.time()
nsg.build_index(train, knn_path, L=40, R=50, C=500)
build_time = time.time() - t0
build_mem = mem_mb()
log(f"   ✅ 构图: {build_time:.1f}s  峰值内存={build_mem}MB")

# ═══════════════════════════════════════
# 4. 优化图
# ═══════════════════════════════════════
log(f"\n4. 优化图布局  mem={mem_mb()}MB")
t0 = time.time()
nsg.optimize_graph(train)
opt_time = time.time() - t0
opt_mem = mem_mb()
log(f"   ✅ 耗时: {opt_time:.1f}s  内存={opt_mem}MB")

# 释放 train 内存（optimize_graph 后不再需要原始向量用于搜索）
log(f"   释放数据前 mem={mem_mb()}MB")

# ═══════════════════════════════════════
# 5. Recall-QPS 评测
# ═══════════════════════════════════════
log(f"\n5. Recall-QPS 曲线 (K={K})")
search_L_list = [20, 40, 80, 160, 320, 640]
results = []

for L in search_L_list:
    t0 = time.time()
    pred = nsg.search_opt(test, k=K, search_L=L)
    elapsed = time.time() - t0

    # 计算 recall
    correct = sum(
        len(set(map(int, pred[i][:K])) & set(map(int, gt[i][:K])))
        for i in range(len(pred))
    )
    recall = correct / (len(pred) * K)
    qps = len(pred) / elapsed

    results.append({
        "search_L": L, "recall@100": round(recall, 4),
        "qps": round(qps, 1), "time_s": round(elapsed, 2)
    })
    log(f"   L={L:>4d}  Recall@100={recall:.4f}  QPS={qps:>8.1f}  ({elapsed:.1f}s)")

# ═══════════════════════════════════════
# 6. 汇总
# ═══════════════════════════════════════
nsg_path = os.path.join(DATA, "sift.nsg")
nsg.save_index(nsg_path)
index_size = file_mb(nsg_path)

summary = {
    "dataset": f"SIFT1M ({N}x{D})",
    "knn_time_s": round(knn_time, 1),
    "knn_size_mb": round(knn_size, 1),
    "nsg_build_time_s": round(build_time, 1),
    "nsg_build_peak_mem_mb": build_mem,
    "nsg_opt_time_s": round(opt_time, 1),
    "nsg_opt_peak_mem_mb": opt_mem,
    "index_size_mb": round(index_size, 1),
    "results": results,
}

print("\n" + "=" * 50)
print("            📊  基 线 汇 总")
print("=" * 50)
print(f"  数据集:     SIFT1M ({N//1000}K x {D})")
print(f"  kNN 图:     {knn_time:.0f}s  {knn_size:.0f}MB")
print(f"  NSG 构图:   {build_time:.0f}s  峰值{mem_mb()}MB")
print(f"  优化图:     {opt_time:.0f}s")
print(f"  索引大小:   {index_size:.0f}MB")
print(f"\n  {'search_L':>8}  {'Recall@100':>10}  {'QPS':>8}")
for r in results:
    print(f"  {r['search_L']:>8d}  {r['recall@100']:>10.4f}  {r['qps']:>8.1f}")

with open(os.path.join(DATA, "baseline_full.json"), "w") as f:
    json.dump(summary, f, indent=2)
with open(os.path.join(DATA, "baseline_full.log"), "w") as f:
    f.write("\n".join(LOG))
log(f"\n✅ 结果: {DATA}/baseline_full.json")
log(f"✅ 日志: {DATA}/baseline_full.log")
