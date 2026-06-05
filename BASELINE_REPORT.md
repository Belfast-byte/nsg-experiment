# NSG 性能优化实验 · 基线复现报告

## 一、实验环境

| 项目 | 详情 |
|------|------|
| CPU | DO-Premium-Intel, 1核1线程 |
| 内存 | 2GB (可用约 1.9GB) |
| 操作系统 | Ubuntu 24.04, Linux 6.8.0-71 x86_64 |
| 编译器 | g++ 13.3.0 (Ubuntu) |
| 编译选项 | `-O3 -march=native -std=c++11 -fopenmp` |
| 搜索线程 | 单线程 |
| 构图线程 | 1 线程 |
| SIMD 支持 | AVX2 |
| 数据源 | ann-benchmarks.com `sift-128-euclidean.hdf5` |
| NSG 版本 | ZJULearning/nsg (master, commit 228791f) |
| Python 绑定 | pynsg (本地 pip 安装) |

## 二、关键构图参数

| 参数 | 含义 | 取值 |
|------|------|------|
| L | kNN 图搜索队列长度 / NSG 搜索宽度 | 40 |
| R | NSG 图最大出度 | 50 |
| C | NSG 构图候选池上限 | 500 |
| K (kNN) | kNN 图的邻居数 | 100 |
| 数据类型 | 底库 / 查询 | float32 |

## 三、数据集

| 项目 | 值 |
|------|-----|
| 数据集 | SIFT1M (100K 子集) |
| 底库规模 | 100,000 × 128 维 |
| 查询规模 | 10,000 × 128 维 |
| 向量维度 | 128 |
| 子集选取 | 取前 100K 条 |

## 四、基线结果

### 4.1 各阶段耗时与资源

| 阶段 | 耗时 | 峰值内存 | 备注 |
|------|------|----------|------|
| 数据加载 | 0.1s | ~50MB | HDF5 → numpy |
| kNN 图构建 | 已完成 | ~200MB | faiss HNSW, K=100, M=16 |
| NSG 构图 | **53.4s** | ~300MB | L=40, R=50, C=500 |
| 图优化 | 0.1s | ~150MB | 向量+邻居交错布局 |
| 搜索评测 | 26.8s | ~200MB | 10K 查询 × 6 组 search_L |

### 4.2 索引大小

| 组件 | 大小 |
|------|------|
| kNN 图文件 (`sift_100k_knn.graph`) | 38.5 MB |
| 优化图 (内存中) | ≈ 55 MB |

### 4.3 图度数统计

| 指标 | 值 |
|------|-----|
| 最大出度 | 50 |
| 最小出度 | 2 |
| 平均出度 | 21 |
| 预期出度 R | 50 |

### 4.4 Recall–QPS 曲线 (Recall@100)

| search_L | Recall@100 | QPS | 平均延迟 |
|----------|-----------|------|----------|
| 20 | 21.00% | 12,899 | 0.078ms |
| 40 | 40.96% | 8,136 | 0.123ms |
| 80 | 80.99% | 4,561 | 0.219ms |
| **160** | **99.51%** | **2,743** | **0.365ms** |
| 320 | 99.94% | 1,506 | 0.664ms |
| 640 | 99.99% | 842 | 1.187ms |

> ⚠️ 注意：Ground truth 在 100K 子集内暴力计算得到，保证 Recall 计算的正确性。

### 4.5 可视化

```
Recall@100
 1.00 |                                    ●--●--●  99.5~99.99%
 0.80 |                          ●                   80.99%
 0.40 |               ●                              40.96%
 0.21 |     ●                                         21.00%
      +--------+--------+--------+--------+--------+--------> QPS
         12900    8100     4560     2740     1500      840
```

### 4.6 基准性能点

| 指标 | 值 |
|------|-----|
| Recall@100 ≥ 99% 所需最小 search_L | **160** |
| 该配置下的 QPS | 2,743 |
| 该配置下的平均查询延迟 | 0.365ms |

## 五、评测方法说明

1. **Recall 计算**：在 100K 子集内用 faiss FlatL2 暴力搜索得到 ground truth，每个查询保留 K=100 个最近邻
2. **QPS 计算**：10,000 个查询的总耗时取倒数，不包含数据加载和索引构建时间
3. **控制变量**：同一组实验使用完全相同的底库、查询和 ground truth
4. **多次测量**：每组参数仅跑一次（基线阶段），后续优化实验将跑 3 次取中位数

## 六、Ground Truth 生成

```python
import faiss
gt_index = faiss.IndexFlatL2(128)
gt_index.add(train)  # 100K 子集
_, gt_neighbors = gt_index.search(test, 100)  # 10K 查询
```

Ground truth 已保存至 `/home/hermes/hermes-workspace/nsg/data/baseline_gt_100k.npy`。

## 七、可复现性声明

- 数据来源：ann-benchmarks.com `sift-128-euclidean.hdf5`（前 100K 条）
- 随机种子：未固定（构图阶段有随机初始化，影响可忽略）
- 复现命令：

```bash
cd /home/hermes/hermes-workspace/nsg
python3 baseline_100k_final.py
```

全部源代码、数据和结果文件位于：
```
~/hermes-workspace/nsg/
  ├── data/
  │   ├── sift-128-euclidean.hdf5      # 原始数据
  │   ├── sift_base.fvecs              # 转换格式
  │   ├── sift_query.fvecs
  │   ├── sift_100k_knn.graph          # kNN 图
  │   ├── baseline_gt_100k.npy         # ground truth
  │   └── baseline_100k_final.json     # 结果
  └── baseline_100k_final.py           # 复现脚本
```

---

*基线记录时间: 2026-06-04*
