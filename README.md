# NSG 性能优化实验

> **高级数据库课程实验** — 对 NSG（Navigating Spread-out Graph）索引进行单线程查询性能优化
>
> 原始仓库: [ZJULearning/nsg](https://github.com/ZJULearning/nsg) | Python 绑定: [twuebker/nsg](https://github.com/twuebker/nsg)

## 实验简介

对 NSG 近似最近邻搜索索引进行两项查询性能优化，并通过消融实验量化各优化项的贡献。

**优化方向 A：单线程查询性能**

| 版本 | 优化内容 | L=160 QPS | 总提升 |
|------|---------|----------|--------|
| 基线 | 原始 NSG (Python 绑定) | 1,799 | — |
| V1 | Version Array 替代 boost::dynamic_bitset | 2,242 | +24.6% |
| V2 | V1 + AVX2 FMA 融合乘加 | 2,742 | **+52.4%** |

详细结果见 [REPORT.md](REPORT.md) 和 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)。

## 依赖

### 系统依赖 (apt)

```bash
sudo apt-get install -y cmake g++ libboost-dev libgoogle-perftools-dev libopenblas-dev pybind11-dev
```

### Python 依赖 (pip)

```bash
pip install numpy faiss-cpu h5py matplotlib
```

### 硬件要求

- CPU 支持 AVX2 (`grep avx2 /proc/cpuinfo`)
- 建议内存 ≥ 2GB（100K 子集）

## 编译

```bash
cd nsg-experiment
pip install --no-build-isolation -e .
```

编译产物：`pynsg/_bindings.cpython-*.so`

核心修改文件：
- `src/index_nsg.cpp` — Version Array 搜索 + 32B 对齐内存布局
- `include/efanna2e/distance.h` — AVX2 FMA 距离计算

## 运行实验

### 基线复现

```bash
# 1. 下载 SIFT1M 数据集
wget http://ann-benchmarks.com/sift-128-euclidean.hdf5 -O data/sift-128-euclidean.hdf5

# 2. 生成 kNN 图（首次运行）
python3 -c "
import numpy as np, h5py
from pynsg.graph_creator import create_graph_file
with h5py.File('data/sift-128-euclidean.hdf5','r') as f:
    train = f['train'][:100000]
create_graph_file('data/sift_100k_knn.graph', train, k=100, hnsw_M=16)
"

# 3. 运行基线评测
python3 baseline_100k_correct.py
```

### 优化版评测

```bash
python3 test_optimized.py
```

输出 `data/optimized_v1_results.json`，包含各 search_L 的 Recall@100 与 QPS。

### 参数说明

| 脚本 | 参数 | 说明 |
|------|------|------|
| `baseline_100k_correct.py` | — | 基线复现，100K 子集，子集内暴力 ground truth |
| `test_optimized.py` | — | 优化版评测，对比基线输出提升百分比 |

默认构图参数：L=40, R=50, C=500, 单线程

## 项目结构

```
nsg-experiment/
├── src/index_nsg.cpp          # NSG 核心实现（已修改 SearchWithOptGraph + OptimizeGraph）
├── include/efanna2e/
│   ├── distance.h             # AVX2 FMA 距离计算（已修改）
│   └── neighbor.h             # 数据结构 + InsertIntoPool
├── pynsg/                     # Python 绑定
│   ├── bindings.cpp
│   └── graph_creator.py
├── data/                      # 数据集 + 结果 JSON
├── images/                    # Recall-QPS 曲线图
├── baseline_100k_correct.py   # 基线评测脚本
├── test_optimized.py          # 优化版评测脚本
├── REPORT.md                  # 完整实验报告
├── EXPERIMENT_LOG.md          # 实验日志（增量记录）
└── README.md                  # 本文件
```

## 复现说明

所有实验在相同环境下进行：

| 项目 | 值 |
|------|-----|
| CPU | DO-Premium-Intel, 1核1线程 |
| 内存 | 2GB |
| 系统 | Ubuntu 24.04, Linux 6.8.0 |
| 编译器 | g++ 13.3.0, -O3 -march=native -fopenmp |
| 数据集 | SIFT1M 前 100K 向量 |
| Ground Truth | faiss IndexFlatL2 子集内暴力计算 |

评测方法：扫描 search_L ∈ {20,40,80,160,320,640}，绘制 Recall–QPS 曲线，在相同 Recall 处比较 QPS。

## License

基于 NSG (MIT) 修改，本实验代码同样 MIT 授权。
