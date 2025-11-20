# 音乐Token Embedding聚类分析

基于UMAP降维和HDBSCAN密度聚类的音乐Token序列分析项目

## 📋 项目概述

本项目使用Transformer编码器-解码器结构对符号音乐数据进行处理，提取83,362个768维的hidden state向量，并通过UMAP降维和HDBSCAN密度聚类对这些数据进行深入分析。

### 研究目标

- 对83,362个768维音乐token embedding进行聚类分析
- 目标簇数：512个
- 探索音乐数据中的潜在结构和模式
- 评估不同参数对聚类效果的影响

## 🗂️ 项目结构

```
FinalWork/
├── data/
│   ├── bos_vectors_dim_83362_768.npy    # 原始768维数据
│   ├── measure_bos_tsne.png              # 初步t-SNE可视化
│   └── read_me.txt                       # 数据说明
├── clustering_analysis.py                 # 主分析脚本
├── advanced_parameter_search.py           # 高级参数搜索脚本
├── compare_results.py                     # 结果对比工具
├── requirements.txt                       # Python依赖
└── README.md                              # 项目文档
```

## 🔧 环境配置

### 系统要求

- Python 3.8+
- 16GB+ RAM（建议）
- 多核CPU（加速聚类）

### 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 依赖包说明

- **numpy**: 数值计算
- **umap-learn**: UMAP降维算法
- **hdbscan**: 基于密度的层次聚类
- **scikit-learn**: 评估指标计算
- **matplotlib, seaborn**: 数据可视化
- **pandas**: 数据处理（参数搜索）
- **tqdm**: 进度条显示

## 🚀 使用指南

### 1. 基础单次分析

运行主分析脚本，使用预设参数进行聚类：

```bash
python clustering_analysis.py
```

**交互提示：**
- 选择 `n` 进行单次分析（使用默认参数）
- 选择 `y` 进行参数搜索模式

**输出结果：**
- `results/run_YYYYMMDD_HHMMSS/`
  - `cluster_labels.npy`: 聚类标签
  - `data_reduced.npy`: UMAP降维后的数据
  - `data_2d.npy`: 2D可视化数据
  - `clustering_visualization.png`: 可视化图表
  - `summary.json`: 结果摘要
  - `analysis_report.txt`: 文本报告

### 2. 高级参数搜索

全面搜索参数空间，寻找最优参数组合：

```bash
python advanced_parameter_search.py
```

**搜索参数空间：**
- UMAP降维维度: 50, 75, 100
- UMAP邻居数: 15, 30
- UMAP最小距离: 0.0, 0.1
- HDBSCAN最小簇大小: 50, 80, 100, 120, 150
- HDBSCAN最小样本数: 5, 10, 15, 20
- 簇选择阈值: 0.0, 0.1, 0.2

**输出结果：**
- `parameter_search_results/search_YYYYMMDD_HHMMSS/`
  - `search_results.csv`: 所有实验结果
  - `top10_results.csv`: Top 10最佳参数组合
  - `parameter_search_visualization.png`: 参数分析图
  - `search_summary.json`: 搜索摘要

### 3. 结果对比分析

对比多次运行的结果：

```bash
python compare_results.py
```

**使用方式：**
- 自动查找`results/`目录下所有运行结果
- 或手动输入多个结果目录路径

**输出结果：**
- `comparison_results.png`: 对比可视化
- `metrics_comparison.png`: 指标对比图
- `comparison_report.txt`: 对比报告

## 📊 方法说明

### 方案B：密度聚类流程

```
原始数据 (83362 × 768)
    ↓
UMAP降维 (→ 50-100维)
    ↓
HDBSCAN聚类
    ↓
评估与可视化
```

### UMAP降维

**优势：**
- 保留全局和局部结构
- 速度快于t-SNE
- 适合大规模数据

**关键参数：**
- `n_components`: 降维后维度（50-100）
- `n_neighbors`: 局部邻域大小（15-50）
- `min_dist`: 点间最小距离（0.0-0.2）

### HDBSCAN聚类

**优势：**
- 自动确定簇数
- 基于密度，能识别任意形状
- 能检测噪声点

**关键参数：**
- `min_cluster_size`: 簇的最小样本数（影响簇数量）
- `min_samples`: 核心点的最小邻居数（影响噪声比例）
- `cluster_selection_epsilon`: 合并阈值（较大值减少簇数）

### 评估指标

1. **Silhouette Score** (轮廓系数)
   - 范围: [-1, 1]
   - 越接近1越好，表示簇内紧密且簇间分离

2. **Davies-Bouldin Index** (DB指数)
   - 范围: [0, +∞)
   - 越小越好，0为理想值

3. **Calinski-Harabasz Index** (CH指数)
   - 范围: [0, +∞)
   - 越大越好，表示簇间方差/簇内方差比值

## 📈 参数调优建议

### 增加簇数
- ✅ 减小 `min_cluster_size` (如: 100 → 50)
- ✅ 减小 `min_samples` (如: 20 → 10)
- ✅ 减小 `cluster_selection_epsilon` (如: 0.2 → 0.0)
- ✅ 增加 UMAP `n_neighbors` (如: 15 → 50)

### 减少噪声点
- ✅ 增大 `min_samples`
- ✅ 增大 `min_cluster_size`
- ✅ 减小 UMAP `min_dist`

### 提高聚类质量
- ✅ 尝试不同的UMAP降维维度（50-100）
- ✅ 使用cosine距离度量（UMAP）
- ✅ 观察Silhouette Score变化

## 🎯 典型工作流程

### 新手流程

```bash
# 1. 快速体验
python clustering_analysis.py
# 选择 n（单次分析）

# 2. 查看结果
# 打开 results/run_*/clustering_visualization.png
# 阅读 results/run_*/analysis_report.txt
```

### 研究流程

```bash
# 1. 参数搜索
python advanced_parameter_search.py
# 等待完成（可能需要1-2小时）

# 2. 查看Top 10参数
# 打开 parameter_search_results/search_*/top10_results.csv

# 3. 使用最佳参数重新运行
python clustering_analysis.py
# 在代码中修改参数或使用参数搜索模式

# 4. 对比多次结果
python compare_results.py
```

## 📝 常见问题

### Q1: 内存不足怎么办？

**解决方案：**
- 减小UMAP降维维度
- 使用数据采样（修改代码加载部分）
- 增加系统swap空间

### Q2: 簇数远少于512怎么办？

**调整策略：**
1. 减小 `min_cluster_size`: 100 → 50 → 30
2. 减小 `min_samples`: 10 → 5
3. 设置 `cluster_selection_epsilon = 0.0`
4. 增加 UMAP `n_neighbors`: 15 → 30 → 50

### Q3: 噪声点太多怎么办？

**调整策略：**
1. 增大 `min_samples`: 5 → 10 → 20
2. 增大 `min_cluster_size`
3. 调整 UMAP `min_dist` 到 0.0

### Q4: 运行太慢怎么办？

**优化方案：**
- 确保安装了 `numba` (HDBSCAN依赖)
- 使用多核CPU（`core_dist_n_jobs=-1`已启用）
- 减少参数搜索空间
- 考虑使用GPU加速UMAP（需安装cuml）

## 🔬 进阶功能

### 自定义参数空间

修改 `advanced_parameter_search.py` 中的参数定义：

```python
umap_params = {
    'n_components': [60, 80],  # 自定义维度
    'n_neighbors': [20, 40],   # 自定义邻居数
    'min_dist': [0.05, 0.15]   # 自定义距离
}

hdbscan_params = {
    'min_cluster_size': [40, 60, 80],  # 针对512簇优化
    'min_samples': [8, 12, 16],
    'cluster_selection_epsilon': [0.0, 0.05, 0.1]
}
```

### 使用已有标签验证

如果有ground truth标签（如high voice / low voice）：

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# 计算外部评估指标
ari = adjusted_rand_score(true_labels, predicted_labels)
nmi = normalized_mutual_info_score(true_labels, predicted_labels)
```

### 簇内分析

提取特定簇的样本进行深入分析：

```python
# 加载结果
labels = np.load('results/run_*/cluster_labels.npy')

# 获取簇5的所有样本索引
cluster_5_indices = np.where(labels == 5)[0]

# 分析该簇的音乐特征
cluster_5_data = original_data[cluster_5_indices]
```

## 📚 参考资料

- **UMAP论文**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.
- **HDBSCAN论文**: Campello, R. J., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates.
- **UMAP文档**: https://umap-learn.readthedocs.io/
- **HDBSCAN文档**: https://hdbscan.readthedocs.io/

## 📧 联系方式

如有问题或建议，请联系项目团队。

## 📄 许可证

本项目仅用于学术研究和教育目的。

---

**祝您分析愉快！** 🎵📊
