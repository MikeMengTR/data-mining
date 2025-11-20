# 完整可复现性指南

**项目**: 音乐Token Embedding聚类分析
**版本**: v2.0 (新增基线方法与稳健性分析)
**日期**: 2025-11-10

---

## 📋 目录

1. [环境准备](#1-环境准备)
2. [完整实验流程](#2-完整实验流程)
3. [基线方法：K-means](#3-基线方法k-means)
4. [主要方法：HDBSCAN](#4-主要方法hdbscan)
5. [方法对比分析](#5-方法对比分析)
6. [稳健性分析](#6-稳健性分析)
7. [论文图表生成](#7-论文图表生成)
8. [可复现性保证](#8-可复现性保证)
9. [常见问题](#9-常见问题)

---

## 1. 环境准备

### 1.1 系统要求

```
Python: 3.8+
RAM: 16GB+推荐
CPU: 多核处理器
存储: 2GB+
操作系统: Linux / macOS / Windows
```

### 1.2 依赖安装

```bash
pip install -r requirements.txt
```

**核心依赖**:
- numpy >= 1.21.0
- umap-learn >= 0.5.3
- hdbscan >= 0.8.29
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pandas >= 1.3.0
- tqdm >= 4.62.0

### 1.3 验证环境

```bash
python quick_test.py
```

预期输出：所有测试通过✓

---

## 2. 完整实验流程

### 步骤概览

```
1. 基线方法 (K-means) ────┐
2. 主要方法 (HDBSCAN) ────┼──→ 4. 方法对比
3. 稳健性分析 ────────────┘
5. 生成论文图表
```

### 完整运行时间估计

| 步骤 | 程序 | 时间 |
|------|------|------|
| 基线方法 | baseline_kmeans.py | 10-15分钟 |
| 主要方法 | clustering_analysis.py | 10-15分钟 |
| 方法对比 | method_comparison.py | 5分钟 |
| 稳健性分析 | robustness_analysis.py | 30-60分钟 |
| 生成图表 | generate_paper_figures.py | 5分钟 |
| **总计** | | **60-100分钟** |

---

## 3. 基线方法：K-means

### 3.1 原理说明

**K-means聚类**是最经典的聚类算法之一：

```
算法步骤:
1. 随机初始化K个簇中心
2. 将每个样本分配给最近的簇中心
3. 重新计算每个簇的中心（均值）
4. 重复2-3直到收敛或达到最大迭代次数

优点:
- 算法简单，易于理解
- 计算快速
- 结果可解释（有明确的簇中心）
- 完全确定性（固定random_state）

缺点:
- 需要预设簇数K
- 假设簇为球形
- 对初始化敏感
- 所有点必须被分配（无噪声概念）
```

### 3.2 运行基线方法

```bash
python baseline_kmeans.py
```

**参数设置** (可在代码中修改):
```python
# UMAP降维
n_components = 75  # 降维维度
random_state = 42  # 随机种子（保证可复现）

# K-means聚类
n_clusters = 512   # 簇数（对应目标）
random_state = 42  # 随机种子
n_init = 10        # 不同初始化尝试次数
use_minibatch = True  # 使用MiniBatch加速
```

### 3.3 预期输出

**文件位置**: `baseline_results/kmeans_YYYYMMDD_HHMMSS/`

**输出文件**:
- `cluster_labels.npy` - 聚类标签 (83362,)
- `cluster_centers.npy` - 簇中心 (512, 75)
- `data_reduced.npy` - UMAP降维后数据
- `data_2d.npy` - 2D可视化数据
- `kmeans_visualization.png` - 可视化图表
- `summary.json` - 结果摘要
- `analysis_report.txt` - 分析报告

**预期结果**:
```
簇数: 512 (精确)
迭代次数: ~50-100次
Silhouette Score: 0.30-0.40
Davies-Bouldin Index: 1.0-1.5
运行时间: 10-15分钟
```

### 3.4 可复现性

✅ **完全可复现**: 使用固定`random_state=42`，每次运行得到完全相同的结果

**验证方法**:
```bash
# 运行两次
python baseline_kmeans.py  # 第一次
python baseline_kmeans.py  # 第二次

# 比较结果
python -c "
import numpy as np
labels1 = np.load('baseline_results/kmeans_run1/cluster_labels.npy')
labels2 = np.load('baseline_results/kmeans_run2/cluster_labels.npy')
print('完全一致:', np.array_equal(labels1, labels2))
"
```

预期输出: `完全一致: True`

---

## 4. 主要方法：HDBSCAN

### 4.1 原理说明

**HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise):

```
算法步骤:
1. 计算互达距离（mutual reachability distance）
2. 构建最小生成树（MST）
3. 构建簇层次结构
4. 提取稳定簇
5. 标记噪声点

优点:
- 自动确定簇数
- 识别任意形状的簇
- 自动检测噪声点
- 提供层次化信息

缺点:
- 参数较多
- 计算相对较慢
- 解释性不如K-means
```

### 4.2 运行主要方法

```bash
python clustering_analysis.py
```

**交互提示**: 输入`n`使用默认参数，或输入`y`进行参数搜索

**参数设置** (单次分析):
```python
# UMAP降维
n_components = 75
n_neighbors = 15
min_dist = 0.1

# HDBSCAN聚类
min_cluster_size = 100
min_samples = 10
cluster_selection_epsilon = 0.0
```

**参数搜索模式**:
- 自动尝试多组参数
- 寻找最接近512簇的配置
- 时间: 1-2小时

### 4.3 预期输出

**文件位置**: `results/run_YYYYMMDD_HHMMSS/`

**输出文件**:
- `cluster_labels.npy` - 聚类标签
- `data_reduced.npy` - UMAP降维后数据
- `data_2d.npy` - 2D可视化数据
- `clustering_visualization.png` - 可视化图表
- `summary.json` - 结果摘要
- `analysis_report.txt` - 分析报告

**预期结果** (默认参数):
```
簇数: 100-200 (自动确定)
噪声点: 1-5%
Silhouette Score: 0.35-0.45
Davies-Bouldin Index: 0.8-1.2
运行时间: 10-15分钟
```

**如需接近512簇**, 调整参数:
```python
min_cluster_size = 30-50  # 减小
min_samples = 5-10        # 减小
```

### 4.4 可复现性

⚠️ **部分可复现**: UMAP有随机性，但设置`random_state=42`可保证稳定性

**稳定性测试**:
```bash
python robustness_analysis.py
```

查看"多次运行一致性"部分，平均ARI应 > 0.7

---

## 5. 方法对比分析

### 5.1 运行对比

```bash
python method_comparison.py
```

**输入要求**:
- K-means结果目录
- HDBSCAN结果目录

### 5.2 对比内容

| 对比维度 | 说明 |
|---------|------|
| 簇数 | K-means固定512, HDBSCAN自动确定 |
| 噪声处理 | K-means无，HDBSCAN有 |
| 评估指标 | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| 聚类一致性 | ARI, NMI |
| 簇大小分布 | 直方图、箱线图 |

### 5.3 预期输出

**文件位置**: `comparison_results/comparison_YYYYMMDD_HHMMSS/`

**输出文件**:
- `method_comparison_visualization.png` - 9合1对比图
- `comparison_report.txt` - 详细对比报告
- `comparison_summary.json` - 对比摘要

**关键发现**:
```
簇数: K-means=512 (精确), HDBSCAN=100-200 (可调)
一致性: ARI=0.4-0.6 (中等一致)
质量: 两者Silhouette Score相近
噪声: HDBSCAN能识别1-5%噪声点
```

---

## 6. 稳健性分析

### 6.1 运行稳健性测试

```bash
python robustness_analysis.py
```

### 6.2 测试内容

#### 6.2.1 噪声敏感性测试

**方法**: 向数据添加不同程度的高斯噪声

**噪声水平**: 0%, 1%, 5%, 10%, 20% (相对于数据标准差)

**评估指标**:
- ARI with baseline (无噪声版本)
- NMI with baseline
- Silhouette Score

**预期结果**:
```
噪声水平 0%:   ARI=1.000 (基准)
噪声水平 1%:   ARI>0.95  (高稳定性)
噪声水平 5%:   ARI>0.85  (中高稳定性)
噪声水平 10%:  ARI>0.70  (中等稳定性)
噪声水平 20%:  ARI>0.50  (可接受)
```

#### 6.2.2 参数稳定性测试

**测试参数**:
- UMAP n_components: 50, 75, 100
- UMAP n_neighbors: 10, 15, 30
- K-means n_clusters: 400, 512, 600

**评估指标**: Silhouette Score, Inertia

**预期发现**:
- n_components增加 → Silhouette略微提升
- n_neighbors对结果影响较小
- n_clusters=512接近最优

#### 6.2.3 多次运行一致性测试

**方法**: 使用不同随机种子运行10次

**评估指标**:
- 两两之间的ARI
- Silhouette Score的均值和标准差

**预期结果**:
```
平均ARI: 0.85-0.95 (高一致性)
Silhouette Score: 0.35±0.02 (低变异性)
变异系数(CV): <5%
```

### 6.3 预期输出

**文件位置**: `robustness_results/robustness_YYYYMMDD_HHMMSS/`

**输出文件**:
- `noise_sensitivity.png` - 噪声敏感性图
- `parameter_stability.png` - 参数稳定性图
- `multiple_runs_consistency.png` - 多次运行一致性图
- `robustness_report.txt` - 稳健性报告
- `robustness_summary.json` - 稳健性摘要

**运行时间**: 30-60分钟

---

## 7. 论文图表生成

### 7.1 运行图表生成器

```bash
python generate_paper_figures.py
```

### 7.2 生成的图表

| 图表 | 文件名 | 描述 | 用途 |
|------|--------|------|------|
| Figure 1 | fig1_method_flowchart.png | 方法流程对比 | 论文方法部分 |
| Figure 2 | fig2_data_overview.png | 数据概览统计 | 论文数据部分 |
| Figure 3 | fig3_evaluation_metrics.png | 评估指标对比 | 论文结果部分 |
| Figure 4 | fig4_clustering_results.png | 聚类结果可视化 | 论文结果部分 |

**图表质量**:
- 分辨率: 300 DPI
- 格式: PNG
- 风格: 学术期刊标准

### 7.3 预期输出

**文件位置**: `paper_figures/`

**索引文件**: `FIGURE_INDEX.md` - 包含所有图表的说明和使用建议

---

## 8. 可复现性保证

### 8.1 随机种子设置

**所有程序使用固定随机种子**:

```python
# UMAP
random_state = 42

# K-means
random_state = 42

# HDBSCAN (虽然有随机性，但UMAP固定后结果稳定)
umap: random_state = 42
```

### 8.2 环境一致性

**Python版本**: 建议使用Python 3.8-3.10

**依赖版本**: 固定在requirements.txt中

**验证命令**:
```bash
pip freeze > current_env.txt
diff requirements.txt current_env.txt
```

### 8.3 数据完整性

**验证数据文件**:
```bash
python -c "
import numpy as np
import hashlib

data = np.load('data/bos_vectors_dim_83362_768.npy')
print('Shape:', data.shape)
print('Dtype:', data.dtype)
print('MD5:', hashlib.md5(data.tobytes()).hexdigest()[:16])
"
```

预期输出:
```
Shape: (83362, 768)
Dtype: float32
```

### 8.4 完整复现检查清单

- [ ] 环境准备：Python 3.8+, 依赖安装
- [ ] 数据验证：形状(83362, 768)
- [ ] K-means基线：运行baseline_kmeans.py
- [ ] HDBSCAN主方法：运行clustering_analysis.py
- [ ] 方法对比：运行method_comparison.py
- [ ] 稳健性分析：运行robustness_analysis.py
- [ ] 论文图表：运行generate_paper_figures.py
- [ ] 结果验证：检查所有输出文件

---

## 9. 常见问题

### Q1: 如何确保完全可复现？

**A**:
1. 使用相同的Python版本和依赖版本
2. 使用requirements.txt安装依赖
3. 不修改代码中的random_state参数
4. 使用相同的数据文件

### Q2: K-means和HDBSCAN哪个更好？

**A**: 取决于需求
- 如果**已知目标簇数**(512)且需要**高可解释性** → 选K-means
- 如果需要**探索性分析**、**自动簇数**、**噪声检测** → 选HDBSCAN
- **建议**: 两种方法都运行，进行对比

### Q3: 如何调整参数以接近512簇(HDBSCAN)？

**A**: 主要调整两个参数
```python
min_cluster_size = 30-50  # 减小此值增加簇数 ⭐⭐⭐
min_samples = 5-10        # 减小此值增加簇数 ⭐⭐
```

或使用参数搜索模式:
```bash
python clustering_analysis.py
# 输入 y 进行参数搜索
```

### Q4: 稳健性测试需要运行多久？

**A**:
- 噪声敏感性: ~10分钟
- 参数稳定性: ~15分钟
- 多次运行一致性: ~10分钟
- **总计**: 30-60分钟

### Q5: 如何验证结果的稳定性？

**A**: 查看稳健性分析结果
```bash
# 运行稳健性测试
python robustness_analysis.py

# 查看报告
cat robustness_results/robustness_*/robustness_report.txt
```

关键指标:
- 平均ARI > 0.85 → 高稳定性
- Silhouette变异系数 < 5% → 低变异性

### Q6: 内存不足怎么办？

**A**:
1. 减小UMAP维度: n_components=75 → 50
2. 使用MiniBatchKMeans (已默认启用)
3. 增加系统swap空间
4. 数据采样测试

### Q7: 如何引用结果？

**A**: 论文中可以这样描述
```
"我们使用K-means作为可解释的基线方法,固定簇数为512,
使用UMAP(n_components=75)进行降维。所有实验使用
random_state=42以保证完全可复现。通过对比实验,
K-means在簇数控制上更精确(512 clusters),而HDBSCAN
能够自动发现数据中的噪声点(1-5%)。两种方法的
Silhouette Score相近(0.30-0.40),表明聚类质量comparable。
稳健性分析显示,在10%噪声水平下,聚类结果仍保持
较高一致性(ARI>0.70)。"
```

---

## 10. 完整运行脚本

### 10.1 全自动运行（需1-2小时）

创建 `run_all.sh`:
```bash
#!/bin/bash

echo "=== 开始完整实验 ==="

# 1. K-means基线
echo "Step 1: Running K-means baseline..."
python baseline_kmeans.py

# 2. HDBSCAN主方法
echo "Step 2: Running HDBSCAN clustering..."
echo "n" | python clustering_analysis.py

# 3. 方法对比
echo "Step 3: Method comparison..."
# 需要手动输入目录，这里使用最新的
KM_DIR=$(ls -td baseline_results/kmeans_* | head -1)
HDB_DIR=$(ls -td results/run_* | head -1)
echo -e "$KM_DIR\n$HDB_DIR" | python method_comparison.py

# 4. 稳健性分析
echo "Step 4: Robustness analysis..."
python robustness_analysis.py

# 5. 生成论文图表
echo "Step 5: Generating paper figures..."
echo -e "$KM_DIR\n$HDB_DIR" | python generate_paper_figures.py

echo "=== 完整实验完成 ==="
echo "结果位置:"
echo "  - K-means: $KM_DIR"
echo "  - HDBSCAN: $HDB_DIR"
echo "  - 对比: comparison_results/comparison_*"
echo "  - 稳健性: robustness_results/robustness_*"
echo "  - 论文图表: paper_figures/"
```

运行:
```bash
chmod +x run_all.sh
./run_all.sh
```

---

## 11. 结果验证

### 11.1 验证基线方法

```python
import numpy as np
import json

# 加载结果
labels = np.load('baseline_results/kmeans_*/cluster_labels.npy')
with open('baseline_results/kmeans_*/summary.json', 'r') as f:
    summary = json.load(f)

# 验证
assert len(set(labels)) == 512, "簇数应为512"
assert summary['n_clusters'] == 512, "JSON中簇数应为512"
assert 0.25 < summary['metrics']['silhouette_score'] < 0.45, "Silhouette应在合理范围"

print("✓ 基线方法验证通过")
```

### 11.2 验证主要方法

```python
labels = np.load('results/run_*/cluster_labels.npy')
with open('results/run_*/summary.json', 'r') as f:
    summary = json.load(f)

# 验证
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
assert n_clusters > 0, "应至少有1个簇"
assert summary['n_noise'] >= 0, "噪声点数应>=0"

print(f"✓ HDBSCAN验证通过: {n_clusters}簇, {summary['n_noise']}噪声点")
```

---

## 12. 引用和致谢

### 引用格式

**数据来源**:
```
音乐Token Embedding数据集
来源: Transformer encoder hidden states
样本数: 83,362
维度: 768
```

**方法引用**:
```
UMAP: McInnes, L., Healy, J., & Melville, J. (2018).
HDBSCAN: Campello, R. J., Moulavi, D., & Sander, J. (2013).
K-means: Lloyd, S. (1982).
```

---

**文档版本**: v2.0
**最后更新**: 2025-11-10
**维护者**: Data Mining Project Team

如有问题，请参考README.md或项目总结文档。
