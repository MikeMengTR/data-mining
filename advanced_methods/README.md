# 进阶聚类方法

## 方法说明
实现适配音乐编码数据特点的进阶聚类算法。

## 实现的算法
- **HDBSCAN**: 分层密度聚类（处理不规则形状）
- **Spectral Clustering**: 谱聚类（处理非凸簇）
- **GMM**: 高斯混合模型（概率聚类）

## 主要脚本
- `hdbscan_clustering.py`: HDBSCAN实现
- `spectral_clustering.py`: 谱聚类实现
- `parameter_search.py`: 超参数搜索
- `advanced_evaluation.py`: 深度评估分析

## 消融实验
- 降维方法对比（UMAP vs t-SNE）
- 超参数敏感性分析
- 噪声鲁棒性测试

## 运行方式
```bash
python run_advanced.py --method hdbscan
python parameter_search.py --search_space config.yaml
```

## 结果
详见 `results/advanced/` 目录

