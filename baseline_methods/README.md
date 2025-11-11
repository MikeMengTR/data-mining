# 基线聚类方法

## 方法说明
实现简单可解释的基线聚类算法，作为性能对比基准。

## 实现的算法
- **K-means**: 经典聚类算法
- **层次聚类**: 凝聚式层次聚类
- **DBSCAN**: 基于密度的聚类

## 主要脚本
- `kmeans_clustering.py`: K-means实现
- `hierarchical_clustering.py`: 层次聚类
- `dbscan_clustering.py`: DBSCAN实现
- `evaluate_baseline.py`: 基线评估

## 评估指标
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

## 运行方式
```bash
python run_baseline.py --method kmeans --n_clusters 100
```

## 结果
详见 `results/baseline/` 目录

