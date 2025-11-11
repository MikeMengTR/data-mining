# 数据预处理模块

## 功能说明
本模块负责原始数据的加载、清洗、转换和降维处理。

## 主要脚本
- `load_data.py`: 数据加载
- `dimensionality_reduction.py`: 降维处理（UMAP/t-SNE）
- `data_validation.py`: 数据质量检查

## 数据流程
1. 加载原始768维向量
2. 数据验证（缺失值、异常值检测）
3. 降维到2D/3D用于可视化
4. 保存处理后的数据

## 使用方法
```python
python preprocess_pipeline.py
```

## 输出
- `data/processed/`: 处理后的数据
- `data/figures/`: 数据分布可视化

