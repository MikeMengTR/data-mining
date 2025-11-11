#!/bin/bash
# 一键复现脚本

echo "======================================"
echo "数据挖掘课程设计 - 一键复现"
echo "======================================"

# 1. 数据预处理
echo "[1/3] 数据预处理..."
cd data_preprocessing
python preprocess_pipeline.py
cd ..

# 2. 基线方法
echo "[2/3] 运行基线聚类方法..."
cd baseline_methods
python run_baseline.py
cd ..

# 3. 进阶方法
echo "[3/3] 运行进阶聚类方法..."
cd advanced_methods
python run_advanced.py
cd ..

echo "======================================"
echo "运行完成！结果已保存到对应目录。"
echo "======================================"

