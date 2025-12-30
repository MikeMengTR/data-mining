# 数据挖掘课程设计 - 音乐编码聚类分析

## 项目概述
本项目针对音乐编码数据进行聚类分析，旨在探索高维音乐特征的内在结构，并对比多种聚类算法的表现。代码包含数据预处理、基线方法以及进阶方法三大模块，可用于复现课程实验或作为后续研究的起点。

## 环境配置
1. 建议使用虚拟环境隔离依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 数据准备
- 数据集格式及获取方式见 [DATASET.md](DATASET.md)。
- 将原始数据放置到 `data_preprocessing` 模块期望的输入目录（参见其中的注释），预处理脚本会输出清洗后的特征文件供后续步骤使用。

## 运行流程
按模块顺序运行即可完成完整实验流程：

1. **数据预处理**
   ```bash
   cd data_preprocessing
   python preprocess_pipeline.py
   cd ..
   ```
   输出：标准化与特征工程后的数据文件。

2. **基线聚类方法**（K-means 等）
   ```bash
   cd baseline_methods
   python run_baseline.py
   cd ..
   ```
   输出：基线模型的聚类结果与评估指标。

3. **进阶聚类方法**（HDBSCAN 等）
   ```bash
   cd advanced_methods
   python run_advanced.py
   cd ..
   ```
   输出：进阶模型的聚类结果与评估指标，可与基线方法对比。

如需自定义参数，可直接在对应脚本中调整配置或添加命令行参数。

## 项目结构
```
├── data_preprocessing/          # 数据预处理
├── baseline_methods/            # 基础聚类方法（K-means等）
├── advanced_methods/            # 进阶聚类方法（HDBSCAN等）
├── latex/                       # LaTeX报告
├── DATASET.md                   # 数据集说明
├── CONTRIBUTION.md              # 贡献指南
├── AI_USAGE.md                  # AI 工具使用声明
└── requirements.txt             # 依赖包
```

## 团队贡献
详见 [CONTRIBUTION.md](CONTRIBUTION.md)。

## AI工具使用声明
详见 [AI_USAGE.md](AI_USAGE.md)。

