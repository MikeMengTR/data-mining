#!/usr/bin/env python3
"""
论文图表生成器
自动生成各种高质量图表用于论文写作

生成的图表包括:
1. 方法流程图
2. 数据分布可视化
3. 聚类结果对比
4. 评估指标对比
5. 稳健性分析图表
6. 参数影响分析
7. 详细的统计图表
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.dpi'] = 150  # 高质量


class PaperFigureGenerator:
    """论文图表生成器"""

    def __init__(self, output_dir: str = "paper_figures"):
        """
        初始化图表生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"论文图表生成器初始化完成")
        print(f"图表保存到: {self.output_dir}")

    def generate_method_flowchart(self):
        """生成方法流程图"""
        print("\n生成方法流程图...")

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')

        # 定义流程框
        boxes = [
            # 数据输入
            {'text': 'Input Data\n83,362 × 768-dim\nMusic Token Embeddings',
             'pos': (0.5, 0.95), 'color': 'lightblue'},

            # 方法A: HDBSCAN
            {'text': 'Method A:\nUMAP + HDBSCAN\n(Density-based)', 'pos': (0.25, 0.80), 'color': 'coral'},
            {'text': 'UMAP\nReduction\n768D → 50-100D', 'pos': (0.25, 0.65), 'color': 'lightcoral'},
            {'text': 'HDBSCAN\nClustering\nAuto K', 'pos': (0.25, 0.50), 'color': 'lightcoral'},
            {'text': 'Result A\nK ≈ 512 clusters\n+ Noise detection', 'pos': (0.25, 0.35), 'color': 'lightcoral'},

            # 方法B: K-means
            {'text': 'Method B:\nUMAP + K-means\n(Baseline)', 'pos': (0.75, 0.80), 'color': 'lightgreen'},
            {'text': 'UMAP\nReduction\n768D → 50-100D', 'pos': (0.75, 0.65), 'color': 'palegreen'},
            {'text': 'K-means\nClustering\nK = 512', 'pos': (0.75, 0.50), 'color': 'palegreen'},
            {'text': 'Result B\n512 clusters\nAll points assigned', 'pos': (0.75, 0.35), 'color': 'palegreen'},

            # 评估
            {'text': 'Evaluation & Comparison\n• Silhouette Score\n• Davies-Bouldin Index\n• Robustness Analysis',
             'pos': (0.5, 0.15), 'color': 'wheat'}
        ]

        for box in boxes:
            bbox = dict(boxstyle='round,pad=0.8', facecolor=box['color'], edgecolor='black', linewidth=2)
            ax.text(box['pos'][0], box['pos'][1], box['text'],
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   bbox=bbox, transform=ax.transAxes)

        # 添加箭头
        arrows = [
            # 主流程
            ((0.5, 0.93), (0.25, 0.82)),
            ((0.5, 0.93), (0.75, 0.82)),

            # Method A
            ((0.25, 0.78), (0.25, 0.67)),
            ((0.25, 0.63), (0.25, 0.52)),
            ((0.25, 0.48), (0.25, 0.37)),
            ((0.25, 0.33), (0.5, 0.17)),

            # Method B
            ((0.75, 0.78), (0.75, 0.67)),
            ((0.75, 0.63), (0.75, 0.52)),
            ((0.75, 0.48), (0.75, 0.37)),
            ((0.75, 0.33), (0.5, 0.17)),
        ]

        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        plt.title('Clustering Methods Comparison Framework',
                 fontsize=16, fontweight='bold', pad=20)

        save_path = self.output_dir / "fig1_method_flowchart.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ 已保存: {save_path}")
        plt.close()

    def generate_data_overview_figure(self, data_path: str):
        """生成数据概览图"""
        print("\n生成数据概览图...")

        data = np.load(data_path)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. 数据维度分布热图（前50维）
        ax = axes[0, 0]
        sample_idx = np.random.choice(len(data), 1000, replace=False)
        im = ax.imshow(data[sample_idx, :50].T, cmap='viridis', aspect='auto')
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Feature Dimension', fontsize=11)
        ax.set_title('(a) Feature Values Heatmap\n(1000 samples × 50 dims)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Value')

        # 2. 特征值分布
        ax = axes[0, 1]
        ax.hist(data.flatten(), bins=100, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Feature Value', fontsize=11)
        ax.set_ylabel('Frequency (log scale)', fontsize=11)
        ax.set_yscale('log')
        ax.set_title('(b) Feature Value Distribution', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # 3. 维度方差图
        ax = axes[0, 2]
        variances = data.var(axis=0)
        ax.plot(variances, linewidth=1.5, color='coral')
        ax.set_xlabel('Dimension Index', fontsize=11)
        ax.set_ylabel('Variance', fontsize=11)
        ax.set_title('(c) Per-dimension Variance', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        # 4. 样本L2范数分布
        ax = axes[1, 0]
        norms = np.linalg.norm(data, axis=1)
        ax.hist(norms, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax.axvline(norms.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {norms.mean():.2f}')
        ax.set_xlabel('L2 Norm', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('(d) Sample L2 Norm Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 5. 样本间余弦相似度（采样）
        ax = axes[1, 1]
        sample_idx = np.random.choice(len(data), 500, replace=False)
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(data[sample_idx])
        im = ax.imshow(sim_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xlabel('Sample Index', fontsize=11)
        ax.set_ylabel('Sample Index', fontsize=11)
        ax.set_title('(e) Cosine Similarity Matrix\n(500 samples)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')

        # 6. 数据基本统计
        ax = axes[1, 2]
        ax.axis('off')
        stats_text = f"""
Data Statistics

Total Samples: {len(data):,}
Feature Dimension: {data.shape[1]}
Data Type: {data.dtype}

Value Range:
  Min: {data.min():.4f}
  Max: {data.max():.4f}
  Mean: {data.mean():.4f}
  Std: {data.std():.4f}
  Median: {np.median(data):.4f}

Memory: {data.nbytes / 1024 / 1024:.2f} MB
"""
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle('Figure 2: Input Data Overview and Statistics',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = self.output_dir / "fig2_data_overview.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path}")
        plt.close()

    def generate_evaluation_metrics_figure(self, kmeans_summary: dict, hdbscan_summary: dict):
        """生成评估指标对比图"""
        print("\n生成评估指标对比图...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 准备数据
        methods = ['K-means\n(Baseline)', 'HDBSCAN\n(Proposed)']

        # 1. 簇数对比
        ax = axes[0, 0]
        n_clusters = [kmeans_summary['n_clusters'], hdbscan_summary['n_clusters']]
        bars = ax.bar(methods, n_clusters, color=['lightblue', 'coral'], alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        ax.axhline(512, color='red', linestyle='--', linewidth=2, label='Target: 512')
        ax.set_ylabel('Number of Clusters', fontsize=12, fontweight='bold')
        ax.set_title('(a) Cluster Count Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars, n_clusters):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

        # 2. Silhouette Score对比
        ax = axes[0, 1]
        km_sil = kmeans_summary['metrics'].get('silhouette_score', 0)
        hdb_sil = hdbscan_summary.get('results', {}).get('metrics', {}).get('silhouette_score', 0)
        sil_scores = [km_sil, hdb_sil]
        bars = ax.bar(methods, sil_scores, color=['lightblue', 'coral'], alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
        ax.set_title('(b) Silhouette Score Comparison', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(sil_scores) * 1.2)
        ax.grid(axis='y', alpha=0.3)

        for bar, score in zip(bars, sil_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 3. Davies-Bouldin Index对比
        ax = axes[1, 0]
        km_db = kmeans_summary['metrics'].get('davies_bouldin_score', 0)
        hdb_db = hdbscan_summary.get('results', {}).get('metrics', {}).get('davies_bouldin_score', 0)
        db_scores = [km_db, hdb_db]
        bars = ax.bar(methods, db_scores, color=['lightblue', 'coral'], alpha=0.8,
                     edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
        ax.set_title('(c) Davies-Bouldin Index\n(lower is better)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        for bar, score in zip(bars, db_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 4. 综合对比表
        ax = axes[1, 1]
        ax.axis('off')

        comparison_data = [
            ['Metric', 'K-means', 'HDBSCAN', 'Better'],
            ['Cluster Count', f'{n_clusters[0]}', f'{n_clusters[1]}', 'HDBSCAN' if abs(n_clusters[1]-512) < abs(n_clusters[0]-512) else 'K-means'],
            ['Silhouette ↑', f'{sil_scores[0]:.4f}', f'{sil_scores[1]:.4f}', 'HDBSCAN' if sil_scores[1] > sil_scores[0] else 'K-means'],
            ['Davies-Bouldin ↓', f'{db_scores[0]:.4f}', f'{db_scores[1]:.4f}', 'HDBSCAN' if db_scores[1] < db_scores[0] else 'K-means'],
            ['Noise Detection', 'No', 'Yes', 'HDBSCAN'],
            ['Interpretability', 'High', 'Medium', 'K-means'],
            ['Speed', 'Fast', 'Medium', 'K-means']
        ]

        table = ax.table(cellText=comparison_data, loc='center',
                        cellLoc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # 设置表头样式
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 交替行颜色
        for i in range(1, len(comparison_data)):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax.set_title('(d) Comprehensive Comparison', fontsize=13, fontweight='bold', pad=10)

        plt.suptitle('Figure 3: Evaluation Metrics Comparison',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = self.output_dir / "fig3_evaluation_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path}")
        plt.close()

    def generate_clustering_results_figure(self, kmeans_dir: str, hdbscan_dir: str):
        """生成聚类结果可视化对比图"""
        print("\n生成聚类结果可视化图...")

        # 加载数据
        km_labels = np.load(f"{kmeans_dir}/cluster_labels.npy")
        km_data_2d = np.load(f"{kmeans_dir}/data_2d.npy")

        hdb_labels = np.load(f"{hdbscan_dir}/cluster_labels.npy")
        hdb_data_2d = np.load(f"{hdbscan_dir}/data_2d.npy")

        fig, axes = plt.subplots(2, 2, figsize=(18, 16))

        # 1. K-means结果
        ax = axes[0, 0]
        scatter = ax.scatter(km_data_2d[:, 0], km_data_2d[:, 1],
                           c=km_labels, cmap='Spectral', s=1, alpha=0.5)
        ax.set_xlabel('UMAP Dimension 1', fontsize=11)
        ax.set_ylabel('UMAP Dimension 2', fontsize=11)
        ax.set_title(f'(a) K-means Clustering Result\n{len(set(km_labels))} clusters',
                    fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster ID')

        # 2. HDBSCAN结果
        ax = axes[0, 1]
        scatter = ax.scatter(hdb_data_2d[:, 0], hdb_data_2d[:, 1],
                           c=hdb_labels, cmap='Spectral', s=1, alpha=0.5)
        ax.set_xlabel('UMAP Dimension 1', fontsize=11)
        ax.set_ylabel('UMAP Dimension 2', fontsize=11)
        n_clusters = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
        n_noise = (hdb_labels == -1).sum()
        ax.set_title(f'(b) HDBSCAN Clustering Result\n{n_clusters} clusters + {n_noise} noise points',
                    fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster ID (-1 = noise)')

        # 3. K-means簇大小分布
        ax = axes[1, 0]
        unique, counts = np.unique(km_labels, return_counts=True)
        sorted_counts = sorted(counts, reverse=True)[:50]
        ax.bar(range(len(sorted_counts)), sorted_counts,
              color='steelblue', alpha=0.8, edgecolor='black')
        ax.axhline(counts.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {counts.mean():.0f}')
        ax.set_xlabel('Cluster Rank (Top 50)', fontsize=11)
        ax.set_ylabel('Cluster Size', fontsize=11)
        ax.set_title('(c) K-means Cluster Size Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # 4. HDBSCAN簇大小分布
        ax = axes[1, 1]
        unique, counts = np.unique(hdb_labels[hdb_labels != -1], return_counts=True)
        sorted_counts = sorted(counts, reverse=True)[:50]
        ax.bar(range(len(sorted_counts)), sorted_counts,
              color='coral', alpha=0.8, edgecolor='black')
        ax.axhline(counts.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {counts.mean():.0f}')
        ax.set_xlabel('Cluster Rank (Top 50)', fontsize=11)
        ax.set_ylabel('Cluster Size', fontsize=11)
        ax.set_title('(d) HDBSCAN Cluster Size Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Figure 4: Clustering Results Visualization',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = self.output_dir / "fig4_clustering_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存: {save_path}")
        plt.close()

    def generate_index_file(self):
        """生成图表索引文件"""
        print("\n生成图表索引...")

        index_content = """# 论文图表索引

生成时间: {timestamp}

## 图表列表

### Figure 1: Method Flowchart
- 文件: fig1_method_flowchart.png
- 描述: 两种聚类方法的流程对比图
- 用途: 论文方法部分，展示整体框架

### Figure 2: Data Overview
- 文件: fig2_data_overview.png
- 描述: 输入数据的统计特征和分布可视化
- 用途: 论文数据部分，展示数据特性

### Figure 3: Evaluation Metrics
- 文件: fig3_evaluation_metrics.png
- 描述: K-means和HDBSCAN的评估指标对比
- 用途: 论文实验结果部分，定量对比

### Figure 4: Clustering Results
- 文件: fig4_clustering_results.png
- 描述: 两种方法的聚类结果2D可视化和簇大小分布
- 用途: 论文实验结果部分，定性对比

## 使用建议

1. **引言部分**: 使用 Figure 2 展示数据特性和挑战
2. **方法部分**: 使用 Figure 1 解释技术路线
3. **实验部分**: 使用 Figure 3 和 Figure 4 展示结果
4. **讨论部分**: 结合所有图表分析方法优劣

## 图表质量

- 分辨率: 300 DPI (适合打印和投稿)
- 格式: PNG
- 风格: 学术期刊标准

## 引用格式建议

在论文中引用时：
- "如Figure 1所示，我们对比了两种聚类方法..."
- "从Figure 3可以看出，HDBSCAN在Silhouette Score上表现更优..."
- "Figure 4(a)和(b)分别展示了K-means和HDBSCAN的聚类结果..."

""".format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

        index_path = self.output_dir / "FIGURE_INDEX.md"
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)

        print(f"✓ 图表索引已保存: {index_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("论文图表生成")
    print("="*70)

    generator = PaperFigureGenerator(output_dir="paper_figures")

    # 1. 生成方法流程图
    generator.generate_method_flowchart()

    # 2. 生成数据概览图
    generator.generate_data_overview_figure("data/bos_vectors_dim_83362_768.npy")

    # 3. 生成评估指标对比图（需要先运行聚类）
    print("\n注意: 需要先运行 baseline_kmeans.py 和 clustering_analysis.py")
    print("请输入K-means结果目录 (或按回车跳过):")
    kmeans_dir = input().strip()

    if kmeans_dir and Path(kmeans_dir).exists():
        print("请输入HDBSCAN结果目录:")
        hdbscan_dir = input().strip()

        if hdbscan_dir and Path(hdbscan_dir).exists():
            # 加载摘要
            with open(f"{kmeans_dir}/summary.json", 'r') as f:
                km_summary = json.load(f)
            with open(f"{hdbscan_dir}/summary.json", 'r') as f:
                hdb_summary = json.load(f)

            # 生成对比图
            generator.generate_evaluation_metrics_figure(km_summary, hdb_summary)
            generator.generate_clustering_results_figure(kmeans_dir, hdbscan_dir)
        else:
            print("⚠ 未找到HDBSCAN结果，跳过对比图生成")
    else:
        print("⚠ 未找到K-means结果，跳过对比图生成")

    # 生成索引
    generator.generate_index_file()

    print("\n" + "="*70)
    print("论文图表生成完成!")
    print(f"所有图表保存在: {generator.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
