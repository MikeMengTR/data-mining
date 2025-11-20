#!/usr/bin/env python3
"""
方法对比分析：K-means vs HDBSCAN
详细对比两种聚类方法的性能、结果质量和适用场景
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

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


class MethodComparison:
    """方法对比分析器"""

    def __init__(self, kmeans_dir: str, hdbscan_dir: str, output_dir: str = "comparison_results"):
        """
        初始化对比分析器

        Args:
            kmeans_dir: K-means结果目录
            hdbscan_dir: HDBSCAN结果目录
            output_dir: 对比结果输出目录
        """
        self.kmeans_dir = Path(kmeans_dir)
        self.hdbscan_dir = Path(hdbscan_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"comparison_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        # 加载结果
        self.load_results()

        print(f"方法对比分析初始化完成")
        print(f"结果保存到: {self.run_dir}")

    def load_results(self):
        """加载两种方法的结果"""
        print("\n加载结果...")

        # 加载K-means结果
        self.kmeans_labels = np.load(self.kmeans_dir / "cluster_labels.npy")
        self.kmeans_data_2d = np.load(self.kmeans_dir / "data_2d.npy")
        with open(self.kmeans_dir / "summary.json", 'r') as f:
            self.kmeans_summary = json.load(f)

        print(f"✓ K-means结果已加载")
        print(f"  簇数: {self.kmeans_summary['n_clusters']}")

        # 加载HDBSCAN结果
        self.hdbscan_labels = np.load(self.hdbscan_dir / "cluster_labels.npy")
        self.hdbscan_data_2d = np.load(self.hdbscan_dir / "data_2d.npy")
        with open(self.hdbscan_dir / "summary.json", 'r') as f:
            self.hdbscan_summary = json.load(f)

        print(f"✓ HDBSCAN结果已加载")
        print(f"  簇数: {self.hdbscan_summary['n_clusters']}")
        print(f"  噪声点: {self.hdbscan_summary['n_noise']}")

    def compare_basic_stats(self):
        """对比基本统计信息"""
        print("\n" + "="*70)
        print("基本统计对比")
        print("="*70)

        stats = {
            'K-means': {
                'n_clusters': self.kmeans_summary['n_clusters'],
                'n_noise': 0,  # K-means没有噪声点概念
                'silhouette': self.kmeans_summary['metrics'].get('silhouette_score', None),
                'davies_bouldin': self.kmeans_summary['metrics'].get('davies_bouldin_score', None),
                'calinski_harabasz': self.kmeans_summary['metrics'].get('calinski_harabasz_score', None)
            },
            'HDBSCAN': {
                'n_clusters': self.hdbscan_summary['n_clusters'],
                'n_noise': self.hdbscan_summary.get('n_noise', 0),
                'silhouette': self.hdbscan_summary.get('results', {}).get('metrics', {}).get('silhouette_score', None),
                'davies_bouldin': self.hdbscan_summary.get('results', {}).get('metrics', {}).get('davies_bouldin_score', None),
                'calinski_harabasz': self.hdbscan_summary.get('results', {}).get('metrics', {}).get('calinski_harabasz_score', None)
            }
        }

        print("\n簇数对比:")
        print(f"  K-means:  {stats['K-means']['n_clusters']}")
        print(f"  HDBSCAN:  {stats['HDBSCAN']['n_clusters']}")

        print("\n噪声点:")
        print(f"  K-means:  0 (所有点都被分配)")
        print(f"  HDBSCAN:  {stats['HDBSCAN']['n_noise']}")

        print("\n评估指标对比:")
        for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
            km_val = stats['K-means'][metric]
            hdb_val = stats['HDBSCAN'][metric]
            print(f"\n  {metric}:")
            print(f"    K-means:  {km_val:.4f}" if km_val is not None else "    K-means:  N/A")
            print(f"    HDBSCAN:  {hdb_val:.4f}" if hdb_val is not None else "    HDBSCAN:  N/A")

        return stats

    def compare_cluster_agreement(self):
        """对比两种方法的聚类一致性"""
        print("\n" + "="*70)
        print("聚类一致性分析")
        print("="*70)

        # 过滤HDBSCAN的噪声点
        mask = self.hdbscan_labels != -1
        labels1 = self.kmeans_labels[mask]
        labels2 = self.hdbscan_labels[mask]

        # 计算ARI和NMI
        ari = adjusted_rand_score(labels1, labels2)
        nmi = normalized_mutual_info_score(labels1, labels2)

        print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
        print(f"  范围: [-1, 1], 1表示完全一致")
        print(f"  解释: {'高度一致' if ari > 0.7 else '中等一致' if ari > 0.4 else '低一致性'}")

        print(f"\nNormalized Mutual Information (NMI): {nmi:.4f}")
        print(f"  范围: [0, 1], 1表示完全一致")
        print(f"  解释: {'高度一致' if nmi > 0.7 else '中等一致' if nmi > 0.4 else '低一致性'}")

        return {'ari': ari, 'nmi': nmi}

    def visualize_comparison(self):
        """生成对比可视化"""
        print("\n" + "="*70)
        print("生成对比可视化")
        print("="*70)

        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. K-means 2D散点图
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(
            self.kmeans_data_2d[:, 0],
            self.kmeans_data_2d[:, 1],
            c=self.kmeans_labels,
            cmap='Spectral',
            s=1,
            alpha=0.5
        )
        ax1.set_title(f'K-means\n{self.kmeans_summary["n_clusters"]} Clusters',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('UMAP Dim 1')
        ax1.set_ylabel('UMAP Dim 2')
        plt.colorbar(scatter1, ax=ax1, label='Cluster ID')

        # 2. HDBSCAN 2D散点图
        ax2 = fig.add_subplot(gs[0, 1])
        scatter2 = ax2.scatter(
            self.hdbscan_data_2d[:, 0],
            self.hdbscan_data_2d[:, 1],
            c=self.hdbscan_labels,
            cmap='Spectral',
            s=1,
            alpha=0.5
        )
        ax2.set_title(f'HDBSCAN\n{self.hdbscan_summary["n_clusters"]} Clusters',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('UMAP Dim 1')
        ax2.set_ylabel('UMAP Dim 2')
        plt.colorbar(scatter2, ax=ax2, label='Cluster ID')

        # 3. 簇数对比
        ax3 = fig.add_subplot(gs[0, 2])
        methods = ['K-means', 'HDBSCAN']
        n_clusters = [
            self.kmeans_summary['n_clusters'],
            self.hdbscan_summary['n_clusters']
        ]
        bars = ax3.bar(methods, n_clusters, color=['steelblue', 'coral'], alpha=0.7)
        ax3.axhline(512, color='red', linestyle='--', linewidth=2, label='Target: 512')
        ax3.set_ylabel('Number of Clusters', fontsize=12)
        ax3.set_title('Cluster Count Comparison', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, count in zip(bars, n_clusters):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        # 4. 簇大小分布对比 - K-means
        ax4 = fig.add_subplot(gs[1, 0])
        unique, counts = np.unique(self.kmeans_labels, return_counts=True)
        ax4.hist(counts, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.set_title('K-means: Cluster Size Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Cluster Size')
        ax4.set_ylabel('Frequency')
        ax4.axvline(counts.mean(), color='red', linestyle='--', label=f'Mean: {counts.mean():.0f}')
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. 簇大小分布对比 - HDBSCAN
        ax5 = fig.add_subplot(gs[1, 1])
        unique, counts = np.unique(self.hdbscan_labels[self.hdbscan_labels != -1], return_counts=True)
        ax5.hist(counts, bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax5.set_title('HDBSCAN: Cluster Size Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Cluster Size')
        ax5.set_ylabel('Frequency')
        ax5.axvline(counts.mean(), color='red', linestyle='--', label=f'Mean: {counts.mean():.0f}')
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. 评估指标对比
        ax6 = fig.add_subplot(gs[1, 2])
        metrics_names = ['Silhouette\nScore', 'Davies-\nBouldin', 'Calinski-\nHarabasz']
        km_metrics = [
            self.kmeans_summary['metrics'].get('silhouette_score', 0),
            self.kmeans_summary['metrics'].get('davies_bouldin_score', 0),
            self.kmeans_summary['metrics'].get('calinski_harabasz_score', 0) / 1000  # 缩放
        ]
        hdb_metrics_dict = self.hdbscan_summary.get('results', {}).get('metrics', {})
        hdb_metrics = [
            hdb_metrics_dict.get('silhouette_score', 0),
            hdb_metrics_dict.get('davies_bouldin_score', 0),
            hdb_metrics_dict.get('calinski_harabasz_score', 0) / 1000
        ]

        x = np.arange(len(metrics_names))
        width = 0.35
        ax6.bar(x - width/2, km_metrics, width, label='K-means', color='steelblue', alpha=0.7)
        ax6.bar(x + width/2, hdb_metrics, width, label='HDBSCAN', color='coral', alpha=0.7)
        ax6.set_ylabel('Metric Value', fontsize=12)
        ax6.set_title('Evaluation Metrics Comparison', fontsize=14, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics_names, fontsize=10)
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)

        # 7. 簇大小箱线图对比
        ax7 = fig.add_subplot(gs[2, 0])
        km_counts = [len(self.kmeans_labels[self.kmeans_labels == i]) for i in range(self.kmeans_summary['n_clusters'])]
        hdb_counts = [len(self.hdbscan_labels[self.hdbscan_labels == i])
                     for i in range(self.hdbscan_summary['n_clusters'])]

        bp = ax7.boxplot([km_counts, hdb_counts], labels=['K-means', 'HDBSCAN'],
                         patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('coral')
        ax7.set_ylabel('Cluster Size', fontsize=12)
        ax7.set_title('Cluster Size Distribution (Boxplot)', fontsize=14, fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)

        # 8. 方法特点对比表
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')

        comparison_data = [
            ['特性', 'K-means', 'HDBSCAN'],
            ['簇数', f'{self.kmeans_summary["n_clusters"]} (预设)', f'{self.hdbscan_summary["n_clusters"]} (自动)'],
            ['噪声处理', '无（所有点必须分配）', f'有 ({self.hdbscan_summary.get("n_noise", 0)} 噪声点)'],
            ['簇形状', '球形假设', '任意形状'],
            ['可解释性', '高（中心点明确）', '中（基于密度）'],
            ['计算速度', '快', '中等'],
            ['参数敏感度', '低（主要是K）', '中（多个参数）'],
            ['适用场景', '已知簇数、球形簇', '探索性分析、复杂形状']
        ]

        table = ax8.table(cellText=comparison_data, loc='center',
                         cellLoc='left', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # 设置表头样式
        for i in range(3):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 交替行颜色
        for i in range(1, len(comparison_data)):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax8.set_title('Method Characteristics Comparison',
                     fontsize=14, fontweight='bold', pad=20)

        # 保存
        save_path = self.run_dir / "method_comparison_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对比可视化已保存: {save_path}")

        plt.close()

    def generate_comparison_report(self, stats, agreement):
        """生成对比报告"""
        print("\n生成对比报告...")

        report_path = self.run_dir / "comparison_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("K-means vs HDBSCAN 方法对比报告\n")
            f.write("="*70 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("一、方法简介\n")
            f.write("-"*70 + "\n")
            f.write("K-means:\n")
            f.write("  - 经典的划分聚类算法\n")
            f.write("  - 需要预设簇数K\n")
            f.write("  - 假设球形簇，对中心点进行迭代优化\n")
            f.write("  - 优点: 简单、快速、可解释性强\n")
            f.write("  - 缺点: 需要预设K、假设球形、对初始化敏感\n\n")

            f.write("HDBSCAN:\n")
            f.write("  - 基于密度的层次聚类算法\n")
            f.write("  - 自动确定簇数\n")
            f.write("  - 能识别任意形状的簇和噪声点\n")
            f.write("  - 优点: 自动簇数、识别噪声、任意形状\n")
            f.write("  - 缺点: 参数较多、计算较慢、解释性中等\n\n")

            f.write("二、聚类结果对比\n")
            f.write("-"*70 + "\n")
            f.write(f"簇数:\n")
            f.write(f"  K-means:  {stats['K-means']['n_clusters']}\n")
            f.write(f"  HDBSCAN:  {stats['HDBSCAN']['n_clusters']}\n")
            f.write(f"  目标:     512\n\n")

            f.write(f"噪声点:\n")
            f.write(f"  K-means:  0 (所有点都必须被分配)\n")
            f.write(f"  HDBSCAN:  {stats['HDBSCAN']['n_noise']}\n\n")

            f.write("三、质量指标对比\n")
            f.write("-"*70 + "\n")
            for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
                f.write(f"{metric}:\n")
                km_val = stats['K-means'][metric]
                hdb_val = stats['HDBSCAN'][metric]
                f.write(f"  K-means:  {km_val:.4f}\n" if km_val is not None else "  K-means:  N/A\n")
                f.write(f"  HDBSCAN:  {hdb_val:.4f}\n" if hdb_val is not None else "  HDBSCAN:  N/A\n")

                # 判断哪个更好
                if km_val is not None and hdb_val is not None:
                    if metric == 'silhouette' or metric == 'calinski_harabasz':
                        winner = 'K-means' if km_val > hdb_val else 'HDBSCAN'
                        f.write(f"  更优: {winner}\n")
                    else:  # davies_bouldin越小越好
                        winner = 'K-means' if km_val < hdb_val else 'HDBSCAN'
                        f.write(f"  更优: {winner}\n")
                f.write("\n")

            f.write("四、聚类一致性\n")
            f.write("-"*70 + "\n")
            f.write(f"Adjusted Rand Index: {agreement['ari']:.4f}\n")
            f.write(f"  解释: 两种方法的聚类{'高度一致' if agreement['ari'] > 0.7 else '中等一致' if agreement['ari'] > 0.4 else '差异较大'}\n\n")
            f.write(f"Normalized Mutual Information: {agreement['nmi']:.4f}\n")
            f.write(f"  解释: 信息共享程度{'高' if agreement['nmi'] > 0.7 else '中等' if agreement['nmi'] > 0.4 else '较低'}\n\n")

            f.write("五、方法选择建议\n")
            f.write("-"*70 + "\n")
            f.write("选择K-means如果:\n")
            f.write("  1. 已知目标簇数（如本项目的512簇）\n")
            f.write("  2. 需要高度可解释性（簇中心明确）\n")
            f.write("  3. 计算资源有限（速度更快）\n")
            f.write("  4. 数据簇相对均匀且接近球形\n\n")

            f.write("选择HDBSCAN如果:\n")
            f.write("  1. 探索性分析，不确定簇数\n")
            f.write("  2. 数据包含噪声点\n")
            f.write("  3. 簇形状复杂、密度不均\n")
            f.write("  4. 需要层次化的簇结构信息\n\n")

            f.write("="*70 + "\n")

        print(f"✓ 对比报告已保存: {report_path}")

    def save_comparison_summary(self, stats, agreement):
        """保存对比摘要"""
        summary = {
            'timestamp': self.timestamp,
            'kmeans_dir': str(self.kmeans_dir),
            'hdbscan_dir': str(self.hdbscan_dir),
            'statistics': stats,
            'agreement': agreement
        }

        summary_path = self.run_dir / "comparison_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ 对比摘要已保存: {summary_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("K-means vs HDBSCAN 方法对比分析")
    print("="*70)

    # 示例：需要提供K-means和HDBSCAN的结果目录
    print("\n请提供结果目录路径:")
    print("K-means结果目录 (例如: baseline_results/kmeans_20251110_160000):")
    kmeans_dir = input().strip()
    if not kmeans_dir:
        kmeans_dir = "baseline_results/kmeans_20251110_160000"  # 默认值
        print(f"使用默认值: {kmeans_dir}")

    print("HDBSCAN结果目录 (例如: results/run_20251110_150000):")
    hdbscan_dir = input().strip()
    if not hdbscan_dir:
        hdbscan_dir = "results/run_20251110_150000"  # 默认值
        print(f"使用默认值: {hdbscan_dir}")

    # 初始化对比器
    try:
        comparator = MethodComparison(
            kmeans_dir=kmeans_dir,
            hdbscan_dir=hdbscan_dir,
            output_dir="comparison_results"
        )

        # 对比分析
        stats = comparator.compare_basic_stats()
        agreement = comparator.compare_cluster_agreement()

        # 可视化
        comparator.visualize_comparison()

        # 生成报告
        comparator.generate_comparison_report(stats, agreement)
        comparator.save_comparison_summary(stats, agreement)

        print("\n" + "="*70)
        print("方法对比分析完成!")
        print(f"结果保存在: {comparator.run_dir}")
        print("="*70)

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("1. 已经运行过 baseline_kmeans.py")
        print("2. 已经运行过 clustering_analysis.py")
        print("3. 提供的目录路径正确")


if __name__ == "__main__":
    main()
