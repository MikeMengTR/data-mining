#!/usr/bin/env python3
"""
聚类结果对比分析工具
比较不同参数下的聚类结果
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


class ClusteringResultsComparator:
    """聚类结果对比器"""

    def __init__(self, results_dirs: List[str]):
        """
        初始化对比器

        Args:
            results_dirs: 结果目录列表
        """
        self.results_dirs = [Path(d) for d in results_dirs]
        self.results = []
        self.load_all_results()

    def load_all_results(self):
        """加载所有结果"""
        print("\n加载结果...")
        for result_dir in self.results_dirs:
            try:
                # 加载summary.json
                summary_path = result_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)

                    # 加载标签
                    labels_path = result_dir / "cluster_labels.npy"
                    labels = np.load(labels_path) if labels_path.exists() else None

                    # 加载2D数据
                    data_2d_path = result_dir / "data_2d.npy"
                    data_2d = np.load(data_2d_path) if data_2d_path.exists() else None

                    self.results.append({
                        'dir': result_dir,
                        'name': result_dir.name,
                        'summary': summary,
                        'labels': labels,
                        'data_2d': data_2d
                    })
                    print(f"✓ 加载: {result_dir.name}")
                else:
                    print(f"✗ 未找到summary.json: {result_dir}")

            except Exception as e:
                print(f"✗ 加载失败 {result_dir}: {e}")

        print(f"\n成功加载 {len(self.results)} 个结果")

    def compare_statistics(self):
        """比较统计信息"""
        print("\n" + "="*70)
        print("统计对比")
        print("="*70)

        for i, result in enumerate(self.results, 1):
            summary = result['summary']
            print(f"\n{i}. {result['name']}")
            print(f"   簇数: {summary['n_clusters']}")
            print(f"   噪声点: {summary['n_noise']}")
            print(f"   噪声比例: {summary['n_noise']/summary['data_shape'][0]*100:.2f}%")

            if 'metrics' in summary.get('results', {}):
                metrics = summary['results']['metrics']
                print(f"   评估指标:")
                for metric_name, value in metrics.items():
                    if value is not None:
                        print(f"     - {metric_name}: {value:.4f}")

    def visualize_comparison(self, save_path: str = "comparison_results.png"):
        """可视化对比"""
        print("\n" + "="*70)
        print("生成对比可视化")
        print("="*70)

        n_results = len(self.results)
        fig, axes = plt.subplots(2, n_results, figsize=(6*n_results, 12))

        if n_results == 1:
            axes = axes.reshape(-1, 1)

        for i, result in enumerate(self.results):
            labels = result['labels']
            data_2d = result['data_2d']
            summary = result['summary']

            # 第一行：2D散点图
            ax = axes[0, i]
            if data_2d is not None and labels is not None:
                scatter = ax.scatter(
                    data_2d[:, 0],
                    data_2d[:, 1],
                    c=labels,
                    cmap='Spectral',
                    s=1,
                    alpha=0.5
                )
                ax.set_title(
                    f"{result['name']}\n{summary['n_clusters']} Clusters",
                    fontsize=12,
                    fontweight='bold'
                )
                ax.set_xlabel('UMAP Dim 1')
                ax.set_ylabel('UMAP Dim 2')
                plt.colorbar(scatter, ax=ax, label='Cluster ID')
            else:
                ax.text(0.5, 0.5, 'No 2D data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(result['name'])

            # 第二行：簇大小分布
            ax = axes[1, i]
            if labels is not None:
                unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
                sorted_counts = sorted(counts, reverse=True)[:50]  # Top 50

                ax.bar(range(len(sorted_counts)), sorted_counts, color='steelblue', alpha=0.7)
                ax.set_title('Cluster Size Distribution (Top 50)', fontsize=11)
                ax.set_xlabel('Cluster Rank')
                ax.set_ylabel('Size')
                ax.grid(axis='y', alpha=0.3)

                # 添加统计信息
                stats_text = (
                    f"Mean: {counts.mean():.0f}\n"
                    f"Median: {np.median(counts):.0f}\n"
                    f"Max: {counts.max()}"
                )
                ax.text(
                    0.95, 0.95, stats_text,
                    transform=ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=9
                )
            else:
                ax.text(0.5, 0.5, 'No labels', ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对比图已保存: {save_path}")
        plt.show()

    def compare_metrics(self, save_path: str = "metrics_comparison.png"):
        """对比评估指标"""
        print("\n生成指标对比图...")

        # 提取指标
        metrics_data = []
        for result in self.results:
            if 'metrics' in result['summary'].get('results', {}):
                metrics = result['summary']['results']['metrics']
                metrics_data.append({
                    'name': result['name'],
                    'n_clusters': result['summary']['n_clusters'],
                    **{k: v for k, v in metrics.items() if v is not None}
                })

        if not metrics_data:
            print("⚠ 没有找到评估指标")
            return

        # 创建图表
        metric_names = [k for k in metrics_data[0].keys() if k not in ['name', 'n_clusters']]
        n_metrics = len(metric_names)

        fig, axes = plt.subplots(1, n_metrics + 1, figsize=(5*(n_metrics+1), 5))

        # 簇数对比
        ax = axes[0]
        names = [d['name'] for d in metrics_data]
        n_clusters = [d['n_clusters'] for d in metrics_data]
        bars = ax.bar(range(len(names)), n_clusters, color='steelblue', alpha=0.7)
        ax.axhline(512, color='red', linestyle='--', linewidth=2, label='Target: 512')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Number of Clusters')
        ax.set_title('Cluster Count Comparison', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 为每个柱子添加数值标签
        for i, (bar, count) in enumerate(zip(bars, n_clusters)):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f'{count}',
                ha='center', va='bottom', fontsize=10
            )

        # 其他指标
        for i, metric_name in enumerate(metric_names):
            ax = axes[i + 1]
            values = [d.get(metric_name, 0) for d in metrics_data]

            bars = ax.bar(range(len(names)), values, color='coral', alpha=0.7)
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # 添加数值标签
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height(),
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=9
                )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 指标对比图已保存: {save_path}")
        plt.show()

    def generate_comparison_report(self, save_path: str = "comparison_report.txt"):
        """生成对比报告"""
        print("\n生成对比报告...")

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("聚类结果对比报告\n")
            f.write("="*70 + "\n\n")

            for i, result in enumerate(self.results, 1):
                summary = result['summary']

                f.write(f"{i}. {result['name']}\n")
                f.write("-" * 70 + "\n")
                f.write(f"  时间戳: {summary['timestamp']}\n")
                f.write(f"  样本数: {summary['data_shape'][0]:,}\n")
                f.write(f"  簇数: {summary['n_clusters']}\n")
                f.write(f"  噪声点: {summary['n_noise']:,}\n")
                f.write(f"  噪声比例: {summary['n_noise']/summary['data_shape'][0]*100:.2f}%\n")

                if 'metrics' in summary.get('results', {}):
                    f.write(f"\n  评估指标:\n")
                    for metric_name, value in summary['results']['metrics'].items():
                        if value is not None:
                            f.write(f"    - {metric_name}: {value:.4f}\n")

                if 'parameter_search' in summary.get('results', {}):
                    ps = summary['results']['parameter_search']
                    if ps.get('best_result'):
                        br = ps['best_result']
                        f.write(f"\n  参数:\n")
                        f.write(f"    - UMAP维度: {br.get('umap_dim', 'N/A')}\n")
                        f.write(f"    - min_cluster_size: {br.get('min_cluster_size', 'N/A')}\n")
                        f.write(f"    - min_samples: {br.get('min_samples', 'N/A')}\n")

                f.write("\n")

        print(f"✓ 对比报告已保存: {save_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("聚类结果对比分析")
    print("="*70)

    # 示例：对比多个结果目录
    # 用户需要替换为实际的结果目录路径
    print("\n请输入要对比的结果目录（用空格分隔）:")
    print("例如: results/run_20251110_143021 results/run_20251110_150532")
    print("或直接回车使用示例目录")

    user_input = input("\n目录路径: ").strip()

    if user_input:
        results_dirs = user_input.split()
    else:
        # 自动查找results目录下的所有运行
        results_path = Path("results")
        if results_path.exists():
            results_dirs = [str(d) for d in results_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
            if results_dirs:
                print(f"\n找到 {len(results_dirs)} 个结果目录:")
                for d in results_dirs:
                    print(f"  - {d}")
            else:
                print("\n⚠ 未找到任何结果目录，请先运行聚类分析")
                return
        else:
            print("\n⚠ results目录不存在，请先运行聚类分析")
            return

    if not results_dirs:
        print("\n⚠ 没有指定对比目录")
        return

    # 创建对比器
    comparator = ClusteringResultsComparator(results_dirs)

    if not comparator.results:
        print("\n⚠ 没有成功加载任何结果")
        return

    # 统计对比
    comparator.compare_statistics()

    # 可视化对比
    comparator.visualize_comparison()

    # 指标对比
    comparator.compare_metrics()

    # 生成报告
    comparator.generate_comparison_report()

    print("\n" + "="*70)
    print("对比分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
