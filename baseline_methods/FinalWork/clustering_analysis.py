#!/usr/bin/env python3
"""
音乐Token Embedding聚类分析 - 方案B：密度聚类
使用UMAP降维 + HDBSCAN聚类，探索最优簇数

作者：Data Mining Project
日期：2025-11-10
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# 导入核心库
import umap
import hdbscan
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100


class MusicEmbeddingClustering:
    """音乐Embedding聚类分析类"""

    def __init__(self, data_path: str, output_dir: str = "results"):
        """
        初始化聚类分析器

        Args:
            data_path: 数据文件路径
            output_dir: 结果输出目录
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 创建时间戳子目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        # 数据
        self.data = None
        self.data_reduced = None
        self.data_2d = None
        self.labels = None

        # 模型
        self.umap_model = None
        self.hdbscan_model = None

        # 结果
        self.results = {}

        print(f"初始化完成，结果将保存到: {self.run_dir}")

    def load_data(self):
        """加载数据"""
        print("\n" + "="*60)
        print("步骤 1: 加载数据")
        print("="*60)

        self.data = np.load(self.data_path)
        print(f"✓ 数据加载成功")
        print(f"  - 样本数: {self.data.shape[0]:,}")
        print(f"  - 维度: {self.data.shape[1]}")
        print(f"  - 数据类型: {self.data.dtype}")
        print(f"  - 内存占用: {self.data.nbytes / 1024 / 1024:.2f} MB")
        print(f"  - 统计信息:")
        print(f"    * Mean: {self.data.mean():.4f}")
        print(f"    * Std: {self.data.std():.4f}")
        print(f"    * Min: {self.data.min():.4f}")
        print(f"    * Max: {self.data.max():.4f}")

        return self.data

    def umap_dimensionality_reduction(
        self,
        n_components: int = 50,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'cosine',
        random_state: int = 42
    ):
        """
        使用UMAP进行降维

        Args:
            n_components: 降维后的维度 (50-100)
            n_neighbors: UMAP邻居数
            min_dist: UMAP最小距离
            metric: 距离度量方式
            random_state: 随机种子
        """
        print("\n" + "="*60)
        print(f"步骤 2: UMAP降维 (降至 {n_components} 维)")
        print("="*60)

        start_time = time.time()

        self.umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            verbose=True
        )

        self.data_reduced = self.umap_model.fit_transform(self.data)

        elapsed = time.time() - start_time

        print(f"\n✓ UMAP降维完成")
        print(f"  - 原始维度: {self.data.shape[1]} → 降维后: {self.data_reduced.shape[1]}")
        print(f"  - 耗时: {elapsed:.2f} 秒")
        print(f"  - 参数:")
        print(f"    * n_neighbors: {n_neighbors}")
        print(f"    * min_dist: {min_dist}")
        print(f"    * metric: {metric}")

        return self.data_reduced

    def hdbscan_clustering(
        self,
        min_cluster_size: int = 50,
        min_samples: int = 10,
        cluster_selection_epsilon: float = 0.0,
        metric: str = 'euclidean'
    ):
        """
        使用HDBSCAN进行聚类

        Args:
            min_cluster_size: 最小簇大小
            min_samples: 最小样本数
            cluster_selection_epsilon: 簇选择阈值
            metric: 距离度量
        """
        print("\n" + "="*60)
        print("步骤 3: HDBSCAN聚类")
        print("="*60)

        start_time = time.time()

        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            core_dist_n_jobs=-1  # 使用所有CPU核心
        )

        self.labels = self.hdbscan_model.fit_predict(self.data_reduced)

        elapsed = time.time() - start_time

        # 统计信息
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)

        print(f"\n✓ HDBSCAN聚类完成")
        print(f"  - 簇数: {n_clusters}")
        print(f"  - 噪声点数: {n_noise} ({n_noise/len(self.labels)*100:.2f}%)")
        print(f"  - 耗时: {elapsed:.2f} 秒")
        print(f"  - 参数:")
        print(f"    * min_cluster_size: {min_cluster_size}")
        print(f"    * min_samples: {min_samples}")
        print(f"    * cluster_selection_epsilon: {cluster_selection_epsilon}")
        print(f"    * metric: {metric}")

        # 簇大小分布
        unique, counts = np.unique(self.labels[self.labels != -1], return_counts=True)
        if len(unique) > 0:
            print(f"  - 簇大小统计:")
            print(f"    * 平均大小: {counts.mean():.0f}")
            print(f"    * 最大簇: {counts.max()}")
            print(f"    * 最小簇: {counts.min()}")
            print(f"    * 中位数: {np.median(counts):.0f}")

        return self.labels, n_clusters

    def evaluate_clustering(self):
        """评估聚类质量"""
        print("\n" + "="*60)
        print("步骤 4: 聚类质量评估")
        print("="*60)

        # 过滤噪声点
        mask = self.labels != -1
        data_filtered = self.data_reduced[mask]
        labels_filtered = self.labels[mask]

        if len(set(labels_filtered)) < 2:
            print("⚠ 警告: 簇数少于2，无法计算评估指标")
            return {}

        # 计算评估指标
        metrics = {}

        try:
            metrics['silhouette_score'] = silhouette_score(
                data_filtered, labels_filtered, sample_size=min(10000, len(labels_filtered))
            )
            print(f"✓ Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"  (范围: [-1, 1], 越接近1越好)")
        except Exception as e:
            print(f"✗ Silhouette Score计算失败: {e}")

        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(
                data_filtered, labels_filtered
            )
            print(f"✓ Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}")
            print(f"  (越小越好，0为理想值)")
        except Exception as e:
            print(f"✗ Davies-Bouldin Index计算失败: {e}")

        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                data_filtered, labels_filtered
            )
            print(f"✓ Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.2f}")
            print(f"  (越大越好)")
        except Exception as e:
            print(f"✗ Calinski-Harabasz Index计算失败: {e}")

        self.results['metrics'] = metrics
        return metrics

    def umap_visualization(self, random_state: int = 42):
        """降维到2D用于可视化"""
        print("\n" + "="*60)
        print("步骤 5: 2D可视化降维")
        print("="*60)

        start_time = time.time()

        umap_2d = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=random_state,
            verbose=False
        )

        self.data_2d = umap_2d.fit_transform(self.data)

        elapsed = time.time() - start_time
        print(f"✓ 2D降维完成，耗时: {elapsed:.2f} 秒")

        return self.data_2d

    def visualize_results(self, save: bool = True):
        """可视化聚类结果"""
        print("\n" + "="*60)
        print("步骤 6: 生成可视化图表")
        print("="*60)

        if self.data_2d is None:
            print("⚠ 先执行2D降维...")
            self.umap_visualization()

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 聚类结果散点图
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.data_2d[:, 0],
            self.data_2d[:, 1],
            c=self.labels,
            cmap='Spectral',
            s=1,
            alpha=0.6
        )
        ax.set_title(f'HDBSCAN Clustering Results\n{n_clusters} Clusters Found',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Cluster ID')

        # 2. 簇大小分布直方图
        ax = axes[0, 1]
        unique_labels, counts = np.unique(self.labels[self.labels != -1], return_counts=True)
        ax.bar(range(len(counts)), sorted(counts, reverse=True), color='steelblue', alpha=0.7)
        ax.set_title(f'Cluster Size Distribution (Top 50)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster Rank', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_xlim(0, min(50, len(counts)))
        ax.grid(axis='y', alpha=0.3)

        # 3. 簇大小箱线图
        ax = axes[1, 0]
        ax.boxplot(counts, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax.set_title('Cluster Size Statistics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cluster Size', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # 添加统计文本
        stats_text = f"Mean: {counts.mean():.0f}\n"
        stats_text += f"Median: {np.median(counts):.0f}\n"
        stats_text += f"Std: {counts.std():.0f}\n"
        stats_text += f"Min: {counts.min()}\n"
        stats_text += f"Max: {counts.max()}"
        ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4. 噪声点分布
        ax = axes[1, 1]
        n_noise = (self.labels == -1).sum()
        n_clustered = (self.labels != -1).sum()

        wedges, texts, autotexts = ax.pie(
            [n_clustered, n_noise],
            labels=['Clustered', 'Noise'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            startangle=90,
            textprops={'fontsize': 12}
        )
        ax.set_title(f'Data Distribution\nTotal: {len(self.labels):,} samples',
                    fontsize=14, fontweight='bold')

        # 添加总结信息
        summary_text = f"Clustering Summary:\n"
        summary_text += f"• Total Clusters: {n_clusters}\n"
        summary_text += f"• Clustered Samples: {n_clustered:,}\n"
        summary_text += f"• Noise Points: {n_noise:,}\n"
        if 'metrics' in self.results:
            metrics = self.results['metrics']
            if 'silhouette_score' in metrics:
                summary_text += f"• Silhouette: {metrics['silhouette_score']:.4f}\n"
            if 'davies_bouldin_score' in metrics:
                summary_text += f"• Davies-Bouldin: {metrics['davies_bouldin_score']:.4f}"

        fig.text(0.5, 0.02, summary_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout(rect=[0, 0.05, 1, 1])

        if save:
            save_path = self.run_dir / "clustering_visualization.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")

        plt.show()

        return fig

    def parameter_search(
        self,
        target_clusters: int = 512,
        tolerance: int = 50,
        umap_dims: List[int] = [50, 75, 100],
        min_cluster_sizes: List[int] = [30, 50, 100, 150],
        min_samples_list: List[int] = [5, 10, 20],
        max_iterations: int = 20
    ):
        """
        参数网格搜索，寻找接近目标簇数的参数组合

        Args:
            target_clusters: 目标簇数
            tolerance: 可接受的簇数偏差
            umap_dims: UMAP降维维度候选
            min_cluster_sizes: 最小簇大小候选
            min_samples_list: 最小样本数候选
            max_iterations: 最大尝试次数
        """
        print("\n" + "="*60)
        print(f"步骤 7: 参数搜索 (目标: {target_clusters} ± {tolerance} 簇)")
        print("="*60)

        results_list = []
        best_result = None
        best_diff = float('inf')

        iteration = 0

        for umap_dim in umap_dims:
            # 降维
            print(f"\n--- 尝试 UMAP 维度: {umap_dim} ---")
            self.umap_dimensionality_reduction(n_components=umap_dim)

            for min_cluster_size in min_cluster_sizes:
                for min_samples in min_samples_list:
                    if iteration >= max_iterations:
                        break

                    iteration += 1

                    print(f"\n[{iteration}/{max_iterations}] 测试参数组合:")
                    print(f"  UMAP dim={umap_dim}, min_cluster_size={min_cluster_size}, min_samples={min_samples}")

                    # 聚类
                    labels, n_clusters = self.hdbscan_clustering(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples
                    )

                    # 评估
                    metrics = self.evaluate_clustering()

                    # 记录结果
                    result = {
                        'iteration': iteration,
                        'umap_dim': umap_dim,
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples,
                        'n_clusters': n_clusters,
                        'metrics': metrics,
                        'diff_from_target': abs(n_clusters - target_clusters)
                    }
                    results_list.append(result)

                    # 更新最佳结果
                    diff = abs(n_clusters - target_clusters)
                    if diff < best_diff:
                        best_diff = diff
                        best_result = result
                        print(f"  ⭐ 新的最佳结果! 簇数: {n_clusters} (差距: {diff})")

                    # 如果在容忍范围内，可以提前停止
                    if diff <= tolerance:
                        print(f"  ✓ 找到满足条件的参数组合!")
                        break

                if iteration >= max_iterations:
                    break

            if iteration >= max_iterations:
                break

        # 保存搜索结果
        self.results['parameter_search'] = {
            'target_clusters': target_clusters,
            'tolerance': tolerance,
            'best_result': best_result,
            'all_results': results_list
        }

        # 打印最佳结果
        print("\n" + "="*60)
        print("参数搜索完成!")
        print("="*60)
        print(f"\n最佳参数组合:")
        print(f"  - UMAP维度: {best_result['umap_dim']}")
        print(f"  - min_cluster_size: {best_result['min_cluster_size']}")
        print(f"  - min_samples: {best_result['min_samples']}")
        print(f"  - 得到簇数: {best_result['n_clusters']}")
        print(f"  - 与目标差距: {best_result['diff_from_target']}")

        return best_result, results_list

    def save_results(self):
        """保存所有结果"""
        print("\n" + "="*60)
        print("步骤 8: 保存结果")
        print("="*60)

        # 保存聚类标签
        labels_path = self.run_dir / "cluster_labels.npy"
        np.save(labels_path, self.labels)
        print(f"✓ 聚类标签已保存: {labels_path}")

        # 保存降维数据
        if self.data_reduced is not None:
            reduced_path = self.run_dir / "data_reduced.npy"
            np.save(reduced_path, self.data_reduced)
            print(f"✓ 降维数据已保存: {reduced_path}")

        # 保存2D数据
        if self.data_2d is not None:
            data_2d_path = self.run_dir / "data_2d.npy"
            np.save(data_2d_path, self.data_2d)
            print(f"✓ 2D数据已保存: {data_2d_path}")

        # 保存结果摘要
        summary = {
            'timestamp': self.timestamp,
            'data_shape': self.data.shape,
            'n_clusters': int(len(set(self.labels)) - (1 if -1 in self.labels else 0)),
            'n_noise': int((self.labels == -1).sum()),
            'results': self.results
        }

        # 转换numpy类型为Python原生类型
        summary_json = json.dumps(summary, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else None)

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            f.write(summary_json)
        print(f"✓ 结果摘要已保存: {summary_path}")

        # 生成文本报告
        self.generate_report()

        print(f"\n所有结果已保存到: {self.run_dir}")

    def generate_report(self):
        """生成分析报告"""
        report_path = self.run_dir / "analysis_report.txt"

        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = (self.labels == -1).sum()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("音乐Token Embedding聚类分析报告\n")
            f.write("="*70 + "\n\n")

            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {self.data_path}\n")
            f.write(f"输出目录: {self.run_dir}\n\n")

            f.write("数据信息:\n")
            f.write(f"  样本数: {self.data.shape[0]:,}\n")
            f.write(f"  原始维度: {self.data.shape[1]}\n")
            f.write(f"  降维后维度: {self.data_reduced.shape[1] if self.data_reduced is not None else 'N/A'}\n\n")

            f.write("聚类结果:\n")
            f.write(f"  簇数: {n_clusters}\n")
            f.write(f"  聚类样本数: {len(self.labels) - n_noise:,}\n")
            f.write(f"  噪声点数: {n_noise:,} ({n_noise/len(self.labels)*100:.2f}%)\n\n")

            if 'metrics' in self.results:
                f.write("评估指标:\n")
                for metric_name, metric_value in self.results['metrics'].items():
                    f.write(f"  {metric_name}: {metric_value:.4f}\n")
                f.write("\n")

            # 簇大小统计
            unique_labels, counts = np.unique(self.labels[self.labels != -1], return_counts=True)
            if len(counts) > 0:
                f.write("簇大小统计:\n")
                f.write(f"  平均大小: {counts.mean():.0f}\n")
                f.write(f"  中位数: {np.median(counts):.0f}\n")
                f.write(f"  标准差: {counts.std():.0f}\n")
                f.write(f"  最小簇: {counts.min()}\n")
                f.write(f"  最大簇: {counts.max()}\n\n")

            if 'parameter_search' in self.results:
                f.write("参数搜索结果:\n")
                ps = self.results['parameter_search']
                f.write(f"  目标簇数: {ps['target_clusters']}\n")
                if ps['best_result']:
                    br = ps['best_result']
                    f.write(f"  最佳参数:\n")
                    f.write(f"    - UMAP维度: {br['umap_dim']}\n")
                    f.write(f"    - min_cluster_size: {br['min_cluster_size']}\n")
                    f.write(f"    - min_samples: {br['min_samples']}\n")
                    f.write(f"    - 得到簇数: {br['n_clusters']}\n")
                    f.write(f"    - 与目标差距: {br['diff_from_target']}\n")

            f.write("\n" + "="*70 + "\n")

        print(f"✓ 分析报告已保存: {report_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("音乐Token Embedding聚类分析 - 方案B: 密度聚类")
    print("="*70)

    # 初始化
    clustering = MusicEmbeddingClustering(
        data_path="data/bos_vectors_dim_83362_768.npy",
        output_dir="results"
    )

    # 1. 加载数据
    clustering.load_data()

    # 2. 快速单次运行（用于初步探索）
    print("\n是否执行参数搜索模式? (y/n, 默认n进行单次分析)")
    choice = input("请选择: ").strip().lower()

    if choice == 'y':
        # 参数搜索模式
        best_result, all_results = clustering.parameter_search(
            target_clusters=512,
            tolerance=50,
            umap_dims=[50, 75, 100],
            min_cluster_sizes=[30, 50, 100, 150],
            min_samples_list=[5, 10, 20],
            max_iterations=15
        )

        # 使用最佳参数重新运行
        print("\n使用最佳参数重新运行...")
        clustering.umap_dimensionality_reduction(n_components=best_result['umap_dim'])
        clustering.hdbscan_clustering(
            min_cluster_size=best_result['min_cluster_size'],
            min_samples=best_result['min_samples']
        )
        clustering.evaluate_clustering()

    else:
        # 单次分析模式
        clustering.umap_dimensionality_reduction(n_components=75)
        clustering.hdbscan_clustering(min_cluster_size=100, min_samples=10)
        clustering.evaluate_clustering()

    # 可视化
    clustering.umap_visualization()
    clustering.visualize_results()

    # 保存结果
    clustering.save_results()

    print("\n" + "="*70)
    print("分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
