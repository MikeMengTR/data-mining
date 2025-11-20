#!/usr/bin/env python3
"""
可解释基线方法：K-means聚类
简单、经典、完全可解释和可复现

方法原理：
1. 随机初始化K个聚类中心
2. 将每个样本分配到最近的中心
3. 重新计算每个簇的中心
4. 重复2-3直到收敛

优点：
- 算法简单，易于理解和解释
- 计算效率高
- 结果可复现（固定random_state）
- 适合作为对比基线

参数：
- n_clusters: 簇的数量（需预设，这里设为512）
- random_state: 随机种子（保证可复现）
- n_init: 不同初始化的运行次数
- max_iter: 最大迭代次数
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入核心库
import umap
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


class KMeansBaseline:
    """K-means基线聚类方法"""

    def __init__(self, data_path: str, output_dir: str = "baseline_results"):
        """
        初始化K-means基线分析器

        Args:
            data_path: 数据文件路径
            output_dir: 结果输出目录
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"kmeans_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        self.data = None
        self.data_reduced = None
        self.data_2d = None
        self.labels = None
        self.kmeans_model = None

        print(f"K-means基线方法初始化完成")
        print(f"结果保存到: {self.run_dir}")

    def load_data(self):
        """加载数据"""
        print("\n" + "="*70)
        print("步骤 1: 加载数据")
        print("="*70)

        self.data = np.load(self.data_path)
        print(f"✓ 数据加载成功")
        print(f"  - 样本数: {self.data.shape[0]:,}")
        print(f"  - 维度: {self.data.shape[1]}")
        print(f"  - 数据类型: {self.data.dtype}")
        print(f"  - 统计信息: Mean={self.data.mean():.4f}, Std={self.data.std():.4f}")

        return self.data

    def umap_reduction(self, n_components=50, random_state=42):
        """
        UMAP降维（与HDBSCAN方法保持一致）

        Args:
            n_components: 降维后的维度
            random_state: 随机种子
        """
        print("\n" + "="*70)
        print(f"步骤 2: UMAP降维 (降至 {n_components} 维)")
        print("="*70)

        start_time = time.time()

        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=random_state,
            verbose=False
        )

        self.data_reduced = umap_model.fit_transform(self.data)

        elapsed = time.time() - start_time

        print(f"✓ UMAP降维完成")
        print(f"  - 原始维度: {self.data.shape[1]} → 降维后: {self.data_reduced.shape[1]}")
        print(f"  - 耗时: {elapsed:.2f} 秒")

        return self.data_reduced

    def kmeans_clustering(
        self,
        n_clusters=512,
        random_state=42,
        n_init=10,
        max_iter=300,
        use_minibatch=False
    ):
        """
        K-means聚类

        Args:
            n_clusters: 簇的数量
            random_state: 随机种子（保证可复现）
            n_init: 不同初始化的运行次数
            max_iter: 最大迭代次数
            use_minibatch: 是否使用MiniBatchKMeans（大数据集更快）
        """
        print("\n" + "="*70)
        print("步骤 3: K-means聚类")
        print("="*70)

        start_time = time.time()

        if use_minibatch:
            print(f"使用 MiniBatchKMeans (适合大数据集)")
            self.kmeans_model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
                batch_size=1024,
                verbose=0
            )
        else:
            print(f"使用 标准 K-means")
            self.kmeans_model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
                verbose=0
            )

        self.labels = self.kmeans_model.fit_predict(self.data_reduced)

        elapsed = time.time() - start_time

        # 统计信息
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        n_clusters_actual = len(unique_labels)

        print(f"\n✓ K-means聚类完成")
        print(f"  - 簇数: {n_clusters_actual}")
        print(f"  - 迭代次数: {self.kmeans_model.n_iter_}")
        print(f"  - 惯性 (inertia): {self.kmeans_model.inertia_:.2f}")
        print(f"  - 耗时: {elapsed:.2f} 秒")
        print(f"  - 参数:")
        print(f"    * n_clusters: {n_clusters}")
        print(f"    * random_state: {random_state}")
        print(f"    * n_init: {n_init}")
        print(f"    * max_iter: {max_iter}")

        # 簇大小分布
        print(f"  - 簇大小统计:")
        print(f"    * 平均大小: {counts.mean():.0f}")
        print(f"    * 最大簇: {counts.max()}")
        print(f"    * 最小簇: {counts.min()}")
        print(f"    * 中位数: {np.median(counts):.0f}")
        print(f"    * 标准差: {counts.std():.0f}")

        return self.labels

    def evaluate_clustering(self):
        """评估聚类质量"""
        print("\n" + "="*70)
        print("步骤 4: 聚类质量评估")
        print("="*70)

        metrics = {}

        try:
            # Silhouette Score (可能较慢，采样计算)
            sample_size = min(10000, len(self.labels))
            metrics['silhouette_score'] = silhouette_score(
                self.data_reduced,
                self.labels,
                sample_size=sample_size
            )
            print(f"✓ Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"  (范围: [-1, 1], 越接近1越好)")
        except Exception as e:
            print(f"✗ Silhouette Score计算失败: {e}")

        try:
            metrics['davies_bouldin_score'] = davies_bouldin_score(
                self.data_reduced,
                self.labels
            )
            print(f"✓ Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}")
            print(f"  (越小越好，0为理想值)")
        except Exception as e:
            print(f"✗ Davies-Bouldin Index计算失败: {e}")

        try:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                self.data_reduced,
                self.labels
            )
            print(f"✓ Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.2f}")
            print(f"  (越大越好)")
        except Exception as e:
            print(f"✗ Calinski-Harabasz Index计算失败: {e}")

        # K-means特有指标
        metrics['inertia'] = self.kmeans_model.inertia_
        print(f"✓ Inertia (惯性): {metrics['inertia']:.2f}")
        print(f"  (簇内平方和，越小越好)")

        return metrics

    def umap_2d_visualization(self, random_state=42):
        """2D降维用于可视化"""
        print("\n" + "="*70)
        print("步骤 5: 2D可视化降维")
        print("="*70)

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

    def visualize_results(self, save=True):
        """可视化聚类结果"""
        print("\n" + "="*70)
        print("步骤 6: 生成可视化图表")
        print("="*70)

        if self.data_2d is None:
            self.umap_2d_visualization()

        n_clusters = len(set(self.labels))

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))

        # 1. 2D散点图
        ax = axes[0, 0]
        scatter = ax.scatter(
            self.data_2d[:, 0],
            self.data_2d[:, 1],
            c=self.labels,
            cmap='Spectral',
            s=1,
            alpha=0.6
        )
        ax.set_title(f'K-means Clustering Results\n{n_clusters} Clusters',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1', fontsize=12)
        ax.set_ylabel('UMAP Dimension 2', fontsize=12)
        plt.colorbar(scatter, ax=ax, label='Cluster ID')

        # 2. 簇大小分布
        ax = axes[0, 1]
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        ax.bar(range(50), sorted(counts, reverse=True)[:50],
              color='steelblue', alpha=0.7)
        ax.set_title('Cluster Size Distribution (Top 50)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Cluster Rank', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # 3. 簇中心距离分布
        ax = axes[1, 0]
        center_distances = np.zeros(len(self.labels))
        for i in range(n_clusters):
            mask = self.labels == i
            if mask.sum() > 0:
                center = self.kmeans_model.cluster_centers_[i]
                distances = np.linalg.norm(
                    self.data_reduced[mask] - center, axis=1
                )
                center_distances[mask] = distances

        ax.hist(center_distances, bins=100, color='coral', alpha=0.7, edgecolor='black')
        ax.set_title('Distribution of Distances to Cluster Centers',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance to Center', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # 4. 簇大小统计
        ax = axes[1, 1]
        ax.boxplot(counts, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax.set_title('Cluster Size Statistics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cluster Size', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        stats_text = f"Mean: {counts.mean():.0f}\n"
        stats_text += f"Median: {np.median(counts):.0f}\n"
        stats_text += f"Std: {counts.std():.0f}\n"
        stats_text += f"Min: {counts.min()}\n"
        stats_text += f"Max: {counts.max()}"
        ax.text(1.15, 0.5, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save:
            save_path = self.run_dir / "kmeans_visualization.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")

        return fig

    def save_results(self, metrics):
        """保存结果"""
        print("\n" + "="*70)
        print("步骤 7: 保存结果")
        print("="*70)

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

        # 保存簇中心
        centers_path = self.run_dir / "cluster_centers.npy"
        np.save(centers_path, self.kmeans_model.cluster_centers_)
        print(f"✓ 簇中心已保存: {centers_path}")

        # 保存摘要
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        summary = {
            'timestamp': self.timestamp,
            'method': 'K-means',
            'data_shape': self.data.shape,
            'n_clusters': int(len(unique_labels)),
            'n_iter': int(self.kmeans_model.n_iter_),
            'inertia': float(self.kmeans_model.inertia_),
            'metrics': {k: float(v) for k, v in metrics.items()},
            'cluster_sizes': {
                'mean': float(counts.mean()),
                'std': float(counts.std()),
                'min': int(counts.min()),
                'max': int(counts.max()),
                'median': float(np.median(counts))
            }
        }

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ 结果摘要已保存: {summary_path}")

        # 生成报告
        self.generate_report(summary)

    def generate_report(self, summary):
        """生成分析报告"""
        report_path = self.run_dir / "analysis_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("K-means基线聚类分析报告\n")
            f.write("="*70 + "\n\n")

            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据文件: {self.data_path}\n")
            f.write(f"输出目录: {self.run_dir}\n\n")

            f.write("方法说明:\n")
            f.write("  K-means聚类是一种简单、经典的聚类算法\n")
            f.write("  原理: 迭代优化簇内平方和，将样本分配到最近的簇中心\n")
            f.write("  优点: 算法简单、计算快速、完全可解释\n")
            f.write("  缺点: 需要预设簇数、假设球形簇、对初始化敏感\n\n")

            f.write("数据信息:\n")
            f.write(f"  样本数: {summary['data_shape'][0]:,}\n")
            f.write(f"  原始维度: {summary['data_shape'][1]}\n")
            f.write(f"  降维后维度: {self.data_reduced.shape[1]}\n\n")

            f.write("聚类结果:\n")
            f.write(f"  簇数: {summary['n_clusters']}\n")
            f.write(f"  迭代次数: {summary['n_iter']}\n")
            f.write(f"  惯性 (Inertia): {summary['inertia']:.2f}\n\n")

            f.write("评估指标:\n")
            for metric_name, value in summary['metrics'].items():
                f.write(f"  {metric_name}: {value:.4f}\n")
            f.write("\n")

            f.write("簇大小统计:\n")
            for stat_name, value in summary['cluster_sizes'].items():
                f.write(f"  {stat_name}: {value:.2f}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("可复现性说明:\n")
            f.write("  本方法使用固定的random_state=42，保证结果完全可复现\n")
            f.write("  只需使用相同的参数运行，即可获得完全相同的结果\n")
            f.write("="*70 + "\n")

        print(f"✓ 分析报告已保存: {report_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("K-means基线聚类分析")
    print("="*70)

    # 初始化
    baseline = KMeansBaseline(
        data_path="data/bos_vectors_dim_83362_768.npy",
        output_dir="baseline_results"
    )

    # 加载数据
    baseline.load_data()

    # UMAP降维
    baseline.umap_reduction(n_components=75)

    # K-means聚类 (512簇)
    baseline.kmeans_clustering(
        n_clusters=512,
        random_state=42,
        n_init=10,
        use_minibatch=True  # 使用MiniBatch加速
    )

    # 评估
    metrics = baseline.evaluate_clustering()

    # 可视化
    baseline.umap_2d_visualization()
    baseline.visualize_results()

    # 保存结果
    baseline.save_results(metrics)

    print("\n" + "="*70)
    print("K-means基线分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
