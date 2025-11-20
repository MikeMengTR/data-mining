#!/usr/bin/env python3
"""
高级参数搜索脚本 - 全面探索HDBSCAN参数空间
目标：找到接近512个簇的最优参数组合
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import umap
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm


class AdvancedParameterSearch:
    """高级参数搜索器"""

    def __init__(self, data_path: str, output_dir: str = "parameter_search_results"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"search_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        print(f"初始化参数搜索器")
        print(f"结果保存到: {self.run_dir}")

    def load_data(self):
        """加载数据"""
        print("\n加载数据...")
        self.data = np.load(self.data_path)
        print(f"✓ 数据形状: {self.data.shape}")
        return self.data

    def grid_search(
        self,
        target_clusters: int = 512,
        umap_params: Dict = None,
        hdbscan_params: Dict = None
    ):
        """
        网格搜索

        Args:
            target_clusters: 目标簇数
            umap_params: UMAP参数候选字典
            hdbscan_params: HDBSCAN参数候选字典
        """
        # 默认参数空间
        if umap_params is None:
            umap_params = {
                'n_components': [50, 75, 100],
                'n_neighbors': [15, 30, 50],
                'min_dist': [0.0, 0.1, 0.2]
            }

        if hdbscan_params is None:
            hdbscan_params = {
                'min_cluster_size': [30, 50, 80, 100, 150, 200],
                'min_samples': [5, 10, 15, 20, 30],
                'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.3]
            }

        print(f"\n{'='*70}")
        print("开始网格搜索")
        print(f"{'='*70}")
        print(f"目标簇数: {target_clusters}")
        print(f"\nUMAP参数空间:")
        for key, values in umap_params.items():
            print(f"  {key}: {values}")
        print(f"\nHDBSCAN参数空间:")
        for key, values in hdbscan_params.items():
            print(f"  {key}: {values}")

        # 计算总组合数
        total_combinations = (
            len(umap_params['n_components']) *
            len(umap_params['n_neighbors']) *
            len(umap_params['min_dist']) *
            len(hdbscan_params['min_cluster_size']) *
            len(hdbscan_params['min_samples']) *
            len(hdbscan_params['cluster_selection_epsilon'])
        )
        print(f"\n总组合数: {total_combinations}")
        print("开始搜索...\n")

        results = []
        best_result = None
        best_diff = float('inf')

        # 进度条
        pbar = tqdm(total=total_combinations, desc="参数搜索")

        for umap_n_comp in umap_params['n_components']:
            for umap_n_neigh in umap_params['n_neighbors']:
                for umap_min_dist in umap_params['min_dist']:
                    # UMAP降维
                    try:
                        umap_model = umap.UMAP(
                            n_components=umap_n_comp,
                            n_neighbors=umap_n_neigh,
                            min_dist=umap_min_dist,
                            metric='cosine',
                            random_state=42,
                            verbose=False
                        )
                        data_reduced = umap_model.fit_transform(self.data)

                    except Exception as e:
                        pbar.write(f"UMAP失败: {e}")
                        pbar.update(
                            len(hdbscan_params['min_cluster_size']) *
                            len(hdbscan_params['min_samples']) *
                            len(hdbscan_params['cluster_selection_epsilon'])
                        )
                        continue

                    for min_cls_size in hdbscan_params['min_cluster_size']:
                        for min_samp in hdbscan_params['min_samples']:
                            for eps in hdbscan_params['cluster_selection_epsilon']:
                                pbar.update(1)

                                try:
                                    # HDBSCAN聚类
                                    clusterer = hdbscan.HDBSCAN(
                                        min_cluster_size=min_cls_size,
                                        min_samples=min_samp,
                                        cluster_selection_epsilon=eps,
                                        metric='euclidean',
                                        core_dist_n_jobs=-1
                                    )
                                    labels = clusterer.fit_predict(data_reduced)

                                    # 统计
                                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                    n_noise = (labels == -1).sum()
                                    noise_ratio = n_noise / len(labels)

                                    # 计算评估指标
                                    metrics = {}
                                    if n_clusters >= 2:
                                        mask = labels != -1
                                        if mask.sum() > 0:
                                            try:
                                                metrics['silhouette'] = silhouette_score(
                                                    data_reduced[mask],
                                                    labels[mask],
                                                    sample_size=min(5000, mask.sum())
                                                )
                                            except:
                                                metrics['silhouette'] = None

                                            try:
                                                metrics['davies_bouldin'] = davies_bouldin_score(
                                                    data_reduced[mask],
                                                    labels[mask]
                                                )
                                            except:
                                                metrics['davies_bouldin'] = None

                                    # 记录结果
                                    result = {
                                        'umap_n_components': umap_n_comp,
                                        'umap_n_neighbors': umap_n_neigh,
                                        'umap_min_dist': umap_min_dist,
                                        'min_cluster_size': min_cls_size,
                                        'min_samples': min_samp,
                                        'cluster_selection_epsilon': eps,
                                        'n_clusters': n_clusters,
                                        'n_noise': int(n_noise),
                                        'noise_ratio': float(noise_ratio),
                                        'diff_from_target': abs(n_clusters - target_clusters),
                                        **metrics
                                    }
                                    results.append(result)

                                    # 更新最佳结果
                                    diff = abs(n_clusters - target_clusters)
                                    if diff < best_diff:
                                        best_diff = diff
                                        best_result = result
                                        pbar.write(
                                            f"⭐ 新最佳: {n_clusters}簇 "
                                            f"(UMAP:{umap_n_comp}D, "
                                            f"min_cls:{min_cls_size}, "
                                            f"min_samp:{min_samp}, "
                                            f"eps:{eps})"
                                        )

                                except Exception as e:
                                    pbar.write(f"聚类失败: {e}")
                                    continue

        pbar.close()

        # 保存结果
        self.results_df = pd.DataFrame(results)
        self.best_result = best_result

        print(f"\n{'='*70}")
        print("搜索完成!")
        print(f"{'='*70}")
        print(f"有效结果数: {len(results)}")
        print(f"\n最佳参数组合:")
        for key, value in best_result.items():
            print(f"  {key}: {value}")

        return self.results_df, best_result

    def analyze_results(self):
        """分析搜索结果"""
        print(f"\n{'='*70}")
        print("结果分析")
        print(f"{'='*70}")

        df = self.results_df

        # Top 10结果
        print("\nTop 10 最接近目标的参数组合:")
        top10 = df.nsmallest(10, 'diff_from_target')
        print(top10[[
            'n_clusters', 'diff_from_target',
            'umap_n_components', 'min_cluster_size', 'min_samples',
            'noise_ratio', 'silhouette'
        ]].to_string(index=False))

        # 统计分析
        print(f"\n簇数分布:")
        print(f"  最小: {df['n_clusters'].min()}")
        print(f"  最大: {df['n_clusters'].max()}")
        print(f"  平均: {df['n_clusters'].mean():.1f}")
        print(f"  中位数: {df['n_clusters'].median():.0f}")

        print(f"\n噪声比例分布:")
        print(f"  最小: {df['noise_ratio'].min():.2%}")
        print(f"  最大: {df['noise_ratio'].max():.2%}")
        print(f"  平均: {df['noise_ratio'].mean():.2%}")

        # 保存详细结果
        csv_path = self.run_dir / "search_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ 详细结果已保存: {csv_path}")

        # 保存Top 10
        top10_path = self.run_dir / "top10_results.csv"
        top10.to_csv(top10_path, index=False)
        print(f"✓ Top10结果已保存: {top10_path}")

    def visualize_results(self):
        """可视化搜索结果"""
        print(f"\n{'='*70}")
        print("生成可视化")
        print(f"{'='*70}")

        df = self.results_df

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        # 1. 簇数分布直方图
        ax = axes[0, 0]
        ax.hist(df['n_clusters'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(512, color='red', linestyle='--', linewidth=2, label='Target: 512')
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Cluster Numbers', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. 簇数 vs UMAP维度
        ax = axes[0, 1]
        for dim in df['umap_n_components'].unique():
            subset = df[df['umap_n_components'] == dim]
            ax.scatter(subset.index, subset['n_clusters'], label=f'UMAP {dim}D', alpha=0.6, s=20)
        ax.axhline(512, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Experiment Index', fontsize=12)
        ax.set_ylabel('Number of Clusters', fontsize=12)
        ax.set_title('Cluster Count by UMAP Dimension', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 3. 簇数 vs min_cluster_size
        ax = axes[0, 2]
        scatter = ax.scatter(
            df['min_cluster_size'],
            df['n_clusters'],
            c=df['noise_ratio'],
            cmap='viridis',
            alpha=0.6,
            s=30
        )
        ax.axhline(512, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('min_cluster_size', fontsize=12)
        ax.set_ylabel('Number of Clusters', fontsize=12)
        ax.set_title('Clusters vs min_cluster_size', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Noise Ratio')
        ax.grid(alpha=0.3)

        # 4. 噪声比例分布
        ax = axes[1, 0]
        ax.hist(df['noise_ratio'], bins=50, color='coral', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Noise Ratio', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Noise Ratio', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # 5. Silhouette Score vs 簇数
        ax = axes[1, 1]
        valid_silhouette = df[df['silhouette'].notna()]
        scatter = ax.scatter(
            valid_silhouette['n_clusters'],
            valid_silhouette['silhouette'],
            c=valid_silhouette['noise_ratio'],
            cmap='plasma',
            alpha=0.6,
            s=30
        )
        ax.axvline(512, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title('Silhouette Score vs Cluster Count', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Noise Ratio')
        ax.grid(alpha=0.3)

        # 6. 参数热图：min_cluster_size vs min_samples
        ax = axes[1, 2]
        pivot = df.pivot_table(
            values='n_clusters',
            index='min_samples',
            columns='min_cluster_size',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Avg Clusters'})
        ax.set_title('Avg Clusters: min_samples vs min_cluster_size', fontsize=14, fontweight='bold')

        plt.tight_layout()

        # 保存
        save_path = self.run_dir / "parameter_search_visualization.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 可视化已保存: {save_path}")

        plt.show()

    def save_summary(self):
        """保存搜索摘要"""
        summary = {
            'timestamp': self.timestamp,
            'total_experiments': len(self.results_df),
            'best_result': self.best_result,
            'statistics': {
                'n_clusters': {
                    'min': int(self.results_df['n_clusters'].min()),
                    'max': int(self.results_df['n_clusters'].max()),
                    'mean': float(self.results_df['n_clusters'].mean()),
                    'median': float(self.results_df['n_clusters'].median())
                },
                'noise_ratio': {
                    'min': float(self.results_df['noise_ratio'].min()),
                    'max': float(self.results_df['noise_ratio'].max()),
                    'mean': float(self.results_df['noise_ratio'].mean())
                }
            }
        }

        summary_path = self.run_dir / "search_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ 搜索摘要已保存: {summary_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("高级参数搜索 - 目标: 512个簇")
    print("="*70)

    # 初始化
    searcher = AdvancedParameterSearch(
        data_path="data/bos_vectors_dim_83362_768.npy",
        output_dir="parameter_search_results"
    )

    # 加载数据
    searcher.load_data()

    # 定义参数空间（根据目标512簇调整）
    umap_params = {
        'n_components': [50, 75, 100],  # 尝试不同降维维度
        'n_neighbors': [15, 30],  # 邻居数
        'min_dist': [0.0, 0.1]  # 最小距离
    }

    hdbscan_params = {
        'min_cluster_size': [50, 80, 100, 120, 150],  # 簇的最小大小
        'min_samples': [5, 10, 15, 20],  # 核心点的最小样本数
        'cluster_selection_epsilon': [0.0, 0.1, 0.2]  # 簇选择阈值
    }

    # 网格搜索
    results_df, best_result = searcher.grid_search(
        target_clusters=512,
        umap_params=umap_params,
        hdbscan_params=hdbscan_params
    )

    # 分析结果
    searcher.analyze_results()

    # 可视化
    searcher.visualize_results()

    # 保存摘要
    searcher.save_summary()

    print(f"\n{'='*70}")
    print("参数搜索完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
