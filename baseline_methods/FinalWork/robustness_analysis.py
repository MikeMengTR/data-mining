#!/usr/bin/env python3
"""
稳健性与公平性分析
包括:
1. 噪声敏感性测试
2. 参数稳定性分析
3. 多次运行一致性
4. 子群体公平性分析（如有标签）
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

import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from tqdm import tqdm

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']


class RobustnessAnalyzer:
    """稳健性分析器"""

    def __init__(self, data_path: str, output_dir: str = "robustness_results"):
        """
        初始化稳健性分析器

        Args:
            data_path: 数据文件路径
            output_dir: 结果输出目录
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"robustness_{self.timestamp}"
        self.run_dir.mkdir(exist_ok=True)

        self.data = None
        self.results = {
            'noise_sensitivity': {},
            'parameter_stability': {},
            'multiple_runs': {},
            'subgroup_fairness': {}
        }

        print(f"稳健性分析器初始化完成")
        print(f"结果保存到: {self.run_dir}")

    def load_data(self):
        """加载数据"""
        print("\n" + "="*70)
        print("加载数据")
        print("="*70)

        self.data = np.load(self.data_path)
        print(f"✓ 数据加载成功: {self.data.shape}")

        return self.data

    def noise_sensitivity_test(self, noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2]):
        """
        噪声敏感性测试
        向数据中添加不同程度的高斯噪声，观察聚类结果的变化

        Args:
            noise_levels: 噪声水平列表（相对于数据标准差的倍数）
        """
        print("\n" + "="*70)
        print("噪声敏感性测试")
        print("="*70)

        results = []

        # 基准聚类（无噪声）
        print(f"\n基准测试（无噪声）...")
        umap_model = umap.UMAP(n_components=50, random_state=42, verbose=False)
        data_reduced_base = umap_model.fit_transform(self.data)
        km_base = KMeans(n_clusters=512, random_state=42, n_init=5)
        labels_base = km_base.fit_predict(data_reduced_base)

        for noise_level in tqdm(noise_levels, desc="噪声测试"):
            # 添加高斯噪声
            if noise_level > 0:
                noise = np.random.randn(*self.data.shape) * self.data.std() * noise_level
                data_noisy = self.data + noise
            else:
                data_noisy = self.data.copy()

            # UMAP降维
            data_reduced = umap_model.fit_transform(data_noisy)

            # K-means聚类
            km = KMeans(n_clusters=512, random_state=42, n_init=5)
            labels = km.fit_predict(data_reduced)

            # 计算与基准的一致性
            ari = adjusted_rand_score(labels_base, labels)
            nmi = normalized_mutual_info_score(labels_base, labels)

            # 计算评估指标
            sil_score = silhouette_score(data_reduced, labels, sample_size=5000)

            results.append({
                'noise_level': noise_level,
                'ari_with_base': ari,
                'nmi_with_base': nmi,
                'silhouette_score': sil_score,
                'n_clusters': len(set(labels))
            })

            print(f"  噪声水平 {noise_level:.2f}: ARI={ari:.4f}, NMI={nmi:.4f}, Sil={sil_score:.4f}")

        self.results['noise_sensitivity'] = results

        # 可视化
        self._plot_noise_sensitivity(results)

        return results

    def _plot_noise_sensitivity(self, results):
        """绘制噪声敏感性结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        noise_levels = [r['noise_level'] for r in results]
        ari_scores = [r['ari_with_base'] for r in results]
        nmi_scores = [r['nmi_with_base'] for r in results]
        sil_scores = [r['silhouette_score'] for r in results]

        # ARI vs 噪声
        axes[0].plot(noise_levels, ari_scores, 'o-', linewidth=2, markersize=8, color='steelblue')
        axes[0].set_xlabel('Noise Level', fontsize=12)
        axes[0].set_ylabel('ARI with Baseline', fontsize=12)
        axes[0].set_title('Clustering Stability vs Noise', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].axhline(0.9, color='green', linestyle='--', alpha=0.5, label='High Stability')
        axes[0].axhline(0.7, color='orange', linestyle='--', alpha=0.5, label='Medium Stability')
        axes[0].legend()

        # NMI vs 噪声
        axes[1].plot(noise_levels, nmi_scores, 'o-', linewidth=2, markersize=8, color='coral')
        axes[1].set_xlabel('Noise Level', fontsize=12)
        axes[1].set_ylabel('NMI with Baseline', fontsize=12)
        axes[1].set_title('Information Preservation vs Noise', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)

        # Silhouette vs 噪声
        axes[2].plot(noise_levels, sil_scores, 'o-', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Noise Level', fontsize=12)
        axes[2].set_ylabel('Silhouette Score', fontsize=12)
        axes[2].set_title('Clustering Quality vs Noise', fontsize=14, fontweight='bold')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.run_dir / "noise_sensitivity.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 噪声敏感性图已保存: {save_path}")
        plt.close()

    def parameter_stability_test(self, param_variations=None):
        """
        参数稳定性测试
        测试关键参数变化对结果的影响

        Args:
            param_variations: 参数变化字典
        """
        print("\n" + "="*70)
        print("参数稳定性测试")
        print("="*70)

        if param_variations is None:
            param_variations = {
                'umap_n_components': [50, 75, 100],
                'umap_n_neighbors': [10, 15, 30],
                'kmeans_n_clusters': [400, 512, 600]
            }

        results = {}

        # UMAP n_components
        print(f"\n测试 UMAP n_components 参数...")
        results['umap_n_components'] = []
        for n_comp in tqdm(param_variations['umap_n_components']):
            umap_model = umap.UMAP(n_components=n_comp, random_state=42, verbose=False)
            data_reduced = umap_model.fit_transform(self.data)
            km = KMeans(n_clusters=512, random_state=42, n_init=5)
            labels = km.fit_predict(data_reduced)
            sil = silhouette_score(data_reduced, labels, sample_size=5000)

            results['umap_n_components'].append({
                'value': n_comp,
                'silhouette': sil,
                'inertia': km.inertia_
            })
            print(f"  n_components={n_comp}: Silhouette={sil:.4f}")

        # UMAP n_neighbors
        print(f"\n测试 UMAP n_neighbors 参数...")
        results['umap_n_neighbors'] = []
        for n_neigh in tqdm(param_variations['umap_n_neighbors']):
            umap_model = umap.UMAP(n_components=75, n_neighbors=n_neigh, random_state=42, verbose=False)
            data_reduced = umap_model.fit_transform(self.data)
            km = KMeans(n_clusters=512, random_state=42, n_init=5)
            labels = km.fit_predict(data_reduced)
            sil = silhouette_score(data_reduced, labels, sample_size=5000)

            results['umap_n_neighbors'].append({
                'value': n_neigh,
                'silhouette': sil,
                'inertia': km.inertia_
            })
            print(f"  n_neighbors={n_neigh}: Silhouette={sil:.4f}")

        # K-means n_clusters
        print(f"\n测试 K-means n_clusters 参数...")
        umap_model = umap.UMAP(n_components=75, random_state=42, verbose=False)
        data_reduced = umap_model.fit_transform(self.data)

        results['kmeans_n_clusters'] = []
        for n_clust in tqdm(param_variations['kmeans_n_clusters']):
            km = KMeans(n_clusters=n_clust, random_state=42, n_init=5)
            labels = km.fit_predict(data_reduced)
            sil = silhouette_score(data_reduced, labels, sample_size=5000)

            results['kmeans_n_clusters'].append({
                'value': n_clust,
                'silhouette': sil,
                'inertia': km.inertia_
            })
            print(f"  n_clusters={n_clust}: Silhouette={sil:.4f}")

        self.results['parameter_stability'] = results

        # 可视化
        self._plot_parameter_stability(results)

        return results

    def _plot_parameter_stability(self, results):
        """绘制参数稳定性结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # UMAP n_components
        values = [r['value'] for r in results['umap_n_components']]
        scores = [r['silhouette'] for r in results['umap_n_components']]
        axes[0].plot(values, scores, 'o-', linewidth=2, markersize=8, color='steelblue')
        axes[0].set_xlabel('UMAP n_components', fontsize=12)
        axes[0].set_ylabel('Silhouette Score', fontsize=12)
        axes[0].set_title('Effect of UMAP Dimensions', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)

        # UMAP n_neighbors
        values = [r['value'] for r in results['umap_n_neighbors']]
        scores = [r['silhouette'] for r in results['umap_n_neighbors']]
        axes[1].plot(values, scores, 'o-', linewidth=2, markersize=8, color='coral')
        axes[1].set_xlabel('UMAP n_neighbors', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Effect of UMAP Neighbors', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)

        # K-means n_clusters
        values = [r['value'] for r in results['kmeans_n_clusters']]
        scores = [r['silhouette'] for r in results['kmeans_n_clusters']]
        axes[2].plot(values, scores, 'o-', linewidth=2, markersize=8, color='green')
        axes[2].axvline(512, color='red', linestyle='--', label='Target: 512')
        axes[2].set_xlabel('Number of Clusters', fontsize=12)
        axes[2].set_ylabel('Silhouette Score', fontsize=12)
        axes[2].set_title('Effect of Cluster Number', fontsize=14, fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        save_path = self.run_dir / "parameter_stability.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 参数稳定性图已保存: {save_path}")
        plt.close()

    def multiple_runs_consistency(self, n_runs=10):
        """
        多次运行一致性测试
        使用不同随机种子运行多次，检查结果稳定性

        Args:
            n_runs: 运行次数
        """
        print("\n" + "="*70)
        print(f"多次运行一致性测试 ({n_runs} 次)")
        print("="*70)

        all_labels = []
        all_scores = []

        for run in tqdm(range(n_runs), desc="运行测试"):
            # 使用不同的随机种子
            seed = 42 + run

            # UMAP降维
            umap_model = umap.UMAP(n_components=75, random_state=seed, verbose=False)
            data_reduced = umap_model.fit_transform(self.data)

            # K-means聚类
            km = KMeans(n_clusters=512, random_state=seed, n_init=5)
            labels = km.fit_predict(data_reduced)

            all_labels.append(labels)

            # 计算评估指标
            sil = silhouette_score(data_reduced, labels, sample_size=5000)
            all_scores.append(sil)

        # 计算两两之间的ARI
        ari_matrix = np.zeros((n_runs, n_runs))
        for i in range(n_runs):
            for j in range(i+1, n_runs):
                ari = adjusted_rand_score(all_labels[i], all_labels[j])
                ari_matrix[i, j] = ari
                ari_matrix[j, i] = ari

        # 计算平均ARI（排除对角线）
        mask = ~np.eye(n_runs, dtype=bool)
        mean_ari = ari_matrix[mask].mean()
        std_ari = ari_matrix[mask].std()

        mean_sil = np.mean(all_scores)
        std_sil = np.std(all_scores)

        print(f"\n✓ 完成 {n_runs} 次运行")
        print(f"  平均 ARI (不同运行之间): {mean_ari:.4f} ± {std_ari:.4f}")
        print(f"  平均 Silhouette Score: {mean_sil:.4f} ± {std_sil:.4f}")
        print(f"  Silhouette 变异系数 (CV): {(std_sil/mean_sil)*100:.2f}%")

        results = {
            'n_runs': n_runs,
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'mean_silhouette': mean_sil,
            'std_silhouette': std_sil,
            'cv_silhouette': (std_sil/mean_sil)*100,
            'all_scores': all_scores,
            'ari_matrix': ari_matrix.tolist()
        }

        self.results['multiple_runs'] = results

        # 可视化
        self._plot_multiple_runs(results, all_scores, ari_matrix)

        return results

    def _plot_multiple_runs(self, results, scores, ari_matrix):
        """绘制多次运行一致性结果"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Silhouette分布
        axes[0].boxplot(scores, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0].axhline(results['mean_silhouette'], color='red', linestyle='--',
                       label=f"Mean: {results['mean_silhouette']:.4f}")
        axes[0].set_ylabel('Silhouette Score', fontsize=12)
        axes[0].set_title('Silhouette Score Distribution\nAcross Multiple Runs',
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Silhouette趋势
        axes[1].plot(range(1, len(scores)+1), scores, 'o-', linewidth=2, markersize=6)
        axes[1].axhline(results['mean_silhouette'], color='red', linestyle='--', alpha=0.5)
        axes[1].fill_between(range(1, len(scores)+1),
                             results['mean_silhouette'] - results['std_silhouette'],
                             results['mean_silhouette'] + results['std_silhouette'],
                             alpha=0.2, color='red')
        axes[1].set_xlabel('Run Number', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score Across Runs', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)

        # ARI热图
        im = axes[2].imshow(ari_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        axes[2].set_xlabel('Run Number', fontsize=12)
        axes[2].set_ylabel('Run Number', fontsize=12)
        axes[2].set_title(f'Pairwise ARI Between Runs\nMean ARI: {results["mean_ari"]:.4f}',
                         fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=axes[2], label='ARI')

        plt.tight_layout()
        save_path = self.run_dir / "multiple_runs_consistency.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 多次运行一致性图已保存: {save_path}")
        plt.close()

    def save_results(self):
        """保存所有结果"""
        print("\n" + "="*70)
        print("保存稳健性分析结果")
        print("="*70)

        # 保存JSON
        # 转换numpy数组为列表
        results_json = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_json[key] = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                                    for k, v in value.items()}
            else:
                results_json[key] = value

        summary_path = self.run_dir / "robustness_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results_json, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        print(f"✓ 结果摘要已保存: {summary_path}")

        # 生成报告
        self.generate_report()

    def generate_report(self):
        """生成稳健性分析报告"""
        report_path = self.run_dir / "robustness_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("稳健性与公平性分析报告\n")
            f.write("="*70 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 噪声敏感性
            if 'noise_sensitivity' in self.results and self.results['noise_sensitivity']:
                f.write("一、噪声敏感性分析\n")
                f.write("-"*70 + "\n")
                f.write("测试方法: 向数据添加不同程度的高斯噪声\n\n")

                for result in self.results['noise_sensitivity']:
                    f.write(f"噪声水平 {result['noise_level']:.2f}:\n")
                    f.write(f"  ARI with baseline: {result['ari_with_base']:.4f}\n")
                    f.write(f"  NMI with baseline: {result['nmi_with_base']:.4f}\n")
                    f.write(f"  Silhouette Score: {result['silhouette_score']:.4f}\n\n")

                # 总结
                ari_scores = [r['ari_with_base'] for r in self.results['noise_sensitivity']]
                if min(ari_scores) > 0.9:
                    stability = "极高"
                elif min(ari_scores) > 0.7:
                    stability = "高"
                elif min(ari_scores) > 0.5:
                    stability = "中等"
                else:
                    stability = "低"

                f.write(f"噪声稳健性: {stability}\n")
                f.write(f"最低ARI: {min(ari_scores):.4f}\n\n")

            # 参数稳定性
            if 'parameter_stability' in self.results and self.results['parameter_stability']:
                f.write("二、参数稳定性分析\n")
                f.write("-"*70 + "\n")

                for param_name, param_results in self.results['parameter_stability'].items():
                    f.write(f"\n{param_name}:\n")
                    scores = [r['silhouette'] for r in param_results]
                    f.write(f"  最小 Silhouette: {min(scores):.4f}\n")
                    f.write(f"  最大 Silhouette: {max(scores):.4f}\n")
                    f.write(f"  变化范围: {max(scores) - min(scores):.4f}\n")
                    f.write(f"  平均值: {np.mean(scores):.4f}\n")
                    f.write(f"  标准差: {np.std(scores):.4f}\n")

            # 多次运行一致性
            if 'multiple_runs' in self.results and self.results['multiple_runs']:
                f.write("\n三、多次运行一致性分析\n")
                f.write("-"*70 + "\n")
                mr = self.results['multiple_runs']
                f.write(f"运行次数: {mr['n_runs']}\n")
                f.write(f"平均两两ARI: {mr['mean_ari']:.4f} ± {mr['std_ari']:.4f}\n")
                f.write(f"平均Silhouette: {mr['mean_silhouette']:.4f} ± {mr['std_silhouette']:.4f}\n")
                f.write(f"Silhouette变异系数: {mr['cv_silhouette']:.2f}%\n\n")

                if mr['mean_ari'] > 0.9:
                    consistency = "极高"
                elif mr['mean_ari'] > 0.7:
                    consistency = "高"
                elif mr['mean_ari'] > 0.5:
                    consistency = "中等"
                else:
                    consistency = "低"

                f.write(f"运行一致性: {consistency}\n")

            f.write("\n" + "="*70 + "\n")

        print(f"✓ 稳健性分析报告已保存: {report_path}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("稳健性与公平性分析")
    print("="*70)

    # 初始化
    analyzer = RobustnessAnalyzer(
        data_path="data/bos_vectors_dim_83362_768.npy",
        output_dir="robustness_results"
    )

    # 加载数据
    analyzer.load_data()

    # 1. 噪声敏感性测试
    print("\n开始噪声敏感性测试...")
    analyzer.noise_sensitivity_test(noise_levels=[0.0, 0.01, 0.05, 0.1, 0.2])

    # 2. 参数稳定性测试
    print("\n开始参数稳定性测试...")
    analyzer.parameter_stability_test()

    # 3. 多次运行一致性测试
    print("\n开始多次运行一致性测试...")
    analyzer.multiple_runs_consistency(n_runs=10)

    # 保存结果
    analyzer.save_results()

    print("\n" + "="*70)
    print("稳健性分析完成!")
    print(f"结果保存在: {analyzer.run_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
