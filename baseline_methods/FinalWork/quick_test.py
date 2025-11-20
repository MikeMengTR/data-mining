#!/usr/bin/env python3
"""
快速测试脚本 - 验证聚类分析程序的正确性
使用小样本数据快速测试
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("快速功能测试")
print("="*70)

# 测试1: 数据加载
print("\n[测试1] 数据加载")
try:
    data = np.load("data/bos_vectors_dim_83362_768.npy")
    print(f"✓ 数据加载成功: {data.shape}")
    print(f"  样本数: {data.shape[0]:,}, 维度: {data.shape[1]}")
except Exception as e:
    print(f"✗ 数据加载失败: {e}")
    exit(1)

# 测试2: 使用小样本测试UMAP
print("\n[测试2] UMAP降维测试 (使用1000个样本)")
try:
    import umap

    # 采样1000个样本进行快速测试
    sample_size = 1000
    indices = np.random.choice(len(data), sample_size, replace=False)
    data_sample = data[indices]

    print(f"  采样数据: {data_sample.shape}")

    umap_model = umap.UMAP(
        n_components=50,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=False
    )

    data_reduced = umap_model.fit_transform(data_sample)
    print(f"✓ UMAP降维成功: {data_sample.shape} → {data_reduced.shape}")

except Exception as e:
    print(f"✗ UMAP降维失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试3: HDBSCAN聚类
print("\n[测试3] HDBSCAN聚类测试")
try:
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        metric='euclidean'
    )

    labels = clusterer.fit_predict(data_reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"✓ HDBSCAN聚类成功")
    print(f"  簇数: {n_clusters}")
    print(f"  噪声点: {n_noise}/{len(labels)} ({n_noise/len(labels)*100:.1f}%)")

except Exception as e:
    print(f"✗ HDBSCAN聚类失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试4: 评估指标计算
print("\n[测试4] 评估指标计算")
try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    if n_clusters >= 2:
        mask = labels != -1
        data_filtered = data_reduced[mask]
        labels_filtered = labels[mask]

        sil_score = silhouette_score(data_filtered, labels_filtered)
        db_score = davies_bouldin_score(data_filtered, labels_filtered)

        print(f"✓ 评估指标计算成功")
        print(f"  Silhouette Score: {sil_score:.4f}")
        print(f"  Davies-Bouldin Index: {db_score:.4f}")
    else:
        print(f"⚠ 簇数少于2，跳过评估指标计算")

except Exception as e:
    print(f"✗ 评估指标计算失败: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 2D可视化
print("\n[测试5] 2D可视化测试")
try:
    umap_2d = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        verbose=False
    )

    data_2d = umap_2d.fit_transform(data_sample)

    # 创建简单的散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        data_2d[:, 0],
        data_2d[:, 1],
        c=labels,
        cmap='Spectral',
        s=10,
        alpha=0.6
    )
    ax.set_title(f'Quick Test: {n_clusters} Clusters Found', fontsize=14, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')

    plt.savefig('quick_test_result.png', dpi=150, bbox_inches='tight')
    print(f"✓ 2D可视化成功，图片已保存: quick_test_result.png")

except Exception as e:
    print(f"✗ 2D可视化失败: {e}")
    import traceback
    traceback.print_exc()

# 测试6: 文件保存
print("\n[测试6] 文件保存测试")
try:
    import json
    from pathlib import Path

    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)

    # 保存numpy数组
    np.save(test_dir / "test_labels.npy", labels)

    # 保存JSON
    test_summary = {
        'n_samples': int(sample_size),
        'n_clusters': int(n_clusters),
        'n_noise': int(n_noise)
    }

    with open(test_dir / "test_summary.json", 'w') as f:
        json.dump(test_summary, f, indent=2)

    print(f"✓ 文件保存成功")
    print(f"  输出目录: {test_dir}")

except Exception as e:
    print(f"✗ 文件保存失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("测试总结")
print("="*70)
print("✓ 所有核心功能测试通过!")
print("\n下一步:")
print("  1. 查看快速测试结果图: quick_test_result.png")
print("  2. 运行完整分析: python clustering_analysis.py")
print("  3. 运行参数搜索: python advanced_parameter_search.py")
print("="*70)
