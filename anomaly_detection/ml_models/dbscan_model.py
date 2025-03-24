import numpy as np
import pandas as pd
import logging
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter

logger = logging.getLogger(__name__)


class DBSCANModel:
    """DBSCAN(基于密度的空间聚类)模型封装类"""

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', algorithm='auto'):
        """
        初始化DBSCAN模型

        Args:
            eps: 邻域半径，定义两个样本被视为邻居的最大距离
            min_samples: 一个核心样本的邻居数量，包括样本本身
            metric: 用于计算距离的指标('euclidean', 'manhattan', 'cosine'等)
            algorithm: 用于计算最近邻的算法('auto', 'ball_tree', 'kd_tree', 'brute')
        """
        self.model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            algorithm=algorithm,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.is_fitted = False
        self.labels_ = None
        self.n_clusters_ = 0
        self.noise_ratio_ = 0.0

    def fit(self, X):
        """
        训练模型

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)

        Returns:
            self: 返回模型实例
        """
        # 确保X不包含NaN或inf
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("Input data contains NaN or inf values. Replacing with 0.")
            X = np.nan_to_num(X)

        # 拟合模型
        self.model.fit(X)
        self.labels_ = self.model.labels_

        # 计算聚类数量（不包括噪声）
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)

        # 计算被标记为噪声的样本比例
        self.noise_ratio_ = np.sum(self.labels_ == -1) / len(self.labels_)

        logger.info(
            f"DBSCAN clustering complete. Found {self.n_clusters_} clusters, {self.noise_ratio_:.2%} noise points")
        self.is_fitted = True

        # 如果有多于一个聚类，计算轮廓分数
        if self.n_clusters_ > 1 and self.noise_ratio_ < 1.0:
            try:
                # 排除噪声点
                non_noise_mask = self.labels_ != -1
                if np.sum(non_noise_mask) > 1:
                    silhouette = silhouette_score(
                        X[non_noise_mask],
                        self.labels_[non_noise_mask]
                    )
                    logger.info(f"Silhouette score (excluding noise): {silhouette:.4f}")
            except Exception as e:
                logger.warning(f"Failed to calculate silhouette score: {str(e)}")

        return self

    def predict(self, X):
        """
        预测新数据点的簇标签

        注意: DBSCAN不支持直接predict，这里使用最近邻启发式方法

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)

        Returns:
            numpy.ndarray: 簇标签，-1表示噪声点
        """
        if not self.is_fitted:
            logger.warning("Model is not fitted yet")
            return None

        # 确保X不包含NaN或inf
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("Input data contains NaN or inf values. Replacing with 0.")
            X = np.nan_to_num(X)

        # DBSCAN本身不支持predict，这里返回拟合过程中的标签
        logger.warning("DBSCAN does not support predicting new samples. Returning fitted labels.")
        return self.labels_

    def calculate_anomaly_scores(self, X):
        """
        计算异常分数

        DBSCAN中，-1标签表示异常(噪声点)
        这里将噪声点的分数设为100，其他设为0

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)

        Returns:
            numpy.ndarray: 异常分数，100表示异常，0表示正常
        """
        if not self.is_fitted:
            logger.warning("Model is not fitted yet")
            return None

        # 创建异常分数数组，噪声点设为100，其他设为0
        anomaly_scores = np.where(self.labels_ == -1, 100.0, 0.0)
        return anomaly_scores

    def get_cluster_statistics(self):
        """
        获取聚类统计信息

        Returns:
            dict: 聚类统计信息字典
        """
        if not self.is_fitted:
            logger.warning("Model is not fitted yet")
            return {}

        # 计算每个簇的样本数量
        cluster_counts = Counter(self.labels_)

        # 创建统计信息字典
        stats = {
            'n_clusters': self.n_clusters_,
            'noise_ratio': self.noise_ratio_,
            'cluster_sizes': {label: count for label, count in cluster_counts.items() if label != -1},
            'noise_count': cluster_counts.get(-1, 0)
        }

        return stats

    def find_optimal_eps(self, X, eps_range=None, min_samples=5):
        """
        寻找最优的eps参数

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)
            eps_range: eps参数的候选值列表，如果为None则使用默认范围
            min_samples: min_samples参数

        Returns:
            dict: 最优参数和评估结果
        """
        # 确保X不包含NaN或inf
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("Input data contains NaN or inf values. Replacing with 0.")
            X = np.nan_to_num(X)

        # 默认eps范围
        if eps_range is None:
            eps_range = np.linspace(0.1, 2.0, 10)

        results = []

        for eps in eps_range:
            # 创建和拟合模型
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(X)

            # 获取标签
            labels = model.labels_

            # 计算聚类数量和噪声比例
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.sum(labels == -1) / len(labels)

            # 计算评分（如果可能）
            silhouette = None
            if n_clusters > 1 and noise_ratio < 1.0:
                try:
                    # 排除噪声点
                    non_noise_mask = labels != -1
                    if np.sum(non_noise_mask) > 1:
                        silhouette = silhouette_score(
                            X[non_noise_mask],
                            labels[non_noise_mask]
                        )
                except Exception:
                    pass

            # 记录结果
            results.append({
                'eps': eps,
                'n_clusters': n_clusters,
                'noise_ratio': noise_ratio,
                'silhouette': silhouette
            })

        # 找出最佳参数
        valid_results = [r for r in results if r['n_clusters'] > 1 and r['silhouette'] is not None]

        if valid_results:
            # 按轮廓分数排序
            best_result = max(valid_results, key=lambda r: r['silhouette'])
        else:
            # 如果没有有效结果，选择聚类数最合理且噪声比适中的结果
            best_result = min(results, key=lambda r:
            abs(r['n_clusters'] - 3) + abs(r['noise_ratio'] - 0.05)
                              )

        logger.info(
            f"Optimal eps: {best_result['eps']}, clusters: {best_result['n_clusters']}, noise ratio: {best_result['noise_ratio']:.2%}")

        # 使用最佳参数更新模型
        self.model = DBSCAN(eps=best_result['eps'], min_samples=min_samples)

        return best_result