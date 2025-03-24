import numpy as np
import pandas as pd
import logging
from sklearn.neighbors import LocalOutlierFactor

logger = logging.getLogger(__name__)


class LOFModel:
    """局部异常因子(Local Outlier Factor)模型封装类"""

    def __init__(self, n_neighbors=20, contamination=0.05, novelty=True, algorithm='auto'):
        """
        初始化LOF模型

        Args:
            n_neighbors: 用于计算局部密度的邻居数量
            contamination: 数据集中异常值的比例
            novelty: 是否以novelty模式运行(可以对新数据点进行预测)
            algorithm: 用于计算最近邻的算法('auto', 'ball_tree', 'kd_tree', 'brute')
        """
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty,
            algorithm=algorithm,
            n_jobs=-1  # 使用所有CPU核心
        )
        self.is_fitted = False

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

        # 检查novelty参数，只有在novelty=True时才能进行fit
        if not self.model.get_params()['novelty']:
            logger.warning("LOF model with novelty=False cannot be fitted. Using fit_predict instead.")
            self.model.fit_predict(X)
        else:
            self.model.fit(X)

        self.is_fitted = True
        return self

    def predict(self, X):
        """
        预测数据点是否为异常

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)

        Returns:
            numpy.ndarray: -1表示异常点，1表示正常点
        """
        if not self.is_fitted:
            logger.warning("Model is not fitted yet")
            return None

        # 确保X不包含NaN或inf
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("Input data contains NaN or inf values. Replacing with 0.")
            X = np.nan_to_num(X)

        return self.model.predict(X)

    def decision_function(self, X):
        """
        计算每个样本的异常分数

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)

        Returns:
            numpy.ndarray: 负异常分数，越低表示越异常
        """
        if not self.is_fitted:
            logger.warning("Model is not fitted yet")
            return None

        # 确保X不包含NaN或inf
        if np.isnan(X).any() or np.isinf(X).any():
            logger.warning("Input data contains NaN or inf values. Replacing with 0.")
            X = np.nan_to_num(X)

        return self.model.decision_function(X)

    def calculate_normalized_scores(self, X):
        """
        计算归一化的异常分数(0-100)，其中越高表示越异常

        Args:
            X: 特征矩阵，形状为(n_samples, n_features)

        Returns:
            numpy.ndarray: 0-100范围内的异常分数
        """
        scores = self.decision_function(X)
        if scores is None:
            return None

        # LOF的分数是负数，越低表示越异常，需要反转并归一化
        # 首先取反，使得越高表示越异常
        reversed_scores = -scores

        # 计算分数范围
        min_score = np.min(reversed_scores)
        max_score = np.max(reversed_scores)
        range_score = max_score - min_score

        # 避免除以零
        if range_score == 0:
            normalized_scores = np.zeros_like(reversed_scores)
        else:
            # 归一化到0-100
            normalized_scores = 100 * (reversed_scores - min_score) / range_score

        return normalized_scores

    def get_params(self):
        """
        获取模型参数

        Returns:
            dict: 模型参数字典
        """
        return self.model.get_params()

    def set_params(self, **params):
        """
        设置模型参数

        Args:
            **params: 参数键值对

        Returns:
            self: 返回模型实例
        """
        self.model.set_params(**params)
        return self

    def get_n_neighbors(self):
        """
        获取邻居数量

        Returns:
            int: 邻居数量
        """
        return self.model.n_neighbors_