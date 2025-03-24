"""
机器学习模型包装器，为各种模型提供统一的接口
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelWrapper:
    """基础模型包装器类"""

    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.is_fitted = False

    def fit(self, X):
        """训练模型"""
        try:
            self.model.fit(X)
            self.is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error fitting {self.name} model: {str(e)}")
            return self

    def predict(self, X):
        """预测异常"""
        if not self.is_fitted:
            logger.warning(f"{self.name} model is not fitted yet")
            return np.zeros(X.shape[0])
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error predicting with {self.name} model: {str(e)}")
            return np.zeros(X.shape[0])

    def calculate_normalized_scores(self, X):
        """计算归一化的异常分数"""
        raise NotImplementedError("子类必须实现calculate_normalized_scores方法")


class IsolationForestWrapper(ModelWrapper):
    """IsolationForest模型包装器"""

    def __init__(self, model):
        super().__init__(model, "isolation_forest")

    def calculate_normalized_scores(self, X):
        """计算归一化异常分数，分数越高表示越异常"""
        if not self.is_fitted:
            logger.warning(f"{self.name} model is not fitted yet")
            return np.zeros(X.shape[0])

        try:
            # decision_function返回负的异常分数，越负表示越异常
            # 我们取负值，这样分数越高表示越异常
            raw_scores = -self.model.decision_function(X)
            return self._normalize_scores(raw_scores)
        except Exception as e:
            logger.error(f"Error calculating scores with {self.name}: {str(e)}")
            return np.zeros(X.shape[0])

    def _normalize_scores(self, scores):
        """将分数归一化到0-1范围"""
        if len(scores) == 0 or np.max(scores) == np.min(scores):
            return np.zeros_like(scores)

        normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return normalized


class LOFWrapper(ModelWrapper):
    """LocalOutlierFactor模型包装器"""

    def __init__(self, model):
        super().__init__(model, "lof")

    def calculate_normalized_scores(self, X):
        """计算归一化异常分数，分数越高表示越异常"""
        if not self.is_fitted:
            logger.warning(f"{self.name} model is not fitted yet")
            return np.zeros(X.shape[0])

        try:
            # decision_function返回负的异常分数，越负表示越异常
            # 我们取负值，这样分数越高表示越异常
            raw_scores = -self.model.decision_function(X)
            return self._normalize_scores(raw_scores)
        except Exception as e:
            logger.error(f"Error calculating scores with {self.name}: {str(e)}")
            return np.zeros(X.shape[0])

    def _normalize_scores(self, scores):
        """将分数归一化到0-1范围"""
        if len(scores) == 0 or np.max(scores) == np.min(scores):
            return np.zeros_like(scores)

        normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        return normalized


class DBSCANWrapper(ModelWrapper):
    """DBSCAN模型包装器"""

    def __init__(self, model):
        super().__init__(model, "dbscan")
        self.labels_ = None

    def fit(self, X):
        """训练模型并保存标签"""
        try:
            self.model.fit(X)
            self.labels_ = self.model.labels_
            self.is_fitted = True
            return self
        except Exception as e:
            logger.error(f"Error fitting {self.name} model: {str(e)}")
            return self

    def predict(self, X):
        """
        DBSCAN没有原生的predict方法，实现一个简单版本
        返回-1表示异常点（噪声点），>= 0表示正常点（簇标签）
        """
        if not self.is_fitted or self.labels_ is None:
            logger.warning(f"{self.name} model is not fitted yet")
            return np.zeros(X.shape[0])

        try:
            # 这是一个简化版本，将所有新点都视为异常
            # 实际应用中可能需要更复杂的逻辑
            return np.ones(X.shape[0], dtype=int) * -1
        except Exception as e:
            logger.error(f"Error predicting with {self.name}: {str(e)}")
            return np.zeros(X.shape[0])

    def calculate_normalized_scores(self, X):
        """
        计算归一化异常分数
        由于DBSCAN没有内置的异常分数，使用一个简单的方法
        """
        if not self.is_fitted or self.labels_ is None:
            logger.warning(f"{self.name} model is not fitted yet")
            return np.zeros(X.shape[0])

        try:
            # 简化实现：将所有点视为同等异常
            # 实际应用中可能需要基于距离计算
            return np.ones(X.shape[0]) * 0.5
        except Exception as e:
            logger.error(f"Error calculating scores with {self.name}: {str(e)}")
            return np.zeros(X.shape[0])
