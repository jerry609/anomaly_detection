import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class IsolationForestModel:
    """隔离森林模型封装类"""

    def __init__(self, n_estimators=100, contamination=0.05, random_state=42):
        """初始化隔离森林模型"""
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples='auto',
            contamination=contamination,
            random_state=random_state
        )
        self.is_fitted = False

    def fit(self, X):
        """训练模型"""
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, X):
        """预测离群点"""
        if not self.is_fitted:
            logger.warning("Model is not fitted yet")
            return None
        return self.model.predict(X)

    def decision_function(self, X):
        """计算异常分数"""
        if not self.is_fitted:
            logger.warning("Model is not fitted yet")
            return None
        return self.model.decision_function(X)

    def calculate_normalized_scores(self, X):
        """计算归一化的异常分数 (0-100)"""
        scores = self.decision_function(X)
        if scores is None:
            return None

        # 归一化分数到0-100范围，其中越低的分数表示越异常
        min_score = np.min(scores)
        max_score = np.max(scores)
        range_score = max_score - min_score

        if range_score == 0:
            return np.zeros_like(scores)

        # 转换为0-100的分数，数值越高表示越异常
        normalized_scores = 100 * (1 - ((scores - min_score) / range_score))
        return normalized_scores