"""
组合多个机器学习模型的集成异常检测器
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnsembleAnomalyDetector:
    """集成多个异常检测模型"""

    def __init__(self, model_weights=None):
        """初始化集成模型

        Args:
            model_weights: 模型权重字典，如 {'isolation_forest': 0.4, 'lof': 0.4, 'dbscan': 0.2}
        """
        self.models = {}
        self.model_weights = model_weights or {'isolation_forest': 0.4, 'lof': 0.4, 'dbscan': 0.2}

    def add_model(self, name, model):
        """添加模型到集成"""
        self.models[name] = model
        return self

    def fit(self, X):
        """训练所有模型"""
        for name, model in self.models.items():
            try:
                model.fit(X)
                logger.debug(f"Fitted model {name}")
            except Exception as e:
                logger.error(f"Error fitting model {name}: {str(e)}")
        return self

    def predict_anomaly_scores(self, X):
        """预测综合异常分数"""
        scores = {}

        # 获取每个模型的分数
        for name, model in self.models.items():
            try:
                if hasattr(model, 'calculate_normalized_scores'):
                    # 如果模型有标准化分数计算方法，直接使用
                    model_scores = model.calculate_normalized_scores(X)
                elif name == 'dbscan' or hasattr(model, 'labels_'):
                    # DBSCAN处理特殊情况
                    labels = model.predict(X) if hasattr(model, 'predict') else np.ones(X.shape[0]) * -1
                    # -1表示异常点，将其设为1.0，其他设为0.0
                    model_scores = np.array([1.0 if label == -1 else 0.0 for label in labels])
                else:
                    # 默认情况，尝试使用predict函数
                    predictions = model.predict(X)
                    # 通常，-1表示异常，1表示正常，转换为分数
                    model_scores = np.array([1.0 if pred == -1 else 0.0 for pred in predictions])

                scores[name] = model_scores
            except Exception as e:
                logger.error(f"Error predicting with model {name}: {str(e)}")
                scores[name] = np.zeros(X.shape[0])

        # 计算加权平均分数
        weighted_scores = np.zeros(X.shape[0])
        total_weight = 0

        for name, model_scores in scores.items():
            weight = self.model_weights.get(name, 0)
            weighted_scores += weight * model_scores
            total_weight += weight

        # 归一化权重
        if total_weight > 0:
            weighted_scores /= total_weight

        return weighted_scores

    # 为了兼容性，添加predict方法
    def predict(self, X):
        """预测异常（兼容方法）

        Returns:
            numpy.ndarray: -1表示异常，1表示正常
        """
        scores = self.predict_anomaly_scores(X)
        # 通常异常分数阈值为0.5，高于阈值判定为异常
        return np.where(scores > 0.5, -1, 1)
