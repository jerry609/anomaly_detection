import numpy as np
import pandas as pd
import logging
from datetime import datetime
from anomaly_detection.ml_models.model_factory import MLModelFactory
from anomaly_detection.ml_models.ensemble_model import EnsembleAnomalyDetector
from anomaly_detection.utils.feature_engineering import prepare_features_for_ml

logger = logging.getLogger(__name__)


class MLAnalyzer:
    """使用机器学习模型进行异常检测的分析器"""

    def __init__(self):
        """初始化ML分析器"""
        self.model_factory = MLModelFactory()
        self.ensemble_model = None
        self.features_by_user = {}
        self.anomaly_scores = {}

    def initialize_models(self):
        """初始化所有需要的机器学习模型"""
        self.model_factory.initialize_all_models()

        # 创建集成模型
        ensemble = EnsembleAnomalyDetector()
        ensemble.add_model('isolation_forest', self.model_factory.get_model('isolation_forest'))
        ensemble.add_model('lof', self.model_factory.get_model('lof'))
        ensemble.add_model('dbscan', self.model_factory.get_model('dbscan'))

        self.ensemble_model = ensemble
        return self

    def prepare_features(self, datasets, user_profiles):
        """准备用于机器学习的特征

        Args:
            datasets: 包含各类数据集的字典
            user_profiles: 用户配置文件字典

        Returns:
            dict: 用户ID到特征向量的映射
        """
        self.features_by_user = prepare_features_for_ml(datasets, user_profiles)
        return self.features_by_user

    def detect_anomalies(self, datasets, user_profiles, anomaly_threshold=60):
        """检测异常

        Args:
            datasets: 包含各类数据集的字典
            user_profiles: 用户配置文件字典
            anomaly_threshold: 异常分数阈值，超过此值被视为异常

        Returns:
            dict: 用户ID到异常列表的映射
        """
        # 准备特征
        self.prepare_features(datasets, user_profiles)

        if not self.features_by_user:
            logger.warning("No sufficient data for ML-based anomaly detection")
            return {}

        # 确保模型已初始化
        if self.ensemble_model is None:
            self.initialize_models()

        # 合并所有用户的特征为一个数据集
        user_ids = []
        feature_rows = []

        for user_id, features in self.features_by_user.items():
            user_ids.append(user_id)
            feature_rows.append(features)

        # 创建特征矩阵
        X = np.array(feature_rows)

        # 标准化特征
        scaler = self.model_factory.get_preprocessor('scaler')
        X_scaled = scaler.fit_transform(X)

        # 降维
        if X_scaled.shape[1] > 2:
            pca = self.model_factory.get_preprocessor('pca')
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled

        # 使用集成模型检测异常
        self.ensemble_model.fit(X_reduced)
        anomaly_scores = self.ensemble_model.predict_anomaly_scores(X_reduced)

        # 记录每个用户的异常分数
        self.anomaly_scores = {user_id: score for user_id, score in zip(user_ids, anomaly_scores)}

        # 基于阈值创建异常结果
        results = {}

        for i, user_id in enumerate(user_ids):
            score = anomaly_scores[i]

            if score > anomaly_threshold:
                anomaly = {
                    'type': 'ml_pattern',
                    'score': float(score),
                    'timestamp': pd.Timestamp.now(),
                    'description': "Machine learning detected unusual behavior pattern",
                    'details': {
                        'anomaly_score': float(score),
                        'pca_coordinates': X_reduced[i].tolist() if X_reduced.ndim > 1 else [float(X_reduced[i])]
                    }
                }

                if user_id not in results:
                    results[user_id] = []

                results[user_id].append(anomaly)

        logger.info(f"ML models detected anomalies for {len(results)} users")
        return results
