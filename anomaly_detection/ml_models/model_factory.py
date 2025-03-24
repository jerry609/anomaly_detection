import logging
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .model_wrappers import IsolationForestWrapper, LOFWrapper, DBSCANWrapper
from .ensemble_model import EnsembleAnomalyDetector

logger = logging.getLogger(__name__)


class MLModelFactory:
    """机器学习模型工厂，负责创建和管理各种机器学习模型"""

    def __init__(self):
        """初始化模型工厂"""
        self.models = {}
        self.wrapped_models = {}  # 新增：存储包装后的模型
        self.preprocessors = {}

    def create_isolation_forest(self, n_estimators=100, contamination=0.05, random_state=42):
        """创建隔离森林模型"""
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples='auto',
            contamination=contamination,
            random_state=random_state
        )
        self.models['isolation_forest'] = model
        # 创建并存储包装后的模型
        wrapped_model = IsolationForestWrapper(model)
        self.wrapped_models['isolation_forest'] = wrapped_model
        return wrapped_model  # 返回包装后的模型

    def create_lof(self, n_neighbors=20, contamination=0.05):
        """创建局部异常因子模型"""
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )
        self.models['lof'] = model
        # 创建并存储包装后的模型
        wrapped_model = LOFWrapper(model)
        self.wrapped_models['lof'] = wrapped_model
        return wrapped_model  # 返回包装后的模型

    def create_dbscan(self, eps=0.5, min_samples=5, metric='euclidean'):
        """创建DBSCAN聚类模型"""
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )
        self.models['dbscan'] = model
        # 创建并存储包装后的模型
        wrapped_model = DBSCANWrapper(model)
        self.wrapped_models['dbscan'] = wrapped_model
        return wrapped_model  # 返回包装后的模型

    def create_scaler(self):
        """创建标准化器"""
        scaler = StandardScaler()
        self.preprocessors['scaler'] = scaler
        return scaler

    def create_pca(self, n_components=0.95):
        """创建PCA降维器"""
        pca = PCA(n_components=n_components)
        self.preprocessors['pca'] = pca
        return pca

    def get_model(self, model_name):
        """获取原始模型"""
        return self.models.get(model_name)

    def get_wrapped_model(self, model_name):
        """获取包装后的模型"""
        return self.wrapped_models.get(model_name)

    def get_preprocessor(self, preprocessor_name):
        """获取预处理器"""
        return self.preprocessors.get(preprocessor_name)

    def create_ensemble_detector(self, model_weights=None):
        """创建集成异常检测器"""
        ensemble = EnsembleAnomalyDetector(model_weights)

        # 添加所有可用的包装后模型
        for name, model in self.wrapped_models.items():
            ensemble.add_model(name, model)

        return ensemble

    def initialize_all_models(self):
        """初始化所有机器学习模型"""
        self.create_isolation_forest()
        self.create_lof()
        self.create_dbscan()
        self.create_scaler()
        self.create_pca()
        logger.info("Initialized all machine learning models")
