"""
预处理协调器，协调各种预处理操作。
"""

import logging
from anomaly_detection.preprocessors.data_loader import DataLoader
from anomaly_detection.profile.user_profile_builder import UserProfileBuilder
from anomaly_detection.utils.timestamp_processor import convert_timestamps
import config

logger = logging.getLogger(__name__)


class PreprocessingCoordinator:
    """协调和执行所有数据预处理步骤"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.datasets = {}
        self.preprocessed_datasets = {}
        self.user_profiles = {}

    def set_datasets(self, datasets):
        """
        设置数据集，用于替换当前的数据集

        Args:
            datasets: 数据集字典

        Returns:
            self: 返回自身以支持链式调用
        """
        self.datasets = datasets
        return self

    def get_raw_datasets(self):
        """
        获取原始数据集

        Returns:
            dict: 包含原始数据集的字典
        """
        return self.datasets  # 假设原始数据集存储在datasets属性中

    def load_datasets(self, dataset_paths):
        """加载所有数据集"""
        self.datasets = self.data_loader.load_datasets(dataset_paths)
        return self

    def preprocess_all(self):
        """执行所有预处理步骤"""
        self._preprocess_timestamps()
        self._build_user_profiles()
        return self

    def _preprocess_timestamps(self):
        """预处理所有数据集中的时间戳"""
        self.preprocessed_datasets = {}

        for name, dataset in self.datasets.items():
            # 处理时间戳
            processed_df = convert_timestamps(dataset,
                                              date_column='date',
                                              format=config.DATE_FORMAT,
                                              target_column='timestamp')
            self.preprocessed_datasets[name] = processed_df

        logger.info("Timestamp preprocessing complete")
        return self

    def _build_user_profiles(self):
        """构建用户配置文件"""
        profile_builder = UserProfileBuilder(self.preprocessed_datasets)
        self.user_profiles = profile_builder.build_all_profiles()
        logger.info(f"Built profiles for {len(self.user_profiles)} users")
        return self

    def get_datasets(self):
        """获取预处理后的数据集"""
        return self.preprocessed_datasets

    def get_user_profiles(self):
        """获取用户配置文件"""
        return self.user_profiles
