"""
数据加载组件，用于加载和初步验证数据。
"""

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """数据加载器，负责加载和验证原始数据"""

    def __init__(self):
        self.datasets = {}
        self.required_columns = {
            'logon': ['id', 'date', 'user', 'pc', 'activity'],
            'device': ['id', 'date', 'user', 'pc', 'activity'],
            'email': ['id', 'date', 'user', 'pc', 'to', 'cc', 'bcc', 'from', 'activity', 'size', 'attachments',
                      'content'],
            'file': ['id', 'date', 'user', 'pc', 'filename', 'content', 'activity'],
            'http': ['id', 'date', 'user', 'pc', 'url', 'content', 'activity'],
            # 更新LDAP列名以匹配实际数据结构
            'ldap': ['employee_name', 'user_id', 'email', 'role', 'business_unit', 'functional_unit', 'department',
                     'team', 'supervisor'],
            'psychometric': ['user_id', 'O', 'C', 'E', 'A', 'N']
        }

    def load_datasets(self, dataset_paths):
        """加载所有数据集并执行基本验证"""
        for name, path in dataset_paths.items():
            self._load_single_dataset(name, path)

        logger.info(f"Loaded {len(self.datasets)} datasets")
        return self.datasets

    def _load_single_dataset(self, name, path):
        """加载单个数据集并验证"""
        if not os.path.exists(path):
            logger.error(f"Dataset file not found: {path}")
            raise FileNotFoundError(f"Dataset file not found: {path}")

        try:
            df = pd.read_csv(path)
            logger.info(f"Loaded {name} dataset with {len(df)} rows")

            # 验证必要的列是否存在
            if name in self.required_columns:
                missing_columns = [col for col in self.required_columns[name] if col not in df.columns]
                if missing_columns:
                    logger.warning(f"Missing columns in {name} dataset: {missing_columns}")

            self.datasets[name] = df
        except Exception as e:
            logger.error(f"Error loading dataset {name}: {str(e)}")
            raise

        return df

    def get_datasets(self):
        """返回加载的所有数据集"""
        return self.datasets
