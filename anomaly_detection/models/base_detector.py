#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础异常检测器接口，定义所有检测器类应实现的方法。
"""

import abc
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING

# 使用TYPE_CHECKING条件导入以避免循环导入
if TYPE_CHECKING:
    from ..analyzers.time_analyzer import TimeAnalyzer
    from ..analyzers.access_analyzer import AccessAnalyzer
    from ..analyzers.email_analyzer import EmailAnalyzer
    from ..analyzers.org_analyzer import OrgAnalyzer


class BaseAnomalyDetector(abc.ABC):
    """
    异常检测器基类，定义所有检测器必须实现的接口。
    """

    def __init__(self):
        """初始化基础异常检测器"""
        self.user_profiles = {}
        self.preprocessed_data = {}
        self.anomaly_results = {}
        self.risk_scores = {}

    def set_preprocessed_data(self, preprocessed_data: Dict, user_profiles: Dict) -> None:
        """
        设置预处理后的数据和用户配置文件

        Args:
            preprocessed_data: 预处理后的数据集字典
            user_profiles: 用户配置文件字典
        """
        self.preprocessed_data = preprocessed_data
        self.user_profiles = user_profiles

    @abc.abstractmethod
    def detect_time_anomalies(self, time_analyzer=None):
        """
        检测时间相关的异常

        Args:
            time_analyzer: 时间分析器实例（如果None则创建新实例）

        Returns:
            Dict: 按用户ID组织的时间异常词典
        """
        pass

    @abc.abstractmethod
    def detect_access_anomalies(self, access_analyzer=None):
        """
        检测访问相关的异常

        Args:
            access_analyzer: 访问分析器实例（如果None则创建新实例）

        Returns:
            Dict: 按用户ID组织的访问异常词典
        """
        pass

    @abc.abstractmethod
    def detect_email_anomalies(self, email_analyzer=None):
        """
        检测电子邮件相关的异常

        Args:
            email_analyzer: 电子邮件分析器实例（如果None则创建新实例）

        Returns:
            Dict: 按用户ID组织的电子邮件异常词典
        """
        pass

    @abc.abstractmethod
    def detect_org_anomalies(self, org_analyzer=None):
        """
        检测组织相关的异常

        Args:
            org_analyzer: 组织分析器实例（如果None则创建新实例）

        Returns:
            Dict: 按用户ID组织的组织异常词典
        """
        pass

    @abc.abstractmethod
    def detect_all_anomalies(self):
        """
        运行所有异常检测分析器并更新异常结果

        Returns:
            Dict: 按类型和用户ID组织的所有异常结果
        """
        pass

    @abc.abstractmethod
    def calculate_risk_scores(self):
        """
        基于检测到的异常计算用户风险评分

        Returns:
            Dict: 用户ID到风险评分的映射
        """
        pass

    def get_high_risk_users(self, threshold=70):
        """
        获取高风险用户列表

        Args:
            threshold: 风险评分阈值（默认70）

        Returns:
            List: 高风险用户ID列表，按风险评分降序排序
        """
        # 确保已计算风险评分
        if not self.risk_scores:
            self.calculate_risk_scores()

        # 过滤并排序高风险用户
        high_risk = [(user_id, score) for user_id, score in self.risk_scores.items()
                     if score >= threshold]
        return sorted(high_risk, key=lambda x: x[1], reverse=True)

    def get_user_anomalies(self, user_id):
        """
        获取特定用户的所有异常

        Args:
            user_id: 用户ID

        Returns:
            Dict: 按异常类型组织的用户异常
        """
        result = {}
        for anomaly_type, anomalies_by_user in self.anomaly_results.items():
            if user_id in anomalies_by_user:
                result[anomaly_type] = anomalies_by_user[user_id]
        return result
