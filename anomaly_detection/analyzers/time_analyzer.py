import pandas as pd
import numpy as np
from ..utils.timestamp_processor import convert_timestamps, extract_time_features, create_time_windows


class TimeAnalyzer:
    """
    分析用户登录和活动模式，检测基于时间的异常。
    检测不寻常的访问时间、工作模式变化等。
    """

    def __init__(self, detector_model=None):
        """
        使用检测器模型初始化时间分析器。

        Args:
            detector_model (BaseDetector, optional): 用于异常检测的模型。
                                                    如果为None，将使用默认模型。
        """
        self.detector = detector_model
        self.user_time_profiles = {}  # 存储每个用户的时间配置文件

    def fit(self, logon_data):
        """
        基于历史数据构建用户时间配置文件。

        Args:
            logon_data (pd.DataFrame): 包含用户登录事件的DataFrame

        Returns:
            self: 返回实例自身
        """
        # 转换时间戳为datetime对象
        logon_data = convert_timestamps(logon_data, date_column='timestamp')

        # 提取时间特征，如一天中的小时、星期几等
        logon_data = extract_time_features(logon_data)

        # 按用户分组并构建时间配置文件
        for user_id, user_data in logon_data.groupby('user_id'):
            # 计算常规模式
            hour_distribution = user_data['hour'].value_counts(normalize=True)
            day_distribution = user_data['day_of_week'].value_counts(normalize=True)

            # 存储用户配置文件
            self.user_time_profiles[user_id] = {
                'hour_distribution': hour_distribution,
                'day_distribution': day_distribution,
                'avg_login_time': user_data['hour'].mean(),
                'std_login_time': user_data['hour'].std(),
                'usual_days': day_distribution[day_distribution > 0.1].index.tolist(),
                'is_business_hours_ratio': user_data['is_business_hours'].mean(),
                'is_weekend_ratio': user_data['is_weekend'].mean()
            }

        # 如果提供了检测器，则训练它
        if self.detector:
            self.detector.fit(logon_data)

        return self

    def detect_anomalies(self, new_logon_data):
        """
        在新的登录事件中检测基于时间的异常。

        Args:
            new_logon_data (pd.DataFrame): 要分析的新登录事件

        Returns:
            pd.DataFrame: 每个事件的异常分数和标志
        """
        # 准备数据
        new_logon_data = convert_timestamps(new_logon_data, date_column='timestamp')
        new_logon_data = extract_time_features(new_logon_data)

        results = []

        for idx, event in new_logon_data.iterrows():
            user_id = event['user_id']
            anomaly_score = 0
            anomaly_reason = []

            # 如果我们没有此用户的配置文件，则跳过
            if user_id not in self.user_time_profiles:
                results.append({
                    'event_id': idx,
                    'user_id': user_id,
                    'anomaly_score': 0,
                    'is_anomaly': False,
                    'reason': "没有用户配置文件"
                })
                continue

            profile = self.user_time_profiles[user_id]

            # 检查不寻常的登录时间
            hour = event['hour']
            if profile['hour_distribution'].get(hour, 0) < 0.05:
                anomaly_score += 0.5
                anomaly_reason.append(f"不寻常的登录小时: {hour}")

            # 检查不寻常的日期
            day = event['day_of_week']
            if day not in profile['usual_days']:
                anomaly_score += 0.3
                anomaly_reason.append(f"不寻常的登录日期: {day}")

            # 检查登录时间是否超出标准偏差
            if abs(hour - profile['avg_login_time']) > 2 * max(profile['std_login_time'], 1):
                anomaly_score += 0.4
                anomaly_reason.append("登录时间超出正常范围")

            # 检查非工作时间登录
            if not event['is_business_hours'] and profile['is_business_hours_ratio'] > 0.9:
                anomaly_score += 0.5
                anomaly_reason.append("非工作时间登录")

            # 检查周末登录
            if event['is_weekend'] and profile['is_weekend_ratio'] < 0.1:
                anomaly_score += 0.5
                anomaly_reason.append("周末登录")

            # 使用检测器模型（如果可用）
            if self.detector:
                model_score = self.detector.score(pd.DataFrame([event]))[0]
                anomaly_score = max(anomaly_score, model_score)

            results.append({
                'event_id': idx,
                'user_id': user_id,
                'timestamp': event['timestamp'],
                'anomaly_score': anomaly_score,
                'is_anomaly': anomaly_score > 0.7,  # 阈值
                'reason': ", ".join(anomaly_reason) if anomaly_reason else "正常行为"
            })

        return pd.DataFrame(results)
