import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class TimeSeriesAnalyzer:
    """时间序列分析器，用于检测用户行为随时间的渐变异常"""

    def __init__(self):
        """初始化时间序列分析器"""
        self.historical_data = {}
        self.min_data_points = 7  # 至少需要7个数据点才能进行趋势分析

    def set_historical_data(self, historical_data):
        """设置历史数据"""
        self.historical_data = historical_data
        return self

    def detect_anomalies(self, datasets, user_profiles, risk_history=None):
        """检测时间序列异常

        Args:
            datasets: 当前数据集字典
            user_profiles: 用户配置文件字典
            risk_history: 风险历史记录，可选

        Returns:
            dict: 用户ID到异常列表的映射
        """
        logger.info("Detecting time series anomalies")

        # 获取或构建历史数据
        if not self.historical_data:
            self._build_historical_data(datasets, user_profiles, risk_history)

        # 如果仍然没有足够的历史数据，则返回空结果
        if not self.historical_data:
            logger.warning("Insufficient historical data for time series analysis")
            return {}

        # 检测每个用户的趋势异常
        anomalies = {}

        for user_id, user_profile in user_profiles.items():
            # 获取用户的历史数据
            user_history = self._get_user_historical_data(user_id)

            # 检查是否有足够的历史数据点
            if not user_history or len(user_history) < self.min_data_points:
                continue

            # 检测三种趋势
            login_time_trend = self._analyze_login_time_trend(user_history)
            access_trend = self._analyze_access_pattern_trend(user_history)
            communication_trend = self._analyze_communication_trend(user_history)

            # 收集显著趋势
            significant_trends = []

            # 登录时间趋势
            if login_time_trend > 0.6:
                significant_trends.append({
                    'type': 'login_time_drift',
                    'score': login_time_trend * 100,
                    'description': "Gradual shift in login times detected"
                })

            # 资源访问趋势
            if access_trend > 0.6:
                significant_trends.append({
                    'type': 'access_pattern_drift',
                    'score': access_trend * 100,
                    'description': "Gradual change in resource access patterns"
                })

            # 通信模式趋势
            if communication_trend > 0.6:
                significant_trends.append({
                    'type': 'communication_drift',
                    'score': communication_trend * 100,
                    'description': "Gradual shift in communication patterns"
                })

            # 如果检测到显著趋势，创建异常
            if significant_trends:
                user_anomalies = []

                for trend in significant_trends:
                    anomaly = {
                        'type': 'time_series',
                        'subtype': trend['type'],
                        'score': trend['score'],
                        'timestamp': pd.Timestamp.now(),
                        'description': trend['description'],
                        'details': {
                            'drift_type': trend['type'],
                            'trend_strength': trend['score'] / 100
                        }
                    }
                    user_anomalies.append(anomaly)

                if user_anomalies:
                    anomalies[user_id] = user_anomalies

        logger.info(f"Time series analysis detected anomalies for {len(anomalies)} users")
        return anomalies

    def _build_historical_data(self, datasets, user_profiles, risk_history):
        """构建历史数据用于趋势分析"""
        import logging
        logger = logging.getLogger(__name__)

        self.historical_data = {}

        # 根据实际数据集定义各数据集的用户ID和时间戳列
        dataset_columns = {
            'logon': {'user_id': 'user', 'timestamp': 'date'},
            'file': {'user_id': 'user', 'timestamp': 'date'},
            'email': {'user_id': 'from', 'timestamp': 'date'},
            'http': {'user_id': 'user', 'timestamp': 'date'},
            'device': {'user_id': 'user', 'timestamp': 'date'},
            'ldap': {'user_id': 'user', 'timestamp': None},  # LDAP可能没有时间戳
            'psychometric': {'user_id': 'employee_name', 'timestamp': None}  # 假设使用employee_name
        }

        # 处理每个用户
        for user_id in user_profiles:
            user_history = {
                'logon_frequency': [],
                'file_access_frequency': [],
                'email_frequency': [],
                'http_frequency': [],
                'risk_scores': []
            }

            # 遍历所有数据集类型
            for dataset_name in ['logon', 'file', 'email', 'http']:
                if dataset_name in datasets and not datasets[dataset_name].empty:
                    cols = dataset_columns[dataset_name]
                    try:
                        user_col = cols['user_id']
                        time_col = cols['timestamp']

                        # 确保列存在
                        if user_col in datasets[dataset_name].columns and time_col in datasets[dataset_name].columns:
                            # 筛选该用户的数据
                            user_data = datasets[dataset_name][datasets[dataset_name][user_col] == user_id]
                            if not user_data.empty:
                                frequency_key = f"{dataset_name}_frequency"
                                if dataset_name == 'logon':
                                    frequency_key = 'logon_frequency'
                                elif dataset_name == 'file':
                                    frequency_key = 'file_access_frequency'

                                user_history[frequency_key] = self._calculate_daily_frequency(user_data, time_col)
                        else:
                            missing_cols = []
                            if user_col not in datasets[dataset_name].columns:
                                missing_cols.append(user_col)
                            if time_col not in datasets[dataset_name].columns:
                                missing_cols.append(time_col)
                            logger.warning(f"Missing columns in {dataset_name} dataset: {', '.join(missing_cols)}")
                    except Exception as e:
                        logger.error(f"Error processing {dataset_name} data for user {user_id}: {str(e)}")

            # 添加风险得分历史
            if user_id in risk_history:
                user_history['risk_scores'] = risk_history[user_id]

            # 保存该用户的历史数据
            if any(len(series) > 0 for series in user_history.values()):
                self.historical_data[user_id] = user_history

        logger.info(f"Built historical data for {len(self.historical_data)} users")

    def _get_user_historical_data(self, user_id):
        """获取用户的历史行为数据"""
        return self.historical_data.get(user_id, [])

    def _analyze_login_time_trend(self, user_history):
        """分析登录时间的趋势变化"""
        # 简化的实现，在实际系统中应使用更复杂的时间序列分析
        # 如ARIMA, Prophet或其他时间序列模型

        # 假设user_history是一个包含每日行为的列表
        # 每个元素包含login_times键，值为当天登录时间的列表
        if not user_history or 'login_times' not in user_history[0]:
            return 0

        # 计算每天登录时间的平均值
        daily_avg_times = []

        for day_data in user_history:
            login_times = day_data.get('login_times', [])
            if login_times:
                # 转换为小时表示
                hour_values = [t.hour + t.minute / 60 for t in login_times]
                daily_avg_times.append(np.mean(hour_values))

        if len(daily_avg_times) < 5:
            return 0

        # 计算趋势的简单线性回归
        x = np.arange(len(daily_avg_times))
        y = np.array(daily_avg_times)

        # 计算线性回归的斜率
        if len(x) > 1:
            slope, _ = np.polyfit(x, y, 1)

            # 归一化斜率，转换为0-1的趋势强度
            # 假设4小时的变化是显著的
            normalized_slope = min(1.0, abs(slope * len(daily_avg_times) / 4))
            return normalized_slope

        return 0

    def _analyze_access_pattern_trend(self, user_history):
        """分析资源访问模式的趋势变化"""
        # 简化实现
        if not user_history or 'accessed_resources' not in user_history[0]:
            return 0

        # 计算资源访问的Jaccard相似度变化
        jaccard_scores = []

        for i in range(1, len(user_history)):
            prev_resources = set(user_history[i - 1].get('accessed_resources', []))
            curr_resources = set(user_history[i].get('accessed_resources', []))

            # 计算Jaccard相似度
            if prev_resources or curr_resources:
                intersection = len(prev_resources.intersection(curr_resources))
                union = len(prev_resources.union(curr_resources))
                similarity = intersection / union if union > 0 else 1
                jaccard_scores.append(similarity)

        if not jaccard_scores:
            return 0

        # 计算相似度的平均变化率
        avg_similarity = np.mean(jaccard_scores)

        # 转换为趋势强度 (1 - 平均相似度)
        trend_strength = 1 - avg_similarity
        return trend_strength

    def _analyze_communication_trend(self, user_history):
        """分析通信模式的趋势变化"""
        # 简化实现
        if not user_history or 'communication_partners' not in user_history[0]:
            return 0

        # 计算通信对象的变化趋势
        changes = []

        for i in range(1, len(user_history)):
            prev_partners = set(user_history[i - 1].get('communication_partners', []))
            curr_partners = set(user_history[i].get('communication_partners', []))

            # 计算新增的通信对象比例
            new_partners = curr_partners - prev_partners
            new_ratio = len(new_partners) / len(curr_partners) if curr_partners else 0
            changes.append(new_ratio)

        if not changes:
            return 0

        # 计算平均变化率
        avg_change = np.mean(changes)
        return avg_change

    def _calculate_daily_frequency(self, data, timestamp_column):
        """
        计算每日活动频率

        Args:
            data: 包含时间戳的数据框
            timestamp_column: 时间戳列名

        Returns:
            list: 按日期排序的活动频率列表
        """
        import pandas as pd

        # 确保时间戳列是datetime类型
        try:
            if pd.api.types.is_string_dtype(data[timestamp_column]):
                data['date'] = pd.to_datetime(data[timestamp_column]).dt.date
            else:
                data.loc[:, 'date'] = data[timestamp_column].dt.date

            # 按日期分组并计算每天的活动数量
            daily_counts = data.groupby('date').size()

            # 按日期排序并转换为列表
            sorted_counts = daily_counts.sort_index().tolist()

            return sorted_counts
        except Exception as e:
            self.logger.warning(f"Error calculating daily frequency: {str(e)}")
            return []
