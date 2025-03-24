import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from scipy import stats
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class TrendDetector:
    """
    趋势检测器 - 用于检测用户行为和风险分数的趋势变化
    可以检测线性趋势、突变点和周期性模式
    """

    def __init__(self):
        """初始化趋势检测器"""
        self.min_data_points = 7  # 最少需要的数据点数量
        self.significance_level = 0.05  # 统计显著性水平
        self.change_threshold = 0.6  # 变化显著性阈值

    def detect_linear_trend(self, time_series, min_slope=0.1):
        """
        检测时间序列中的线性趋势

        Args:
            time_series: 时间序列数据的列表或数组
            min_slope: 最小斜率阈值，低于此值不被视为显著趋势

        Returns:
            dict: 包含趋势信息的字典
        """
        if len(time_series) < self.min_data_points:
            return {'has_trend': False, 'slope': 0, 'p_value': 1, 'trend_type': None}

        # 创建时间索引
        x = np.arange(len(time_series))
        y = np.array(time_series)

        # 计算线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # 判断趋势类型和显著性
        has_trend = (abs(slope) >= min_slope) and (p_value <= self.significance_level)
        trend_type = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else None

        # 计算趋势强度 (0-1之间)
        # R² 是确定系数，表示模型解释的方差比例
        trend_strength = r_value ** 2

        return {
            'has_trend': has_trend,
            'slope': slope,
            'intercept': intercept,
            'p_value': p_value,
            'r_squared': r_value ** 2,
            'std_err': std_err,
            'trend_type': trend_type,
            'trend_strength': trend_strength
        }

    def detect_change_points(self, time_series, window_size=3):
        """
        检测时间序列中的变化点

        Args:
            time_series: 时间序列数据的列表或数组
            window_size: 滑动窗口大小，用于计算均值变化

        Returns:
            list: 变化点索引列表
        """
        if len(time_series) < (window_size * 2):
            return []

        # 计算滑动窗口均值
        change_points = []
        ts = np.array(time_series)

        for i in range(window_size, len(ts) - window_size):
            window1 = ts[i - window_size:i]
            window2 = ts[i:i + window_size]

            # 计算窗口均值
            mean1 = np.mean(window1)
            mean2 = np.mean(window2)

            # 计算窗口标准差
            std1 = np.std(window1) or 1.0  # 避免除零
            std2 = np.std(window2) or 1.0

            # 计算均值变化的Z分数
            mean_change = abs(mean2 - mean1)
            pooled_std = np.sqrt((std1 ** 2 + std2 ** 2) / 2)
            z_score = mean_change / pooled_std

            # 如果变化超过阈值，则标记为变化点
            if z_score > 1.96:  # 95%置信度对应的Z分数
                change_points.append({
                    'index': i,
                    'z_score': z_score,
                    'before_mean': mean1,
                    'after_mean': mean2,
                    'change_magnitude': mean_change,
                    'relative_change': mean_change / (mean1 if mean1 != 0 else 1)
                })

        # 按变化幅度排序
        change_points.sort(key=lambda x: x['z_score'], reverse=True)
        return change_points

    def detect_seasonality(self, time_series, periods_to_test=[7, 14, 30]):
        """
        检测时间序列中的季节性或周期性模式

        Args:
            time_series: 时间序列数据的列表或数组
            periods_to_test: 要测试的周期长度列表

        Returns:
            dict: 包含季节性信息的字典
        """
        if len(time_series) < max(periods_to_test) * 2:
            return {'has_seasonality': False, 'period': None, 'strength': 0}

        # 自相关检测
        best_period = None
        best_corr = 0
        ts = np.array(time_series)

        for period in periods_to_test:
            if len(ts) <= period * 2:
                continue

            # 计算滞后为period的自相关系数
            lagged_ts = ts[:-period]
            current_ts = ts[period:]

            if len(lagged_ts) > 1:
                corr = np.corrcoef(lagged_ts, current_ts)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_period = period

        # 判断是否存在季节性
        has_seasonality = abs(best_corr) > self.change_threshold

        return {
            'has_seasonality': has_seasonality,
            'period': best_period,
            'correlation': best_corr,
            'strength': abs(best_corr)
        }

    def analyze_risk_trend(self, risk_history, days=90):
        """
        分析风险分数的趋势

        Args:
            risk_history: 包含日期和分数的历史数据列表
            days: 要分析的天数

        Returns:
            dict: 包含趋势分析结果的字典
        """
        # 确保风险历史数据是按日期排序的
        if not risk_history:
            return {
                'trend': None,
                'change_points': [],
                'seasonality': {'has_seasonality': False}
            }

        # 提取日期和分数
        dates = []
        scores = []

        for entry in risk_history:
            if isinstance(entry, dict):
                dates.append(entry.get('date'))
                scores.append(entry.get('score', 0))
            elif isinstance(entry, tuple) and len(entry) >= 2:
                dates.append(entry[0])
                scores.append(entry[1])

        if not scores:
            return {
                'trend': None,
                'change_points': [],
                'seasonality': {'has_seasonality': False}
            }

        # 将日期转换为相对天数
        if isinstance(dates[0], (datetime, pd.Timestamp)):
            start_date = min(dates)
            days_from_start = [(d - start_date).days for d in dates]
        else:
            days_from_start = list(range(len(scores)))

        # 检测线性趋势
        trend = self.detect_linear_trend(scores)

        # 检测变化点
        change_points = self.detect_change_points(scores)

        # 检测季节性
        seasonality = self.detect_seasonality(scores)

        return {
            'trend': trend,
            'change_points': change_points,
            'seasonality': seasonality,
            'days_analyzed': len(scores),
            'start_date': dates[0] if dates else None,
            'end_date': dates[-1] if dates else None,
            'average_score': np.mean(scores),
            'max_score': max(scores),
            'current_score': scores[-1] if scores else None
        }

    def predict_future_risk(self, risk_history, days_ahead=30):
        """
        基于历史趋势预测未来风险分数

        Args:
            risk_history: 包含日期和分数的历史数据列表
            days_ahead: 预测未来的天数

        Returns:
            list: 预测的风险分数列表
        """
        # 分析趋势
        analysis = self.analyze_risk_trend(risk_history)

        # 提取当前分数
        if not risk_history:
            return [0] * days_ahead

        current_score = 0
        if isinstance(risk_history[-1], dict):
            current_score = risk_history[-1].get('score', 0)
        elif isinstance(risk_history[-1], tuple) and len(risk_history[-1]) >= 2:
            current_score = risk_history[-1][1]

        # 如果没有显著趋势，使用当前分数作为预测
        if not analysis['trend']['has_trend']:
            if analysis['seasonality']['has_seasonality']:
                # 如果有季节性，则使用最近一个周期的数据
                period = analysis['seasonality']['period']
                if period and len(risk_history) >= period:
                    recent_period = [
                        entry[1] if isinstance(entry, tuple) else entry.get('score', 0)
                        for entry in risk_history[-period:]
                    ]
                    # 循环使用季节性模式
                    predictions = []
                    for i in range(days_ahead):
                        predictions.append(recent_period[i % len(recent_period)])
                    return predictions
            return [current_score] * days_ahead

        # 使用线性趋势进行预测
        trend = analysis['trend']
        slope = trend['slope']
        intercept = trend['intercept']

        # 计算预测值
        last_index = len(risk_history) - 1
        predictions = []

        for i in range(days_ahead):
            predicted_score = slope * (last_index + i + 1) + intercept
            # 确保预测值在有效范围内 (0-100)
            predicted_score = max(0, min(100, predicted_score))
            predictions.append(predicted_score)

        return predictions