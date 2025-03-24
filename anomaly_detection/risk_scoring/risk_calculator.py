import numpy as np
import pandas as pd
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskCalculator:
    """风险计算器，负责基于不同维度的异常计算综合风险分数"""

    def __init__(self):
        """初始化风险计算器"""
        self.logger = logger

    def calculate_risk_scores(self, anomalies, user_profiles, risk_config):
        """计算多维度风险评分

        Args:
            anomalies: 按类型和用户分组的异常字典
            user_profiles: 用户配置文件字典
            risk_config: 风险评分配置

        Returns:
            dict: 包含风险评分结果的字典
        """
        logger.info("Calculating multi-dimensional risk scores")

        # 初始化结果结构
        results = {
            'scores': {},  # 用户风险评分
            'dimensions': {},  # 各维度评分
            'factors': {}  # 详细风险因素
        }

        # 获取配置项
        category_weights = risk_config.get('category_weights', {
            'organizational': 3.0,
            'access': 2.5,
            'email': 2.0,
            'time': 1.5,
            'ml': 2.0,
            'timeseries': 2.2,
            'graph': 2.3
        })

        # 简化配置，直接固定分数
        base_score = 30  # 每个用户的基础分数
        anomaly_value = 10  # 每个异常的基础加分

        # 调试输出
        logger.info(f"处理 {len(user_profiles)} 个用户的风险评分")

        # 记录异常总数及其结构
        total_anomalies = 0
        categories_found = set()
        for category, user_anomalies_dict in anomalies.items():
            categories_found.add(category)
            for user_id, user_anomalies in user_anomalies_dict.items():
                total_anomalies += len(user_anomalies)
                # 检查第一个异常的结构
                if user_anomalies and len(user_anomalies) > 0:
                    logger.debug(f"异常结构示例: {str(user_anomalies[0])[:200]}...")

        logger.info(f"总共处理 {total_anomalies} 个异常，类别: {categories_found}")

        # 直接检测和记录异常数量
        user_anomaly_counts = {}
        user_anomalies_by_category = defaultdict(lambda: defaultdict(list))
        user_all_anomalies = defaultdict(list)

        # 收集所有用户的所有异常
        for category, user_anomalies_dict in anomalies.items():
            for user_id, user_anomalies in user_anomalies_dict.items():
                # 确保user_anomalies是列表类型
                if not isinstance(user_anomalies, list):
                    logger.warning(f"用户 {user_id} 的 {category} 异常不是列表类型，尝试转换")
                    try:
                        user_anomalies = list(user_anomalies)
                    except TypeError:
                        logger.warning(f"无法将用户 {user_id} 的 {category} 异常转换为列表，创建单元素列表")
                        user_anomalies = [{'type': category, 'score': 30.0, 'reason': "Unknown anomaly"}]

                # 计数
                if user_id not in user_anomaly_counts:
                    user_anomaly_counts[user_id] = 0
                user_anomaly_counts[user_id] += len(user_anomalies)

                # 收集异常
                for anomaly in user_anomalies:
                    # 确保anomaly是字典类型
                    if not isinstance(anomaly, dict):
                        logger.warning(f"跳过非字典类型的异常: {type(anomaly)}")
                        continue

                    # 确保类型字段存在
                    if 'type' not in anomaly:
                        anomaly['type'] = category

                    # 添加到对应的分类中
                    user_anomalies_by_category[user_id][category].append(anomaly)

                    # 添加到所有异常列表
                    user_all_anomalies[user_id].append(anomaly)

        # 给每个用户分配风险分数
        for user_id, count in user_anomaly_counts.items():
            # 获取该用户的所有异常
            all_anomalies = user_all_anomalies[user_id]

            # 计算平均异常分数
            avg_severity = 0
            severity_count = 0

            for anomaly in all_anomalies:
                if 'score' in anomaly:
                    try:
                        severity = float(anomaly['score'])
                        avg_severity += severity
                        severity_count += 1
                    except (ValueError, TypeError):
                        pass

            # 计算平均严重度
            if severity_count > 0:
                avg_severity = avg_severity / severity_count
            else:
                avg_severity = 30.0  # 默认严重度

            # 直接根据异常数量和严重度计算风险分数
            risk_score = min(100, base_score + (anomaly_value * count) + (avg_severity / 3))

            # 记录风险分数
            results['scores'][user_id] = risk_score

            # 记录维度信息
            results['dimensions'][user_id] = {
                'anomaly_count': count,
                'average_severity': avg_severity,
                'base_score': base_score,
                'computed_risk': risk_score
            }

            # 记录详细风险因素
            results['factors'][user_id] = {
                'anomaly_count': count,
                'average_severity': avg_severity,
                'categories': {}
            }

            # 记录各分类的异常数量
            for category, anomalies_list in user_anomalies_by_category[user_id].items():
                category_count = len(anomalies_list)
                results['factors'][user_id]['categories'][category] = category_count

                # 如果需要，这里可以记录每个分类的具体异常

        # 如果没有用户有异常，确保至少初始化结果
        if not user_anomaly_counts:
            for user_id in user_profiles.keys():
                results['scores'][user_id] = 0
                results['dimensions'][user_id] = {'anomaly_count': 0, 'average_severity': 0}
                results['factors'][user_id] = {'anomaly_count': 0, 'categories': {}}

        high_risk_users = sum(1 for s in results['scores'].values() if s > 70)
        medium_risk_users = sum(1 for s in results['scores'].values() if 40 <= s <= 70)
        logger.info(
            f"Risk score calculation complete. Found {high_risk_users} high-risk users and {medium_risk_users} medium-risk users")

        # 记录风险分数统计
        if results['scores']:
            max_score = max(results['scores'].values())
            min_score = min(results['scores'].values())
            avg_score = sum(results['scores'].values()) / len(results['scores'])
            logger.info(f"风险分数范围: 最低 {min_score}, 最高 {max_score}, 平均 {avg_score:.2f}")

        return results

    # 保留这些方法作为未来可能的扩展，但不在当前的calculate_risk_scores中使用
    def _calculate_time_sensitivity(self, anomaly):
        """计算异常的时间敏感度"""
        # 获取异常时间
        ts = self._get_anomaly_timestamp(anomaly)
        if ts is None:
            return 1.0  # 默认敏感度

        # 检查是否在非工作时间
        hour = ts.hour
        is_weekend = ts.dayofweek >= 5  # 5=周六，6=周日

        # 非工作时间敏感度更高
        if is_weekend:
            return 2.0
        elif hour < 6 or hour > 20:  # 凌晨或夜间
            return 1.8
        elif hour < 8 or hour > 18:  # 早晨或傍晚
            return 1.4
        else:
            return 1.0  # 工作时间

    def _calculate_data_sensitivity(self, anomaly, user_id, user_profiles):
        """计算数据敏感度"""
        # 检查涉及的资源类型
        resource_type = anomaly.get('resource_type', '')
        resource_id = anomaly.get('resource_id', '')

        # 默认敏感度
        sensitivity = 1.0

        # 检查敏感资源类型
        sensitive_types = ['financial', 'hr', 'customer', 'secret', 'confidential', 'pii']

        # 如果资源类型或ID包含敏感词，增加敏感度
        for s_type in sensitive_types:
            if (resource_type and s_type in resource_type.lower()) or \
                    (resource_id and s_type in str(resource_id).lower()):
                sensitivity *= 1.5
                break

        # 检查用户是否有权限访问该资源
        if anomaly.get('permission_level', '') == 'unauthorized':
            sensitivity *= 2.0

        return sensitivity

    def _calculate_behavior_deviation(self, anomaly, user_id, user_profiles):
        """计算行为偏差程度"""
        # 获取用户的基准行为
        user_profile = user_profiles.get(user_id, {})

        # 默认偏差度
        deviation = 1.0

        # 获取异常评分
        anomaly_score = anomaly.get('score', 0)

        # 从用户配置文件获取基准行为
        if 'behavior_baselines' in user_profile:
            baselines = user_profile['behavior_baselines']

            # 检查异常类型
            anomaly_type = anomaly.get('type', '')

            # 获取该类型的基准
            if anomaly_type in baselines:
                baseline = baselines[anomaly_type]

                # 计算偏差 (可使用自定义逻辑)
                if 'mean' in baseline and 'std' in baseline:
                    mean_val = baseline['mean']
                    std_val = baseline['std']

                    # 如果有标准差，计算z分数
                    if std_val > 0 and anomaly_score > 0:
                        z_score = abs(anomaly_score - mean_val) / std_val
                        deviation = 1.0 + min(z_score / 3.0, 2.0)  # 最多增加3倍

        # 检查是否为首次发生的行为
        if anomaly.get('is_first_occurrence', False):
            deviation *= 1.5

        return deviation

    def _get_anomaly_timestamp(self, anomaly):
        """从异常中提取时间戳"""
        # 尝试多种时间戳字段
        for field in ['timestamp', 'date', 'time', 'occurred_at']:
            if field in anomaly:
                ts = anomaly[field]
                # 转换为pandas Timestamp
                if isinstance(ts, str):
                    try:
                        return pd.to_datetime(ts)
                    except:
                        pass
                elif isinstance(ts, (pd.Timestamp, datetime)):
                    return pd.Timestamp(ts)
        return None

    def _get_user_sensitivity(self, user_id, user_profiles):
        """基于用户角色和部门计算敏感度系数"""
        user_profile = user_profiles.get(user_id, {})
        role = str(user_profile.get('role', '')).lower()
        dept = str(user_profile.get('department', '')).lower()

        # 基础敏感度
        sensitivity = 1.0

        # 敏感角色加权
        if any(r in role for r in ['admin', 'executive', 'finance', 'hr', 'legal']):
            sensitivity *= 1.5

        # 敏感部门加权
        if any(d in dept for d in ['finance', 'executive', 'legal', 'hr', 'it security']):
            sensitivity *= 1.3

        return sensitivity

    def _calculate_diversity_factor(self, category_counts):
        """计算用户异常的多样性因子"""
        # 计算存在异常的类别数量
        category_count = len(category_counts)

        # 计算多样性因子 (1.0-2.0)
        return 1.0 + min(1.0, (category_count - 1) * 0.25)

    def _calculate_time_concentration(self, anomalies):
        """计算异常的时间集中度"""
        if not anomalies:
            return 0

        # 收集所有异常的时间戳
        timestamps = []
        for anomaly in anomalies:
            # 尝试从不同的字段获取时间信息
            if 'timestamp' in anomaly:
                ts = anomaly['timestamp']
            elif 'date' in anomaly:
                ts = anomaly['date']
            else:
                continue

            # 确保是datetime对象
            if isinstance(ts, str):
                try:
                    ts = pd.to_datetime(ts)
                except:
                    continue

            timestamps.append(ts)

        if len(timestamps) <= 1:
            return 0

        # 计算时间戳的标准差，标准差越小表示越集中
        timestamps = pd.to_datetime(timestamps)
        std_days = timestamps.std().total_seconds() / (24 * 3600)

        # 转换为0-1的集中度分数
        if std_days > 30:  # 如果分布在30天以上
            return 0
        else:
            return 1 - (std_days / 30)
