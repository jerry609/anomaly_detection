import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict

logger = logging.getLogger(__name__)


class RiskHistoryTracker:
    """
    风险历史跟踪器 - 负责记录、持久化和分析用户风险分数的历史变化
    """

    def __init__(self, history_file=None, max_history_days=180):
        """
        初始化风险历史跟踪器

        Args:
            history_file: 风险历史数据文件路径，可选
            max_history_days: 保留的最大历史天数
        """
        self.history_file = history_file
        self.max_history_days = max_history_days
        self.risk_history = defaultdict(list)
        self.last_update = None
        self.loaded = False

        # 如果提供了历史文件且文件存在，则加载历史数据
        if history_file and os.path.exists(history_file):
            self.load_history()

    def load_history(self):
        """从文件加载风险历史数据"""
        if not self.history_file:
            logger.warning("No history file specified, cannot load history")
            return False

        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)

                # 转换日期字符串为日期对象
                for user_id, history in data['history'].items():
                    user_history = []
                    for entry in history:
                        # 确保日期是日期对象
                        if 'date' in entry and isinstance(entry['date'], str):
                            entry['date'] = datetime.fromisoformat(entry['date']).date()
                        user_history.append(entry)
                    self.risk_history[user_id] = user_history

                # 获取最后更新时间
                if 'last_update' in data and data['last_update']:
                    self.last_update = datetime.fromisoformat(data['last_update'])

                logger.info(f"Loaded risk history for {len(self.risk_history)} users from {self.history_file}")
                self.loaded = True
                return True

        except Exception as e:
            logger.error(f"Failed to load risk history: {str(e)}")
            return False

    def save_history(self):
        """将风险历史数据保存到文件"""
        if not self.history_file:
            logger.warning("No history file specified, cannot save history")
            return False

        try:
            # 创建包含历史数据的字典
            data = {
                'history': {},
                'last_update': datetime.now().isoformat()
            }

            # 转换日期对象为ISO格式字符串
            for user_id, history in self.risk_history.items():
                user_history = []
                for entry in history:
                    entry_copy = entry.copy()
                    # 确保日期是字符串
                    if 'date' in entry_copy and not isinstance(entry_copy['date'], str):
                        entry_copy['date'] = entry_copy['date'].isoformat()
                    user_history.append(entry_copy)
                data['history'][user_id] = user_history

            # 写入文件
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved risk history for {len(self.risk_history)} users to {self.history_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save risk history: {str(e)}")
            return False

    def update_risk_scores(self, risk_scores, dimensions=None, factors=None):
        """
        更新用户风险分数历史

        Args:
            risk_scores: 用户ID到风险分数的映射字典
            dimensions: 用户ID到风险维度的映射字典，可选
            factors: 用户ID到风险因素的映射字典，可选

        Returns:
            bool: 更新是否成功
        """
        update_date = datetime.now().date()

        for user_id, score in risk_scores.items():
            # 创建历史记录条目
            entry = {
                'date': update_date,
                'score': score
            }

            # 添加维度信息(如果有)
            if dimensions and user_id in dimensions:
                entry['dimensions'] = dimensions[user_id]

            # 添加因素信息(如果有)
            if factors and user_id in factors:
                # 仅保存关键信息以节省空间
                entry['factors'] = {
                    'primary_factors': factors[user_id].get('primary_factors', []),
                    'user_sensitivity': factors[user_id].get('user_sensitivity', 1.0),
                    'diversity_factor': factors[user_id].get('diversity_factor', 1.0)
                }

            # 添加到用户历史
            self.risk_history[user_id].append(entry)

        # 更新最后更新时间
        self.last_update = datetime.now()

        # 清理历史数据
        self._cleanup_history()

        # 尝试保存历史
        if self.history_file:
            return self.save_history()

        return True

    def _cleanup_history(self):
        """清理过旧的历史数据"""
        cutoff_date = datetime.now().date() - timedelta(days=self.max_history_days)

        for user_id in self.risk_history:
            # 过滤掉老于cutoff_date的记录
            self.risk_history[user_id] = [
                entry for entry in self.risk_history[user_id]
                if entry['date'] >= cutoff_date
            ]

    def get_user_history(self, user_id, days=None):
        """
        获取用户的风险历史

        Args:
            user_id: 用户ID
            days: 要获取的天数，可选

        Returns:
            list: 用户的风险历史记录
        """
        history = self.risk_history.get(user_id, [])

        if days:
            cutoff_date = datetime.now().date() - timedelta(days=days)
            history = [entry for entry in history if entry['date'] >= cutoff_date]

        return history

    def get_all_user_history(self, days=None):
        """
        获取所有用户的风险历史

        Args:
            days: 要获取的天数，可选

        Returns:
            dict: 用户ID到风险历史的映射
        """
        if days:
            cutoff_date = datetime.now().date() - timedelta(days=days)

            result = {}
            for user_id, history in self.risk_history.items():
                result[user_id] = [entry for entry in history if entry['date'] >= cutoff_date]
            return result
        else:
            return dict(self.risk_history)

    def get_average_risk_trend(self, days=90, department=None, role=None, user_profiles=None):
        """
        获取平均风险趋势

        Args:
            days: 要分析的天数
            department: 筛选特定部门，可选
            role: 筛选特定角色，可选
            user_profiles: 用户配置信息字典，用于部门/角色筛选

        Returns:
            dict: 包含日期和平均分数的趋势数据
        """
        cutoff_date = datetime.now().date() - timedelta(days=days)

        # 按日期组织所有分数
        daily_scores = defaultdict(list)

        for user_id, history in self.risk_history.items():
            # 如果需要按部门/角色筛选
            if (department or role) and user_profiles:
                profile = user_profiles.get(user_id, {})
                user_dept = profile.get('department', '')
                user_role = profile.get('role', '')

                if department and user_dept != department:
                    continue
                if role and user_role != role:
                    continue

            # 收集符合日期范围的分数
            for entry in history:
                if entry['date'] >= cutoff_date:
                    daily_scores[entry['date']].append(entry['score'])

        # 计算每日平均分数
        avg_scores = {}
        for date, scores in daily_scores.items():
            avg_scores[date] = np.mean(scores)

        # 转换为排序后的列表
        trend_data = [
            {'date': date, 'avg_score': score, 'count': len(daily_scores[date])}
            for date, score in avg_scores.items()
        ]

        # 按日期排序
        trend_data.sort(key=lambda x: x['date'])

        return trend_data

    def get_risk_distribution(self, date=None):
        """
        获取特定日期的风险分布

        Args:
            date: 要分析的日期，默认为最新日期

        Returns:
            dict: 风险分布统计
        """
        if not date:
            # 找出最新的日期
            all_dates = set()
            for user_history in self.risk_history.values():
                all_dates.update(entry['date'] for entry in user_history)

            if not all_dates:
                return {
                    'date': None,
                    'high_risk': 0,
                    'medium_risk': 0,
                    'low_risk': 0,
                    'total_users': 0
                }

            date = max(all_dates)

        # 收集指定日期的所有分数
        scores = []
        for user_id, history in self.risk_history.items():
            # 查找指定日期的记录
            for entry in history:
                if entry['date'] == date:
                    scores.append(entry['score'])
                    break

        # 计算分布
        high_risk = sum(1 for s in scores if s >= 70)
        medium_risk = sum(1 for s in scores if 40 <= s < 70)
        low_risk = sum(1 for s in scores if s < 40)

        return {
            'date': date,
            'high_risk': high_risk,
            'medium_risk': medium_risk,
            'low_risk': low_risk,
            'total_users': len(scores),
            'distribution': {
                'high': high_risk / len(scores) if scores else 0,
                'medium': medium_risk / len(scores) if scores else 0,
                'low': low_risk / len(scores) if scores else 0
            }
        }

    def detect_significant_changes(self, threshold_percent=20, days=7):
        """
        检测显著的风险分数变化

        Args:
            threshold_percent: 显著变化的百分比阈值
            days: 比较的天数范围

        Returns:
            list: 包含显著变化的用户记录
        """
        significant_changes = []
        today = datetime.now().date()
        comparison_date = today - timedelta(days=days)

        for user_id, history in self.risk_history.items():
            # 按日期排序
            sorted_history = sorted(history, key=lambda x: x['date'])

            # 如果历史记录不足，则跳过
            if len(sorted_history) < 2:
                continue

            # 获取当前和之前的分数
            current_entry = None
            previous_entry = None

            for entry in reversed(sorted_history):
                if not current_entry and entry['date'] >= comparison_date:
                    current_entry = entry
                elif not previous_entry and entry['date'] < comparison_date:
                    previous_entry = entry
                    break

            if not current_entry or not previous_entry:
                continue

            # 计算变化百分比
            current_score = current_entry['score']
            previous_score = previous_entry['score']

            if previous_score == 0:
                change_percent = 100 if current_score > 0 else 0
            else:
                change_percent = ((current_score - previous_score) / previous_score) * 100

            # 如果变化超过阈值，记录
            if abs(change_percent) >= threshold_percent:
                significant_changes.append({
                    'user_id': user_id,
                    'current_date': current_entry['date'],
                    'current_score': current_score,
                    'previous_date': previous_entry['date'],
                    'previous_score': previous_score,
                    'change_percent': change_percent,
                    'is_increase': change_percent > 0
                })

        # 按变化百分比排序
        significant_changes.sort(key=lambda x: abs(x['change_percent']), reverse=True)
        return significant_changes