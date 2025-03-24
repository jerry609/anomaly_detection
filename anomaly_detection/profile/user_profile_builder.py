"""
用户配置文件构建器，从预处理数据中构建用户行为配置文件。
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import logging
from anomaly_detection.utils.timestamp_processor import extract_time_features
from anomaly_detection.utils.text_processor import normalize_text, extract_email_parts
from anomaly_detection.utils.feature_engineering import create_activity_distribution, calculate_entropy

logger = logging.getLogger(__name__)


class UserProfileBuilder:
    """构建用户行为配置文件的组件"""

    def __init__(self, datasets):
        self.datasets = datasets

    def build_all_profiles(self):
        """为所有用户构建配置文件"""
        # 创建用户ID到组织信息的映射
        ldap_users = self.datasets['ldap'].set_index('user_id') if 'ldap' in self.datasets else pd.DataFrame()

        # 收集所有用户
        all_users = self._collect_all_users()

        # 为每个用户创建配置文件
        user_profiles = {}
        for user_id in all_users:
            user_profiles[user_id] = self._build_user_profile(user_id, ldap_users)

        logger.info(f"Built profiles for {len(user_profiles)} users")
        return user_profiles

    def _collect_all_users(self):
        """收集所有数据集中的用户ID"""
        all_users = set()
        for dataset_name in ['logon', 'device', 'email', 'file', 'http']:
            if dataset_name in self.datasets and 'user' in self.datasets[dataset_name].columns:
                all_users.update(self.datasets[dataset_name]['user'].unique())
        return all_users

    def _build_user_profile(self, user_id, ldap_users):
        """为单个用户构建配置文件"""
        profile = {
            'user_id': user_id,
            'org_info': self._get_org_info(user_id, ldap_users),
            'psychometric': self._get_psychometric_data(user_id),
            'activity': self._build_activity_profile(user_id),
            'email': self._build_email_profile(user_id),
            'file': self._build_file_profile(user_id),
            'web': self._build_web_profile(user_id),
            'device': self._build_device_profile(user_id)
        }

        return profile

    def _get_org_info(self, user_id, ldap_users):
        """获取用户的组织信息"""
        if ldap_users.empty or user_id not in ldap_users.index:
            return {}
        return ldap_users.loc[user_id].to_dict()

    def _get_psychometric_data(self, user_id):
        """获取用户的心理测量数据"""
        if 'psychometric' not in self.datasets:
            return {}

        psych_data = self.datasets['psychometric']
        user_data = psych_data[psych_data['user_id'] == user_id]

        if user_data.empty:
            return {}

        row = user_data.iloc[0]
        return {
            'O': row['O'] if 'O' in row else None,  # 开放性
            'C': row['C'] if 'C' in row else None,  # 尽责性
            'E': row['E'] if 'E' in row else None,  # 外向性
            'A': row['A'] if 'A' in row else None,  # 宜人性
            'N': row['N'] if 'N' in row else None  # 神经质
        }

    def _build_activity_profile(self, user_id):
        """构建用户活动相关的配置文件部分"""
        activity_data = []

        for dataset_name in ['logon', 'device', 'email', 'file', 'http']:
            if dataset_name not in self.datasets:
                continue

            dataset = self.datasets[dataset_name]
            if 'user' in dataset.columns and 'timestamp' in dataset.columns:
                user_activities = dataset[dataset['user'] == user_id]
                if not user_activities.empty:
                    activity_data.append(user_activities)

        if not activity_data:
            return {'activity_hours': [0] * 24, 'activity_days': [0] * 7}

        # 合并所有活动
        all_activities = pd.concat(activity_data)

        # 创建活动分布
        timestamps = all_activities['timestamp'].tolist()
        hours = [ts.hour for ts in timestamps]
        days = [ts.dayofweek for ts in timestamps]

        hour_dist = [0] * 24
        for hour in hours:
            hour_dist[hour] += 1

        day_dist = [0] * 7
        for day in days:
            day_dist[day] += 1

        # 计算熵和其他指标
        return {
            'activity_hours': hour_dist,
            'activity_days': day_dist,
            'hour_entropy': calculate_entropy(hour_dist),
            'day_entropy': calculate_entropy(day_dist),
            'total_activities': len(timestamps)
        }

    def _build_email_profile(self, user_id):
        """构建用户电子邮件相关的配置文件部分"""
        if 'email' not in self.datasets:
            return {}

        email_data = self.datasets['email']
        user_emails = email_data[email_data['user'] == user_id]

        if user_emails.empty:
            return {'recipients': [], 'email_count': 0}

        # 收集所有收件人
        recipients = set()
        for _, row in user_emails.iterrows():
            if 'to' in row and pd.notna(row['to']):
                for recipient in extract_email_parts(row['to']):
                    recipients.add(recipient)

        return {
            'recipients': list(recipients),
            'recipient_count': len(recipients),
            'email_count': len(user_emails)
        }

    def _build_file_profile(self, user_id):
        """构建用户文件相关的配置文件部分"""
        if 'file' not in self.datasets:
            return {}

        file_data = self.datasets['file']
        user_files = file_data[file_data['user'] == user_id]

        if user_files.empty:
            return {'accessed_files': Counter()}

        # 统计访问的文件
        accessed_files = Counter()
        for _, row in user_files.iterrows():
            if 'filename' in row and pd.notna(row['filename']):
                accessed_files[normalize_text(row['filename'])] += 1

        return {
            'accessed_files': accessed_files,
            'unique_files': len(accessed_files),
            'total_file_operations': sum(accessed_files.values())
        }

    def _build_web_profile(self, user_id):
        """构建用户Web活动相关的配置文件部分"""
        if 'http' not in self.datasets:
            return {}

        http_data = self.datasets['http']
        user_http = http_data[http_data['user'] == user_id]

        if user_http.empty:
            return {'accessed_websites': Counter()}

        # 统计访问的网站
        accessed_websites = Counter()
        for _, row in user_http.iterrows():
            if 'url' in row and pd.notna(row['url']):
                accessed_websites[normalize_text(row['url'])] += 1

        return {
            'accessed_websites': accessed_websites,
            'unique_websites': len(accessed_websites),
            'total_web_requests': sum(accessed_websites.values())
        }

    def _build_device_profile(self, user_id):
        """构建用户设备相关的配置文件部分"""
        devices = set()

        for dataset_name in ['logon', 'device']:
            if dataset_name not in self.datasets:
                continue

            dataset = self.datasets[dataset_name]
            if 'user' in dataset.columns and 'pc' in dataset.columns:
                user_records = dataset[dataset['user'] == user_id]

                for _, row in user_records.iterrows():
                    if pd.notna(row['pc']):
                        devices.add(row['pc'])

        return {
            'devices_used': list(devices),
            'device_count': len(devices)
        }
