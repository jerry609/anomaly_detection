import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Union
import networkx as nx
from ..utils.text_processor import normalize_text, extract_email_parts, detect_patterns

# 避免循环导入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models.base_detector import BaseAnomalyDetector

class EmailAnalyzer:
    """
    分析电子邮件模式，检测可疑行为。
    检测异常电子邮件量、内容、收件人等。
    """

    def __init__(self, detector_model=None):
        """
        使用检测器模型初始化电子邮件分析器。

        Args:
            detector_model (BaseDetector, optional): 用于异常检测的模型。
                                                    如果为None，将使用默认模型。
        """
        self.detector = detector_model
        self.user_email_profiles = {}
        self.org_domain = None  # 组织的默认域名
        self.sensitive_patterns = [
            r'confidential',
            r'secret',
            r'password',
            r'credentials',
            r'ssn',
            r'social security'
        ]

    def set_org_domain(self, domain):
        """设置组织的电子邮件域名，用于内部/外部分析。"""
        self.org_domain = domain

    def set_sensitive_patterns(self, patterns):
        """设置敏感模式列表，用于检测敏感内容。"""
        self.sensitive_patterns = patterns

    def fit(self, email_data):
        """
        基于历史数据构建用户电子邮件配置文件。

        Args:
            email_data (pd.DataFrame): 包含电子邮件事件的DataFrame

        Returns:
            self: 返回实例自身
        """
        # 预处理数据
        processed_emails = self._preprocess_emails(email_data)

        # 按用户分组并构建电子邮件配置文件
        for user_id, user_data in processed_emails.groupby('from_user'):
            # 获取常用收件人
            all_recipients = []
            for recipients in user_data['recipients_list']:
                all_recipients.extend(recipients)

            recipient_counter = Counter(all_recipients)
            frequent_recipients = [r for r, c in recipient_counter.most_common(10)]

            # 计算内部与外部电子邮件
            if self.org_domain and 'domains' in user_data.columns:
                internal_emails = user_data[user_data['domains'].apply(
                    lambda domains: all(domain == self.org_domain for domain in domains))]
                internal_ratio = len(internal_emails) / len(user_data) if len(user_data) > 0 else 0
            else:
                internal_ratio = None

            # 计算电子邮件量统计数据
            emails_per_day = user_data.groupby('date').size()

            # 构建配置文件
            self.user_email_profiles[user_id] = {
                'frequent_recipients': frequent_recipients,
                'avg_emails_per_day': emails_per_day.mean(),
                'std_emails_per_day': emails_per_day.std(),
                'internal_ratio': internal_ratio,
                'avg_recipient_count': user_data['recipient_count'].mean(),
                'max_recipient_count': user_data['recipient_count'].max(),
                'avg_attachment_count': user_data[
                    'attachment_count'].mean() if 'attachment_count' in user_data.columns else 0,
                'avg_email_length': user_data[
                    'content_length'].mean() if 'content_length' in user_data.columns else None,
                'has_sent_sensitive': user_data[
                    'has_sensitive'].any() if 'has_sensitive' in user_data.columns else False
            }

        # 如果提供了检测器，则训练它
        if self.detector:
            self.detector.fit(processed_emails)

        return self

    def _preprocess_emails(self, email_data):
        """预处理电子邮件数据以进行分析。"""
        df = email_data.copy()

        # 标准化文本
        if 'subject' in df.columns:
            df['subject_norm'] = df['subject'].apply(normalize_text)

        if 'content' in df.columns:
            df['content_norm'] = df['content'].apply(normalize_text)
            df['content_length'] = df['content'].apply(lambda x: len(str(x)) if not pd.isna(x) else 0)
            # 检测敏感内容
            df['has_sensitive'] = df['content_norm'].apply(
                lambda text: len(detect_patterns(text, self.sensitive_patterns)) > 0)

        # 提取收件人列表
        if 'to_users' in df.columns:
            df['recipients_list'] = df['to_users'].apply(extract_email_parts)
            df['recipient_count'] = df['recipients_list'].apply(len)

        # 提取域名
        if 'to_domain' in df.columns:
            df['domains'] = df['to_domain'].apply(lambda x: [x] if not pd.isna(x) else [])

        # 提取日期
        if 'timestamp' in df.columns and 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date

        return df

    def detect_anomalies(self, new_email_data):
        """
        在新事件中检测基于电子邮件的异常。

        Args:
            new_email_data (pd.DataFrame): 要分析的新电子邮件事件

        Returns:
            pd.DataFrame: 每个事件的异常分数和标志
        """
        # 预处理数据
        processed_emails = self._preprocess_emails(new_email_data)

        results = []

        # 按用户和日期分组进行上下文分析
        for (user_id, date), events in processed_emails.groupby(['from_user', 'date']):
            # 如果我们没有此用户的配置文件，则跳过
            if user_id not in self.user_email_profiles:
                for idx, event in events.iterrows():
                    results.append({
                        'event_id': idx,
                        'user_id': user_id,
                        'date': date,
                        'anomaly_score': 0,
                        'is_anomaly': False,
                        'reason': "没有用户配置文件"
                    })
                continue

            profile = self.user_email_profiles[user_id]
            daily_email_count = len(events)

            # 检查每个事件的异常
            for idx, event in events.iterrows():
                anomaly_score = 0
                anomaly_reason = []

                # 检查不寻常的电子邮件量
                if profile['std_emails_per_day'] > 0:
                    z_score = (daily_email_count - profile['avg_emails_per_day']) / profile['std_emails_per_day']
                    if z_score > 2:
                        anomaly_score += 0.3
                        anomaly_reason.append(f"不寻常的电子邮件量: 今天{daily_email_count}封电子邮件")

                # 检查不寻常的收件人
                recipients = event['recipients_list']
                unusual_recipients = [r for r in recipients if r not in profile['frequent_recipients']]
                if unusual_recipients and len(unusual_recipients) > 0.7 * len(recipients):
                    anomaly_score += 0.4
                    anomaly_reason.append(f"不寻常的收件人: {', '.join(unusual_recipients[:3])}")

                # 检查不寻常的收件人数量
                if 'recipient_count' in event and event['recipient_count'] > 2 * profile['max_recipient_count']:
                    anomaly_score += 0.4
                    anomaly_reason.append(f"不寻常的收件人数量: {event['recipient_count']}")

                # 检查不寻常的附件数量
                if 'attachment_count' in event and event['attachment_count'] > 2 * profile['avg_attachment_count']:
                    anomaly_score += 0.3
                    anomaly_reason.append(f"不寻常的附件数量: {event['attachment_count']}")

                # 检查不寻常的电子邮件大小或内容
                if 'content_length' in event and profile['avg_email_length']:
                    if event['content_length'] > 3 * profile['avg_email_length']:
                        anomaly_score += 0.2
                        anomaly_reason.append(f"不寻常的长电子邮件: {event['content_length']}字符")

                # 检查不寻常的内部/外部比率
                if self.org_domain and 'domains' in event and profile['internal_ratio'] is not None:
                    is_external = not all(domain == self.org_domain for domain in event['domains'])
                    if profile['internal_ratio'] > 0.9 and is_external:
                        anomaly_score += 0.5
                        anomaly_reason.append(f"不寻常的外部电子邮件到{', '.join(event['domains'])}")

                # 检查敏感内容
                if 'has_sensitive' in event and event['has_sensitive'] and not profile['has_sent_sensitive']:
                    anomaly_score += 0.6
                    anomaly_reason.append("电子邮件包含敏感内容")

                # 使用检测器模型（如果可用）
                if self.detector:
                    model_score = self.detector.score(pd.DataFrame([event]))[0]
                    anomaly_score = max(anomaly_score, model_score)

                results.append({
                    'event_id': idx,
                    'user_id': user_id,
                    'recipients': recipients,
                    'date': date,
                    'anomaly_score': anomaly_score,
                    'is_anomaly': anomaly_score > 0.7,  # 阈值
                    'reason': ", ".join(anomaly_reason) if anomaly_reason else "正常行为"
                })

        return pd.DataFrame(results)
