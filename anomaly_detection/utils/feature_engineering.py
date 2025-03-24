import logging

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_activity_distribution(timestamps, bins=24):
    """创建活动时间分布特征"""
    if not timestamps:
        return [0] * bins

    hours = [ts.hour for ts in timestamps]
    distribution = [0] * bins

    for hour in hours:
        distribution[hour] += 1

    return distribution


def create_count_features(items):
    """从列表项目创建计数特征"""
    if not items:
        return {}

    counter = Counter(items)
    return {
        'total_count': sum(counter.values()),
        'unique_count': len(counter),
        'most_common': counter.most_common(3),
        'diversity_ratio': len(counter) / max(1, sum(counter.values()))
    }


def calculate_entropy(distribution):
    """计算分布的熵"""
    total = sum(distribution)
    if total == 0:
        return 0

    probs = [count / total for count in distribution if count > 0]
    return -sum(p * np.log2(p) for p in probs)


def normalize_features(features_df):
    """标准化特征值"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    return pd.DataFrame(scaled_features, columns=features_df.columns, index=features_df.index)


def create_user_activity_features(user_data, timestamp_col='timestamp'):
    """为用户创建活动相关的特征"""
    if timestamp_col not in user_data.columns:
        return {}

    timestamps = user_data[timestamp_col].tolist()

    hour_dist = create_activity_distribution(timestamps, 24)
    day_dist = create_activity_distribution([ts.dayofweek for ts in timestamps], 7)

    return {
        'activity_hours': hour_dist,
        'activity_days': day_dist,
        'activity_entropy': calculate_entropy(hour_dist),
        'day_entropy': calculate_entropy(day_dist),
        'total_activities': sum(hour_dist)
    }


def prepare_features_for_ml(datasets, user_profiles):
    """
    从不同的数据集准备用于机器学习的特征

    Args:
        datasets: 包含各类数据集的字典
        user_profiles: 用户配置文件字典

    Returns:
        dict: 用户ID到特征向量的映射
    """
    logger.info("Preparing features for machine learning models")
    features_by_user = {}

    # 确保有足够的数据集
    if not datasets or not user_profiles:
        logger.warning("Insufficient data for feature engineering")
        return features_by_user

    # 处理登录数据
    logon_features = extract_logon_features(datasets.get('logon'), user_profiles)

    # 处理邮件数据
    email_features = extract_email_features(datasets.get('email'), user_profiles)

    # 处理文件访问数据
    file_features = extract_file_features(datasets.get('file'), user_profiles)

    # 处理HTTP数据
    http_features = extract_http_features(datasets.get('http'), user_profiles)

    # 合并所有用户的特征
    all_user_ids = set(user_profiles.keys())

    for user_id in all_user_ids:
        # 收集该用户的所有特征
        user_feature_vector = []

        # 添加登录特征
        if user_id in logon_features:
            user_feature_vector.extend(logon_features[user_id])
        else:
            # 如果没有特征，添加默认值
            user_feature_vector.extend([0] * 5)  # 假设有5个登录特征

        # 添加邮件特征
        if user_id in email_features:
            user_feature_vector.extend(email_features[user_id])
        else:
            user_feature_vector.extend([0] * 5)  # 假设有5个邮件特征

        # 添加文件访问特征
        if user_id in file_features:
            user_feature_vector.extend(file_features[user_id])
        else:
            user_feature_vector.extend([0] * 3)  # 假设有3个文件访问特征

        # 添加HTTP特征
        if user_id in http_features:
            user_feature_vector.extend(http_features[user_id])
        else:
            user_feature_vector.extend([0] * 3)  # 假设有3个HTTP特征

        # 添加用户配置文件特征
        profile_features = extract_profile_features(user_profiles.get(user_id, {}))
        user_feature_vector.extend(profile_features)

        # 存储该用户的特征向量
        features_by_user[user_id] = user_feature_vector

    logger.info(f"Generated features for {len(features_by_user)} users")
    return features_by_user


def extract_logon_features(logon_data, user_profiles):
    """从登录数据中提取特征"""
    features = {}
    import logging
    logger = logging.getLogger(__name__)

    if logon_data is None or logon_data.empty:
        return features

    # 创建数据副本
    logon_data = logon_data.copy()

    # 根据实际数据集使用正确的用户列名
    user_col = 'user'  # 根据实际数据集格式

    # 确保有user_id列
    if user_col not in logon_data.columns:
        logger.warning(f"User column '{user_col}' not found in logon data")
        return features

    # 使用正确的用户列名创建user_id列（如果不存在）
    if 'user_id' not in logon_data.columns:
        logon_data['user_id'] = logon_data[user_col]

    # 按用户分组
    for user_id, group in logon_data.groupby('user_id'):
        # 提取特征
        # 1. 登录次数
        login_count = len(group)

        # 2. 非工作时间登录比例
        after_hours_ratio = 0
        if 'date' in group.columns or 'timestamp' in group.columns:
            time_col = 'timestamp' if 'timestamp' in group.columns else 'date'
            try:
                # 转换为datetime类型
                group['hour'] = pd.to_datetime(group[time_col]).dt.hour
                after_hours_logins = sum((group['hour'] < 8) | (group['hour'] > 18))
                after_hours_ratio = after_hours_logins / login_count if login_count > 0 else 0
            except Exception as e:
                logger.warning(f"Error processing timestamp in logon data: {str(e)}")

        # 3. 远程登录比例 (如果有这个字段)
        remote_ratio = 0
        if 'remote' in group.columns:
            try:
                remote_logins = sum(group['remote'] == True)
                remote_ratio = remote_logins / login_count if login_count > 0 else 0
            except:
                pass

        # 4. 失败登录比例 (如果有这个字段)
        failed_ratio = 0
        if 'activity' in group.columns:
            try:
                # 假设'Logoff Failure'或'Logon Failure'表示失败
                failed_patterns = ['failure', 'failed', 'fail']
                failed_logins = sum(group['activity'].str.lower().str.contains('|'.join(failed_patterns)))
                failed_ratio = failed_logins / login_count if login_count > 0 else 0
            except:
                pass

        # 5. 设备多样性
        device_diversity = 0
        if 'pc' in group.columns:
            try:
                device_diversity = len(group['pc'].unique())
            except:
                pass

        # 组合特征
        features[user_id] = [
            login_count,
            after_hours_ratio,
            remote_ratio,
            failed_ratio,
            device_diversity
        ]

    return features


def extract_email_features(email_data, user_profiles):
    """从邮件数据中提取特征"""
    features = {}
    import logging
    logger = logging.getLogger(__name__)

    if email_data is None or email_data.empty:
        return features

    # 创建一个数据副本以避免修改原始数据
    email_data = email_data.copy()

    # 根据示例数据，用户信息可能在'from'字段或'user'字段
    if 'user_id' not in email_data.columns:
        if 'from' in email_data.columns:
            email_data['user_id'] = email_data['from']
            logger.info("Using 'from' field as user_id in email data")
        elif 'user' in email_data.columns:
            email_data['user_id'] = email_data['user']
            logger.info("Using 'user' field as user_id in email data")
        else:
            logger.warning("No suitable user ID column found in email data")
            return features

    # 按用户分组
    try:
        for user_id, group in email_data.groupby('user_id'):
            # 提取特征
            # 1. 邮件发送频率
            email_count = len(group)

            # 2. 附件使用率
            if 'attachments' in group.columns:
                try:
                    # 根据示例，附件字段是数值型，0表示无附件
                    attachment_ratio = sum(
                        group['attachments'].astype(float) > 0) / email_count if email_count > 0 else 0
                except Exception as e:
                    logger.warning(f"Error processing attachments: {e}")
                    attachment_ratio = 0
            else:
                attachment_ratio = 0

            # 3. 外部邮件比例
            if 'to' in group.columns:
                try:
                    # 从示例可见，to字段包含完整邮件地址，可能有多个收件人
                    external_count = 0
                    company_domains = ['dtaa.com', 'company.com']  # 根据示例添加公司域名

                    for recipients in group['to']:
                        if pd.isna(recipients) or recipients == '':
                            continue

                        # 分割多个收件人
                        email_list = recipients.split(';')
                        for email in email_list:
                            if email and '@' in email:
                                domain = email.split('@')[1].lower()
                                if domain not in company_domains:
                                    external_count += 1

                    external_ratio = external_count / email_count if email_count > 0 else 0
                except Exception as e:
                    logger.warning(f"Error processing external emails: {e}")
                    external_ratio = 0
            else:
                external_ratio = 0

            # 4. 邮件大小平均值
            if 'size' in group.columns:
                try:
                    avg_size = group['size'].astype(float).mean()
                except Exception as e:
                    logger.warning(f"Error processing email size: {e}")
                    avg_size = 0
            else:
                avg_size = 0

            # 5. 收件人平均数量
            recipient_counts = []
            for idx, row in group.iterrows():
                count = 0

                # 计算to字段中的收件人
                if 'to' in row and pd.notna(row['to']) and row['to']:
                    count += len(row['to'].split(';'))

                # 计算cc字段中的收件人
                if 'cc' in row and pd.notna(row['cc']) and row['cc']:
                    count += len(row['cc'].split(';'))

                # 计算bcc字段中的收件人
                if 'bcc' in row and pd.notna(row['bcc']) and row['bcc']:
                    count += len(row['bcc'].split(';'))

                recipient_counts.append(count)

            avg_recipients = sum(recipient_counts) / len(recipient_counts) if recipient_counts else 0

            # 组合特征
            features[user_id] = [
                email_count,
                attachment_ratio,
                external_ratio,
                avg_size,
                avg_recipients
            ]
    except Exception as e:
        logger.error(f"Error in extract_email_features: {str(e)}")

    return features


def extract_file_features(file_data, user_profiles):
    """从文件访问数据中提取特征"""
    features = {}

    if file_data is None or file_data.empty:
        return features

    # 确保有user_id列
    if 'user_id' not in file_data.columns and 'user' in file_data.columns:
        file_data = file_data.rename(columns={'user': 'user_id'})

    # 按用户分组
    for user_id, group in file_data.groupby('user_id'):
        # 提取特征
        # 1. 文件访问总数
        file_access_count = len(group)

        # 2. 访问的唯一文件数
        if 'filename' in group.columns:
            unique_files = len(group['filename'].unique())
        else:
            unique_files = 0

        # 3. 敏感文件访问比例
        if 'sensitive' in group.columns:
            sensitive_accesses = sum(group['sensitive'] == True)
            sensitive_ratio = sensitive_accesses / file_access_count if file_access_count > 0 else 0
        else:
            sensitive_ratio = 0

        # 组合特征
        features[user_id] = [
            file_access_count,
            unique_files,
            sensitive_ratio
        ]

    return features


def extract_http_features(http_data, user_profiles):
    """从HTTP数据中提取特征"""
    features = {}

    if http_data is None or http_data.empty:
        return features

    # 确保有user_id列
    if 'user_id' not in http_data.columns and 'user' in http_data.columns:
        http_data = http_data.rename(columns={'user': 'user_id'})

    # 按用户分组
    for user_id, group in http_data.groupby('user_id'):
        # 提取特征
        # 1. HTTP请求总数
        http_count = len(group)

        # 2. 唯一域名数量
        if 'domain' in group.columns:
            unique_domains = len(group['domain'].unique())
        elif 'url' in group.columns:
            # 从URL提取域名
            group['domain'] = group['url'].str.extract(r'https?://([^/]+)')
            unique_domains = len(group['domain'].unique())
        else:
            unique_domains = 0

        # 3. 非工作相关网站比例
        if 'category' in group.columns:
            # 假设非工作相关类别
            non_work_categories = ['social', 'entertainment', 'shopping']
            non_work_requests = sum(group['category'].isin(non_work_categories))
            non_work_ratio = non_work_requests / http_count if http_count > 0 else 0
        else:
            non_work_ratio = 0

        # 组合特征
        features[user_id] = [
            http_count,
            unique_domains,
            non_work_ratio
        ]

    return features


def extract_profile_features(user_profile):
    """从用户配置文件中提取特征"""
    # 如果配置文件为空，返回默认特征
    if not user_profile:
        return [0, 0, 0, 0]

    # 特征1: 用户角色重要性
    role_importance = {
        'admin': 5,
        'manager': 4,
        'developer': 3,
        'analyst': 3,
        'user': 2,
        'contractor': 1
    }
    role = user_profile.get('role', 'user').lower()
    importance = role_importance.get(role, 2)

    # 特征2: 权限级别
    access_level = user_profile.get('access_level', 1)

    # 特征3: 入职时间（转换为特征）
    if 'hire_date' in user_profile:
        try:
            hire_date = pd.to_datetime(user_profile['hire_date'])
            today = pd.Timestamp.now()
            tenure_days = (today - hire_date).days
            # 标准化到0-5范围
            tenure_feature = min(5, tenure_days / 365)
        except:
            tenure_feature = 2.5  # 默认中间值
    else:
        tenure_feature = 2.5

    # 特征4: 历史风险评分
    risk_history = user_profile.get('risk_history', [])
    if risk_history:
        avg_risk = sum(risk_history) / len(risk_history)
    else:
        avg_risk = 0

    return [importance, access_level, tenure_feature, avg_risk]