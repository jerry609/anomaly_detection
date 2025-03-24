import numpy as np
import pandas as pd
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import re

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    数据标准化工具 - 用于对各种数据进行预处理和标准化，
    确保不同来源的数据格式统一，可用于机器学习模型
    """

    def __init__(self):
        """初始化数据标准化工具"""
        self.scalers = {}
        self.email_pattern = re.compile(r'[^@]+@[^@]+\.[^@]+')

    def normalize_timestamps(self, df, timestamp_cols=None):
        """
        将时间戳列标准化为pandas datetime格式

        Args:
            df: 要处理的DataFrame
            timestamp_cols: 时间戳列名列表，如果为None则自动检测

        Returns:
            DataFrame: 处理后的数据框
        """
        df = df.copy()

        # 如果未指定列，尝试自动检测
        if timestamp_cols is None:
            timestamp_cols = []
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower() or 'stamp' in col.lower():
                    if df[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[col]):
                        timestamp_cols.append(col)

        # 转换所有时间戳列
        for col in timestamp_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Failed to convert column {col} to datetime: {str(e)}")

        return df

    def normalize_user_ids(self, df, id_cols=None, email_cols=None):
        """
        标准化用户ID和电子邮件列

        Args:
            df: 要处理的DataFrame
            id_cols: 用户ID列名列表，如果为None则自动检测
            email_cols: 电子邮件列名列表，如果为None则自动检测

        Returns:
            DataFrame: 处理后的数据框
        """
        df = df.copy()

        # 自动检测ID列
        if id_cols is None:
            id_cols = []
            for col in df.columns:
                if 'user' in col.lower() and 'id' in col.lower():
                    id_cols.append(col)
                elif col.lower() == 'user':
                    id_cols.append(col)

        # 自动检测电子邮件列
        if email_cols is None:
            email_cols = []
            for col in df.columns:
                if 'email' in col.lower() or 'mail' in col.lower():
                    email_cols.append(col)

        # 标准化用户ID列
        for col in id_cols:
            if col in df.columns:
                # 将用户ID转换为字符串
                df[col] = df[col].astype(str)
                # 清理用户ID
                df[col] = df[col].str.strip().str.lower()

        # 处理电子邮件列
        for col in email_cols:
            if col in df.columns:
                # 将邮件地址转换为字符串
                df[col] = df[col].astype(str)
                # 清理邮件地址
                df[col] = df[col].str.strip().str.lower()

                # 创建用户名和域名列
                if not f"{col}_username" in df.columns:
                    df[f"{col}_username"] = df[col].apply(
                        lambda x: x.split('@')[0] if '@' in x else x
                    )

                if not f"{col}_domain" in df.columns:
                    df[f"{col}_domain"] = df[col].apply(
                        lambda x: x.split('@')[1] if '@' in x else ''
                    )

        return df

    def normalize_categorical_columns(self, df, cat_cols=None):
        """
        标准化分类列(清理、小写化等)

        Args:
            df: 要处理的DataFrame
            cat_cols: 分类列名列表，如果为None则自动检测

        Returns:
            DataFrame: 处理后的数据框
        """
        df = df.copy()

        # 自动检测分类列
        if cat_cols is None:
            cat_cols = []
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() < 100:
                    # 排除看起来像ID或邮件的列
                    col_lower = col.lower()
                    if not ('id' in col_lower or 'email' in col_lower or 'mail' in col_lower):
                        cat_cols.append(col)

        # 标准化分类列
        for col in cat_cols:
            if col in df.columns:
                # 将分类转换为字符串
                df[col] = df[col].astype(str)
                # 清理和小写化
                df[col] = df[col].str.strip().str.lower()

        return df

    def scale_numerical_features(self, df, num_cols=None, method='standard', fit=True, scaler_key='default'):
        """
        对数值特征进行缩放

        Args:
            df: 要处理的DataFrame
            num_cols: 数值列名列表，如果为None则自动检测
            method: 缩放方法('standard', 'minmax', 'robust')
            fit: 是否拟合新的缩放器
            scaler_key: 缩放器的键名，用于存储和重用缩放器

        Returns:
            DataFrame: 处理后的数据框
        """
        df = df.copy()

        # 自动检测数值列
        if num_cols is None:
            num_cols = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and not (
                        'id' in col.lower() or df[col].nunique() < 10
                ):
                    num_cols.append(col)

        if not num_cols:
            return df

        # 创建或获取缩放器
        if fit or scaler_key not in self.scalers:
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:  # standard
                scaler = StandardScaler()

            self.scalers[scaler_key] = scaler
        else:
            scaler = self.scalers[scaler_key]

        # 提取数值特征
        X = df[num_cols].values

        # 处理缺失值
        X = np.nan_to_num(X)

        # 拟合和转换
        if fit:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)

        # 将缩放后的特征放回DataFrame
        for i, col in enumerate(num_cols):
            df[f"{col}_scaled"] = X_scaled[:, i]

        return df

    def normalize_file_paths(self, df, path_cols=None):
        """
        标准化文件路径列

        Args:
            df: 要处理的DataFrame
            path_cols: 文件路径列名列表，如果为None则自动检测

        Returns:
            DataFrame: 处理后的数据框
        """
        df = df.copy()

        # 自动检测文件路径列
        if path_cols is None:
            path_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if 'path' in col_lower or 'file' in col_lower or 'location' in col_lower:
                    if df[col].dtype == 'object':
                        path_cols.append(col)

        # 标准化文件路径
        for col in path_cols:
            if col in df.columns:
                # 转换为字符串
                df[col] = df[col].astype(str)

                # 规范化路径分隔符
                df[col] = df[col].str.replace('\\', '/')

                # 提取文件名和扩展名
                df[f"{col}_filename"] = df[col].apply(
                    lambda x: x.split('/')[-1] if '/' in x else x
                )

                df[f"{col}_extension"] = df[col].apply(
                    lambda x: x.split('.')[-1].lower() if '.' in x else ''
                )

                # 根据扩展名创建文件类型
                def get_file_type(ext):
                    if not ext:
                        return 'unknown'
                    ext = ext.lower()
                    if ext in ['doc', 'docx', 'txt', 'pdf', 'rtf']:
                        return 'document'
                    elif ext in ['xls', 'xlsx', 'csv']:
                        return 'spreadsheet'
                    elif ext in ['ppt', 'pptx']:
                        return 'presentation'
                    elif ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
                        return 'image'
                    elif ext in ['mp3', 'wav', 'ogg']:
                        return 'audio'
                    elif ext in ['mp4', 'avi', 'mov', 'wmv']:
                        return 'video'
                    elif ext in ['zip', 'tar', 'gz', 'rar']:
                        return 'archive'
                    elif ext in ['exe', 'dll', 'bat']:
                        return 'executable'
                    else:
                        return 'other'

                df[f"{col}_filetype"] = df[f"{col}_extension"].apply(get_file_type)

        return df

    def extract_email_parts(self, df, email_cols=None):
        """
        从邮件数据中提取发件人、收件人、主题等信息

        Args:
            df: 要处理的DataFrame
            email_cols: 邮件内容列名列表，如果为None则自动检测

        Returns:
            DataFrame: 处理后的数据框
        """
        df = df.copy()

        # 自动检测邮件内容列
        if email_cols is None:
            email_cols = []
            for col in df.columns:
                col_lower = col.lower()
                if 'email' in col_lower or 'mail' in col_lower or 'message' in col_lower:
                    if df[col].dtype == 'object':
                        email_cols.append(col)

        # 处理发件人列
        from_cols = [col for col in df.columns if 'from' in col.lower()]
        for col in from_cols:
            if col in df.columns and df[col].dtype == 'object':
                # 清理发件人
                df[col] = df[col].astype(str).str.strip().str.lower()

                # 提取域名
                df[f"{col}_domain"] = df[col].apply(
                    lambda x: x.split('@')[1] if '@' in x else ''
                )

                # 检测是否为外部邮件
                if 'internal_domains' in df.columns:
                    internal_domains = df['internal_domains'].unique()
                    df[f"{col}_is_external"] = ~df[f"{col}_domain"].isin(internal_domains)

        # 处理收件人列
        to_cols = [col for col in df.columns if col.lower() in ['to', 'recipient', 'recipients']]
        for col in to_cols:
            if col in df.columns and df[col].dtype == 'object':
                # 清理收件人
                df[col] = df[col].astype(str).str.strip().str.lower()

                # 计算收件人数量
                df[f"{col}_count"] = df[col].apply(
                    lambda x: len(x.split(';')) if ';' in x else (1 if x else 0)
                )

                # 检测是否包含外部收件人
                if 'internal_domains' in df.columns:
                    internal_domains = df['internal_domains'].unique()

                    def has_external_recipient(recipients):
                        if not recipients:
                            return False
                        for r in recipients.split(';'):
                            r = r.strip()
                            if '@' in r and r.split('@')[1] not in internal_domains:
                                return True
                        return False

                    df[f"{col}_has_external"] = df[col].apply(has_external_recipient)

        # 处理主题列
        subject_cols = [col for col in df.columns if 'subject' in col.lower()]
        for col in subject_cols:
            if col in df.columns and df[col].dtype == 'object':
                # 清理主题
                df[col] = df[col].astype(str).str.strip()

                # 创建主题长度
                df[f"{col}_length"] = df[col].str.len()

                # 检测主题是否包含关键词
                sensitive_keywords = ['confidential', 'private', 'secret', 'sensitive']
                df[f"{col}_is_sensitive"] = df[col].str.lower().apply(
                    lambda x: any(kw in x for kw in sensitive_keywords)
                )

        return df

    def normalize_dataset(self, df, dataset_type):
        """
        根据数据集类型执行特定的标准化

        Args:
            df: 要处理的DataFrame
            dataset_type: 数据集类型('logon', 'email', 'file', etc.)

        Returns:
            DataFrame: 标准化后的DataFrame
        """
        if df.empty:
            return df

        # 基础标准化 - 适用于所有数据集
        df = self.normalize_timestamps(df)
        df = self.normalize_user_ids(df)

        # 特定数据集的处理
        if dataset_type == 'logon':
            # 标准化登录相关字段
            device_cols = [col for col in df.columns if 'device' in col.lower()]
            df = self.normalize_categorical_columns(df, device_cols)

            # 添加登录时间特征
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6])
                df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)) & ~df['is_weekend']

        elif dataset_type == 'email':
            # 邮件特定处理
            df = self.extract_email_parts(df)

            # 添加敏感度评分
            if 'subject' in df.columns and 'body' in df.columns:
                sensitive_keywords = [
                    'confidential', 'private', 'secret', 'sensitive', 'personal',
                    'financial', 'password', 'credentials'
                ]

                def calculate_sensitivity(subject, body):
                    score = 0
                    # 检查主题和正文是否包含敏感关键词
                    subject = str(subject).lower()
                    body = str(body).lower()

                    # 主题中的敏感词权重更高
                    for keyword in sensitive_keywords:
                        if keyword in subject:
                            score += 2
                        if keyword in body:
                            score += 1

                    # 是否包含附件
                    if 'attachment' in body or 'attached' in body:
                        score += 1

                    # 归一化到0-1范围
                    return min(1.0, score / 10.0)

                df['sensitivity_score'] = df.apply(
                    lambda row: calculate_sensitivity(row.get('subject', ''), row.get('body', '')),
                    axis=1
                )

        elif dataset_type == 'file':
            # 文件访问特定处理
            df = self.normalize_file_paths(df)

            # 添加敏感度评分
            if 'filename' in df.columns:
                sensitive_paths = [
                    'financial', 'hr', 'executive', 'confidential', 'secret',
                    'password', 'credential', 'userdata', 'personal'
                ]

                def file_sensitivity(path):
                    path = str(path).lower()

                    # 检查路径是否包含敏感词
                    for keyword in sensitive_paths:
                        if keyword in path:
                            return 0.8

                    # 检查特定的文件类型
                    if path.endswith(('.xls', '.xlsx', '.csv', '.db', '.mdb', '.accdb')):
                        return 0.7  # 数据库和电子表格
                    elif path.endswith(('.doc', '.docx', '.pdf')):
                        return 0.5  # 文档
                    elif path.endswith(('.ppt', '.pptx')):
                        return 0.5  # 演示文稿
                    elif path.endswith(('.exe', '.bat', '.ps1', '.vbs', '.sh')):
                        return 0.9  # 可执行文件
                    else:
                        return 0.3  # 其他类型

                df['file_sensitivity'] = df['filename'].apply(file_sensitivity)

        elif dataset_type == 'http':
            # HTTP访问特定处理
            if 'url' in df.columns:
                # 清理URL
                df['url'] = df['url'].astype(str).str.strip().str.lower()

                # 提取域名
                df['domain'] = df['url'].apply(
                    lambda x: x.split('/')[2] if '://' in x and len(x.split('/')) > 2 else ''
                )

                # 检测是否为外部网站
                if 'internal_domains' in df.columns:
                    internal_domains = df['internal_domains'].unique()
                    df['is_external'] = ~df['domain'].isin(internal_domains)
                else:
                    # 假设公司域名通常使用专有域名
                    df['is_external'] = ~df['domain'].str.contains('internal|intranet|corp|company')

                # URL敏感度评分
                sensitive_parts = [
                    'admin', 'config', 'setting', 'password', 'login', 'user',
                    'account', 'secure', 'payment', 'financial'
                ]

                def url_sensitivity(url):
                    url = str(url).lower()

                    # 检查URL是否包含敏感部分
                    for part in sensitive_parts:
                        if part in url:
                            return 0.7

                    # 检查特定的URL模式
                    if 'login' in url or 'signin' in url:
                        return 0.6
                    elif 'download' in url or 'upload' in url:
                        return 0.5
                    elif 'admin' in url or 'manage' in url:
                        return 0.8
                    else:
                        return 0.3

                df['url_sensitivity'] = df['url'].apply(url_sensitivity)

        # 处理缺失值
        df = self._handle_missing_values(df)

        return df

    def _handle_missing_values(self, df):
        """处理DataFrame中的缺失值"""
        # 创建一个副本以避免修改原始数据
        df = df.copy()

        # 对不同类型的列使用不同的填充策略
        for col in df.columns:
            # 跳过ID列
            if 'id' in col.lower():
                continue

            # 处理数值列
            if pd.api.types.is_numeric_dtype(df[col]):
                # 用中位数填充数值列
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)

            # 处理分类列
            elif df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # 用最常见值填充分类列，如果没有值则用"unknown"
                if df[col].count() > 0:
                    mode_val = df[col].mode()[0]
                    df[col].fillna(mode_val, inplace=True)
                else:
                    df[col].fillna("unknown", inplace=True)

            # 处理日期时间列
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # 日期时间通常不填充，但如果需要，可以使用最近的日期
                pass

        return df

    def create_feature_matrix(self, df, feature_cols=None, categorical_cols=None):
        """
        创建用于机器学习的特征矩阵

        Args:
            df: 要处理的DataFrame
            feature_cols: 要包含的特征列列表，如果为None则使用所有合适的列
            categorical_cols: 分类列列表，将进行one-hot编码

        Returns:
            tuple: (X, feature_names), 其中X是特征矩阵，feature_names是特征名称列表
        """
        df = df.copy()

        # 自动检测特征列
        if feature_cols is None:
            feature_cols = []
            for col in df.columns:
                # 排除ID列和其他不适合作为特征的列
                if 'id' in col.lower() or 'name' in col.lower() or 'timestamp' in col.lower():
                    continue

                # 包含数值和布尔列
                if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                    feature_cols.append(col)

        # 自动检测分类列
        if categorical_cols is None:
            categorical_cols = []
            for col in feature_cols:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    categorical_cols.append(col)

        # 创建特征DataFrame
        feature_df = df[feature_cols].copy()

        # 处理分类特征
        for col in categorical_cols:
            if col in feature_df.columns:
                # 创建虚拟变量（one-hot编码）
                dummies = pd.get_dummies(feature_df[col], prefix=col, drop_first=True)

                # 添加到特征DataFrame
                feature_df = pd.concat([feature_df, dummies], axis=1)

                # 删除原始分类列
                feature_df.drop(col, axis=1, inplace=True)

        # 确保所有列都是数值型
        for col in feature_df.columns:
            if not pd.api.types.is_numeric_dtype(feature_df[col]):
                try:
                    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
                    feature_df[col].fillna(feature_df[col].median(), inplace=True)
                except:
                    logger.warning(f"Could not convert column {col} to numeric, dropping it")
                    feature_df.drop(col, axis=1, inplace=True)

        # 获取特征名称
        feature_names = feature_df.columns.tolist()

        # 转换为numpy数组
        X = feature_df.values

        # 确保没有NaN值
        X = np.nan_to_num(X)

        return X, feature_names