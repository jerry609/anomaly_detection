"""
多维异常检测器实现。
"""

import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
from datetime import datetime, timedelta
import warnings
from anomaly_detection.models.base_detector import BaseAnomalyDetector
from anomaly_detection.analyzers.time_analyzer import TimeAnalyzer
from anomaly_detection.analyzers.access_analyzer import AccessAnalyzer
from anomaly_detection.analyzers.email_analyzer import EmailAnalyzer
from anomaly_detection.analyzers.org_analyzer import OrgAnalyzer
import config
from anomaly_detection.utils.text_processor import extract_email_parts

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class MultiDimensionalAnomalyDetector(BaseAnomalyDetector):
    """多维异常检测器"""

    def __init__(self):
        """初始化异常检测器"""
        self.datasets = {}
        self.user_profiles = {}
        self.anomalies = {
            'time': defaultdict(list),
            'access': defaultdict(list),
            'email': defaultdict(list),
            'organizational': defaultdict(list)
        }

        self.ml_models = {  # 初始化ML模型字典
            'isolation_forest': None,
            'lof': None,
            'dbscan': None
        }

        self.risk_scores = {}  # 主要风险评分
        self.risk_dimensions = {}  # 多维风险评分
        self.risk_factors = {}  # 风险因素分析
        self.risk_history = defaultdict(list)  # 风险趋势历史
        self.dynamic_thresholds = {}  # 用户动态阈值
        self.user_behavior_models = {}  # 用户行为模型
        self.detection_date = datetime.now()  # 当前检测日期

        # 初始化各分析器
        self.time_analyzer = TimeAnalyzer()
        self.access_analyzer = AccessAnalyzer()
        self.email_analyzer = EmailAnalyzer()
        self.org_analyzer = OrgAnalyzer()
        # 初始化ML模型
        self._init_ml_models()

    def _init_ml_models(self):
        """初始化机器学习模型"""
        # 隔离森林模型
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=0.05,
            random_state=42
        )

        # 局部异常因子模型
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True
        )

        # DBSCAN聚类模型
        self.dbscan = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric='euclidean'
        )

        # 数据标准化
        self.scaler = StandardScaler()

        # 降维
        self.pca = PCA(n_components=0.95)

        # 用户关系图
        self.user_graph = nx.Graph()

    @property
    def anomaly_results(self):
        """
        提供与anomalies字典相同的访问，用于与ReportGenerator兼容

        Returns:
            dict: 按类型和用户分组的异常结果
        """
        return self.anomalies

    def _prepare_data_for_analyzer(self, dataset_name, column_mappings=None):
        """
        准备用于分析器的数据，确保列名一致

        Args:
            dataset_name: 数据集名称
            column_mappings: 列名映射字典，默认为None

        Returns:
            DataFrame: 准备好的数据副本，或None如果数据不可用
        """
        if dataset_name not in self.datasets or self.datasets[dataset_name].empty:
            return None

        # 创建数据副本
        df = self.datasets[dataset_name].copy()

        # 应用默认的列名映射
        default_mappings = {
            'logon': {'user': 'user_id'},
            'device': {'user': 'user_id'},
            'email': {'user': 'user_id', 'from': 'from_user'},
            'file': {'user': 'user_id'},
            'http': {'user': 'user_id'},
            # ldap和psychometric已经有正确的user_id列
        }

        # 使用默认映射，除非提供了特定映射
        mappings = column_mappings or default_mappings.get(dataset_name, {})

        # 应用列名映射
        rename_dict = {}
        for src, dst in mappings.items():
            if src in df.columns and dst not in df.columns:
                rename_dict[src] = dst

        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
            logger.debug(f"Renamed columns in {dataset_name} dataset: {rename_dict}")

        return df

    def set_preprocessed_data(self, processed_datasets, user_profiles):
        """设置预处理后的数据和用户配置文件"""
        self.datasets = processed_datasets
        self.user_profiles = user_profiles

        # 不再调用不存在的方法，而是直接设置分析器需要的数据
        # 检查各分析器是否有设置数据的属性
        if hasattr(self.time_analyzer, 'datasets'):
            self.time_analyzer.datasets = processed_datasets
        if hasattr(self.time_analyzer, 'user_profiles'):
            self.time_analyzer.user_profiles = user_profiles

        if hasattr(self.access_analyzer, 'datasets'):
            self.access_analyzer.datasets = processed_datasets
        if hasattr(self.access_analyzer, 'user_profiles'):
            self.access_analyzer.user_profiles = user_profiles

        if hasattr(self.email_analyzer, 'datasets'):
            self.email_analyzer.datasets = processed_datasets
        if hasattr(self.email_analyzer, 'user_profiles'):
            self.email_analyzer.user_profiles = user_profiles

        if hasattr(self.org_analyzer, 'datasets'):
            self.org_analyzer.datasets = processed_datasets
        if hasattr(self.org_analyzer, 'user_profiles'):
            self.org_analyzer.user_profiles = user_profiles

        logger.info(f"Loaded preprocessed data for {len(self.user_profiles)} users")
        return self

    def detect_all_anomalies(self):
        """运行所有异常检测功能"""
        logger.info("Starting anomaly detection process")

        self.detect_time_anomalies()
        self.detect_access_anomalies()
        self.detect_email_anomalies()
        self.detect_org_anomalies()

        # 添加新的检测方法
        self.detect_ml_anomalies()
        self.detect_time_series_anomalies()
        self.detect_graph_anomalies()

        # 更新风险评分
        self.calculate_risk_scores()
        self.update_risk_history()
        self.calculate_dynamic_thresholds()

        logger.info("Anomaly detection complete")
        return self

    def detect_time_anomalies(self):
        """检测时间相关的异常行为"""
        logger.info("Detecting time-based anomalies")

        # 使用辅助方法准备数据
        logon_data = self._prepare_data_for_analyzer('logon')
        if logon_data is None:
            logger.warning("No logon data available for time anomaly detection")
            return self

        # 传递处理后的数据
        time_anomalies = self.time_analyzer.detect_anomalies(new_logon_data=logon_data)

        # 合并到主异常字典
        for user_id, anomalies in time_anomalies.items():
            self.anomalies['time'][user_id].extend(anomalies)

        logger.info(f"Detected time anomalies for {len(time_anomalies)} users")
        return self

    def detect_access_anomalies(self):
        """检测访问模式异常"""
        logger.info("Detecting access pattern anomalies")

        # 检查是否有必要的数据集
        required_datasets = {'file', 'http'}
        available = required_datasets.intersection(self.datasets.keys())

        if not available:
            logger.warning("No access data available for access anomaly detection")
            return self

        # 需要合并文件和HTTP数据为一个DataFrame
        access_frames = []

        for key in available:
            processed_data = self._prepare_data_for_analyzer(key)
            if processed_data is not None:
                # 添加数据源标识列
                processed_data['resource_id'] = processed_data.get('url', processed_data.get('filename', 'unknown'))
                processed_data['source'] = key
                access_frames.append(processed_data)

        if not access_frames:
            logger.warning("No valid access data available after processing")
            return self

        # 合并所有DataFrame
        combined_access_data = pd.concat(access_frames, ignore_index=True)

        # 确保有date列
        if 'date' not in combined_access_data.columns and 'timestamp' in combined_access_data.columns:
            combined_access_data['date'] = pd.to_datetime(combined_access_data['timestamp']).dt.date

        # 传递合并后的DataFrame
        access_anomalies = self.access_analyzer.detect_anomalies(new_access_data=combined_access_data)

        # 合并到主异常字典
        for user_id, anomalies in access_anomalies.items():
            self.anomalies['access'][user_id].extend(anomalies)

        logger.info(f"Detected access anomalies for {len(access_anomalies)} users")
        return self

    def detect_email_anomalies(self):
        """检测电子邮件异常"""
        logger.info("Detecting email anomalies")

        # 确保有邮件数据
        email_data = self._prepare_data_for_analyzer('email')
        if email_data is None:
            logger.warning("No email data available for email anomaly detection")
            return self

        # 使用正确的参数名称 - new_email_data 而不是 email_data
        email_anomalies = self.email_analyzer.detect_anomalies(new_email_data=email_data)

        # 合并到主异常字典
        for user_id, anomalies in email_anomalies.items():
            self.anomalies['email'][user_id].extend(anomalies)

        logger.info(f"Detected email anomalies for {len(email_anomalies)} users")
        return self

    def detect_org_anomalies(self):
        """检测组织层面的异常"""
        logger.info("Detecting organizational anomalies")

        # OrgAnalyzer需要email数据包含组织交互信息
        email_data = self._prepare_data_for_analyzer('email')
        if email_data is None:
            logger.warning("No email data available for organizational anomaly detection")
            return self

        # 预处理邮件数据，确保列名正确
        email_data = email_data.copy()

        # 确保存在from_user和to_users列
        if 'from_user' not in email_data.columns and 'user' in email_data.columns:
            email_data['from_user'] = email_data['user']

        if 'to_users' not in email_data.columns and 'to' in email_data.columns:
            # 确保to_users列是列表形式
            email_data['to_users'] = email_data['to'].apply(extract_email_parts)

        # 检查并预处理LDAP数据
        ldap_data = self._prepare_data_for_analyzer('ldap')
        if ldap_data is not None and hasattr(self.org_analyzer, 'set_org_structure'):
            # 创建LDAP数据的副本以避免修改原始数据
            ldap_data = ldap_data.copy()

            # 基于实际LDAP列进行精确映射
            column_map = {
                # 必需列映射
                'employee_id': 'user_id',  # 使用user_id作为employee_id
                'manager_id': 'supervisor',  # 使用supervisor作为manager_id

                # 可选但有用的列映射
                'email_address': 'email',  # 使用email列
                'employee_role': 'role',  # 使用role列
                'team_name': 'team'  # 使用team列
            }

            # 执行列映射
            for target_col, source_col in column_map.items():
                if target_col not in ldap_data.columns and source_col in ldap_data.columns:
                    ldap_data[target_col] = ldap_data[source_col]

            # 创建is_manager标志，基于用户是否在supervisor列中出现
            if 'is_manager' not in ldap_data.columns:
                supervisors = set(ldap_data['supervisor'].dropna().unique())
                ldap_data['is_manager'] = ldap_data['user_id'].isin(supervisors)

            # 创建完整的组织路径，用于分析部门内/外通信
            if 'org_path' not in ldap_data.columns:
                ldap_data['org_path'] = ldap_data.apply(
                    lambda
                        row: f"{row.get('business_unit', '')}/{row.get('functional_unit', '')}/{row.get('department', '')}/{row.get('team', '')}",
                    axis=1
                )

            # 设置组织结构
            try:
                self.org_analyzer.set_org_structure(ldap_data)
                logger.info("Successfully configured organization structure from LDAP data")
            except Exception as e:
                logger.error(f"Error configuring organization structure: {str(e)}")

        # 传递邮件数据用于组织分析
        try:
            # OrgAnalyzer返回DataFrame，需要转换为用户异常字典
            anomaly_df = self.org_analyzer.detect_anomalies(new_interaction_data=email_data)

            # 将DataFrame转换为用户异常字典格式
            org_anomalies = {}

            # 仅处理被标记为异常的行
            anomaly_rows = anomaly_df[anomaly_df['is_anomaly'] == True]

            for _, row in anomaly_rows.iterrows():
                user_id = row['from_user']

                # 创建异常对象
                anomaly = {
                    'type': 'organizational',
                    'score': float(row['anomaly_score']),
                    'timestamp': pd.Timestamp.now(),
                    'description': row['reason'],
                    'details': {
                        'event_id': row['event_id'],
                        'to_users': row['to_users'],
                        'department': row['department']
                    }
                }

                # 初始化用户的异常列表（如果不存在）
                if user_id not in org_anomalies:
                    org_anomalies[user_id] = []

                org_anomalies[user_id].append(anomaly)

            # 合并到主异常字典
            for user_id, anomalies in org_anomalies.items():
                self.anomalies['organizational'][user_id].extend(anomalies)

            logger.info(f"Detected organizational anomalies for {len(org_anomalies)} users")
        except Exception as e:
            logger.error(f"Failed to detect organizational anomalies: {str(e)}")
            # 在调试模式下输出更详细的错误信息
            logger.debug(f"Error details: {e}", exc_info=True)

        return self

    def _get_user_sensitivity(self, user_id):
        """基于用户角色和部门计算敏感度系数"""
        user_profile = self.user_profiles.get(user_id, {})
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


    def get_high_risk_users(self, threshold=None):
        """获取高风险用户列表 - 分层版本"""
        high_threshold = threshold or config.HIGH_RISK_THRESHOLD
        medium_threshold = config.MEDIUM_RISK_THRESHOLD
        low_threshold = config.LOW_RISK_THRESHOLD

        high_risk = {user_id: score for user_id, score in self.risk_scores.items()
                     if score >= high_threshold}
        medium_risk = {user_id: score for user_id, score in self.risk_scores.items()
                       if medium_threshold <= score < high_threshold}
        low_risk = {user_id: score for user_id, score in self.risk_scores.items()
                    if low_threshold <= score < medium_threshold}

        return {
            'high_risk': sorted(high_risk.items(), key=lambda x: x[1], reverse=True),
            'medium_risk': sorted(medium_risk.items(), key=lambda x: x[1], reverse=True),
            'low_risk': sorted(low_risk.items(), key=lambda x: x[1], reverse=True)
        }

    def get_risk_factors(self, user_id=None):
        """获取风险因素构成分析"""
        if user_id:
            return self.risk_factors.get(user_id, {})
        return self.risk_factors

    def get_high_risk_users(self, threshold=None):
        """获取高风险用户列表 - 分层版本"""
        high_threshold = threshold or config.HIGH_RISK_THRESHOLD
        medium_threshold = config.MEDIUM_RISK_THRESHOLD
        low_threshold = config.LOW_RISK_THRESHOLD

        high_risk = {user_id: score for user_id, score in self.risk_scores.items()
                     if score >= high_threshold}
        medium_risk = {user_id: score for user_id, score in self.risk_scores.items()
                       if medium_threshold <= score < high_threshold}
        low_risk = {user_id: score for user_id, score in self.risk_scores.items()
                    if low_threshold <= score < medium_threshold}

        return {
            'high_risk': sorted(high_risk.items(), key=lambda x: x[1], reverse=True),
            'medium_risk': sorted(medium_risk.items(), key=lambda x: x[1], reverse=True),
            'low_risk': sorted(low_risk.items(), key=lambda x: x[1], reverse=True)
        }

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

    def get_user_anomalies(self, user_id):
        """获取特定用户的所有异常"""
        result = {}
        for category in ['time', 'access', 'email', 'organizational']:
            if self.anomalies[category][user_id]:
                result[category] = self.anomalies[category][user_id]
        return result

    def get_all_anomalies(self):
        """获取所有异常"""
        return self.anomalies

    def get_risk_scores(self):
        """获取所有风险评分"""
        return self.risk_scores

    def detect_ml_anomalies(self):
        """使用机器学习模型检测异常模式"""
        logger.info("Detecting anomalies using machine learning models")

        # 收集特征数据
        features_by_user = self._prepare_features_for_ml()
        if not features_by_user:
            logger.warning("No sufficient data for ML-based anomaly detection")
            return self

        # 合并所有用户的特征为一个数据集
        user_ids = []
        feature_rows = []

        for user_id, features in features_by_user.items():
            user_ids.append(user_id)
            feature_rows.append(features)

        # 创建特征矩阵
        X = np.array(feature_rows)

        # 标准化特征
        try:
            X_scaled = self.scaler.fit_transform(X)

            # 降维以处理高维特征
            if X_scaled.shape[1] > 2:
                X_reduced = self.pca.fit_transform(X_scaled)
            else:
                X_reduced = X_scaled

            # 应用隔离森林模型
            self.isolation_forest.fit(X_reduced)
            if_scores = self.isolation_forest.decision_function(X_reduced)
            if_anomalies = self.isolation_forest.predict(X_reduced)

            # 应用LOF模型
            self.lof.fit(X_reduced)
            lof_scores = self.lof.decision_function(X_reduced)

            # 应用DBSCAN模型
            dbscan_labels = self.dbscan.fit_predict(X_reduced)

            # 综合多个模型的结果
            for i, user_id in enumerate(user_ids):
                # 计算综合异常分数 (归一化到0-100)
                if_normalized = (1 - (if_scores[i] + abs(min(if_scores)))) / (
                        abs(min(if_scores)) + max(if_scores)) * 100
                lof_normalized = (1 - (lof_scores[i] + abs(min(lof_scores)))) / (
                        abs(min(lof_scores)) + max(lof_scores)) * 100

                # DBSCAN中-1表示异常
                dbscan_score = 100 if dbscan_labels[i] == -1 else 0

                # 综合评分 (加权平均)
                ml_score = 0.4 * if_normalized + 0.4 * lof_normalized + 0.2 * dbscan_score

                # 仅当分数超过阈值时记录异常
                if ml_score > 60:
                    anomaly = {
                        'type': 'ml_pattern',
                        'score': ml_score,
                        'timestamp': pd.Timestamp.now(),
                        'description': f"Machine learning detected unusual behavior pattern",
                        'details': {
                            'if_score': if_normalized,
                            'lof_score': lof_normalized,
                            'dbscan_label': dbscan_labels[i],
                            'pca_coordinates': X_reduced[i].tolist() if X_reduced.ndim > 1 else [X_reduced[i]]
                        }
                    }

                    # 添加到用户的访问异常列表中
                    self.anomalies['access'][user_id].append(anomaly)

            logger.info(
                f"ML models detected anomalies for {sum(1 for user_id in user_ids if any(a.get('type') == 'ml_pattern' for a in self.anomalies['access'][user_id]))} users")

        except Exception as e:
            logger.error(f"Error in ML anomaly detection: {str(e)}")
            logger.debug(f"Error details: {e}", exc_info=True)

        return self

    def _prepare_features_for_ml(self):
        """准备用于机器学习的特征"""
        features_by_user = {}

        # 确保有必要的数据集
        required_datasets = {'logon', 'file', 'email'}
        if not all(dataset in self.datasets for dataset in required_datasets):
            return {}

        # 为每个用户计算特征
        for user_id in self.user_profiles:
            # 登录特征
            logon_data = self.datasets['logon'][self.datasets['logon']['user_id'] == user_id]
            login_count = len(logon_data)
            unique_devices = logon_data['device_id'].nunique() if 'device_id' in logon_data.columns else 0

            # 登录时间特征 - 计算非工作时间登录比例
            if 'timestamp' in logon_data.columns:
                logon_data['hour'] = pd.to_datetime(logon_data['timestamp']).dt.hour
                non_business_logins = len(logon_data[(logon_data['hour'] < 8) | (logon_data['hour'] > 18)])
                non_business_ratio = non_business_logins / login_count if login_count > 0 else 0
            else:
                non_business_ratio = 0

            # 文件访问特征
            file_data = self.datasets['file'][self.datasets['file']['user_id'] == user_id]
            file_count = len(file_data)
            unique_files = file_data['filename'].nunique() if 'filename' in file_data.columns else 0

            # 敏感文件访问比例
            sensitive_files = 0
            if 'sensitivity' in file_data.columns:
                sensitive_files = len(file_data[file_data['sensitivity'] > 0.7])
            sensitive_ratio = sensitive_files / file_count if file_count > 0 else 0

            # 邮件特征
            email_data = self.datasets['email'][self.datasets['email']['user_id'] == user_id]
            email_count = len(email_data)
            external_emails = 0

            if 'to_domain' in email_data.columns:
                # 检查发送到外部域名的邮件
                company_domain = self.user_profiles[user_id].get('email_domain', 'company.com')
                external_emails = len(email_data[~email_data['to_domain'].str.contains(company_domain, na=False)])

            external_ratio = external_emails / email_count if email_count > 0 else 0

            # 收集特征向量
            features = [
                login_count,
                unique_devices,
                non_business_ratio,
                file_count,
                unique_files,
                sensitive_ratio,
                email_count,
                external_ratio
            ]

            # 用户敏感度作为特征
            sensitivity = self._get_user_sensitivity(user_id)
            features.append(sensitivity)

            # 添加到字典
            features_by_user[user_id] = features

        return features_by_user

    def detect_time_series_anomalies(self):
        """使用时间序列分析检测逐渐变化的行为模式"""
        logger.info("Detecting gradual behavior changes using time series analysis")

        # 确保有足够的历史数据
        if not hasattr(self, 'historical_data') or not self.historical_data:
            logger.warning("No historical data available for time series analysis")
            return self

        try:
            # 针对每个用户进行时间序列分析
            for user_id in self.user_profiles:
                # 获取用户历史行为数据
                user_history = self._get_user_historical_data(user_id)
                if not user_history or len(user_history) < 7:  # 至少需要一周的数据
                    continue

                # 分析行为趋势
                # 1. 登录时间趋势
                login_time_trend = self._analyze_login_time_trend(user_history)

                # 2. 资源访问趋势
                access_trend = self._analyze_access_pattern_trend(user_history)

                # 3. 通信模式趋势
                communication_trend = self._analyze_communication_trend(user_history)

                # 检测显著趋势变化
                significant_trends = []

                if login_time_trend > 0.6:
                    significant_trends.append({
                        'type': 'login_time_drift',
                        'score': login_time_trend * 100,
                        'description': "Gradual shift in login times detected"
                    })

                if access_trend > 0.6:
                    significant_trends.append({
                        'type': 'access_pattern_drift',
                        'score': access_trend * 100,
                        'description': "Gradual change in resource access patterns"
                    })

                if communication_trend > 0.6:
                    significant_trends.append({
                        'type': 'communication_drift',
                        'score': communication_trend * 100,
                        'description': "Gradual shift in communication patterns"
                    })

                # 将显著趋势添加为异常
                for trend in significant_trends:
                    anomaly = {
                        'type': 'time_series',
                        'score': trend['score'],
                        'timestamp': pd.Timestamp.now(),
                        'description': trend['description'],
                        'details': {
                            'drift_type': trend['type'],
                            'trend_strength': trend['score'] / 100
                        }
                    }

                    # 添加到时间异常
                    self.anomalies['time'][user_id].append(anomaly)

            logger.info(
                f"Time series analysis detected behavior drifts for {sum(1 for user_id in self.user_profiles if any(a.get('type') == 'time_series' for a in self.anomalies['time'][user_id]))} users")

        except Exception as e:
            logger.error(f"Error in time series anomaly detection: {str(e)}")
            logger.debug(f"Error details: {e}", exc_info=True)

        return self

    def _get_user_historical_data(self, user_id):
        """获取用户的历史行为数据"""
        # 这里假设有存储历史数据的结构
        # 在实际实现中，应该从数据库或其他存储中获取
        if not hasattr(self, 'historical_data'):
            return []

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

    def detect_graph_anomalies(self):
        """使用图分析检测复杂的关系异常"""
        logger.info("Detecting relationship anomalies using graph analysis")

        # 确保有邮件数据用于构建关系图
        if 'email' not in self.datasets or self.datasets['email'].empty:
            logger.warning("No email data available for graph analysis")
            return self

        try:
            # 构建用户关系图
            self._build_user_graph()

            # 计算图中心性指标
            self._calculate_graph_metrics()

            # 检测异常的关系模式
            self._detect_relationship_anomalies()

            logger.info(
                f"Graph analysis detected relationship anomalies for {sum(1 for user_id in self.user_profiles if any(a.get('type') == 'graph' for a in self.anomalies['organizational'][user_id]))} users")

        except Exception as e:
            logger.error(f"Error in graph anomaly detection: {str(e)}")
            logger.debug(f"Error details: {e}", exc_info=True)

        return self

    def _build_user_graph(self):
        """构建用户关系图"""
        # 重置图
        self.user_graph = nx.Graph()

        # 添加节点 - 所有用户
        for user_id in self.user_profiles:
            self.user_graph.add_node(user_id, type='user')

        # 添加边 - 基于邮件通信
        email_data = self.datasets['email']

        # 确保有必要的列
        if 'from_user' not in email_data.columns or 'to' not in email_data.columns:
            return

        # 添加通信关系
        for _, row in email_data.iterrows():
            from_user = row['from_user']

            # 处理接收者列表
            to_users = row['to']
            if isinstance(to_users, str):
                to_users = extract_email_parts(to_users)

            if not isinstance(to_users, list):
                to_users = [to_users]

            # 添加边
            for to_user in to_users:
                if to_user in self.user_profiles and from_user in self.user_profiles:
                    # 如果边已存在，增加权重
                    if self.user_graph.has_edge(from_user, to_user):
                        self.user_graph[from_user][to_user]['weight'] += 1
                    else:
                        self.user_graph.add_edge(from_user, to_user, weight=1)

    def _calculate_graph_metrics(self):
        """计算图分析指标"""
        # 如果图为空，则返回
        if not self.user_graph.nodes:
            return

        # 计算中心性指标
        try:
            # 度中心性
            degree_centrality = nx.degree_centrality(self.user_graph)

            # 介数中心性 (可能计算较慢)
            if len(self.user_graph) <= 100:  # 限制计算规模
                betweenness_centrality = nx.betweenness_centrality(self.user_graph)
            else:
                # 对大型图采样计算
                betweenness_centrality = nx.betweenness_centrality(self.user_graph, k=min(50, len(self.user_graph)))

            # 接近中心性
            closeness_centrality = nx.closeness_centrality(self.user_graph)

            # 特征向量中心性
            eigenvector_centrality = nx.eigenvector_centrality(self.user_graph, max_iter=100, tol=1e-6)

            # 将指标存储到图中
            for node in self.user_graph.nodes:
                self.user_graph.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
                self.user_graph.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
                self.user_graph.nodes[node]['closeness_centrality'] = closeness_centrality.get(node, 0)
                self.user_graph.nodes[node]['eigenvector_centrality'] = eigenvector_centrality.get(node, 0)

        except Exception as e:
            logger.error(f"Error calculating graph metrics: {str(e)}")

    def _detect_relationship_anomalies(self):
        """检测关系图中的异常模式"""
        # 如果图为空或没有指标，则返回
        if not self.user_graph.nodes or 'degree_centrality' not in next(iter(self.user_graph.nodes(data=True)))[1]:
            return

        # 检测各种关系异常

        # 1. 异常高的中心性
        for user_id in self.user_profiles:
            if user_id not in self.user_graph.nodes:
                continue

            node = self.user_graph.nodes[user_id]

            # 获取中心性指标
            degree = node.get('degree_centrality', 0)
            betweenness = node.get('betweenness_centrality', 0)
            closeness = node.get('closeness_centrality', 0)
            eigenvector = node.get('eigenvector_centrality', 0)

            # 检测异常高的中心性
            # 在实际系统中，应使用机器学习或统计方法确定阈值
            if degree > 0.8 or betweenness > 0.7 or eigenvector > 0.8:
                anomaly = {
                    'type': 'graph',
                    'score': max(degree, betweenness, eigenvector) * 100,
                    'timestamp': pd.Timestamp.now(),
                    'description': "Abnormally high network centrality detected",
                    'details': {
                        'degree_centrality': degree,
                        'betweenness_centrality': betweenness,
                        'closeness_centrality': closeness,
                        'eigenvector_centrality': eigenvector
                    }
                }

                self.anomalies['organizational'][user_id].append(anomaly)

        # 2. 检测异常的三角形结构 (潜在的共谋)
        triangles = nx.triangles(self.user_graph)
        suspicious_triangles = {node: count for node, count in triangles.items() if count > 10}

        for user_id, triangle_count in suspicious_triangles.items():
            # 检查是否同一部门内的三角形过多
            user_dept = self.user_profiles.get(user_id, {}).get('department', '')
            neighbors = list(self.user_graph.neighbors(user_id))

            same_dept_neighbors = 0
            for neighbor in neighbors:
                if neighbor in self.user_profiles:
                    neighbor_dept = self.user_profiles[neighbor].get('department', '')
                    if neighbor_dept == user_dept:
                        same_dept_neighbors += 1

            # 如果与不同部门的人形成了过多三角形，可能是异常
            if same_dept_neighbors < len(neighbors) * 0.7 and triangle_count > 5:
                anomaly = {
                    'type': 'graph',
                    'score': min(90, triangle_count * 5),  # 转换为分数
                    'timestamp': pd.Timestamp.now(),
                    'description': "Suspicious communication triangle patterns across departments",
                    'details': {
                        'triangle_count': triangle_count,
                        'cross_department_ratio': 1 - (same_dept_neighbors / len(neighbors))
                    }
                }

                self.anomalies['organizational'][user_id].append(anomaly)

    def _update_dimension_scores(self, dimensions, anomaly_type, severity):
        """更新风险维度评分"""
        # 行为维度
        if anomaly_type in ['ml_pattern', 'access_pattern_drift', 'communication_drift']:
            dimensions['behavior'] += severity

        # 访问维度
        if anomaly_type in ['access', 'file_access', 'abnormal_resource']:
            dimensions['access'] += severity

        # 通信维度
        if anomaly_type in ['email', 'organizational', 'graph', 'communication']:
            dimensions['communication'] += severity

        # 时间维度
        if anomaly_type in ['time', 'login_time_drift', 'time_series']:
            dimensions['temporal'] += severity

        # 技术维度
        if anomaly_type in ['device', 'vpn', 'technical']:
            dimensions['technical'] += severity

    def update_risk_history(self):
        """更新风险趋势历史"""
        # 记录当前日期的风险评分
        current_date = datetime.now().date()

        for user_id, score in self.risk_scores.items():
            # 添加新的评分记录
            self.risk_history[user_id].append({
                'date': current_date,
                'score': score,
                'dimensions': self.risk_dimensions.get(user_id, {})
            })

            # 限制历史记录长度
            max_history = getattr(config, 'MAX_RISK_HISTORY', 90)  # 默认保留90天
            if len(self.risk_history[user_id]) > max_history:
                self.risk_history[user_id] = self.risk_history[user_id][-max_history:]

        logger.info(f"Updated risk history for {len(self.risk_scores)} users")

    def calculate_risk_scores(self):
        """增强版本：多维度风险评分计算"""
        logger.info("计算多维风险评分...")

        # 初始化风险评分结构
        self.risk_scores = {}
        self.risk_factors = {}

        # 获取配置权重，提供默认值
        category_weights = getattr(config, 'ANOMALY_WEIGHTS', {
            'organizational': 3.0,  # 组织级别异常权重最高
            'access': 2.5,  # 访问级别异常次之
            'email': 2.0,  # 邮件异常
            'time': 1.5  # 时间异常
        })

        # 获取子类型权重
        subtype_weights = getattr(config, 'ANOMALY_SUBTYPE_WEIGHTS', {
            'after_hours_access': 2.0,
            'unusual_file_access': 2.5,
            'data_exfiltration': 3.0,
            'cross_department_email': 1.5
        })

        # 获取其他配置参数
        base_score = getattr(config, 'RISK_BASE_SCORE', 10)
        quantity_weight = getattr(config, 'RISK_QUANTITY_WEIGHT', 0.7)
        time_decay_factor = getattr(config, 'RISK_TIME_DECAY', 0.9)  # 时间衰减因子

        # 遍历所有用户
        for user_id in self.user_profiles.keys():
            # 初始化各维度风险评分
            dimension_scores = {
                'time_sensitivity': 0,
                'data_sensitivity': 0,
                'behavior_deviation': 0,
                'quantity': 0
            }

            # 收集所有用户异常
            all_user_anomalies = []
            category_counts = {}

            # 计算各类别异常的风险贡献
            for category, user_anomalies_dict in self.anomalies.items():
                user_anomalies = user_anomalies_dict.get(user_id, [])
                if not user_anomalies:
                    continue

                category_weight = category_weights.get(category, 1.0)
                category_counts[category] = len(user_anomalies)

                for anomaly in user_anomalies:
                    # 确保每个异常都有基础得分：使用 score 或 severity，或默认值 5
                    base_anomaly_score = anomaly.get('score', 0) or anomaly.get('severity', 5)
                    if base_anomaly_score == 0:  # 处理0分异常
                        base_anomaly_score = 5  # 设置默认的非零得分

                    # 获取子类型权重
                    subtype = anomaly.get('subtype', '')
                    subtype_weight = subtype_weights.get(subtype, 1.0)

                    # 计算时间敏感度
                    time_sensitivity = self._calculate_time_sensitivity(anomaly)
                    dimension_scores['time_sensitivity'] += time_sensitivity * base_anomaly_score * category_weight

                    # 计算数据敏感度
                    data_sensitivity = self._calculate_data_sensitivity(anomaly, user_id)
                    dimension_scores['data_sensitivity'] += data_sensitivity * base_anomaly_score * category_weight

                    # 计算行为偏差程度
                    behavior_deviation = self._calculate_behavior_deviation(anomaly, user_id)
                    dimension_scores[
                        'behavior_deviation'] += behavior_deviation * base_anomaly_score * category_weight * subtype_weight

                    # 添加到所有异常列表
                    all_user_anomalies.append(anomaly)

            # 计算数量维度得分
            anomaly_count = len(all_user_anomalies)
            dimension_scores['quantity'] = min(10, anomaly_count * quantity_weight)

            # 没有异常则分数为0
            if anomaly_count == 0:
                self.risk_scores[user_id] = 0
                self.risk_factors[user_id] = dimension_scores
                continue

            # 获取用户敏感度
            user_sensitivity = self._get_user_sensitivity(user_id)

            # 计算时间集中度
            time_concentration = self._calculate_time_concentration(all_user_anomalies)

            # 计算行为多样性因子
            diversity_factor = self._calculate_diversity_factor(category_counts)

            # 应用机器学习增强的异常检测
            ml_factor = self._apply_ml_enhancement(user_id, all_user_anomalies)

            # 计算最终风险评分 - 将各维度组合起来
            weighted_dimension_sum = (
                    dimension_scores['time_sensitivity'] * 0.25 +
                    dimension_scores['data_sensitivity'] * 0.3 +
                    dimension_scores['behavior_deviation'] * 0.35 +
                    dimension_scores['quantity'] * 0.1
            )

            # 应用增强因子
            risk_score = base_score + (
                    weighted_dimension_sum *
                    user_sensitivity *
                    diversity_factor *
                    (1 + time_concentration) *
                    ml_factor
            )

            # 限制在0-100范围内
            risk_score = max(0, min(100, risk_score))

            # 保存风险评分
            self.risk_scores[user_id] = int(round(risk_score))

            # 保存风险因素构成，用于解释性分析
            self.risk_factors[user_id] = {
                'dimensions': dimension_scores,
                'user_sensitivity': user_sensitivity,
                'diversity_factor': diversity_factor,
                'time_concentration': time_concentration,
                'ml_enhancement': ml_factor,
                'category_breakdown': category_counts
            }

            # 保存历史记录
            timestamp = pd.Timestamp.now()
            self.risk_history[user_id].append((timestamp, risk_score))

            # 如果历史记录过长，保留最近的30条
            if len(self.risk_history[user_id]) > 30:
                self.risk_history[user_id] = self.risk_history[user_id][-30:]

        # 分析用户之间的关联性
        self._analyze_user_correlations()

        logger.info(f"风险评分计算完成，发现{sum(1 for s in self.risk_scores.values() if s > 50)}个高风险用户")
        return self

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

    def _calculate_data_sensitivity(self, anomaly, user_id):
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
        # 通过权限检查，这需要与你的访问控制系统集成
        if anomaly.get('permission_level', '') == 'unauthorized':
            sensitivity *= 2.0

        return sensitivity

    def _calculate_behavior_deviation(self, anomaly, user_id):
        """计算行为偏差程度"""
        # 获取用户的基准行为
        user_profile = self.user_profiles.get(user_id, {})

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

    def _apply_ml_enhancement(self, user_id, anomalies):
        """应用机器学习增强因子"""
        if not anomalies:
            return 1.0

        try:
            # 1. 使用隔离森林检测异常程度
            if self.ml_models['isolation_forest'] is None:
                # 延迟初始化模型
                self.ml_models['isolation_forest'] = IsolationForest(
                    n_estimators=100,
                    contamination=0.1,
                    random_state=42
                )

            # 提取特征
            features = []
            for anomaly in anomalies:
                # 创建特征向量 [score, time_sensitivity, data_sensitivity, ...等]
                feature_vector = [
                    anomaly.get('score', 5),
                    self._calculate_time_sensitivity(anomaly),
                    self._calculate_data_sensitivity(anomaly, user_id)
                ]
                features.append(feature_vector)

            # 如果有足够的样本，进行预测
            if len(features) >= 5:
                # 转换为numpy数组
                features_array = np.array(features)

                # 训练模型
                self.ml_models['isolation_forest'].fit(features_array)

                # 预测异常分数
                scores = -self.ml_models['isolation_forest'].score_samples(features_array)

                # 计算平均异常分数
                avg_anomaly_score = np.mean(scores)

                # 将分数映射到1.0-2.0的范围
                ml_factor = 1.0 + min(avg_anomaly_score, 1.0)
                return ml_factor

            # 2. 时间序列分析 - 这里简化实现
            # 如果有足够的历史记录，分析风险分数趋势
            history = self.risk_history.get(user_id, [])
            if len(history) >= 5:
                # 最近5次的风险评分
                recent_scores = [h[1] for h in history[-5:]]

                # 检查趋势 - 简单版本是检查是否连续上升
                is_increasing = all(recent_scores[i] <= recent_scores[i + 1] for i in range(len(recent_scores) - 1))

                if is_increasing:
                    return 1.5  # 连续上升的风险有额外加成

        except Exception as e:
            logger.error(f"机器学习增强失败: {str(e)}")

        # 默认返回1.0(不加不减)
        return 1.0

    def _analyze_user_correlations(self):
        """分析用户之间的关联性"""
        # 只有在有Email数据集的情况下才进行分析
        if 'email' not in self.datasets or self.datasets['email'].empty:
            logger.warning("没有电子邮件数据，跳过用户关联分析")
            return

        try:
            # 获取邮件数据
            email_data = self.datasets['email'].copy()

            # 确保有必要的列
            required_cols = ['from_user', 'to_users']
            if not all(col in email_data.columns for col in required_cols):
                logger.warning("邮件数据缺少必要的列，跳过用户关联分析")
                return

            # 创建用户交互图
            user_interactions = defaultdict(lambda: defaultdict(int))

            # 统计用户间的邮件交互
            for _, row in email_data.iterrows():
                sender = row['from_user']
                # 确保接收者是列表
                receivers = row['to_users']
                if isinstance(receivers, str):
                    # 尝试转换字符串为列表
                    try:
                        receivers = eval(receivers)
                    except:
                        receivers = [receivers]
                elif not isinstance(receivers, (list, tuple)):
                    receivers = [receivers]

                # 更新交互次数
                for receiver in receivers:
                    user_interactions[sender][receiver] += 1

            # 识别高风险用户之间的关联
            high_risk_users = {user_id for user_id, score in self.risk_scores.items() if score >= 70}

            # 检查高风险用户之间的交互
            for user1 in high_risk_users:
                for user2 in high_risk_users:
                    if user1 != user2:
                        # 检查交互频率
                        interaction_count = user_interactions[user1][user2] + user_interactions[user2][user1]

                        if interaction_count > 0:
                            # 记录关联
                            if user1 not in self.user_correlations:
                                self.user_correlations[user1] = {}

                            self.user_correlations[user1][user2] = {
                                'strength': interaction_count,
                                'combined_risk': self.risk_scores[user1] + self.risk_scores[user2]
                            }

            logger.info(
                f"完成用户关联分析，发现{sum(len(correlations) for correlations in self.user_correlations.values())}个关联")

        except Exception as e:
            logger.error(f"用户关联分析失败: {str(e)}")

    def get_risk_trend(self, user_id=None, days=30):
        """获取用户风险评分趋势"""
        if user_id:
            history = self.risk_history.get(user_id, [])
            # 过滤指定天数内的记录
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            recent_history = [(ts, score) for ts, score in history if ts >= cutoff_date]
            return recent_history
        else:
            # 返回所有用户的趋势
            trends = {}
            for uid, history in self.risk_history.items():
                cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                trends[uid] = [(ts, score) for ts, score in history if ts >= cutoff_date]
            return trends

    def get_user_correlations(self, user_id=None, min_strength=1):
        """获取用户关联信息"""
        if user_id:
            correlations = self.user_correlations.get(user_id, {})
            # 过滤最低强度
            return {uid: data for uid, data in correlations.items()
                    if data['strength'] >= min_strength}
        else:
            # 返回所有关联
            filtered_correlations = {}
            for uid, correlations in self.user_correlations.items():
                filtered_correlations[uid] = {
                    other_uid: data for other_uid, data in correlations.items()
                    if data['strength'] >= min_strength
                }
            return filtered_correlations
