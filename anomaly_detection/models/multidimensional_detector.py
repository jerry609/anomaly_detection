import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import warnings

from anomaly_detection.analyzers.ml_analyzer import MLAnalyzer
from anomaly_detection.graph_analysis.graph_builder import UserGraphBuilder
from anomaly_detection.graph_analysis.graph_metrics import GraphMetricsCalculator
from anomaly_detection.models.base_detector import BaseAnomalyDetector
from anomaly_detection.analyzers.time_analyzer import TimeAnalyzer
from anomaly_detection.analyzers.access_analyzer import AccessAnalyzer
from anomaly_detection.analyzers.email_analyzer import EmailAnalyzer
from anomaly_detection.analyzers.org_analyzer import OrgAnalyzer
import config
from anomaly_detection.risk_scoring.risk_calculator import RiskCalculator
from anomaly_detection.time_series.time_series_analyzer import TimeSeriesAnalyzer
from anomaly_detection.utils.text_processor import extract_email_parts

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


class MultiDimensionalAnomalyDetector(BaseAnomalyDetector):
    """多维异常检测器 - 重构版本"""

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
        self.risk_scores = {}  # 主要风险评分
        self.risk_dimensions = {}  # 多维风险评分
        self.risk_factors = {}  # 风险因素分析
        self.risk_history = defaultdict(list)  # 风险趋势历史
        self.dynamic_thresholds = {}  # 用户动态阈值
        self.user_behavior_models = {}  # 用户行为模型
        self.user_correlations = {}  # 用户关联数据
        self.detection_date = datetime.now()  # 当前检测日期

        # 初始化分析器
        self.time_analyzer = TimeAnalyzer()
        self.access_analyzer = AccessAnalyzer()
        self.email_analyzer = EmailAnalyzer()
        self.org_analyzer = OrgAnalyzer()

        # 初始化机器学习分析器
        self.ml_analyzer = MLAnalyzer()

        # 初始化图分析组件
        self.graph_builder = UserGraphBuilder()
        self.graph_metrics = GraphMetricsCalculator()

        # 初始化时间序列分析器
        self.time_series_analyzer = TimeSeriesAnalyzer()

        # 初始化风险计算器
        self.risk_calculator = RiskCalculator()

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

        # 设置各分析器所需的数据
        for analyzer in [self.time_analyzer, self.access_analyzer, self.email_analyzer, self.org_analyzer]:
            if hasattr(analyzer, 'datasets'):
                analyzer.datasets = processed_datasets
            if hasattr(analyzer, 'user_profiles'):
                analyzer.user_profiles = user_profiles

        logger.info(f"Loaded preprocessed data for {len(self.user_profiles)} users")
        return self

    def detect_all_anomalies(self):
        """运行所有异常检测功能"""
        logger.info("Starting anomaly detection process")

        # 执行基础异常检测
        self.detect_time_anomalies()
        self.detect_access_anomalies()
        self.detect_email_anomalies()
        self.detect_org_anomalies()

        # 执行高级异常检测
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

        # 使用时间分析器检测异常
        time_anomalies = self.time_analyzer.detect_anomalies(new_logon_data=logon_data)

        # 合并到主异常字典
        for user_id, anomalies in time_anomalies.items():
            for anomaly in anomalies:
                # 确保使用_record_anomaly方法记录异常
                if isinstance(anomaly, dict):
                    # 提取关键信息
                    score = anomaly.get('score', 30.0)  # 使用默认值30.0

                    # 确保score是数值
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        score = 30.0

                    # 记录异常，同时生成详细描述
                    self._record_anomaly(user_id, 'time', score, anomaly)
                else:
                    # 如果不是字典，创建一个简单的异常记录
                    self._record_anomaly(user_id, 'time', 30.0, {'description': str(anomaly)})

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

        # 使用访问分析器检测异常
        access_anomalies = self.access_analyzer.detect_anomalies(new_access_data=combined_access_data)

        # 合并到主异常字典
        for user_id, anomalies in access_anomalies.items():
            for anomaly in anomalies:
                # 确保使用_record_anomaly方法记录异常
                if isinstance(anomaly, dict):
                    # 提取关键信息
                    score = anomaly.get('score', 40.0)  # 访问类型异常的默认分数更高

                    # 确保score是数值
                    try:
                        score = float(score)
                    except (ValueError, TypeError):
                        score = 40.0

                    # 记录异常，同时生成详细描述
                    self._record_anomaly(user_id, 'access', score, anomaly)
                else:
                    # 如果不是字典，创建一个简单的异常记录
                    self._record_anomaly(user_id, 'access', 40.0, {'description': str(anomaly)})

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

        # 使用邮件分析器检测异常
        email_anomalies = self.email_analyzer.detect_anomalies(new_email_data=email_data)

        # 合并到主异常字典
        for user_id, anomalies in email_anomalies.items():
            self.anomalies['email'][user_id].extend(anomalies)

        logger.info(f"Detected email anomalies for {len(email_anomalies)} users")
        return self

    def detect_org_anomalies(self):
        """检测组织层面的异常"""
        logger.info("Detecting organizational anomalies")

        # 准备邮件数据
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

        # 处理LDAP数据
        ldap_data = self._prepare_data_for_analyzer('ldap')
        if ldap_data is not None and hasattr(self.org_analyzer, 'set_org_structure'):
            # 预处理LDAP数据
            prepared_ldap_data = self._prepare_ldap_data(ldap_data)

            # 设置组织结构
            try:
                self.org_analyzer.set_org_structure(prepared_ldap_data)
                logger.info("Successfully configured organization structure from LDAP data")
            except Exception as e:
                logger.error(f"Error configuring organization structure: {str(e)}")

        # 使用组织分析器检测异常
        try:
            anomaly_df = self.org_analyzer.detect_anomalies(new_interaction_data=email_data)
            org_anomalies = self._process_org_anomalies(anomaly_df)

            # 合并到主异常字典
            for user_id, anomalies in org_anomalies.items():
                self.anomalies['organizational'][user_id].extend(anomalies)

            logger.info(f"Detected organizational anomalies for {len(org_anomalies)} users")
        except Exception as e:
            logger.error(f"Failed to detect organizational anomalies: {str(e)}")
            logger.debug(f"Error details: {e}", exc_info=True)

        return self

    def _prepare_ldap_data(self, ldap_data):
        """准备LDAP数据用于组织分析"""
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

        return ldap_data

    def _process_org_anomalies(self, anomaly_df):
        """将组织异常DataFrame转换为用户异常字典格式"""
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

        return org_anomalies

    def detect_ml_anomalies(self):
        """使用机器学习模型检测异常模式"""
        logger.info("Detecting anomalies using machine learning models")

        # 使用机器学习分析器检测异常
        ml_anomalies = self.ml_analyzer.detect_anomalies(self.datasets, self.user_profiles)

        # 合并到主异常字典 - ML异常归类为访问异常
        for user_id, anomalies in ml_anomalies.items():
            self.anomalies['access'][user_id].extend(anomalies)

        logger.info(f"ML models detected anomalies for {len(ml_anomalies)} users")
        return self

    def detect_time_series_anomalies(self):
        """使用时间序列分析检测逐渐变化的行为模式"""
        logger.info("Detecting gradual behavior changes using time series analysis")

        # 使用时间序列分析器检测异常
        time_series_anomalies = self.time_series_analyzer.detect_anomalies(
            self.datasets,
            self.user_profiles,
            self.risk_history
        )

        # 合并到主异常字典
        for user_id, anomalies in time_series_anomalies.items():
            self.anomalies['time'][user_id].extend(anomalies)

        logger.info(f"Time series analysis detected behavior drifts for {len(time_series_anomalies)} users")
        return self

    def detect_graph_anomalies(self):
        """使用图分析检测复杂的关系异常"""
        logger.info("Detecting relationship anomalies using graph analysis")

        # 确保有邮件数据
        email_data = self._prepare_data_for_analyzer('email')
        if email_data is None or email_data.empty:
            logger.warning("No email data available for graph analysis")
            return self

        try:
            # 构建用户关系图
            user_graph = self.graph_builder.build_graph(email_data, self.user_profiles)

            # 计算图分析指标
            self.graph_metrics.set_graph(user_graph)
            self.graph_metrics.calculate_metrics()

            # 检测异常节点
            anomalies = self.graph_metrics.detect_anomalies()

            # 创建异常对象并添加到异常字典
            for user_id, score in anomalies.items():
                # 获取图节点的详细信息
                node = user_graph.nodes[user_id]

                anomaly = {
                    'type': 'graph',
                    'score': float(score),
                    'timestamp': pd.Timestamp.now(),
                    'description': "Abnormal network communication patterns detected",
                    'details': {
                        'degree_centrality': float(node.get('degree_centrality', 0)),
                        'betweenness_centrality': float(node.get('betweenness_centrality', 0)),
                        'closeness_centrality': float(node.get('closeness_centrality', 0)),
                        'eigenvector_centrality': float(node.get('eigenvector_centrality', 0))
                    }
                }

                self.anomalies['organizational'][user_id].append(anomaly)

            logger.info(f"Graph analysis detected relationship anomalies for {len(anomalies)} users")

        except Exception as e:
            logger.error(f"Error in graph anomaly detection: {str(e)}")
            logger.debug(f"Error details: {e}", exc_info=True)

        return self

    # ... existing code ...

    def calculate_risk_scores(self):
        """计算综合风险评分"""
        logger.info("Calculating comprehensive risk scores")

        # 使用风险计算器计算风险分数
        risk_results = self.risk_calculator.calculate_risk_scores(
            self.anomalies,
            self.user_profiles,
            config.get_risk_config()  # 假设有一个方法获取风险配置
        )

        # 更新风险评分和维度
        self.risk_scores = risk_results['scores']
        self.risk_dimensions = risk_results['dimensions']
        self.risk_factors = risk_results['factors']

        # 验证风险分数是否有效，如果所有分数为0但存在异常，则应用备用计算方法
        if all(score == 0 for score in self.risk_scores.values()) and self._has_anomalies():
            logger.warning("风险计算器返回全0分数，但存在异常。应用备用风险计算方法。")
            self._calculate_fallback_risk_scores()

        # 分析用户之间的关联性
        self._analyze_user_correlations()

        # 记录风险评分结果
        high_risk_users = sum(1 for s in self.risk_scores.values() if s > 70)
        medium_risk_users = sum(1 for s in self.risk_scores.values() if 40 <= s <= 70)
        logger.info(
            f"Risk score calculation complete. Found {high_risk_users} high-risk users and {medium_risk_users} medium-risk users")

        # 记录风险分数统计
        if self.risk_scores:
            max_score = max(self.risk_scores.values())
            min_score = min(self.risk_scores.values())
            avg_score = sum(self.risk_scores.values()) / len(self.risk_scores)
            logger.info(f"风险分数范围: 最低 {min_score}, 最高 {max_score}, 平均 {avg_score:.2f}")

        return self

    def _has_anomalies(self):
        """检查是否存在任何异常"""
        for category in self.anomalies:
            for user_anomalies in self.anomalies[category].values():
                if user_anomalies:
                    return True
        return False

    def _calculate_fallback_risk_scores(self):
        """备用风险分数计算方法"""
        logger.info("应用备用风险评分计算")

        # 基础配置
        base_score = 30  # 基础分数
        anomaly_value = 10  # 每个异常的基础加分值

        # 收集每个用户的所有异常
        user_anomalies = defaultdict(list)

        for category in self.anomalies:
            for user_id, anomalies in self.anomalies[category].items():
                user_anomalies[user_id].extend(anomalies)

        # 为每个用户计算风险分数
        for user_id, anomalies in user_anomalies.items():
            # 计算异常总数
            anomaly_count = len(anomalies)

            if anomaly_count == 0:
                continue

            # 获取异常严重度平均值（如果可用）
            severity_sum = 0
            severity_count = 0

            for anomaly in anomalies:
                if 'score' in anomaly and anomaly['score']:
                    try:
                        severity = float(anomaly['score'])
                        severity_sum += severity
                        severity_count += 1
                    except (ValueError, TypeError):
                        pass

            # 计算严重度平均值
            avg_severity = severity_sum / max(1, severity_count)

            # 计算风险分数: 基础分数 + 异常数量 * 基础加分 + 平均严重度调整
            risk_score = base_score + (anomaly_count * anomaly_value) + (avg_severity / 2)

            # 限制在100以内
            risk_score = min(100, risk_score)

            # 更新风险分数
            self.risk_scores[user_id] = risk_score

            # 更新风险维度（简化版）
            self.risk_dimensions[user_id] = {
                'anomaly_count': anomaly_count,
                'average_severity': avg_severity,
                'computed_score': risk_score
            }

            # 更新风险因素
            self.risk_factors[user_id] = {
                'anomaly_count': anomaly_count,
                'average_severity': avg_severity,
                'categories': {
                    category: len(self.anomalies[category].get(user_id, []))
                    for category in self.anomalies
                }
            }

            logger.debug(
                f"备用计算: 用户 {user_id} 风险分数 = {risk_score} (异常数: {anomaly_count}, 平均严重度: {avg_severity:.2f})")

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
        return self

    def calculate_dynamic_thresholds(self):
        """根据历史数据计算动态阈值"""
        # 这里可以实现动态阈值的计算逻辑
        # 例如，基于用户历史风险分数的统计特性
        return self

    def _analyze_user_correlations(self):
        """分析用户之间的关联性"""
        # 只有在有Email数据集的情况下才进行分析
        email_data = self._prepare_data_for_analyzer('email')
        if email_data is None or email_data.empty:
            logger.warning("No email data available for user correlation analysis")
            return

        try:
            # 确保有必要的列
            required_cols = ['from_user', 'to_users']
            if not all(col in email_data.columns for col in required_cols):
                # 尝试用to列创建to_users列
                if 'to' in email_data.columns and 'to_users' not in email_data.columns:
                    email_data['to_users'] = email_data['to'].apply(extract_email_parts)

            if not all(col in email_data.columns for col in required_cols):
                logger.warning("Email data lacks required columns for correlation analysis")
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
            high_risk_threshold = getattr(config, 'HIGH_RISK_THRESHOLD', 70)
            high_risk_users = {user_id for user_id, score in self.risk_scores.items() if score >= high_risk_threshold}

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
                f"Completed user correlation analysis, found {sum(len(correlations) for correlations in self.user_correlations.values())} correlations")

        except Exception as e:
            logger.error(f"User correlation analysis failed: {str(e)}")

    # 辅助获取方法

    def get_high_risk_users(self, threshold=None):
        """获取高风险用户列表 - 分层版本"""
        high_threshold = threshold or getattr(config, 'HIGH_RISK_THRESHOLD', 70)
        medium_threshold = getattr(config, 'MEDIUM_RISK_THRESHOLD', 50)
        low_threshold = getattr(config, 'LOW_RISK_THRESHOLD', 30)

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

    def get_risk_trend(self, user_id=None, days=30):
        """获取用户风险评分趋势"""
        if user_id:
            history = self.risk_history.get(user_id, [])
            # 过滤指定天数内的记录
            cutoff_date = datetime.now().date() - timedelta(days=days)
            recent_history = [h for h in history if h['date'] >= cutoff_date]
            return recent_history
        else:
            # 返回所有用户的趋势
            trends = {}
            cutoff_date = datetime.now().date() - timedelta(days=days)
            for uid, history in self.risk_history.items():
                trends[uid] = [h for h in history if h['date'] >= cutoff_date]
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


    def _generate_anomaly_description(self, anomaly_type, user_id, details):
        """
        为异常生成详细描述

        Args:
            anomaly_type: 异常类型
            user_id: 用户ID
            details: 异常详情

        Returns:
            str: 详细异常描述
        """
        # 确保details是字典
        if details is None:
            details = {}

        # 获取时间戳
        timestamp = details.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                # 如果解析失败，使用当前时间
                timestamp = datetime.now()

        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M") if isinstance(timestamp, datetime) else str(timestamp)

        # 基础描述前缀
        description = f"用户 {user_id} 在 {timestamp_str} "

        # 根据异常类型生成具体描述
        if anomaly_type == 'time':
            activity = details.get('activity', '登录系统')
            hour = timestamp.hour if isinstance(timestamp, datetime) else 'N/A'

            # 根据时间段生成描述
            if 0 <= hour < 6:
                time_desc = "凌晨时段"
            elif 6 <= hour < 9:
                time_desc = "早晨非工作时段"
            elif 9 <= hour < 18:
                time_desc = "工作时段"
            elif 18 <= hour < 22:
                time_desc = "晚间时段"
            else:
                time_desc = "深夜时段"

            description += f"在{time_desc}（{hour}时）进行了{activity}操作，这与其正常活动时间模式不符。"

        elif anomaly_type == 'access':
            resource = details.get('resource', '未知资源')
            resource_id = details.get('resource_id', resource)
            action = details.get('action', '访问')

            description += f"{action}了资源 '{resource_id}'，这种访问模式与其历史行为不符。"

        elif anomaly_type == 'email':
            recipients = details.get('recipients', '多位收件人')
            has_attachments = details.get('has_attachments', False)
            subject = details.get('subject', '')
            sensitivity = details.get('sensitivity', 'normal')

            # 构建邮件描述
            email_desc = f"发送了邮件"
            if subject:
                email_desc += f"（主题：{subject}）"
            email_desc += f"给{recipients}"

            if has_attachments:
                email_desc += "，并包含附件"

            if sensitivity != 'normal':
                email_desc += f"，邮件敏感度: {sensitivity}"

            description += f"{email_desc}，该邮件行为与其正常通信模式不符。"

        elif anomaly_type == 'organizational':
            department = details.get('department', '其他部门')
            communication_type = details.get('communication_type', '通信')

            description += f"与{department}进行了异常的{communication_type}，这种跨部门行为不符合其组织角色。"

        else:
            # 默认描述
            description += f"表现出异常的{anomaly_type}类型行为，不符合历史行为模式。"

        return description

    def _record_anomaly(self, user_id, anomaly_type, score, details=None):
        """
        记录检测到的异常

        Args:
            user_id (str): 用户ID
            anomaly_type (str): 异常类型
            score (float): 异常分数
            details (dict): 异常详情

        Returns:
            dict: 创建的异常对象
        """
        if details is None:
            details = {}

        # 确保score是数值类型并且大于零
        try:
            score = float(score)
            if score <= 0:
                score = 30.0  # 设置默认非零分数
        except (ValueError, TypeError):
            score = 30.0  # 转换失败时设置默认分数

        # 生成详细描述
        detailed_reason = self._generate_anomaly_description(anomaly_type, user_id, details)

        # 创建异常记录对象
        anomaly = {
            'user_id': user_id,
            'type': anomaly_type,
            'score': score,  # 确保是数值类型
            'severity': score,  # 添加severity字段作为备用
            'timestamp': details.get('timestamp', datetime.now().isoformat()),
            'description': detailed_reason,
            'reason': detailed_reason,  # 兼容现有代码
            'details': details,
            'is_anomaly': True
        }

        # 记录调试信息
        logger.debug(f"记录异常: 用户={user_id}, 类型={anomaly_type}, 分数={score}, 描述={detailed_reason[:50]}...")

        # 添加到相应的异常集合
        self.anomalies[anomaly_type][user_id].append(anomaly)

        return anomaly


