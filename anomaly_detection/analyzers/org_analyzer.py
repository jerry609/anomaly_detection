import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import config
from collections import defaultdict

logger = logging.getLogger(__name__)


class OrgAnalyzer:
    def __init__(self):
        self.datasets = {}
        self.user_profiles = {}
        self.org_structure = None
        self.dept_mapping = {}
        self.role_mapping = {}
        self.hierarchy_levels = {}
        self.comm_network = defaultdict(lambda: defaultdict(int))  # 通信网络图

    def set_org_structure(self, ldap_data):
        """设置组织结构数据"""
        self.org_structure = ldap_data.copy()

        # 创建部门映射
        self.dept_mapping = dict(zip(ldap_data['user_id'], ldap_data['department']))

        # 创建角色映射
        self.role_mapping = dict(zip(ldap_data['user_id'], ldap_data['role']))

        # 创建层级映射 (基于管理关系)
        self._build_hierarchy_levels()

        logger.info(f"Organization structure set with {len(ldap_data)} users")

    def _build_hierarchy_levels(self):
        """构建组织层级关系"""
        if self.org_structure is None:
            logger.warning("Cannot build hierarchy levels: No org structure data")
            return

        # 找出最高管理层 (没有上级的人)
        top_managers = self.org_structure[self.org_structure['supervisor'].isnull()]['user_id'].tolist()

        # 初始化层级字典
        self.hierarchy_levels = {}

        # 为最高管理层设置级别1
        for manager in top_managers:
            self.hierarchy_levels[manager] = 1

        # 递归构建下级层级
        changed = True
        max_iterations = 20  # 防止无限循环
        iterations = 0

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            for _, row in self.org_structure.iterrows():
                user_id = row['user_id']
                supervisor = row['supervisor']

                if pd.notna(
                        supervisor) and supervisor in self.hierarchy_levels and user_id not in self.hierarchy_levels:
                    self.hierarchy_levels[user_id] = self.hierarchy_levels[supervisor] + 1
                    changed = True

        logger.info(f"Built hierarchy levels for {len(self.hierarchy_levels)} users in {iterations} iterations")

    def detect_anomalies(self, new_interaction_data):
        """检测组织异常 (主方法) - 增强版"""
        if self.org_structure is None:
            logger.warning("Cannot detect org anomalies: No org structure data")
            return pd.DataFrame()  # 返回空DataFrame

        # 确保数据包含所需列
        required_cols = ['from_user', 'to_users', 'subject', 'date']
        if not all(col in new_interaction_data.columns for col in required_cols):
            logger.warning(f"Missing required columns for org anomaly detection: {required_cols}")
            return pd.DataFrame()

        # 更新通信网络图
        self._update_communication_network(new_interaction_data)

        # 创建时间窗口
        time_windows = self._create_time_windows(new_interaction_data,
                                                 days=config.ORG_TIME_WINDOW_DAYS)

        # 对每个窗口进行检测
        all_anomalies = []
        for window_data in time_windows:
            # 1. 检测跨部门异常通信
            dept_anomalies = self._detect_cross_department_anomalies(window_data)

            # 2. 检测层级越级通信
            hierarchy_anomalies = self._detect_hierarchy_violations(window_data)

            # 3. 检测角色不匹配通信
            role_anomalies = self._detect_role_mismatches(window_data)

            # 4. 检测异常通信时间模式 (新增)
            timing_anomalies = self._detect_unusual_timing_patterns(window_data)

            # 5. 检测异常通信频率和网络模式 (新增)
            network_anomalies = self._detect_network_anomalies(window_data)

            # 合并当前窗口的所有异常
            window_anomalies = pd.concat(
                [df for df in [dept_anomalies, hierarchy_anomalies, role_anomalies,
                               timing_anomalies, network_anomalies] if not df.empty],
                ignore_index=True
            )

            all_anomalies.append(window_anomalies)

        # 合并所有窗口的异常
        if not all_anomalies:
            return pd.DataFrame()

        final_anomalies = pd.concat(all_anomalies, ignore_index=True)

        # 为每个异常添加唯一ID和时间戳
        if not final_anomalies.empty:
            final_anomalies['event_id'] = [f"org_{i}" for i in range(len(final_anomalies))]
            final_anomalies['timestamp'] = datetime.now()

            # 进一步分析异常频率和应用威胁情报
            final_anomalies = self._analyze_anomaly_frequency(final_anomalies)
            final_anomalies = self._add_threat_intelligence_score(final_anomalies)

        logger.info(f"Detected {len(final_anomalies)} organizational anomalies")
        return final_anomalies

    def _update_communication_network(self, data):
        """更新用户通信网络图"""
        for _, row in data.iterrows():
            sender = row['from_user']
            receivers = row['to_users']

            for receiver in receivers:
                # 增加发送者到接收者的通信计数
                self.comm_network[sender][receiver] += 1

    def _create_time_windows(self, data, days=7):
        """将数据分割为时间窗口"""
        if 'date' not in data.columns:
            return [data]  # 如果没有日期列，返回整个数据集

        # 转换日期列为datetime
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])

        # 按时间排序
        data = data.sort_values('date')

        # 获取最早和最晚日期
        min_date = data['date'].min()
        max_date = data['date'].max()

        # 创建时间窗口
        windows = []
        current_start = min_date

        while current_start <= max_date:
            current_end = current_start + timedelta(days=days)
            window_data = data[(data['date'] >= current_start) & (data['date'] < current_end)]

            if not window_data.empty:
                windows.append(window_data)

            current_start = current_end

        return windows

    def _detect_cross_department_anomalies(self, data):
        """检测跨部门异常通信"""
        anomalies = []

        # 遍历每封邮件
        for _, row in data.iterrows():
            sender = row['from_user']
            receivers = row['to_users']

            # 跳过没有发件人部门信息的行
            if sender not in self.dept_mapping:
                continue

            sender_dept = self.dept_mapping[sender]

            # 计算接收者中跨部门的比例
            cross_dept_count = 0
            total_known_receivers = 0

            for receiver in receivers:
                if receiver in self.dept_mapping:
                    total_known_receivers += 1
                    if self.dept_mapping[receiver] != sender_dept:
                        cross_dept_count += 1

            if total_known_receivers > 0:
                cross_dept_ratio = cross_dept_count / total_known_receivers

                # 如果跨部门比例超过阈值，标记为异常
                if cross_dept_ratio > config.ORG_DEPT_BOUNDARY_THRESHOLD:
                    anomalies.append({
                        'from_user': sender,
                        'to_users': receivers,
                        'anomaly_score': cross_dept_ratio * 100,
                        'is_anomaly': True,
                        'reason': f"异常的跨部门通信模式 ({cross_dept_ratio:.2f})",
                        'department': sender_dept,
                        'date': row['date']
                    })

        return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

    def _detect_hierarchy_violations(self, data):
        """检测层级越级通信"""
        anomalies = []

        # 遍历每封邮件
        for _, row in data.iterrows():
            sender = row['from_user']
            receivers = row['to_users']

            # 跳过没有发件人层级信息的行
            if sender not in self.hierarchy_levels:
                continue

            sender_level = self.hierarchy_levels[sender]

            # 检测层级跨越
            for receiver in receivers:
                if receiver in self.hierarchy_levels:
                    receiver_level = self.hierarchy_levels[receiver]
                    level_diff = abs(sender_level - receiver_level)

                    # 如果层级差异过大(跨2层以上)，并且是下级给高层发邮件
                    if level_diff > 2 and sender_level > receiver_level:
                        # 计算越级严重程度分数
                        severity_score = (level_diff - 2) * 30  # 每多跨1层加30分

                        anomalies.append({
                            'from_user': sender,
                            'to_users': [receiver],  # 只包含异常接收者
                            'anomaly_score': min(100, severity_score),
                            'is_anomaly': True,
                            'reason': f"组织层级越级通信 (跨越{level_diff}层)",
                            'department': self.dept_mapping.get(sender, 'Unknown'),
                            'date': row['date']
                        })

        return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

    def _detect_role_mismatches(self, data):
        """检测角色不匹配通信"""
        anomalies = []

        # 根据主题词分析邮件内容类型
        sensitive_patterns = {
            'finance': ['budget', 'financial', 'revenue', 'expense', 'forecast', 'profit'],
            'hr': ['hiring', 'interview', 'resume', 'performance', 'salary', 'candidate'],
            'it': ['server', 'infrastructure', 'code', 'software', 'hardware', 'password'],
            'legal': ['contract', 'agreement', 'compliance', 'regulation', 'lawsuit', 'legal']
        }

        for _, row in data.iterrows():
            sender = row['from_user']
            receivers = row['to_users']
            subject = str(row['subject']).lower() if 'subject' in row else ''

            # 跳过没有发件人角色信息的行
            if sender not in self.role_mapping:
                continue

            sender_role = self.role_mapping[sender].lower() if self.role_mapping[sender] else 'unknown'

            # 检测邮件主题是否包含与发件人角色不符的敏感词
            for category, keywords in sensitive_patterns.items():
                # 检查主题是否包含该类别的敏感词
                contains_sensitive = any(keyword in subject for keyword in keywords)

                if contains_sensitive:
                    # 检查发件人角色是否与敏感类别不匹配
                    # 例如：非财务人员发送财务相关邮件
                    role_mismatch = False

                    if category == 'finance' and 'financ' not in sender_role and 'account' not in sender_role:
                        role_mismatch = True
                    elif category == 'hr' and 'hr' not in sender_role and 'human' not in sender_role:
                        role_mismatch = True
                    elif category == 'it' and 'it' not in sender_role and 'tech' not in sender_role and 'develop' not in sender_role:
                        role_mismatch = True
                    elif category == 'legal' and 'legal' not in sender_role and 'law' not in sender_role:
                        role_mismatch = True

                    if role_mismatch:
                        anomalies.append({
                            'from_user': sender,
                            'to_users': receivers,
                            'anomaly_score': 75,  # 固定分数
                            'is_anomaly': True,
                            'reason': f"角色不匹配的敏感内容通信 ({sender_role} 发送 {category} 内容)",
                            'department': self.dept_mapping.get(sender, 'Unknown'),
                            'date': row['date']
                        })
                        break  # 找到一个不匹配就足够了

        return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

    def _detect_unusual_timing_patterns(self, data):
        """检测异常的通信时间模式"""
        anomalies = []

        # 转换时间为小时
        data = data.copy()
        if 'date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['date']):
            data['hour'] = data['date'].dt.hour
            data['weekday'] = data['date'].dt.weekday

            # 定义工作时间和非工作时间
            work_hours = range(config.BUSINESS_HOURS_START, config.BUSINESS_HOURS_END)
            weekend_days = config.WEEKEND_DAYS  # 假设配置中定义了[5,6]表示周六日

            for _, row in data.iterrows():
                is_weekend = row['weekday'] in weekend_days
                is_after_hours = row['hour'] < config.BUSINESS_HOURS_START or row['hour'] >= config.BUSINESS_HOURS_END

                if (is_weekend or is_after_hours):
                    sender = row['from_user']

                    # 检查这是否是一个敏感话题的邮件
                    subject = str(row.get('subject', '')).lower()
                    is_sensitive = any(word in subject for word in [
                        'confidential', 'private', 'secret', 'restricted',
                        'internal', 'sensitive', 'urgent', 'important'
                    ])

                    if is_sensitive:
                        time_desc = "周末" if is_weekend else "非工作时间"

                        anomalies.append({
                            'from_user': sender,
                            'to_users': row['to_users'],
                            'anomaly_score': 85 if is_sensitive else 65,
                            'is_anomaly': True,
                            'reason': f"{time_desc}发送敏感邮件",
                            'department': self.dept_mapping.get(sender, 'Unknown'),
                            'date': row['date']
                        })

        return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

    def _detect_network_anomalies(self, data):
        """检测通信网络异常模式"""
        anomalies = []

        # 跳过如果通信网络为空
        if not self.comm_network:
            return pd.DataFrame()

        # 计算每个发件人的通常通信对象
        for sender, receivers_dict in self.comm_network.items():
            # 按通信频率排序接收者
            sorted_receivers = sorted(receivers_dict.items(), key=lambda x: x[1], reverse=True)

            # 获取前N个最常见接收者作为正常通信对象
            top_n = min(5, len(sorted_receivers))
            common_receivers = {r[0] for r in sorted_receivers[:top_n]}

            # 在当前数据中查找异常通信模式
            sender_data = data[data['from_user'] == sender]

            for _, row in sender_data.iterrows():
                receivers = row['to_users']

                # 计算接收者中不常见的比例
                uncommon_count = sum(1 for r in receivers if r not in common_receivers)

                if receivers and len(receivers) > 0:
                    uncommon_ratio = uncommon_count / len(receivers)

                    # 如果大部分接收者都是不常见的，标记为异常
                    if uncommon_ratio > 0.8 and len(receivers) >= 3:
                        anomalies.append({
                            'from_user': sender,
                            'to_users': receivers,
                            'anomaly_score': min(100, 60 + uncommon_ratio * 30),
                            'is_anomaly': True,
                            'reason': f"通信模式异常 (80%以上接收者为不常见对象)",
                            'department': self.dept_mapping.get(sender, 'Unknown'),
                            'date': row['date']
                        })

        return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

    def _analyze_anomaly_frequency(self, anomalies_df):
        """分析异常的频率模式，识别持续性异常"""
        # 为频繁出现的相同类型异常提高分数
        users = anomalies_df['from_user'].unique()

        for user in users:
            user_anomalies = anomalies_df[anomalies_df['from_user'] == user]

            # 分析用户异常类型的分布
            reason_counts = user_anomalies['reason'].value_counts()

            # 对于重复出现的同类型异常，提高其分数
            for reason, count in reason_counts.items():
                if count > 1:  # 如果同一类型异常出现多次
                    # 计算加权因子 (出现次数越多，加权越高)
                    boost_factor = min(1.5, 1.0 + (count - 1) * 0.1)

                    # 找出这类异常的所有行
                    mask = (anomalies_df['from_user'] == user) & (anomalies_df['reason'] == reason)

                    # 提高分数但不超过100
                    anomalies_df.loc[mask, 'anomaly_score'] = anomalies_df.loc[mask, 'anomaly_score'].apply(
                        lambda x: min(100, x * boost_factor))

                    # 在原因中添加频率信息
                    anomalies_df.loc[mask, 'reason'] = f"{reason} (重复出现{count}次)"

        return anomalies_df

    def _add_threat_intelligence_score(self, anomalies_df):
        """添加基于威胁情报的评分增强"""
        # 这里可以集成外部威胁情报API
        # 简化版：根据部门敏感度调整分数

        # 部门敏感度评分 (示例)
        dept_sensitivity = {
            'Finance': 2.0,
            'HR': 1.8,
            'Legal': 1.8,
            'IT': 1.5,
            'Executive': 2.0,
            'R&D': 1.7,
            'Sales': 1.2,
            'Marketing': 1.2
        }

        # 应用部门敏感度加权
        for idx, row in anomalies_df.iterrows():
            dept = row['department']
            sensitivity = dept_sensitivity.get(dept, 1.0)

            # 根据部门敏感度调整分数
            current_score = row['anomaly_score']
            new_score = min(100, current_score * sensitivity)

            if new_score > current_score:
                anomalies_df.at[idx, 'anomaly_score'] = new_score
                anomalies_df.at[idx, 'reason'] += f" (敏感部门:{dept})"

        return anomalies_df
