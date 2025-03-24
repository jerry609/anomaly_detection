import numpy as np
import pandas as pd
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class RiskFactorAnalyzer:
    """
    风险因素分析器 - 分析用户风险的构成因素，识别主要风险来源，
    并提供风险解释和缓解建议
    """

    def __init__(self):
        """初始化风险因素分析器"""
        self.risk_categories = {
            'time': '时间异常',
            'access': '访问异常',
            'email': '邮件异常',
            'organizational': '组织异常',
            'ml_pattern': '机器学习检测模式',
            'graph': '关系网络异常',
            'time_series': '行为趋势异常'
        }

        self.risk_subtypes = {
            'after_hours_access': '非工作时间访问',
            'unusual_file_access': '异常文件访问',
            'sensitive_data_access': '敏感数据访问',
            'unusual_login_location': '异常登录位置',
            'login_time_drift': '登录时间漂移',
            'access_pattern_drift': '访问模式漂移',
            'communication_drift': '通信模式漂移',
            'data_exfiltration': '数据外泄',
            'cross_department_email': '跨部门邮件',
            'external_communication': '外部通信'
        }

        self.mitigation_strategies = {
            'time': [
                '审查用户的工作时间安排',
                '确认是否有加班或特殊工作需求',
                '实施更严格的非工作时间访问控制',
                '要求非常规时间登录进行双因素认证'
            ],
            'access': [
                '审查用户的访问权限',
                '实施最小权限原则',
                '增加敏感资源的访问审批流程',
                '对敏感操作实施多因素认证'
            ],
            'email': [
                '审查用户的邮件通信模式',
                '实施数据泄露防护(DLP)解决方案',
                '对外发邮件增加内容扫描',
                '对含有敏感信息的邮件实施加密'
            ],
            'organizational': [
                '审查用户的组织关系和职责',
                '确认跨部门沟通的业务需求',
                '定期审核用户权限和职责变更',
                '增强部门间数据共享的审计跟踪'
            ],
            'ml_pattern': [
                '深入分析用户的行为模式',
                '与用户确认是否有行为变化',
                '临时增加对用户的监控级别',
                '考虑实施更细粒度的访问控制'
            ],
            'graph': [
                '分析用户的社交网络关系',
                '确认跨团队协作的合理性',
                '审核关联用户的风险状况',
                '评估是否存在共谋风险'
            ],
            'time_series': [
                '跟踪用户行为随时间的变化',
                '确认行为变化是否与职责变更相符',
                '评估长期行为趋势的安全影响',
                '调整基线行为模型以适应合理变化'
            ]
        }

    def analyze_risk_factors(self, user_id, anomalies, risk_score, risk_factors):
        """
        分析用户的风险因素构成

        Args:
            user_id: 用户ID
            anomalies: 用户的异常字典
            risk_score: 用户的风险评分
            risk_factors: 风险评分的因素构成

        Returns:
            dict: 包含风险分析结果的字典
        """
        # 统计各类异常数量
        category_counts = defaultdict(int)
        subtype_counts = defaultdict(int)
        severity_by_category = defaultdict(list)
        severity_by_subtype = defaultdict(list)

        # 收集所有异常
        all_anomalies = []
        for category, user_anomalies in anomalies.items():
            if user_id in user_anomalies:
                for anomaly in user_anomalies[user_id]:
                    # Check if anomaly is a dictionary before processing
                    if not isinstance(anomaly, dict):
                        # Log the skipped anomaly type and continue
                        logger.debug(f"Skipping non-dictionary anomaly: {type(anomaly)}")
                        continue

                    all_anomalies.append(anomaly)
                    category = anomaly.get('type', 'unknown')
                    subtype = anomaly.get('subtype', '')
                    severity = anomaly.get('score', 0) or anomaly.get('severity', 5)

                    category_counts[category] += 1
                    if subtype:
                        subtype_counts[subtype] += 1

                    severity_by_category[category].append(severity)
                    if subtype:
                        severity_by_subtype[subtype].append(severity)

        # 计算主要风险因素
        primary_risk_factors = []

        # 按异常数量排序类别
        categories_by_count = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 按严重程度排序类别
        categories_by_severity = sorted(
            [(cat, np.mean(sevs)) for cat, sevs in severity_by_category.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # 按数量和严重程度综合排序子类型
        subtypes_by_impact = []
        for subtype, count in subtype_counts.items():
            avg_severity = np.mean(severity_by_subtype[subtype])
            impact = count * avg_severity
            subtypes_by_impact.append((subtype, count, avg_severity, impact))

        subtypes_by_impact.sort(key=lambda x: x[3], reverse=True)

        # 创建主要风险因素列表
        for cat, count in categories_by_count[:3]:  # 取前3个主要类别
            avg_severity = np.mean(severity_by_category[cat])
            factor = {
                'category': cat,
                'display_name': self.risk_categories.get(cat, cat),
                'count': count,
                'avg_severity': avg_severity,
                'impact': count * avg_severity
            }
            primary_risk_factors.append(factor)

        # 添加主要子类型因素
        for subtype, count, avg_severity, impact in subtypes_by_impact[:5]:  # 取前5个主要子类型
            factor = {
                'subtype': subtype,
                'display_name': self.risk_subtypes.get(subtype, subtype),
                'count': count,
                'avg_severity': avg_severity,
                'impact': impact
            }
            primary_risk_factors.append(factor)

        # 生成风险评估
        risk_assessment = self._generate_risk_assessment(risk_score, category_counts, primary_risk_factors)

        # 生成缓解建议
        mitigation_suggestions = self._generate_mitigation_suggestions(category_counts)

        # 准备返回结果
        result = {
            'user_id': user_id,
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'anomaly_counts': dict(category_counts),
            'primary_risk_factors': primary_risk_factors,
            'risk_dimensions': risk_factors.get('dimensions', {}),
            'risk_assessment': risk_assessment,
            'mitigation_suggestions': mitigation_suggestions,
            'has_behavioral_change': any(a.get('type') == 'time_series' for a in all_anomalies),
            'has_relationship_anomalies': any(a.get('type') == 'graph' for a in all_anomalies)
        }

        return result

    def _generate_risk_assessment(self, risk_score, category_counts, primary_factors):
        """生成风险评估文本"""
        risk_level = self._get_risk_level(risk_score)

        if risk_score < 30:
            assessment = f"用户当前风险较低，风险评分为{risk_score}。"
            if category_counts:
                assessment += " 尽管有一些异常行为，但总体风险水平在可接受范围内。建议继续常规监控。"
            else:
                assessment += " 未检测到明显异常行为，建议维持常规监控。"
        elif risk_score < 70:
            assessment = f"用户当前风险中等，风险评分为{risk_score}。"

            if primary_factors:
                main_factor = primary_factors[0]
                if 'category' in main_factor:
                    category = main_factor['display_name']
                    assessment += f" 主要风险来源于{category}，"
                elif 'subtype' in main_factor:
                    subtype = main_factor['display_name']
                    assessment += f" 主要风险来源于{subtype}，"

                assessment += f"共检测到{sum(category_counts.values())}个异常行为。建议增加对该用户的监控频率，并审查相关权限。"
            else:
                assessment += " 建议增加监控频率并定期审查。"
        else:
            assessment = f"用户当前风险高，风险评分为{risk_score}。"

            if primary_factors:
                factor_names = [
                    f.get('display_name') for f in primary_factors[:2]
                    if 'display_name' in f
                ]
                if factor_names:
                    factors_text = "、".join(factor_names)
                    assessment += f" 主要风险因素包括{factors_text}等，"

                assessment += f"共检测到{sum(category_counts.values())}个异常行为。建议立即审查用户行为和权限，并考虑临时限制某些敏感操作权限。"
            else:
                assessment += " 建议立即进行全面的安全审查。"

        return assessment

    def _generate_mitigation_suggestions(self, category_counts):
        """根据异常类别生成缓解建议"""
        suggestions = []

        # 按异常数量排序类别
        sorted_categories = sorted(
            category_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 为主要的风险类别提供建议
        for category, count in sorted_categories[:3]:  # 取前3个主要类别
            if category in self.mitigation_strategies:
                category_suggestions = self.mitigation_strategies[category]
                # 每个类别选取2-3条建议
                selected = category_suggestions[:min(3, len(category_suggestions))]
                for suggestion in selected:
                    suggestions.append({
                        'category': category,
                        'display_name': self.risk_categories.get(category, category),
                        'suggestion': suggestion
                    })

        return suggestions

    def _get_risk_level(self, risk_score):
        """根据风险分数确定风险级别"""
        if risk_score >= 70:
            return 'high'
        elif risk_score >= 40:
            return 'medium'
        else:
            return 'low'

    def analyze_organization_risk_distribution(self, all_user_risks, user_profiles, departments=None):
        """
        分析组织整体的风险分布

        Args:
            all_user_risks: 所有用户的风险评分字典
            user_profiles: 用户配置信息字典
            departments: 部门列表，可选

        Returns:
            dict: 包含组织风险分析结果的字典
        """
        if not all_user_risks:
            return {
                'avg_risk_score': 0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'department_risks': {},
                'role_risks': {}
            }

        # 计算风险人数统计
        high_risk_count = sum(1 for score in all_user_risks.values() if score >= 70)
        medium_risk_count = sum(1 for score in all_user_risks.values() if 40 <= score < 70)
        low_risk_count = sum(1 for score in all_user_risks.values() if score < 40)

        # 按部门统计风险
        department_risks = defaultdict(list)
        role_risks = defaultdict(list)

        for user_id, risk_score in all_user_risks.items():
            profile = user_profiles.get(user_id, {})
            department = profile.get('department', 'Unknown')
            role = profile.get('role', 'Unknown')

            department_risks[department].append(risk_score)
            role_risks[role].append(risk_score)

        # 计算部门平均风险
        dept_avg_risks = {}
        for dept, scores in department_risks.items():
            dept_avg_risks[dept] = {
                'avg_score': np.mean(scores),
                'max_score': max(scores),
                'high_risk_count': sum(1 for s in scores if s >= 70),
                'user_count': len(scores)
            }

        # 计算角色平均风险
        role_avg_risks = {}
        for role, scores in role_risks.items():
            role_avg_risks[role] = {
                'avg_score': np.mean(scores),
                'max_score': max(scores),
                'high_risk_count': sum(1 for s in scores if s >= 70),
                'user_count': len(scores)
            }

        # 找出风险最高的部门
        highest_risk_depts = sorted(
            dept_avg_risks.items(),
            key=lambda x: x[1]['avg_score'],
            reverse=True
        )[:5]  # 取前5个高风险部门

        return {
            'avg_risk_score': np.mean(list(all_user_risks.values())),
            'high_risk_count': high_risk_count,
            'medium_risk_count': medium_risk_count,
            'low_risk_count': low_risk_count,
            'total_users': len(all_user_risks),
            'department_risks': dept_avg_risks,
            'role_risks': role_avg_risks,
            'highest_risk_departments': highest_risk_depts
        }

    def generate_risk_summary_report(self, user_id, risk_analysis):
        """
        生成用户风险摘要报告

        Args:
            user_id: 用户ID
            risk_analysis: 风险分析结果

        Returns:
            str: 风险摘要报告文本
        """
        risk_score = risk_analysis['risk_score']
        risk_level = risk_analysis['risk_level']
        anomaly_counts = risk_analysis['anomaly_counts']
        total_anomalies = sum(anomaly_counts.values())
        primary_factors = risk_analysis['primary_risk_factors']

        # 创建报告
        report = [
            f"用户 {user_id} 风险摘要报告",
            f"=====================",
            f"",
            f"风险评分: {risk_score}/100 ({risk_level.upper()})",
            f"检测到的异常总数: {total_anomalies}",
            f""
        ]

        # 添加异常类别统计
        if anomaly_counts:
            report.append("异常类别分布:")
            for category, count in sorted(anomaly_counts.items(), key=lambda x: x[1], reverse=True):
                display_name = self.risk_categories.get(category, category)
                report.append(f"- {display_name}: {count}个异常")
            report.append("")

        # 添加主要风险因素
        if primary_factors:
            report.append("主要风险因素:")
            for i, factor in enumerate(primary_factors[:3], 1):
                if 'category' in factor:
                    display_name = factor['display_name']
                    report.append(
                        f"{i}. {display_name} - {factor['count']}个异常，平均严重度: {factor['avg_severity']:.1f}")
                elif 'subtype' in factor:
                    display_name = factor['display_name']
                    report.append(
                        f"{i}. {display_name} - {factor['count']}个异常，平均严重度: {factor['avg_severity']:.1f}")
            report.append("")

        # 添加风险评估
        report.append("风险评估:")
        report.append(risk_analysis['risk_assessment'])
        report.append("")

        # 添加缓解建议
        if risk_analysis['mitigation_suggestions']:
            report.append("建议措施:")
            for i, suggestion in enumerate(risk_analysis['mitigation_suggestions'], 1):
                category_name = suggestion['display_name']
                report.append(f"{i}. [{category_name}] {suggestion['suggestion']}")
            report.append("")

        # 特殊警告
        if risk_analysis['has_behavioral_change']:
            report.append("⚠ 警告: 检测到用户行为模式随时间有显著变化，建议密切关注。")

        if risk_analysis['has_relationship_anomalies']:
            report.append("⚠ 警告: 检测到用户关系网络异常，可能存在共谋风险，建议进行关联分析。")

        return "\n".join(report)