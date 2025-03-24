"""
报告生成器：创建异常检测结果报告
"""

import pandas as pd
import os
import json
from datetime import datetime
import logging

# 尝试导入可视化模块，如果不存在则提供空函数
try:
    from .visualizations import create_anomaly_timeline, create_user_risk_heatmap, create_department_network_graph
except ImportError:
    # 定义空函数作为替代
    def create_anomaly_timeline(*args, **kwargs):
        pass


    def create_user_risk_heatmap(*args, **kwargs):
        pass


    def create_department_network_graph(*args, **kwargs):
        pass


    logging.warning("无法导入可视化模块，将使用空函数替代")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportGenerator:
    """异常检测报告生成器类"""

    def __init__(self, detector, output_dir="reports"):
        """
        初始化报告生成器

        Args:
            detector: 异常检测器实例，必须实现BaseAnomalyDetector接口
            output_dir (str): 输出报告的目录
        """
        self.detector = detector
        self.output_dir = output_dir
        self.report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def generate_summary_report(self, title="异常检测摘要报告", future_predictions=None, risk_analysis=None,
                                org_risk_analysis=None):
        """
        生成检测结果的摘要报告

        Args:
            title (str): 报告标题
            future_predictions (dict, optional): 未来风险预测结果
            risk_analysis (dict, optional): 用户风险因素分析结果
            org_risk_analysis (dict, optional): 组织级别风险分析结果

        Returns:
            dict: 摘要报告数据
        """
        # 确保已运行检测
        if not hasattr(self.detector, 'anomaly_results') or not self.detector.anomaly_results:
            self.detector.detect_all_anomalies()

        # 初始化异常类型计数字典，包含所有可能的异常类型
        anomaly_types = {
            'time': 0,
            'access': 0,
            'email': 0,
            'organizational': 0  # 使用正确的键名称
        }

        # 计算汇总统计信息
        total_anomalies = 0
        users_with_anomalies = set()

        for anomaly_type, anomalies_by_user in self.detector.anomaly_results.items():
            for user_id, anomaly_list in anomalies_by_user.items():
                anomaly_count = len(anomaly_list)
                total_anomalies += anomaly_count

                if anomaly_count > 0:
                    users_with_anomalies.add(user_id)

                # 添加类型映射逻辑，以处理可能的名称差异
                if anomaly_type in anomaly_types:
                    anomaly_types[anomaly_type] += anomaly_count
                else:
                    # 处理可能的键名不匹配
                    logger.warning(f"未知的异常类型: {anomaly_type}")

        # 获取高风险用户和中等风险用户 - 调整风险阈值
        high_risk_threshold = 70  # 可以考虑降低此阈值
        medium_risk_threshold = 40  # 可以考虑降低此阈值

        # 获取所有用户的风险分数
        all_risk_scores = self.detector.risk_scores

        # 检查风险分数分布
        if all_risk_scores:
            max_risk = max(all_risk_scores.values()) if all_risk_scores else 0
            min_risk = min(all_risk_scores.values()) if all_risk_scores else 0
            avg_risk = sum(all_risk_scores.values()) / len(all_risk_scores) if all_risk_scores else 0
            logger.info(f"风险分数范围: 最低 {min_risk}, 最高 {max_risk}, 平均 {avg_risk:.2f}")

            # 如果最高风险分数太低，可以自适应降低阈值
            if max_risk < high_risk_threshold and max_risk > 0:
                adjusted_high_threshold = max(20, max_risk * 0.9)  # 至少20，或最高分数的90%
                adjusted_medium_threshold = max(10, max_risk * 0.6)  # 至少10，或最高分数的60%

                logger.info(
                    f"调整风险阈值: 高风险从 {high_risk_threshold} 降至 {adjusted_high_threshold}, 中等风险从 {medium_risk_threshold} 降至 {adjusted_medium_threshold}")

                high_risk_threshold = adjusted_high_threshold
                medium_risk_threshold = adjusted_medium_threshold

        # 获取高风险和中等风险用户
        high_risk_users = []
        medium_risk_users = []

        for user_id, score in all_risk_scores.items():
            if score >= high_risk_threshold:
                high_risk_users.append(user_id)
            elif score >= medium_risk_threshold:
                medium_risk_users.append(user_id)

        # 如果使用调整后的阈值仍然没有高风险用户，则选择前10名风险最高的用户作为高风险用户
        if not high_risk_users and all_risk_scores:
            # 按风险分数降序排序用户
            sorted_users = sorted(all_risk_scores.items(), key=lambda x: x[1], reverse=True)
            top_users = [user_id for user_id, score in sorted_users[:10] if score > 0]

            if top_users:
                logger.info(f"未找到高风险用户，将风险最高的 {len(top_users)} 名用户视为高风险用户")
                high_risk_users = top_users

                # 更新高风险阈值为这些用户中的最低分数
                high_risk_threshold = min(all_risk_scores[user_id] for user_id in high_risk_users)

                # 更新中等风险阈值和用户列表，避免重复
                medium_risk_threshold = high_risk_threshold * 0.6
                medium_risk_users = [user_id for user_id, score in all_risk_scores.items()
                                     if medium_risk_threshold <= score < high_risk_threshold][:20]  # 限制为前20个

        # 过滤掉没有异常的用户
        high_risk_users = [
            user_id for user_id in high_risk_users
            if user_id in users_with_anomalies
        ]

        medium_risk_users = [
            user_id for user_id in medium_risk_users
            if user_id in users_with_anomalies
        ]

        # 创建自定义风险级别描述
        risk_level_description = {
            "high_risk_threshold": high_risk_threshold,
            "medium_risk_threshold": medium_risk_threshold,
            "note": "风险阈值已根据当前数据分布自动调整" if high_risk_threshold != 70 or medium_risk_threshold != 40 else ""
        }

        # 创建报告数据
        report = {
            "title": title,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_anomalies_detected": total_anomalies,
                "total_users_with_anomalies": len(users_with_anomalies),
                "high_risk_users_count": len(high_risk_users),
                "medium_risk_users_count": len(medium_risk_users),
                "anomalies_by_type": anomaly_types,
                "risk_thresholds": risk_level_description
            },
            "high_risk_users": [
                {
                    "user_id": user_id,
                    "risk_score": self.detector.risk_scores.get(user_id, 0),
                    "anomalies_count": sum(
                        len(self.detector.anomaly_results[atype].get(user_id, []))
                        for atype in self.detector.anomaly_results.keys()
                        if user_id in self.detector.anomaly_results[atype]
                    )
                }
                for user_id in high_risk_users
            ],
            "medium_risk_users": [
                {
                    "user_id": user_id,
                    "risk_score": self.detector.risk_scores.get(user_id, 0),
                    "anomalies_count": sum(
                        len(self.detector.anomaly_results[atype].get(user_id, []))
                        for atype in self.detector.anomaly_results.keys()
                        if user_id in self.detector.anomaly_results[atype]
                    )
                }
                for user_id in medium_risk_users[:10]  # 仅包括前10个中等风险用户
            ]
        }

        # 检查是否有高风险或中等风险用户
        if not high_risk_users and not medium_risk_users:
            logger.warning("警告: 没有检测到高风险或中等风险用户，尽管存在异常")

            # 添加一些潜在问题诊断信息
            report["diagnostic_info"] = {
                "warning": "没有检测到高风险或中等风险用户，尽管存在异常",
                "possible_reasons": [
                    "风险评分算法可能需要调整",
                    "异常严重性评估过低",
                    "用户异常数量不足以达到风险阈值"
                ],
                "risk_score_stats": {
                    "max_risk": max_risk if 'max_risk' in locals() else 0,
                    "min_risk": min_risk if 'min_risk' in locals() else 0,
                    "avg_risk": avg_risk if 'avg_risk' in locals() else 0
                },
                "users_with_anomalies": len(users_with_anomalies)
            }

        # 添加未来预测结果，如果有的话
        if future_predictions:
            report["future_predictions"] = future_predictions

        # 添加风险分析结果，如果有的话
        if risk_analysis:
            report["risk_analysis"] = risk_analysis

        # 添加组织风险分析，如果有的话
        if org_risk_analysis:
            report["organization_risk"] = org_risk_analysis

        # 保存报告为JSON
        report_path = os.path.join(self.output_dir, f"summary_report_{self.report_timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"摘要报告已保存至: {report_path}")

        # 导出所有异常到CSV
        csv_path = self.export_all_anomalies_csv()
        if csv_path:
            print(f"\n所有异常已导出至CSV文件: {csv_path}")

        # 打印报告摘要
        print("\n===== 异常检测报告摘要 =====")
        print(
            f"分析用户总数: {len(self.detector.user_profiles) if hasattr(self.detector, 'user_profiles') else 'Unknown'}")
        print(f"检测到的异常总数: {report['summary']['total_anomalies_detected']}")
        print(f"具有异常的用户数: {report['summary']['total_users_with_anomalies']}")
        print(
            f"高风险用户数: {report['summary']['high_risk_users_count']} (阈值: {risk_level_description['high_risk_threshold']:.2f})")
        print(
            f"中等风险用户数: {report['summary']['medium_risk_users_count']} (阈值: {risk_level_description['medium_risk_threshold']:.2f})")

        # 如果启用了自适应阈值，显示通知
        if risk_level_description['note']:
            print(f"注意: {risk_level_description['note']}")

        print("\n异常类型分布:")
        for type_name, count in report['summary']['anomalies_by_type'].items():
            print(f"  - {type_name}: {count}")

        # 打印高风险用户
        if high_risk_users:
            print("\n高风险用户:")
            for i, user_data in enumerate(report['high_risk_users'][:5]):  # 只显示前5个
                print(
                    f"  {i + 1}. 用户 {user_data['user_id']}: 风险分数 {user_data['risk_score']:.2f}, 异常数 {user_data['anomalies_count']}")

        # 打印组织风险分析
        print("\n组织风险分析:")
        if org_risk_analysis:
            # 使用提供的组织风险分析数据
            avg_risk = org_risk_analysis.get('avg_risk_score', 0)
            print(f"  - 组织平均风险分数: {avg_risk:.2f}")

            # 打印风险最高的部门
            highest_risk_depts = org_risk_analysis.get('highest_risk_departments', [])
            if highest_risk_depts:
                print("  - 风险最高的部门:")
                for dept, info in highest_risk_depts[:1]:  # 打印风险最高的部门
                    print(
                        f"    * {dept}: 平均分数 {info.get('avg_score', 0):.2f}, 高风险用户 {info.get('high_risk_count', 0)}")
        elif hasattr(self.detector, 'org_analyzer') and hasattr(self.detector.org_analyzer, 'department_profiles'):
            # 使用检测器中的组织分析器
            dept_risks = {}
            for dept_id, dept_info in self.detector.org_analyzer.department_profiles.items():
                members = dept_info.get('members', [])
                risk_scores = [self.detector.risk_scores.get(m, 0) for m in members]
                avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
                high_risk_count = sum(1 for score in risk_scores if score >= high_risk_threshold)
                dept_risks[dept_id] = (avg_risk, high_risk_count)

            # 计算组织平均风险
            all_risks = [self.detector.risk_scores.get(user_id, 0) for user_id in self.detector.risk_scores]
            org_avg_risk = sum(all_risks) / len(all_risks) if all_risks else 0
            print(f"  - 组织平均风险分数: {org_avg_risk:.2f}")

            # 找出风险最高的部门
            if dept_risks:
                highest_dept = max(dept_risks.items(), key=lambda x: x[1][0])
                dept_id, (avg_risk, high_risk_count) = highest_dept
                print("  - 风险最高的部门:")
                print(f"    * {dept_id}: 平均分数 {avg_risk:.2f}, 高风险用户 {high_risk_count}")
            else:
                print("  - 组织风险分析: 无部门信息")
        else:
            print(f"  - 组织平均风险分数: {0.00:.2f}")
            print("  - 风险最高的部门:")
            print(f"    * Unknown: 平均分数 {0.00:.2f}, 高风险用户 {0}")

        # 打印诊断信息（如果有的话）
        if "diagnostic_info" in report:
            print("\n诊断信息:")
            print(f"  警告: {report['diagnostic_info']['warning']}")
            print("  可能的原因:")
            for reason in report['diagnostic_info']['possible_reasons']:
                print(f"    - {reason}")
            print("  风险分数统计:")
            risk_stats = report['diagnostic_info']['risk_score_stats']
            print(f"    - 最高风险分数: {risk_stats['max_risk']:.2f}")
            print(f"    - 最低风险分数: {risk_stats['min_risk']:.2f}")
            print(f"    - 平均风险分数: {risk_stats['avg_risk']:.2f}")

        # 打印未来风险预测信息（如果有的话）
        if future_predictions:
            print("\n未来风险预测:")
            for user_id, prediction in list(future_predictions.items())[:3]:  # 只显示前3个
                if isinstance(prediction, dict):
                    future_risk = prediction.get('risk_score', 0)
                    print(f"  - 用户 {user_id}: 预测风险 {future_risk:.2f}")

        # 打印风险因素分析（如果有的话）
        if risk_analysis:
            print("\n用户风险因素分析:")
            for user_id, analysis in list(risk_analysis.items())[:3]:  # 只显示前3个
                if isinstance(analysis, dict):
                    top_factors = analysis.get('top_factors', [])
                    if top_factors:
                        print(f"  - 用户 {user_id} 的主要风险因素:")
                        for factor in top_factors[:2]:  # 只显示前2个因素
                            print(f"    * {factor.get('factor', '未知')}: {factor.get('contribution', 0):.2f}%")

        print("\n==============================")

        return report

    def generate_user_report(self, user_id):
        """
        为特定用户生成详细异常报告

        Args:
            user_id (str): 要生成报告的用户ID

        Returns:
            dict: 用户报告数据
        """
        # 获取用户异常
        user_anomalies = self.detector.get_user_anomalies(user_id)
        risk_score = self.detector.risk_scores.get(user_id, 0)

        # 为该用户创建详细的时间线
        timeline_path = os.path.join(self.output_dir, f"user_{user_id}_timeline_{self.report_timestamp}.png")
        create_anomaly_timeline(user_anomalies, save_path=timeline_path)

        # 汇总各类异常
        anomaly_summary = {}
        for anomaly_type, anomalies in user_anomalies.items():
            anomaly_summary[anomaly_type] = {
                "count": len(anomalies),
                "details": [
                    {
                        "score": a.get("score", 0),
                        "reason": a.get("reason", "未知原因"),
                        "timestamp": a.get("timestamp", "未知时间")
                    }
                    for a in anomalies[:5]  # 仅包括每种类型的前5个异常
                ]
            }

        # 创建报告
        report = {
            "user_id": user_id,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_score": risk_score,
            "risk_level": "高" if risk_score >= 70 else "中" if risk_score >= 40 else "低",
            "anomaly_summary": anomaly_summary,
            "visualization_paths": {
                "timeline": timeline_path
            }
        }

        # 保存报告
        report_path = os.path.join(self.output_dir, f"user_{user_id}_report_{self.report_timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"用户报告已保存至: {report_path}")
        return report

    def generate_department_report(self, department_id):
        """
        为特定部门生成异常报告

        Args:
            department_id (str): 部门ID

        Returns:
            dict: 部门报告数据
        """
        # 获取该部门的所有用户
        dept_users = []
        if hasattr(self.detector, 'org_analyzer') and self.detector.org_analyzer.department_profiles:
            if department_id in self.detector.org_analyzer.department_profiles:
                dept_users = self.detector.org_analyzer.department_profiles[department_id]['members']

        # 如果没有用户信息，则返回空报告
        if not dept_users:
            return {"error": f"无法找到部门 {department_id} 的用户"}

        # 收集部门所有用户的异常
        dept_anomalies = {user_id: self.detector.get_user_anomalies(user_id) for user_id in dept_users}

        # 获取风险评分
        dept_risk_scores = {user_id: self.detector.risk_scores.get(user_id, 0) for user_id in dept_users}
        avg_risk_score = sum(dept_risk_scores.values()) / len(dept_risk_scores) if dept_risk_scores else 0

        # 创建部门网络图
        network_path = os.path.join(self.output_dir, f"dept_{department_id}_network_{self.report_timestamp}.png")

        if hasattr(self.detector, 'org_analyzer'):
            create_department_network_graph(
                department_id,
                self.detector.org_analyzer.interaction_graph,
                self.detector.org_analyzer.department_profiles,
                save_path=network_path
            )

        # 创建热力图
        heatmap_path = os.path.join(self.output_dir, f"dept_{department_id}_heatmap_{self.report_timestamp}.png")
        create_user_risk_heatmap(dept_risk_scores, department_id, save_path=heatmap_path)

        # 创建部门报告
        report = {
            "department_id": department_id,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_count": len(dept_users),
            "average_risk_score": avg_risk_score,
            "high_risk_users": [
                {"user_id": user_id, "risk_score": score}
                for user_id, score in dept_risk_scores.items() if score >= 70
            ],
            "anomaly_counts": {
                "time": sum(len(anomalies.get('time', {})) for anomalies in dept_anomalies.values()),
                "access": sum(len(anomalies.get('access', {})) for anomalies in dept_anomalies.values()),
                "email": sum(len(anomalies.get('email', {})) for anomalies in dept_anomalies.values()),
                "org": sum(len(anomalies.get('org', {})) for anomalies in dept_anomalies.values())
            },
            "visualization_paths": {
                "network": network_path,
                "heatmap": heatmap_path
            }
        }

        # 保存报告
        report_path = os.path.join(self.output_dir, f"dept_{department_id}_report_{self.report_timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"部门报告已保存至: {report_path}")
        return report

    def generate_full_report(self):
        """
        生成完整的组织异常报告

        Returns:
            dict: 完整报告数据
        """
        # 生成摘要报告
        summary = self.generate_summary_report("组织全面异常检测报告")

        # 为所有高风险用户生成报告
        high_risk_users = self.detector.get_high_risk_users(threshold=70)
        user_reports = {}
        for user_id in high_risk_users[:10]:  # 仅包括前10个高风险用户
            user_reports[user_id] = self.generate_user_report(user_id)

        # 为所有部门生成报告
        department_reports = {}
        if hasattr(self.detector, 'org_analyzer') and self.detector.org_analyzer.department_profiles:
            for dept_id in self.detector.org_analyzer.department_profiles.keys():
                department_reports[dept_id] = self.generate_department_report(dept_id)

        # 创建完整报告
        full_report = {
            "summary": summary,
            "high_risk_user_reports": user_reports,
            "department_reports": department_reports,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存报告
        report_path = os.path.join(self.output_dir, f"full_report_{self.report_timestamp}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, ensure_ascii=False, indent=2)

        print(f"完整报告已保存至: {report_path}")
        return full_report

    def export_all_anomalies_csv(self):
        """
        将所有检测到的异常导出为CSV文件

        Returns:
            str: 保存的CSV文件路径
        """
        # 确保已运行检测
        if not hasattr(self.detector, 'anomaly_results') or not self.detector.anomaly_results:
            self.detector.detect_all_anomalies()

        # 创建所有异常的列表
        all_anomalies = []

        for anomaly_type, anomalies_by_user in self.detector.anomaly_results.items():
            for user_id, anomaly_list in anomalies_by_user.items():
                for anomaly in anomaly_list:
                    # 处理异常可能是整数或其他非字典类型的情况
                    if isinstance(anomaly, dict):
                        record = {
                            'user_id': user_id,
                            'anomaly_type': anomaly_type,
                            'score': anomaly.get('score', 0),
                            'reason': anomaly.get('reason', ''),
                            'timestamp': anomaly.get('timestamp', ''),
                            'risk_score': self.detector.risk_scores.get(user_id, 0)
                        }
                    else:
                        # 处理非字典类型的异常
                        record = {
                            'user_id': user_id,
                            'anomaly_type': anomaly_type,
                            'score': anomaly if isinstance(anomaly, (int, float)) else 0,
                            'reason': 'Auto-detected anomaly',
                            'timestamp': '',
                            'risk_score': self.detector.risk_scores.get(user_id, 0)
                        }
                    all_anomalies.append(record)

        # 创建DataFrame并导出为CSV
        if all_anomalies:
            df = pd.DataFrame(all_anomalies)
            csv_path = os.path.join(self.output_dir, f"all_anomalies_{self.report_timestamp}.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"所有异常已导出至: {csv_path}")
            return csv_path
        else:
            print("没有异常被检测到，无法导出CSV")
            return None

    def generate_visualizations(self, timestamp=None):
        """
        为检测结果生成标准可视化

        Args:
            timestamp (str, optional): 用于文件名的时间戳

        Returns:
            dict: 生成的可视化文件路径
        """
        if timestamp is None:
            timestamp = self.report_timestamp

        visualizations = {}

        # 确保已运行检测
        if not hasattr(self.detector, 'anomaly_results') or not self.detector.anomaly_results:
            self.detector.detect_all_anomalies()

        # 尝试导入可视化模块
        try:
            from .visualizations import create_anomaly_distribution_chart

            # 1. 生成异常分布图
            dist_path = os.path.join(self.output_dir, f"anomaly_distribution_{timestamp}.png")
            create_anomaly_distribution_chart(self.detector.anomaly_results, save_path=dist_path)
            visualizations['distribution'] = dist_path

            # 2. 为前10个高风险用户生成热力图
            high_risk_users = self.detector.get_high_risk_users(threshold=40)[:10]
            user_risk_scores = {user_id: self.detector.risk_scores.get(user_id, 0) for user_id in high_risk_users}

            if user_risk_scores:
                heatmap_path = os.path.join(self.output_dir, f"risk_heatmap_{timestamp}.png")
                from .visualizations import create_user_risk_heatmap
                create_user_risk_heatmap(user_risk_scores, save_path=heatmap_path)
                visualizations['risk_heatmap'] = heatmap_path

            # 3. 为最高风险用户生成时间线图
            if high_risk_users:
                top_user_id = high_risk_users[0]
                user_anomalies = self.detector.get_user_anomalies(top_user_id)

                if user_anomalies:
                    timeline_path = os.path.join(self.output_dir, f"user_{top_user_id}_timeline_{timestamp}.png")
                    from .visualizations import create_anomaly_timeline
                    create_anomaly_timeline(user_anomalies, save_path=timeline_path)
                    visualizations['top_user_timeline'] = timeline_path
        except ImportError as e:
            logger.warning(f"无法生成可视化: {str(e)}")

        print(f"生成了 {len(visualizations)} 个可视化图表")
        return visualizations


# 兼容性导出
report_generator = ReportGenerator
