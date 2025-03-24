#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测系统的主入口点。
协调数据预处理、用户配置文件创建、异常检测和报告生成。
"""

import os
import json
import logging
from datetime import datetime
from anomaly_detection.preprocessors.preprocessing_coordinator import PreprocessingCoordinator
from anomaly_detection.models.multidimensional_detector import MultiDimensionalAnomalyDetector
from anomaly_detection.reporting.report_generator import ReportGenerator
from anomaly_detection.utils.data_normalizer import DataNormalizer
from anomaly_detection.risk_scoring.risk_factor_analyzer import RiskFactorAnalyzer
from anomaly_detection.risk_scoring.risk_history_tracker import RiskHistoryTracker
from anomaly_detection.time_series.trend_detector import TrendDetector
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程序入口点"""
    start_time = datetime.now()
    logger.info("Starting anomaly detection process")

    # 1. 定义数据路径
    dataset_paths = {
        'logon': os.path.join(config.DATA_DIR, 'logon.csv'),
        'device': os.path.join(config.DATA_DIR, 'device.csv'),
        'email': os.path.join(config.DATA_DIR, 'email.csv'),
        'file': os.path.join(config.DATA_DIR, 'file.csv'),
        'http': os.path.join(config.DATA_DIR, 'http.csv'),
        'ldap': os.path.join(config.DATA_DIR, 'LDAP.csv'),
        'psychometric': os.path.join(config.DATA_DIR, 'psychometric.csv')
    }

    # 2. 初始化数据标准化工具
    logger.info("Initializing data normalizer")
    normalizer = DataNormalizer()

    # 3. 数据预处理阶段
    logger.info("Starting data preprocessing")
    preprocessor = PreprocessingCoordinator()

    # 使用数据标准化工具加载和预处理数据
    # 修改这一部分代码
    preprocessor.load_datasets(dataset_paths)
    raw_datasets = preprocessor.datasets  # 直接使用datasets属性

    # 标准化各数据集
    normalized_datasets = {}
    for name, df in raw_datasets.items():
        if not df.empty:
            normalized_datasets[name] = normalizer.normalize_dataset(df, name)

    # 使用标准化后的数据集继续预处理
    preprocessor.set_datasets(normalized_datasets).preprocess_all()

    # 获取预处理后的数据
    processed_datasets = preprocessor.get_datasets()
    user_profiles = preprocessor.get_user_profiles()
    logger.info(f"Preprocessing complete. Generated profiles for {len(user_profiles)} users")

    # 4. 初始化风险历史跟踪器
    logger.info("Initializing risk history tracker")
    history_file = os.path.join(config.DATA_DIR, 'risk_history.json')
    risk_tracker = RiskHistoryTracker(history_file=history_file)

    # 如果历史文件存在，加载历史数据
    if os.path.exists(history_file):
        risk_tracker.load_history()

    # 5. 异常检测阶段
    logger.info("Starting anomaly detection")
    detector = MultiDimensionalAnomalyDetector()

    # 设置预处理后的数据
    detector.set_preprocessed_data(processed_datasets, user_profiles)

    # 执行异常检测
    detector.detect_all_anomalies()

    # 6. 更新风险历史
    risk_tracker.update_risk_scores(
        detector.risk_scores,
        detector.risk_dimensions,
        detector.risk_factors
    )

    # 保存更新后的风险历史
    risk_tracker.save_history()

    # 7. 趋势分析
    logger.info("Analyzing risk trends")
    trend_detector = TrendDetector()

    # 对高风险用户进行趋势分析
    high_risk_users = [user_id for user_id, score in detector.risk_scores.items() if score >= 70]
    trend_analysis = {}
    future_predictions = {}

    for user_id in high_risk_users:
        user_history = risk_tracker.get_user_history(user_id)
        if user_history:
            # 分析风险趋势
            trend_analysis[user_id] = trend_detector.analyze_risk_trend(user_history)
            # 预测未来风险
            future_predictions[user_id] = trend_detector.predict_future_risk(user_history, days_ahead=30)

    # 8. 风险因素分析
    logger.info("Analyzing risk factors")
    risk_analyzer = RiskFactorAnalyzer()

    risk_analysis_results = {}
    for user_id in high_risk_users:
        # 分析用户风险因素
        risk_analysis = risk_analyzer.analyze_risk_factors(
            user_id,
            detector.anomalies,
            detector.risk_scores.get(user_id, 0),
            detector.risk_factors.get(user_id, {})
        )
        risk_analysis_results[user_id] = risk_analysis

    # 组织级别的风险分析
    org_risk_analysis = risk_analyzer.analyze_organization_risk_distribution(
        detector.risk_scores,
        user_profiles
    )

    # 9. 生成报告
    logger.info("Generating reports")
    report_generator = ReportGenerator(detector)

    # 将趋势分析和风险因素分析添加到报告中
    report = report_generator.generate_summary_report(
        future_predictions=future_predictions,
        risk_analysis=risk_analysis_results,
        org_risk_analysis=org_risk_analysis
    )

    # 确保输出目录存在
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)

    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(config.REPORTS_DIR, f"anomaly_report_{timestamp}.json")

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # 为高风险用户生成风险因素报告
    for user_id, analysis in risk_analysis_results.items():
        risk_report = risk_analyzer.generate_risk_summary_report(user_id, analysis)
        user_report_path = os.path.join(config.REPORTS_DIR, f"risk_report_{user_id}_{timestamp}.txt")

        with open(user_report_path, 'w') as f:
            f.write(risk_report)

        logger.info(f"Risk factor report for user {user_id} saved to {user_report_path}")

    # 导出所有异常到CSV文件
    csv_path = report_generator.export_all_anomalies_csv()
    if csv_path:
        print(f"\n所有异常已导出至CSV文件: {csv_path}")

    # 打印摘要
    print_summary(detector, report, trend_analysis, org_risk_analysis)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Anomaly detection process completed in {duration:.2f} seconds")
    logger.info(f"Report saved to {report_path}")


def print_summary(detector, report, trend_analysis=None, org_risk_analysis=None):
    """打印异常检测报告的摘要信息"""
    print("\n===== 异常检测报告摘要 =====")

    # 获取用户总数 - 从检测器获取而不是从报告
    total_users = len(detector.user_profiles) if hasattr(detector, 'user_profiles') else 0

    # 安全地访问报告数据
    summary = report.get('summary', {})
    total_anomalies = summary.get('total_anomalies_detected', 0)
    high_risk_users = summary.get('high_risk_users_count', 0)
    medium_risk_users = summary.get('medium_risk_users_count', 0)

    print(f"分析用户总数: {total_users}")
    print(f"检测到的异常总数: {total_anomalies}")
    print(f"高风险用户数: {high_risk_users}")
    print(f"中等风险用户数: {medium_risk_users}")

    # 打印异常类型分布
    if 'anomalies_by_type' in summary:
        print("\n异常类型分布:")
        for anomaly_type, count in summary['anomalies_by_type'].items():
            print(f"  - {anomaly_type}: {count}")

    # 打印趋势分析摘要
    if trend_analysis:
        print("\n风险趋势分析:")
        increasing_trends = sum(1 for analysis in trend_analysis.values()
                                if analysis.get('trend', {}).get('trend_type') == 'increasing')

        decreasing_trends = sum(1 for analysis in trend_analysis.values()
                                if analysis.get('trend', {}).get('trend_type') == 'decreasing')

        print(f"  - 风险呈上升趋势的用户: {increasing_trends}")
        print(f"  - 风险呈下降趋势的用户: {decreasing_trends}")

    # 打印组织风险分析
    if org_risk_analysis:
        print("\n组织风险分析:")
        print(f"  - 组织平均风险分数: {org_risk_analysis.get('avg_risk_score', 0):.2f}")

        # 打印风险最高的部门
        highest_risk_depts = org_risk_analysis.get('highest_risk_departments', [])
        if highest_risk_depts:
            print("  - 风险最高的部门:")
            for dept, info in highest_risk_depts[:3]:  # 打印前3个
                print(
                    f"    * {dept}: 平均分数 {info.get('avg_score', 0):.2f}, 高风险用户 {info.get('high_risk_count', 0)}")

    print("\n==============================")


if __name__ == "__main__":
    main()