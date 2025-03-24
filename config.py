#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常检测系统的配置文件。
包含系统路径、阈值和其他常量。
"""

import os
from pathlib import Path

# 基础路径
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

# 风险阈值
HIGH_RISK_THRESHOLD = 70
MEDIUM_RISK_THRESHOLD = 50
LOW_RISK_THRESHOLD = 30

# 历史数据配置
MAX_RISK_HISTORY = 180  # 保留180天的风险历史

# 异常检测配置
ANOMALY_WEIGHTS = {
    'organizational': 3.0,  # 组织级别异常权重最高
    'access': 2.5,          # 访问级别异常次之
    'email': 2.0,           # 邮件异常
    'time': 1.5             # 时间异常
}

# 异常子类型权重
ANOMALY_SUBTYPE_WEIGHTS = {
    'after_hours_access': 2.0,
    'unusual_file_access': 2.5,
    'data_exfiltration': 3.0,
    'cross_department_email': 1.5,
    'login_time_drift': 1.8,
    'access_pattern_drift': 2.0,
    'communication_drift': 2.2,
    'ml_pattern': 1.7,
    'graph': 2.1
}

# 风险评分配置
RISK_BASE_SCORE = 10
RISK_QUANTITY_WEIGHT = 0.7
RISK_TIME_DECAY = 0.9  # 时间衰减因子
# 日期格式配置
DATE_FORMAT = '%m/%d/%Y %H:%M:%S'

# 机器学习模型配置
ML_CONFIG = {
    'isolation_forest': {
        'n_estimators': 100,
        'contamination': 0.05,
        'random_state': 42
    },
    'lof': {
        'n_neighbors': 20,
        'contamination': 0.05
    },
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5
    }
}

# 数据标准化配置
DATA_NORMALIZER_CONFIG = {
    'internal_domains': ['company.com', 'internal.org', 'corp.net'],
    'sensitive_keywords': [
        'confidential', 'private', 'secret', 'sensitive',
        'financial', 'hr', 'personal', 'password', 'credential'
    ]
}

# 组织分析配置
ORG_ANALYSIS_CONFIG = {
    'sensitive_departments': ['finance', 'executive', 'legal', 'hr', 'it security'],
    'sensitive_roles': ['admin', 'executive', 'finance', 'hr', 'legal']
}

# 趋势检测配置
TREND_CONFIG = {
    'threshold_percentile': 95,
    'min_slope': 0.1,
    'window_size': 3,
    'periods_to_test': [7, 14, 30]
}

def get_risk_config():
    """获取风险评分配置"""
    return {
        'category_weights': ANOMALY_WEIGHTS,
        'subtype_weights': ANOMALY_SUBTYPE_WEIGHTS,
        'base_score': RISK_BASE_SCORE,
        'quantity_weight': RISK_QUANTITY_WEIGHT,
        'time_decay': RISK_TIME_DECAY
    }

def get_ml_config():
    """获取机器学习模型配置"""
    return ML_CONFIG

def get_normalizer_config():
    """获取数据标准化配置"""
    return DATA_NORMALIZER_CONFIG

def get_trend_config():
    """获取趋势检测配置"""
    return TREND_CONFIG