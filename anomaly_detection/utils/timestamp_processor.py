import pandas as pd
from datetime import datetime


def convert_timestamps(df, date_column='date', format='%m/%d/%Y %H:%M:%S', target_column='timestamp'):
    """将数据框中的日期字符串转换为datetime对象"""
    if date_column in df.columns:
        df[target_column] = pd.to_datetime(df[date_column], format=format)
    return df


def extract_time_features(df, timestamp_column='timestamp'):
    """从时间戳列提取时间特征（小时、星期几等）"""
    if timestamp_column not in df.columns:
        return df

    result = df.copy()
    result['hour'] = df[timestamp_column].dt.hour
    result['day_of_week'] = df[timestamp_column].dt.dayofweek
    result['is_weekend'] = df[timestamp_column].dt.dayofweek >= 5
    result['is_business_hours'] = (df[timestamp_column].dt.hour >= 9) & \
                                  (df[timestamp_column].dt.hour <= 17) & \
                                  (df[timestamp_column].dt.dayofweek < 5)
    return result


def create_time_windows(df, timestamp_column='timestamp', window_size_minutes=10):
    """为时间序列数据创建滑动窗口标签"""
    if timestamp_column not in df.columns:
        return df

    sorted_df = df.sort_values(timestamp_column)
    window_id = 0
    window_ids = []

    last_time = None
    for time in sorted_df[timestamp_column]:
        if last_time is None or (time - last_time).total_seconds() > (window_size_minutes * 60):
            window_id += 1
        window_ids.append(window_id)
        last_time = time

    result = sorted_df.copy()
    result['window_id'] = window_ids
    return result
