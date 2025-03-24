import re
import pandas as pd
from collections import Counter


def normalize_text(text):
    """标准化文本（小写、去除多余空格等）"""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def extract_email_parts(email_str):
    """从电子邮件字符串中提取收件人列表"""
    if pd.isna(email_str):
        return []
    return [addr.strip() for addr in str(email_str).split(';') if addr.strip()]


def extract_domains(urls):
    """从URL列表中提取域名"""
    domains = []
    for url in urls:
        if pd.isna(url):
            continue
        match = re.search(r'https?://(?:www\.)?([^/]+)', str(url).lower())
        if match:
            domains.append(match.group(1))
        else:
            domains.append(str(url).lower())
    return domains


def detect_patterns(text, patterns):
    """检测文本中的特定模式"""
    if pd.isna(text):
        return []

    text = str(text).lower()
    found_patterns = []

    for pattern in patterns:
        if re.search(pattern, text):
            found_patterns.append(pattern)

    return found_patterns


def tokenize_filename(filename):
    """将文件名分词为有意义的组件"""
    if pd.isna(filename):
        return []

    # 去除扩展名
    base = str(filename).lower().split('.')
    name = '.'.join(base[:-1]) if len(base) > 1 else base[0]

    # 分词
    tokens = re.findall(r'[a-z0-9]+', name)
    return tokens
