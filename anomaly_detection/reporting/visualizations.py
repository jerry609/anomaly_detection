"""
数据可视化工具：为异常检测结果创建各种可视化图表
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import networkx as nx
import pandas as pd
from datetime import datetime
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("警告: 中文字体可能无法正确显示")


def create_anomaly_timeline(user_anomalies, save_path=None, figsize=(12, 8)):
    """
    为用户创建异常时间线可视化

    Args:
        user_anomalies (dict): 用户所有类型的异常
        save_path (str, optional): 保存图表的路径
        figsize (tuple): 图表尺寸

    Returns:
        matplotlib.figure.Figure: 生成的图表
    """
    # 收集所有带时间戳的异常
    time_data = []

    # 处理时间类异常
    if 'time' in user_anomalies:
        for anomaly in user_anomalies['time']:
            if 'timestamp' in anomaly and anomaly['timestamp']:
                time_data.append({
                    'timestamp': pd.to_datetime(anomaly['timestamp']),
                    'type': 'time',
                    'score': anomaly.get('score', 0),
                    'reason': anomaly.get('reason', '未知')
                })

    # 处理其他类型的异常（如果有时间戳）
    for anomaly_type in ['access', 'email', 'org']:
        if anomaly_type in user_anomalies:
            for anomaly in user_anomalies[anomaly_type]:
                if 'timestamp' in anomaly and anomaly['timestamp']:
                    time_data.append({
                        'timestamp': pd.to_datetime(anomaly['timestamp']),
                        'type': anomaly_type,
                        'score': anomaly.get('score', 0),
                        'reason': anomaly.get('reason', '未知')
                    })

    # 如果没有时间数据，则返回空白图表
    if not time_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "该用户没有足够的时间数据来创建时间线",
                horizontalalignment='center', verticalalignment='center')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return fig

    # 创建DataFrame并按时间排序
    df = pd.DataFrame(time_data)
    df = df.sort_values('timestamp')

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 定义颜色映射
    type_colors = {
        'time': 'blue',
        'access': 'red',
        'email': 'green',
        'org': 'purple'
    }

    # 绘制不同类型的异常
    for anomaly_type, color in type_colors.items():
        type_df = df[df['type'] == anomaly_type]
        if not type_df.empty:
            ax.scatter(type_df['timestamp'], type_df['score'],
                       color=color, label=anomaly_type, alpha=0.7, s=100)

    # 添加水平线表示异常阈值
    ax.axhline(y=0.7, color='r', linestyle='--', alpha=0.5, label='异常阈值')

    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # 设置坐标轴标签和图表标题
    ax.set_xlabel('时间')
    ax.set_ylabel('异常评分')
    ax.set_title('用户异常时间线')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 添加注释
    for i, row in df.iterrows():
        if row['score'] > 0.7:  # 仅为高评分异常添加注释
            ax.annotate(row['reason'][:40] + '...' if len(row['reason']) > 40 else row['reason'],
                        xy=(row['timestamp'], row['score']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        fontsize=8)

    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def create_user_risk_heatmap(user_risk_scores, title="用户风险热力图", save_path=None, figsize=(10, 8)):
    """
    创建用户风险热力图

    Args:
        user_risk_scores (dict): 用户ID到风险评分的映射
        title (str): 热力图标题
        save_path (str, optional): 保存图表的路径
        figsize (tuple): 图表尺寸

    Returns:
        matplotlib.figure.Figure: 生成的图表
    """
    # 准备数据
    users = list(user_risk_scores.keys())
    scores = list(user_risk_scores.values())

    # 如果用户太多，只显示前30个高风险用户
    if len(users) > 30:
        data = [(user, score) for user, score in user_risk_scores.items()]
        data.sort(key=lambda x: x[1], reverse=True)
        users = [d[0] for d in data[:30]]
        scores = [d[1] for d in data[:30]]

    # 创建一维热力图数据
    data_array = np.array(scores).reshape(-1, 1)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 创建自定义渐变色
    cmap = LinearSegmentedColormap.from_list('risk_cmap', ['green', 'yellow', 'orange', 'red'])

    # 创建热力图
    sns.heatmap(data_array, annot=True, cmap=cmap, cbar_kws={'label': '风险评分'},
                yticklabels=users, xticklabels=['风险评分'], ax=ax)

    # 设置标题和标签
    ax.set_title(title)
    ax.set_ylabel('用户ID')

    # 旋转y轴标签使其水平显示
    plt.yticks(rotation=0)

    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def create_department_network_graph(department_id, interaction_graph, department_profiles,
                                    save_path=None, figsize=(12, 10)):
    """
    创建部门网络关系图

    Args:
        department_id (str): 部门ID
        interaction_graph (networkx.Graph): 交互图
        department_profiles (dict): 部门配置文件
        save_path (str, optional): 保存图表的路径
        figsize (tuple): 图表尺寸

    Returns:
        matplotlib.figure.Figure: 生成的图表
    """
    # 检查部门是否存在
    if department_id not in department_profiles:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"找不到部门ID: {department_id}",
                horizontalalignment='center', verticalalignment='center')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return fig

    # 获取部门成员
    members = department_profiles[department_id]['members']

    # 创建子图
    subgraph = interaction_graph.subgraph(members)

    # 如果子图为空，则返回空白图表
    if len(subgraph) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"部门 {department_id} 没有交互数据",
                horizontalalignment='center', verticalalignment='center')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        return fig

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 设置网络布局
    pos = nx.spring_layout(subgraph, seed=42)

    # 计算节点大小（基于度中心性）
    node_size = [300 * (1 + nx.degree_centrality(subgraph)[n]) for n in subgraph.nodes()]

    # 获取边的权重（如果有）
    edge_weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
    normalized_weights = [2 * w / max(edge_weights) if edge_weights else 1 for w in edge_weights]

    # 绘制网络
    nx.draw_networkx_nodes(subgraph, pos, node_size=node_size,
                           node_color='skyblue', alpha=0.8, ax=ax)
    nx.draw_networkx_edges(subgraph, pos, width=normalized_weights,
                           alpha=0.5, edge_color='gray', ax=ax)
    nx.draw_networkx_labels(subgraph, pos, font_size=10, font_weight='bold', ax=ax)

    # 设置标题和调整布局
    ax.set_title(f"{department_id}部门网络关系图")
    ax.axis('off')

    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def create_anomaly_distribution_chart(anomaly_results, save_path=None, figsize=(10, 6)):
    """
    创建异常分布饼图

    Args:
        anomaly_results (dict): 按类型分组的异常结果
        save_path (str, optional): 保存图表的路径
        figsize (tuple): 图表尺寸

    Returns:
        matplotlib.figure.Figure: 生成的图表
    """
    # 计算各类型异常数量
    anomaly_counts = {}
    for anomaly_type, anomalies_by_user in anomaly_results.items():
        count = sum(len(anomalies) for anomalies in anomalies_by_user.values())
        anomaly_counts[anomaly_type] = count

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 定义颜色
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    # 创建饼图
    wedges, texts, autotexts = ax.pie(
        anomaly_counts.values(),
        labels=anomaly_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )

    # 设置标题和样式
    ax.set_title('异常类型分布')
    plt.setp(autotexts, size=10, weight='bold')
    ax.axis('equal')  # 确保饼图是圆的

    # 添加图例
    type_labels = {
        'time': '时间异常',
        'access': '访问异常',
        'email': '邮件异常',
        'org': '组织行为异常'
    }
    legend_labels = [f"{type_labels.get(k, k)} ({v})" for k, v in anomaly_counts.items()]
    ax.legend(wedges, legend_labels, title="异常类型", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def create_risk_trend_chart(risk_data, save_path=None, figsize=(12, 6)):
    """
    创建风险趋势图

    Args:
        risk_data (dict): 包含日期和风险分数的字典 {date: {user_id: score, ...}}
        save_path (str, optional): 保存图表的路径
        figsize (tuple): 图表尺寸

    Returns:
        matplotlib.figure.Figure: 生成的图表
    """
    # 准备数据
    dates = sorted(risk_data.keys())

    # 确定所有出现的用户
    all_users = set()
    for date_data in risk_data.values():
        all_users.update(date_data.keys())

    # 创建每个用户的时间序列数据
    user_trends = {}
    for user_id in all_users:
        scores = [risk_data[date].get(user_id, None) for date in dates]
        user_trends[user_id] = scores

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制每个用户的风险趋势
    for user_id, scores in user_trends.items():
        # 填充缺失值为None，使线条断开
        valid_indices = [i for i, s in enumerate(scores) if s is not None]
        valid_dates = [dates[i] for i in valid_indices]
        valid_scores = [scores[i] for i in valid_indices]

        if valid_scores:
            ax.plot(valid_dates, valid_scores, marker='o', linestyle='-', label=user_id)

    # 格式化x轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    # 设置坐标轴标签和图表标题
    ax.set_xlabel('日期')
    ax.set_ylabel('风险评分')
    ax.set_title('用户风险趋势')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 添加风险等级区域
    ax.axhspan(70, 100, alpha=0.2, color='red', label='高风险')
    ax.axhspan(40, 70, alpha=0.2, color='yellow', label='中风险')
    ax.axhspan(0, 40, alpha=0.2, color='green', label='低风险')

    # 添加风险级别的图例
    handles, labels = ax.get_legend_handles_labels()
    risk_handles = [
        plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.2),
        plt.Rectangle((0, 0), 1, 1, color='yellow', alpha=0.2),
        plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.2)
    ]
    risk_labels = ['高风险', '中风险', '低风险']

    # 如果用户太多，则限制图例中显示的用户数量
    if len(all_users) > 10:
        handles = handles[:10]
        labels = labels[:10]
        ax.text(0.5, 0.01, '(仅显示前10名用户)',
                transform=ax.transAxes, horizontalalignment='center')

    # 添加组合图例
    ax.legend(handles + risk_handles, labels + risk_labels,
              loc='center left', bbox_to_anchor=(1, 0.5))

    # 调整布局
    plt.tight_layout()

    # 保存图表（如果指定了路径）
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig
