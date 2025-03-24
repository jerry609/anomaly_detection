import networkx as nx
import pandas as pd
import logging
from anomaly_detection.utils.text_processor import extract_email_parts

logger = logging.getLogger(__name__)


class UserGraphBuilder:
    """构建和分析用户关系图"""

    def __init__(self):
        """初始化图构建器"""
        self.user_graph = nx.Graph()
        self.email_data = None
        self.user_profiles = None

    def build_graph(self, email_data, user_profiles):
        """基于邮件交互构建用户关系图

        Args:
            email_data: 包含邮件交互的DataFrame
            user_profiles: 用户配置文件字典

        Returns:
            nx.Graph: 构建的用户关系图
        """
        self.email_data = email_data
        self.user_profiles = user_profiles

        # 重置图
        self.user_graph = nx.Graph()

        # 添加节点 - 所有用户
        for user_id in user_profiles:
            # 添加用户属性
            profile = user_profiles.get(user_id, {})
            self.user_graph.add_node(
                user_id,
                type='user',
                department=profile.get('department', ''),
                role=profile.get('role', ''),
                is_manager=profile.get('is_manager', False)
            )

        # 确保有必要的列
        if 'from_user' not in email_data.columns or 'to' not in email_data.columns:
            logger.warning("Email data lacks required columns for graph building")
            return self.user_graph

        # 添加边 - 基于邮件通信
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
                if to_user in user_profiles and from_user in user_profiles:
                    # 如果边已存在，增加权重
                    if self.user_graph.has_edge(from_user, to_user):
                        self.user_graph[from_user][to_user]['weight'] += 1
                        self.user_graph[from_user][to_user]['emails'] += 1
                    else:
                        # 检查是否跨部门
                        from_dept = user_profiles[from_user].get('department', '')
                        to_dept = user_profiles[to_user].get('department', '')

                        self.user_graph.add_edge(
                            from_user,
                            to_user,
                            weight=1,
                            emails=1,
                            cross_department=(from_dept != to_dept)
                        )

        logger.info(
            f"Built user graph with {self.user_graph.number_of_nodes()} nodes and {self.user_graph.number_of_edges()} edges")
        return self.user_graph

    def get_graph(self):
        """获取用户关系图"""
        return self.user_graph