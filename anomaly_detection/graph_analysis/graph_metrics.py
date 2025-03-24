import networkx as nx
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class GraphMetricsCalculator:
    """计算图分析相关指标"""

    def __init__(self, graph=None):
        """初始化指标计算器

        Args:
            graph: NetworkX图对象，可选
        """
        self.graph = graph or nx.Graph()
        self.metrics = {}

    def set_graph(self, graph):
        """设置要分析的图

        Args:
            graph: NetworkX图对象
        """
        self.graph = graph
        return self

    def calculate_metrics(self, sample_size=50):
        """计算所有图指标

        Args:
            sample_size: 用于计算高计算成本指标的样本大小

        Returns:
            dict: 包含所有计算的指标
        """
        if not self.graph or not self.graph.nodes:
            logger.warning("Graph is empty, cannot calculate metrics")
            return {}

        metrics = {}

        try:
            # 度中心性 - 衡量节点的连接程度
            metrics['degree_centrality'] = nx.degree_centrality(self.graph)

            # 接近中心性 - 衡量节点到其他节点的平均最短路径长度
            metrics['closeness_centrality'] = nx.closeness_centrality(self.graph)

            # 介数中心性 - 衡量节点位于网络中其他节点之间最短路径上的频率
            # 这是计算成本高的指标，对大型图可能需要采样
            if len(self.graph) <= 100:
                metrics['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
            else:
                metrics['betweenness_centrality'] = nx.betweenness_centrality(
                    self.graph,
                    k=min(sample_size, len(self.graph))
                )

            # 特征向量中心性 - 考虑节点连接的节点重要性
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(
                self.graph,
                max_iter=100,
                tol=1e-6
            )

            # 三角形计数 - 检测潜在的信息共享集团
            metrics['triangles'] = nx.triangles(self.graph)

            # 聚类系数 - 衡量节点的邻居之间的连接程度
            metrics['clustering'] = nx.clustering(self.graph)

            # 社区检测 - 识别紧密连接的组
            try:
                metrics['communities'] = list(nx.community.greedy_modularity_communities(self.graph))
            except:
                # 对于不连通的图，社区检测可能失败
                metrics['communities'] = []

            logger.info(f"Successfully calculated graph metrics for {len(self.graph)} nodes")

        except Exception as e:
            logger.error(f"Error calculating graph metrics: {str(e)}")
            logger.debug("Error details:", exc_info=True)

        self.metrics = metrics
        return metrics

    def detect_anomalies(self, threshold_percentile=95):
        """基于图指标检测异常节点

        Args:
            threshold_percentile: 异常阈值百分位数

        Returns:
            dict: 节点ID到异常分数的映射
        """
        if not self.metrics:
            self.calculate_metrics()

        if not self.metrics:
            return {}

        anomaly_scores = {}

        # 对每个节点，计算综合异常分数
        for node in self.graph.nodes:
            # 获取各项指标分数
            degree = self.metrics['degree_centrality'].get(node, 0)
            betweenness = self.metrics['betweenness_centrality'].get(node, 0)
            eigenvector = self.metrics['eigenvector_centrality'].get(node, 0)
            triangles = self.metrics['triangles'].get(node, 0)

            # 计算异常分数 - 可以根据业务需求调整权重
            score = (
                            0.3 * degree +
                            0.3 * betweenness +
                            0.3 * eigenvector +
                            0.1 * (triangles / (max(self.metrics['triangles'].values()) or 1))
                    ) * 100

            anomaly_scores[node] = score

        # 计算分数阈值
        if anomaly_scores:
            threshold = np.percentile(list(anomaly_scores.values()), threshold_percentile)
        else:
            threshold = 0

        # 筛选异常节点
        anomalies = {
            node: score for node, score in anomaly_scores.items()
            if score > threshold
        }

        return anomalies