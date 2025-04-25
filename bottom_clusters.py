import heapq
import itertools
import numpy as np
import torch
import pandas as pd
from scipy.stats import norm
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class PriorityQueue:
    def __init__(self, query_workload, max_size=100000):
        self.queue = []
        self.query_workload = query_workload
        self.counter = itertools.count()
        self.max_size = max_size

    def enqueue(self, item, priority=None):
        if priority is None:
            priority = calculate_intersecting_queries(item, self.query_workload)
        if len(self.queue) < self.max_size:
            heapq.heappush(self.queue, (-priority, next(self.counter), item))
        else:
            heapq.heappushpop(self.queue, (-priority, next(self.counter), item))

    def dequeue(self):
        return heapq.heappop(self.queue)[-1]

    def is_empty(self):
        return len(self.queue) == 0


class LeafNode:
    """Leaf node for WISK tree containing objects and inverted file index"""

    def __init__(self, layer,mbr, objects, node_id=None):
        self.mbr = mbr  # {min_lat, max_lat, min_lon, max_lon}
        self.is_leaf = True
        self.objects = objects
        self.inverted_file = self._build_inverted_file()
        self.id = node_id
        self.labels = self.get_keywords()  # All keywords in this leaf node
        self.layer = 0

    def _build_inverted_file(self):
        """Build inverted file index for keywords in this leaf node"""
        inverted_file = defaultdict(list)
        for i, obj in enumerate(self.objects):
            for keyword in obj['keywords']:
                inverted_file[keyword].append(i)  # Store object index in the inverted file
        return dict(inverted_file)

    def get_objects_by_keyword(self, keyword):
        """Retrieve objects containing the given keyword"""
        if keyword not in self.inverted_file:
            return []
        return [self.objects[idx] for idx in self.inverted_file[keyword]]

    def get_keywords(self):
        """Return all keywords in this leaf node"""
        return set(self.inverted_file.keys())

    def to_dict(self):
        """Convert to dictionary format for compatibility with existing code"""
        return {
            'layer': 0,
            'MBR': self.mbr,
            'labels': list(self.get_keywords()),
            'children': [],
            'leaf_objects': self.objects,
            'inverted_file': self.inverted_file,
            'is_leaf': True,
            'id': self.id
        }

def calculate_intersecting_queries(subspace, query_workload):
    count = 0

    if isinstance(subspace, dict):
        if not subspace.get('labels'):
            print(f"子空间无关键词labels。")
            return 0
        subspace_keywords = subspace.get('labels', set())
        mbr = {
            'min_lat': subspace['min_lat'],
            'max_lat': subspace['max_lat'],
            'min_lon': subspace['min_lon'],
            'max_lon': subspace['max_lon']
        }
    elif isinstance(subspace, LeafNode):  # New LeafNode format
        subspace_keywords = subspace.get_keywords()
        mbr = subspace.mbr
    else:
        print(f"Unknown subspace type: {type(subspace)}")
        return 0

    for query in query_workload:
        # 检查空间交集
        query_mbr = query['area']
        spatial_intersect = (
                mbr['min_lat'] <= query_mbr['max_lat'] and
                mbr['max_lat'] >= query_mbr['min_lat'] and
                mbr['min_lon'] <= query_mbr['max_lon'] and
                mbr['max_lon'] >= query_mbr['min_lon']
        )
        # 检查关键词交集（使用集合操作）
        keyword_intersect = bool(set(query['keywords']) & subspace_keywords)

        if spatial_intersect and keyword_intersect:
            count += 1
    return count

def generate_subspace(subspace, dim, split_value):
    # 动态计算最小尺寸（避免绝对阈值）
    if dim == 0:
        range_size = subspace['max_lat'] - subspace['min_lat']
    else:
        range_size = subspace['max_lon'] - subspace['min_lon']
    dynamic_min_size = range_size * 0.05

    # 检查分割尺寸有效性
    if dim == 0:
        if (split_value - subspace['min_lat'] < dynamic_min_size) or \
                (subspace['max_lat'] - split_value < dynamic_min_size):
            print(f"reach min size")
            return None, None
    else:
        if (split_value - subspace['min_lon'] < dynamic_min_size) or \
                (subspace['max_lon'] - split_value < dynamic_min_size):
            print(f"reach min size")
            return None, None

    # 分割对象
    left_objects = []
    right_objects = []
    for obj in subspace['objects']:
        if (dim == 0 and obj['latitude'] < split_value) or \
                (dim == 1 and obj['longitude'] < split_value):
            left_objects.append(obj)
        else:
            right_objects.append(obj)

    def create_subspace(objects):
        """创建子空间并自动生成labels"""
        if not objects:
            return None

        # 收集所有唯一关键词
        keywords = set()
        for obj in objects:
            keywords.update(obj['keywords'])

        # 计算空间边界
        lats = [obj['latitude'] for obj in objects]
        lons = [obj['longitude'] for obj in objects]

        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons),
            'objects': objects,
            'labels': keywords
        }

    left_sub = create_subspace(left_objects)
    right_sub = create_subspace(right_objects)
    return left_sub, right_sub



# def SGDLearn(subspace, dim, query_workload, cdf_models, epochs=15, lr=0.01):
#     # 初始分割点
#     initial_value = subspace['min_lat'] + (subspace['max_lat'] - subspace['min_lat']) / 2 if dim == 0 \
#         else subspace['min_lon'] + (subspace['max_lon'] - subspace['min_lon']) / 2
#     split_value = torch.tensor([initial_value], dtype=torch.float32, requires_grad=True)
#     optimizer = torch.optim.Adam([split_value], lr=lr)
#
#     best_cost = float('inf')
#     best_split = initial_value
#     sigmoid = torch.sigmoid
#
#     if dim == 0:
#         subspace_below = subspace['min_lat']
#         subspace_up = subspace['max_lat']
#     else:
#         subspace_below = subspace['min_lon']
#         subspace_up = subspace['max_lon']
#
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         total_cost = torch.tensor(0.0, dtype=torch.float32)
#
#         # 计算总成本：遍历每个查询
#         for query in query_workload:
#             cost = torch.tensor(0.0, dtype=torch.float32)
#             # 根据分割维度获取查询对应的边界
#             if dim == 0:
#                 query_boundary_low = query['area']['min_lat']
#                 query_boundary_high = query['area']['max_lat']
#             else:
#                 query_boundary_low = query['area']['min_lon']
#                 query_boundary_high = query['area']['max_lon']
#
#             # 针对查询中的每个关键词计算成本
#             for kw in query['keywords']:
#                 if kw in cdf_models:
#                     # 根据中频或高频选择不同的 CDF 计算方式
#                     if cdf_models[kw]['gaussian']:
#                         # 中频关键词：使用高斯估计
#                         if dim == 0:
#                             mean = cdf_models[kw]['y']['y_mean']
#                             std = cdf_models[kw]['y']['y_std']
#                         else:
#                             mean = cdf_models[kw]['x']['x_mean']
#                             std = cdf_models[kw]['x']['x_std']
#                         #改成数据subspace的上下界(而非query的上下界)
#                         F_low = torch.tensor(norm.cdf(subspace_below, loc=mean, scale=std), dtype=torch.float32)
#                         F_split = torch.tensor(norm.cdf(split_value.item(), loc=mean, scale=std), dtype=torch.float32)
#                         F_high = torch.tensor(norm.cdf(subspace_up, loc=mean, scale=std), dtype=torch.float32)
#                     else:
#                         # 高频关键词：使用 NN 模型
#                         model = cdf_models[kw]['y'] if dim == 0 else cdf_models[kw]['x']
#                         # 改成subspace的上下界
#                         F_low = model(torch.tensor([[subspace_below]], dtype=torch.float32))
#                         F_split = model(split_value.view(1, 1))
#                         F_high = model(torch.tensor([[subspace_up]], dtype=torch.float32))
#
#
#                     total_kw = cdf_models[kw]['total_kw']
#                     # 估计子区域内包含该关键词的对象数量（根据cdf的单调递增性质，计算的一定是非负的）
#                     O1 = torch.clamp((F_split - F_low) * total_kw, min=0.0)
#                     O2 = torch.clamp((F_high - F_split) * total_kw, min=0.0)
#                     # 成本公式：左右两边使用 sigmoid 调整权重
#                     cost_kw = sigmoid(3 * (split_value - torch.tensor([[query_boundary_low]], dtype=torch.float32))) * O1 + \
#                               sigmoid(3 * (torch.tensor([[query_boundary_high]], dtype=torch.float32) - split_value)) * O2
#                     cost += cost_kw.squeeze()
#
#                     # # 输出调试信息
#                     # print(f"Query: {query}")
#                     # print(f"Keyword: {kw}，total_kw:{total_kw}")
#                     # print(f"boundary_low: {boundary_low}, boundary_high: {boundary_high}")
#                     # print(f"F_low: {F_low.item():.4f}, F_split: {F_split.item():.4f}, F_high: {F_high.item():.4f}")
#                     # print(f"O1: {O1.item():.4f}, O2: {O2.item():.4f}, cost_kw: {cost_kw.item():.4f}")
#                     # print("-" * 50)
#
#                 # else: # 低频关键词直接被忽略了
#
#
#             total_cost = total_cost + cost
#
#         # 反向传播前添加约束：限制 split_value 在合理范围内
#         if dim == 0:
#             split_value.data.clamp_(subspace['min_lat'], subspace['max_lat'])
#         else:
#             split_value.data.clamp_(subspace['min_lon'], subspace['max_lon'])
#         # print(f"total cost:{total_cost}")
#
#         total_cost.backward()
#         optimizer.step()
#
#         if total_cost.item() < best_cost:
#             best_cost = total_cost.item()
#             best_split = split_value.item()
#
#     return best_split, best_cost

def SGDLearn(subspace, dim, query_workload, cdf_models, epochs=15, lr=0.01):
    # 初始分割点
    initial_value = subspace['min_lat'] + (subspace['max_lat'] - subspace['min_lat']) / 2 if dim == 0 \
        else subspace['min_lon'] + (subspace['max_lon'] - subspace['min_lon']) / 2
    split_value = torch.tensor([initial_value], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([split_value], lr=lr)

    best_cost = float('inf')
    best_split = initial_value
    sigmoid = torch.sigmoid

    # 获取子空间的边界
    lat_min, lat_max = subspace['min_lat'], subspace['max_lat']
    lon_min, lon_max = subspace['min_lon'], subspace['max_lon']

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_cost = torch.tensor(0.0, dtype=torch.float32)

        # 计算总成本：遍历每个查询
        for query in query_workload:
            cost = torch.tensor(0.0, dtype=torch.float32)

            # 查询边界
            query_lat_low = query['area']['min_lat']
            query_lat_high = query['area']['max_lat']
            query_lon_low = query['area']['min_lon']
            query_lon_high = query['area']['max_lon']

            # 针对查询中的每个关键词计算成本
            for kw in query['keywords']:
                if kw in cdf_models:
                    # 获取两个维度的CDF值
                    if cdf_models[kw]['gaussian']:
                        # 中频关键词：使用高斯估计
                        # 纬度（y）的CDF值
                        lat_mean = cdf_models[kw]['y']['y_mean']
                        lat_std = cdf_models[kw]['y']['y_std']
                        F_lat_min = torch.tensor(norm.cdf(lat_min, loc=lat_mean, scale=lat_std), dtype=torch.float32)
                        F_lat_max = torch.tensor(norm.cdf(lat_max, loc=lat_mean, scale=lat_std), dtype=torch.float32)

                        # 经度（x）的CDF值
                        lon_mean = cdf_models[kw]['x']['x_mean']
                        lon_std = cdf_models[kw]['x']['x_std']
                        F_lon_min = torch.tensor(norm.cdf(lon_min, loc=lon_mean, scale=lon_std), dtype=torch.float32)
                        F_lon_max = torch.tensor(norm.cdf(lon_max, loc=lon_mean, scale=lon_std), dtype=torch.float32)

                        # 分割点的CDF值
                        if dim == 0:  # 分割纬度
                            F_split = torch.tensor(norm.cdf(split_value.item(), loc=lat_mean, scale=lat_std),dtype=torch.float32)
                        else:  # 分割经度
                            F_split = torch.tensor(norm.cdf(split_value.item(), loc=lon_mean, scale=lon_std),dtype=torch.float32)
                    else:
                        # 高频关键词：使用NN模型
                        lat_model = cdf_models[kw]['y']
                        lon_model = cdf_models[kw]['x']

                        # 纬度的CDF值
                        F_lat_min = lat_model(torch.tensor([[lat_min]], dtype=torch.float32))
                        F_lat_max = lat_model(torch.tensor([[lat_max]], dtype=torch.float32))

                        # 经度的CDF值
                        F_lon_min = lon_model(torch.tensor([[lon_min]], dtype=torch.float32))
                        F_lon_max = lon_model(torch.tensor([[lon_max]], dtype=torch.float32))

                        # 分割点的CDF值
                        if dim == 0:  # 分割纬度
                            F_split = lat_model(split_value.view(1, 1))
                        else:  # 分割经度
                            F_split = lon_model(split_value.view(1, 1))

                    total_kw = cdf_models[kw]['total_kw']

                    # 计算分割后两个子空间内的对象数量（考虑联合概率）
                    if dim == 0:  # 分割纬度
                        # 左侧子空间：[lat_min, split_value] x [lon_min, lon_max]
                        O1 = torch.clamp((F_split - F_lat_min) * (F_lon_max - F_lon_min) * total_kw, min=0.0)
                        # 右侧子空间：[split_value, lat_max] x [lon_min, lon_max]
                        O2 = torch.clamp((F_lat_max - F_split) * (F_lon_max - F_lon_min) * total_kw, min=0.0)
                    else:  # 分割经度
                        # 左侧子空间：[lat_min, lat_max] x [lon_min, split_value]
                        O1 = torch.clamp((F_lat_max - F_lat_min) * (F_split - F_lon_min) * total_kw, min=0.0)
                        # 右侧子空间：[lat_min, lat_max] x [split_value, lon_max]
                        O2 = torch.clamp((F_lat_max - F_lat_min) * (F_lon_max - F_split) * total_kw, min=0.0)

                    # 成本公式：根据查询和分割的关系调整权重
                    if dim == 0:  # 分割纬度
                        cost_kw = sigmoid(3 * (split_value - torch.tensor([[query_lat_low]], dtype=torch.float32))) * O1 + \
                                  sigmoid(3 * (torch.tensor([[query_lat_high]], dtype=torch.float32) - split_value)) * O2
                    else:  # 分割经度
                        cost_kw = sigmoid(3 * (split_value - torch.tensor([[query_lon_low]], dtype=torch.float32))) * O1 + \
                                  sigmoid(3 * (torch.tensor([[query_lon_high]], dtype=torch.float32) - split_value)) * O2

                    cost += cost_kw.squeeze()

            total_cost = total_cost + cost

        # 反向传播前添加约束：限制 split_value 在合理范围内
        if dim == 0:
            split_value.data.clamp_(subspace['min_lat'], subspace['max_lat'])
        else:
            split_value.data.clamp_(subspace['min_lon'], subspace['max_lon'])

        total_cost.backward()
        optimizer.step()

        if total_cost.item() < best_cost:
            best_cost = total_cost.item()
            best_split = split_value.item()

    return best_split, best_cost

def find_optimal_partition_sample(subspace, dim, query_workload, cdf_models):
    min_cost = float('inf')
    best_split_value = None

    objects = subspace['objects']
    values = [obj['latitude'] for obj in objects] if dim == 0 else [obj['longitude'] for obj in objects]
    sorted_values = np.sort(values)
    sampled_values = sorted_values[::10]

    for value in sampled_values:
        # 生成子空间并检查有效性
        left_subspace, right_subspace = generate_subspace(subspace, dim, value)

        # 跳过无效分割
        if left_subspace is None or right_subspace is None:
            continue

        # 跳过空子空间
        if len(left_subspace['objects']) == 0 or len(right_subspace['objects']) == 0:
            continue

        # 计算总成本：遍历每个查询
        total_cost = 0
        for query in query_workload:
            cost = 0
            # 根据分割维度获取查询对应的边界
            if dim == 0:
                boundary_low = query['area']['min_lat']
                boundary_high = query['area']['max_lat']
            else:
                boundary_low = query['area']['min_lon']
                boundary_high = query['area']['max_lon']

            # 针对查询中的每个关键词计算成本
            for kw in query['keywords']:
                if kw in cdf_models:
                    # 根据中频或高频选择不同的 CDF 计算方式
                    if cdf_models[kw]['gaussian']:
                        # 中频关键词：使用高斯估计
                        if dim == 0:
                            mean = cdf_models[kw]['y_mean']
                            std = cdf_models[kw]['y_std']
                        else:
                            mean = cdf_models[kw]['x_mean']
                            std = cdf_models[kw]['x_std']

                        F_low = torch.tensor(norm.cdf(boundary_low, loc=mean, scale=std), dtype=torch.float32)
                        F_split = torch.tensor(norm.cdf(value.item(), loc=mean, scale=std), dtype=torch.float32)
                        F_high = torch.tensor(norm.cdf(boundary_high, loc=mean, scale=std), dtype=torch.float32)
                    else:
                        # 高频关键词：使用 NN 模型
                        model = cdf_models[kw]['y'] if dim == 0 else cdf_models[kw]['x']
                        F_low = model(torch.tensor([[boundary_low]], dtype=torch.float32))
                        F_split = model(value.view(1, 1))
                        F_high = model(torch.tensor([[boundary_high]], dtype=torch.float32))

                    total_kw = cdf_models[kw]['total_kw']
                    # 估计子区域内包含该关键词的对象数量(由于split有可能不在low和high区域内，故取非负值)
                    O1 = torch.clamp((F_split - F_low) * total_kw, min=0.0)
                    O2 = torch.clamp((F_high - F_split) * total_kw, min=0.0)
                    # 成本公式：左右两边使用 sigmoid 调整权重
                    cost_kw = torch.sigmoid(3 * (value - torch.tensor([[boundary_low]], dtype=torch.float32))) * O1 + \
                                  torch.sigmoid(3 * (torch.tensor([[boundary_high]], dtype=torch.float32) - value)) * O2
                    cost += cost_kw.squeeze()
            total_cost = total_cost + cost

        # 更新最优值
        if total_cost < min_cost:
            min_cost = total_cost
            best_split_value = value
    return {'dim': dim, 'cost': min_cost, 'val': best_split_value}


def find_optimal_partition(subspace, dim, query_workload, cdf_models):
    optimal_val, predicted_cost = SGDLearn(subspace, dim, query_workload, cdf_models, epochs=15, lr=0.01)
    # s1, s2 = generate_subspace(subspace, dim, optimal_val)
    # if s1 and s2:
    #     num_queries_s1 = calculate_intersecting_queries(s1, query_workload)
    #     num_queries_s2 = calculate_intersecting_queries(s2, query_workload)
    #     C_split = len(s1['objects']) * num_queries_s1 + len(s2['objects']) * num_queries_s2
    # else:
    #     C_split = float('inf')
    return {'dim': dim, 'cost': predicted_cost, 'val': optimal_val}


# New function to calculate cost using CDF models
def calculate_cost_with_cdf(subspace, query, cdf_models):
    """
    Calculate query evaluation cost using CDF models directly
    """
    cost = 0.0

    # Get spatial boundaries
    if isinstance(subspace, LeafNode):
        mbr = subspace.mbr
    else:
        mbr = {
            'min_lat': subspace['min_lat'],
            'max_lat': subspace['max_lat'],
            'min_lon': subspace['min_lon'],
            'max_lon': subspace['max_lon']
        }

    # Extract query boundaries
    query_mbr = query['area']

    # Check for spatial overlap
    spatial_intersect = (
            mbr['min_lat'] <= query_mbr['max_lat'] and
            mbr['max_lat'] >= query_mbr['min_lat'] and
            mbr['min_lon'] <= query_mbr['max_lon'] and
            mbr['max_lon'] >= query_mbr['min_lon']
    )

    if not spatial_intersect:
        return 0.0

    # For each keyword in the query, estimate cost using CDF models
    for kw in query['keywords']:
        if kw in cdf_models:
            # Latitude (y) dimension
            if cdf_models[kw]['gaussian']:
                # Using Gaussian estimation
                lat_mean = cdf_models[kw]['y']['y_mean']
                lat_std = cdf_models[kw]['y']['y_std']
                lon_mean = cdf_models[kw]['x']['x_mean']
                lon_std = cdf_models[kw]['x']['x_std']

                # Calculate CDF values for latitude
                F_lat_low = norm.cdf(mbr['min_lat'], loc=lat_mean, scale=lat_std)
                F_lat_high = norm.cdf(mbr['max_lat'], loc=lat_mean, scale=lat_std)

                # Calculate CDF values for longitude
                F_lon_low = norm.cdf(mbr['min_lon'], loc=lon_mean, scale=lon_std)
                F_lon_high = norm.cdf(mbr['max_lon'], loc=lon_mean, scale=lon_std)
            else:
                # Using neural network model
                lat_model = cdf_models[kw]['y']
                lon_model = cdf_models[kw]['x']

                # Calculate CDF values for latitude
                F_lat_low = lat_model(
                    torch.tensor([mbr['min_lat']], dtype=torch.float32)).item()
                F_lat_high = lat_model(
                    torch.tensor([mbr['max_lat']], dtype=torch.float32)).item()

                # Calculate CDF values for longitude
                F_lon_low = lon_model(
                    torch.tensor([mbr['min_lon']], dtype=torch.float32)).item()
                F_lon_high = lon_model(
                    torch.tensor([mbr['max_lon']], dtype=torch.float32)).item()

            # Calculate expected number of objects in intersection area (joint probability)
            total_kw = cdf_models[kw]['total_kw']
            objects_in_intersection = (F_lat_high - F_lat_low) * (F_lon_high - F_lon_low) * total_kw

            # Clamp to non-negative values
            objects_in_intersection = max(0.0, objects_in_intersection)

            # Add to total cost
            cost += objects_in_intersection

    return cost

def count_matching_objects(subspace, query):
    """
    Count objects in subspace that match any query keywords
    """
    # if isinstance(subspace, LeafNode):
    #     return subspace.get_matching_objects_count(query['keywords'])
    #
    # # For dictionary-style subspace
    matching_count = 0
    query_keywords = set(query['keywords'])
    #
    # # If inverted file already exists in dictionary
    # if 'inverted_file' in subspace:
    #     matching_indices = set()
    #     for keyword in query_keywords:
    #         if keyword in subspace['inverted_file']:
    #             matching_indices.update(subspace['inverted_file'][keyword])
    #     return len(matching_indices)

    # Otherwise, count manually
    for obj in subspace['objects']:
        if any(kw in query_keywords for kw in obj['keywords']):
            matching_count += 1
    return matching_count

def bottom_clusters_generation(query_workload, data_space, cdf_models, w1=0.1, w2=1, MIN_OBJECTS=10):
    # 确保query_workload是字典列表格式
    if isinstance(query_workload, pd.DataFrame):
        query_workload = query_workload.to_dict('records')

    # 初始化优先队列
    Q = PriorityQueue(query_workload)
    G = []  # 存储最终生成的聚类
    node_id_counter = 0

    # 将整个数据空间添加到队列
    for item in data_space:
        Q.enqueue(item)

    while not Q.is_empty():
        s = Q.dequeue()
        print(f"处理子空间 MBR: [lat: {s['min_lat']:.6f}, {s['max_lat']:.6f}; "
              f"lon: {s['min_lon']:.6f}, {s['max_lon']:.6f}], "
              f"objects: {len(s['objects'])},"
              f"labels:{len(s['labels'])}")

        num_objects = len(s['objects'])
        # 对象数过少，直接加入结果
        if len(s['objects']) < MIN_OBJECTS:
            # 创建叶子节点并添加到结果
            mbr = {
                'min_lat': s['min_lat'],
                'max_lat': s['max_lat'],
                'min_lon': s['min_lon'],
                'max_lon': s['max_lon']
            }
            leaf_node = LeafNode(0,mbr, s['objects'], node_id=node_id_counter)
            node_id_counter += 1
            G.append(leaf_node)
            continue

        # num_queries = calculate_intersecting_queries(s, query_workload) #不仅mbr要交，keyword也得匹配
        # Cs = num_objects * num_queries   # 空间里全部对象均要检验

        Cs = 0.0
        for query in query_workload:
            query_cost = calculate_cost_with_cdf(s, query, cdf_models)
            Cs += query_cost

        # 寻找最优分割
        opt_x = find_optimal_partition(s, dim=0, query_workload=query_workload, cdf_models=cdf_models)
        opt_y = find_optimal_partition(s, dim=1, query_workload=query_workload, cdf_models=cdf_models)

        # 选择成本更低的分割方式
        best = opt_x if opt_x['cost'] <= opt_y['cost'] else opt_y

        s1, s2 = generate_subspace(s, dim=best['dim'], split_value=best['val'])
        if s1 and s2:
            # num_queries_s1 = calculate_intersecting_queries(s1, query_workload)
            # num_queries_s2 = calculate_intersecting_queries(s2, query_workload)
            # C_split = (len(s1['objects']) * num_queries_s1 + len(s2['objects']) * num_queries_s2) * w2
            print(f"分割前对象验证成本: {Cs}, 分割后对象验证成本: {best['cost']}, 增加簇扫描成本: {w1 * len(query_workload)}")
            # 分割条件：(Cs + w1 * 当前集群数 * | W |) - (C_split + w1 * (当前集群数 + 1) * | W |) > 0
            if (Cs - best['cost']) * w2 > (w1 * len(query_workload)):
                # 允许分割并加入队列
                print(f"分割前Cs: {Cs}")
                print(f"分割后对象数: {len(s1['objects'])} + {len(s2['objects'])} = {len(s1['objects']) + len(s2['objects'])}, C_split: {best['cost']}")
                Q.enqueue(s1)
                Q.enqueue(s2)
                print(f"成本缩小，可以分割")
            else:
                # 创建叶子节点并添加到结果
                mbr = {
                    'min_lat': s['min_lat'],
                    'max_lat': s['max_lat'],
                    'min_lon': s['min_lon'],
                    'max_lon': s['max_lon']
                }
                leaf_node = LeafNode(0,mbr, s['objects'], node_id=node_id_counter)
                node_id_counter += 1
                G.append(leaf_node)
        else:
            # 创建叶子节点并添加到结果
            mbr = {
                'min_lat': s['min_lat'],
                'max_lat': s['max_lat'],
                'min_lon': s['min_lon'],
                'max_lon': s['max_lon']
            }
            leaf_node = LeafNode(0,mbr, s['objects'], node_id=node_id_counter)
            node_id_counter += 1
            G.append(leaf_node)

    print(f"已生成聚类数: {len(G)}")
    # 转换为字典格式以兼容当前代码
    bottom_nodes = [leaf_node.to_dict() for leaf_node in G]
    return bottom_nodes


def visualize_clusters(bottom_nodes, original_data, output_file=None, max_points=1000, figsize=(12, 10)):
    """
    Visualize the CDF-based clusters and the original data points.

    Parameters:
    -----------
    bottom_nodes : list
        List of leaf nodes (clusters) generated by bottom_clusters_generation
    original_data : list
        Original data points used for clustering
    output_file : str, optional
        Path to save the visualization. If None, the plot is displayed.
    max_points : int, optional
        Maximum number of original points to display (to prevent overcrowding)
    figsize : tuple, optional
        Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)

    # Create a colormap with distinct colors for different clusters
    colors = plt.cm.tab20(np.linspace(0, 1, len(bottom_nodes)))

    # Sample original data points if there are too many
    if len(original_data) > max_points:
        sampled_data = random.sample(original_data, max_points)
    else:
        sampled_data = original_data

    # Plot original data points (smaller and gray)
    lats = [obj['latitude'] for obj in sampled_data]
    lons = [obj['longitude'] for obj in sampled_data]
    plt.scatter(lons, lats, s=10, color='gray', alpha=0.3, label='Original Data')

    # Plot clusters and their boundaries
    for i, node in enumerate(bottom_nodes):
        # Extract MBR information
        mbr = node['MBR']
        min_lat, max_lat = mbr['min_lat'], mbr['max_lat']
        min_lon, max_lon = mbr['min_lon'], mbr['max_lon']

        # Create a rectangle for the cluster boundary
        rect = patches.Rectangle(
            (min_lon, min_lat),
            max_lon - min_lon,
            max_lat - min_lat,
            linewidth=2,
            edgecolor=colors[i],
            facecolor=colors[i],
            alpha=0.2
        )
        plt.gca().add_patch(rect)

        # If the node has leaf objects, plot their centers with the same color
        if node.get('leaf_objects'):
            cluster_objects = node['leaf_objects']
            cluster_lats = [obj['latitude'] for obj in cluster_objects]
            cluster_lons = [obj['longitude'] for obj in cluster_objects]

            # Calculate the center of the cluster
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2

            # Plot the cluster center
            plt.scatter(center_lon, center_lat, s=100, color=colors[i],
                        marker='x', linewidth=2, label=f'Cluster {i} (n={len(cluster_objects)})')

    # Plot settings
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Spatial Clustering Results ({len(bottom_nodes)} clusters)')

    # Create a custom legend that doesn't show every cluster
    if len(bottom_nodes) > 10:
        # Create a legend with just a few example clusters
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, alpha=0.3,
                       label='Original Data'),
            plt.Line2D([0], [0], marker='x', color=colors[0], markersize=8,
                       label=f'Cluster Centers ({len(bottom_nodes)} total)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    else:
        # Legend might be too big, show outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()


