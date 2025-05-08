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

def generate_subspace(subspace, dim, split_value, recalculate_mbr=True):
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

    # def create_subspace(objects): #收紧mbr
    #     """创建子空间并自动生成labels"""
    #     if not objects:
    #         return None
    #
    #     # 收集所有唯一关键词
    #     keywords = set()
    #     for obj in objects:
    #         keywords.update(obj['keywords'])
    #
    #     # 计算空间边界
    #     lats = [obj['latitude'] for obj in objects]
    #     lons = [obj['longitude'] for obj in objects]
    #
    #     return {
    #         'min_lat': min(lats),
    #         'max_lat': max(lats),
    #         'min_lon': min(lons),
    #         'max_lon': max(lons),
    #         'objects': objects,
    #         'labels': keywords
    #     }
    #
    # left_sub = create_subspace(left_objects)
    # right_sub = create_subspace(right_objects)
    def create_subspace(objects, mbr):
        """创建子空间并自动生成labels"""
        if not objects:
            return None

        # 收集所有唯一关键词
        keywords = set()
        for obj in objects:
            keywords.update(obj['keywords'])

        return {
            'min_lat': mbr['min_lat'],
            'max_lat': mbr['max_lat'],
            'min_lon': mbr['min_lon'],
            'max_lon': mbr['max_lon'],
            'objects': objects,
            'labels': keywords
        }

    # 根据分割维度和分割值确定左右子空间的MBR
    if dim == 0:  # 纬度分割
        left_mbr = {
            'min_lat': subspace['min_lat'],
            'max_lat': split_value,
            'min_lon': subspace['min_lon'],
            'max_lon': subspace['max_lon']
        }
        right_mbr = {
            'min_lat': split_value,
            'max_lat': subspace['max_lat'],
            'min_lon': subspace['min_lon'],
            'max_lon': subspace['max_lon']
        }
    else:  # 经度分割
        left_mbr = {
            'min_lat': subspace['min_lat'],
            'max_lat': subspace['max_lat'],
            'min_lon': subspace['min_lon'],
            'max_lon': split_value
        }
        right_mbr = {
            'min_lat': subspace['min_lat'],
            'max_lat': subspace['max_lat'],
            'min_lon': split_value,
            'max_lon': subspace['max_lon']
        }

    left_sub = create_subspace(left_objects, left_mbr)
    right_sub = create_subspace(right_objects, right_mbr)

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

# def SGDLearn(subspace, dim, query_workload, cdf_models, epochs=10, lr=0.005):
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
#     # 获取子空间的边界
#     lat_min, lat_max = subspace['min_lat'], subspace['max_lat']
#     lon_min, lon_max = subspace['min_lon'], subspace['max_lon']
#
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         total_cost = torch.tensor(0.0, dtype=torch.float32)
#         O1_TEST_sum = 0
#         O2_TEST_sum = 0
#         # 计算总成本：遍历每个查询
#         for query in query_workload:
#             cost = torch.tensor(0.0, dtype=torch.float32)
#             O1_TEST = 0
#             O2_TEST = 0
#             # 查询边界
#             query_lat_low = query['area']['min_lat']
#             query_lat_high = query['area']['max_lat']
#             query_lon_low = query['area']['min_lon']
#             query_lon_high = query['area']['max_lon']
#
#
#             spatial_intersect = (
#                     lat_min <= query_lat_high and
#                     lat_max >= query_lat_low and
#                     lon_min <= query_lon_high and
#                     lon_max >= query_lon_low
#             )
#             # 早停：如果不相交，直接跳过此查询
#             if not spatial_intersect:
#                 continue
#
#             # 针对查询中的每个关键词计算成本，但是会多算
#             for kw in query['keywords']:
#                 if kw in cdf_models:
#                     # 获取两个维度的CDF值
#                     if cdf_models[kw]['gaussian']:
#                         # 中频关键词：使用高斯估计
#                         # 纬度（y）的CDF值
#                         lat_mean = cdf_models[kw]['y']['mean']
#                         lat_std = cdf_models[kw]['y']['std']
#                         F_lat_min = torch.tensor(norm.cdf(lat_min, loc=lat_mean, scale=lat_std), dtype=torch.float32)
#                         F_lat_max = torch.tensor(norm.cdf(lat_max, loc=lat_mean, scale=lat_std), dtype=torch.float32)
#
#                         # 经度（x）的CDF值
#                         lon_mean = cdf_models[kw]['x']['mean']
#                         lon_std = cdf_models[kw]['x']['std']
#                         F_lon_min = torch.tensor(norm.cdf(lon_min, loc=lon_mean, scale=lon_std), dtype=torch.float32)
#                         F_lon_max = torch.tensor(norm.cdf(lon_max, loc=lon_mean, scale=lon_std), dtype=torch.float32)
#
#                         # 分割点的CDF值
#                         if dim == 0:  # 分割纬度
#                             F_split = torch.tensor(norm.cdf(split_value.item(), loc=lat_mean, scale=lat_std),dtype=torch.float32)
#                         else:  # 分割经度
#                             F_split = torch.tensor(norm.cdf(split_value.item(), loc=lon_mean, scale=lon_std),dtype=torch.float32)
#                     else:
#                         # 高频关键词：使用NN模型
#                         lat_model = cdf_models[kw]['y']
#                         lon_model = cdf_models[kw]['x']
#
#                         # 纬度的CDF值
#                         F_lat_min = lat_model(torch.tensor([[lat_min]], dtype=torch.float32))
#                         F_lat_max = lat_model(torch.tensor([[lat_max]], dtype=torch.float32))
#
#                         # 经度的CDF值
#                         F_lon_min = lon_model(torch.tensor([[lon_min]], dtype=torch.float32))
#                         F_lon_max = lon_model(torch.tensor([[lon_max]], dtype=torch.float32))
#
#                         # 分割点的CDF值
#                         if dim == 0:  # 分割纬度
#                             F_split = lat_model(split_value.view(1, 1))
#                         else:  # 分割经度
#                             F_split = lon_model(split_value.view(1, 1))
#
#                     total_kw = cdf_models[kw]['total_kw']
#
#                     # 计算分割后两个子空间内的对象数量（考虑联合概率）
#                     if dim == 0:  # 分割纬度
#                         # 左侧子空间：[lat_min, split_value] x [lon_min, lon_max]
#                         O1 = torch.clamp((F_split - F_lat_min) * (F_lon_max - F_lon_min) * total_kw, min=0.0)
#                         # 右侧子空间：[split_value, lat_max] x [lon_min, lon_max]
#                         O2 = torch.clamp((F_lat_max - F_split) * (F_lon_max - F_lon_min) * total_kw, min=0.0)
#                     else:  # 分割经度
#                         # 左侧子空间：[lat_min, lat_max] x [lon_min, split_value]
#                         O1 = torch.clamp((F_lat_max - F_lat_min) * (F_split - F_lon_min) * total_kw, min=0.0)
#                         # 右侧子空间：[lat_min, lat_max] x [split_value, lon_max]
#                         O2 = torch.clamp((F_lat_max - F_lat_min) * (F_lon_max - F_split) * total_kw, min=0.0)
#
#                     # # 修改：计算2D空间相交性的sigmoid值, 4个sigmoid相乘会增大误差
#                     # if dim == 0:  # 分割纬度
#                     #     # 左子空间相交条件:
#                     #     # 1. split_value > query_lat_min (左子空间的右边缘 > 查询的左边缘)
#                     #     # 2. lat_min < query_lat_max (左子空间的左边缘 < 查询的右边缘)
#                     #     # 3. lon_max > query_lon_min (子空间的右边缘 > 查询的左边缘)
#                     #     # 4. lon_min < query_lon_max (子空间的左边缘 < 查询的右边缘)
#                     #     intersect_left = sigmoid(3 * (split_value - torch.tensor(query_lat_low, dtype=torch.float32))) * \
#                     #                      sigmoid(3 * (torch.tensor(query_lat_high, dtype=torch.float32) - torch.tensor(lat_min, dtype=torch.float32))) * \
#                     #                      sigmoid(3 * (torch.tensor(lon_max, dtype=torch.float32) - torch.tensor(query_lon_low, dtype=torch.float32))) * \
#                     #                      sigmoid(3 * (torch.tensor(query_lon_high, dtype=torch.float32) - torch.tensor(lon_min, dtype=torch.float32)))
#                     #
#                     #     # 正确的右子空间相交测试
#                     #     # 右子空间相交条件:
#                     #     # 1. lat_max > query_lat_min (右子空间的右边缘 > 查询的左边缘)
#                     #     # 2. split_value < query_lat_max (右子空间的左边缘 < 查询的右边缘)
#                     #     # 3. lon_max > query_lon_min (子空间的右边缘 > 查询的左边缘)
#                     #     # 4. lon_min < query_lon_max (子空间的左边缘 < 查询的右边缘)
#                     #     intersect_right = sigmoid(3 * (torch.tensor(lat_max, dtype=torch.float32) - torch.tensor(query_lat_low,dtype=torch.float32))) * \
#                     #                       sigmoid(3 * (torch.tensor(query_lat_high, dtype=torch.float32) - split_value)) * \
#                     #                       sigmoid(3 * (torch.tensor(lon_max, dtype=torch.float32) - torch.tensor(query_lon_low, dtype=torch.float32))) * \
#                     #                       sigmoid(3 * (torch.tensor(query_lon_high, dtype=torch.float32) - torch.tensor(lon_min, dtype=torch.float32)))
#                     #
#                     # else:  # 分割经度
#                     #
#                     #     intersect_left = sigmoid(3 * (torch.tensor(lat_max, dtype=torch.float32) - torch.tensor(query_lat_low,dtype=torch.float32))) * \
#                     #                      sigmoid(3 * (torch.tensor(query_lat_high, dtype=torch.float32) - torch.tensor(lat_min, dtype=torch.float32))) * \
#                     #                      sigmoid(3 * (split_value - torch.tensor(query_lon_low, dtype=torch.float32))) * \
#                     #                      sigmoid(3 * (torch.tensor(query_lon_high, dtype=torch.float32) - torch.tensor(lon_min, dtype=torch.float32)))
#                     #
#                     #
#                     #     intersect_right = sigmoid(3 * (torch.tensor(lat_max, dtype=torch.float32) - torch.tensor(query_lat_low,dtype=torch.float32))) * \
#                     #                       sigmoid(3 * (torch.tensor(query_lat_high, dtype=torch.float32) - torch.tensor(lat_min, dtype=torch.float32))) * \
#                     #                       sigmoid(3 * (torch.tensor(lon_max, dtype=torch.float32) - torch.tensor(query_lon_low, dtype=torch.float32))) * \
#                     #                       sigmoid(3 * (torch.tensor(query_lon_high, dtype=torch.float32) - split_value))
#
#                     # 修改2：预先计算非分割维度的相交，并作为常量处理
#                     # 修改3：检查空间相交，相交再用一个维度分割判断两个子空间是否相交
#
#
#                     if dim == 0:  # 分割纬度
#                         # 预计算经度维度相交（不参与梯度计算）
#                         # with torch.no_grad():
#                             # lon_diff1 = torch.tensor(lon_max, dtype=torch.float32) - torch.tensor(query_lon_low,dtype=torch.float32)
#                             # lon_diff2 = torch.tensor(query_lon_high, dtype=torch.float32) - torch.tensor(lon_min,dtype=torch.float32)
#                             # non_split_intersect = sigmoid(3 * lon_diff1) * sigmoid(3 * lon_diff2)
#                             # 直接判断两个矩形在经度维度是否相交
#                             # lon_intersect = torch.tensor(1.0, dtype=torch.float32) if (lon_max > query_lon_low and lon_min < query_lon_high) else torch.tensor(0.0,dtype=torch.float32)
#                         # non_split_intersect = spatial_intersect
#
#                         # 只对split_value相关的计算保留梯度
#                         intersect_left =  sigmoid(3 * (split_value - torch.tensor(query_lat_low, dtype=torch.float32)))
#                         intersect_right = sigmoid(3 * (torch.tensor(query_lat_high, dtype=torch.float32) - split_value))
#
#                     else:  # 分割经度
#                         # 预计算纬度维度相交（不参与梯度计算）
#                         # with torch.no_grad():
#                         #     # lat_diff1 = torch.tensor(lat_max, dtype=torch.float32) - torch.tensor(query_lat_low,dtype=torch.float32)
#                         #     # lat_diff2 = torch.tensor(query_lat_high, dtype=torch.float32) - torch.tensor(lat_min,dtype=torch.float32)
#                         #     # non_split_intersect = sigmoid(3 * lat_diff1) * sigmoid(3 * lat_diff2)
#                         #     # 直接判断两个矩形在纬度维度是否相交
#                         #     lat_intersect = torch.tensor(1.0, dtype=torch.float32) if (lat_max > query_lat_low and lat_min < query_lat_high) else torch.tensor(0.0,dtype=torch.float32)
#                         # non_split_intersect = spatial_intersect
#
#                         # 只对split_value相关的计算保留梯度
#                         intersect_left =  sigmoid(3 * (split_value - torch.tensor(query_lon_low, dtype=torch.float32)))
#                         intersect_right =  sigmoid(3 * (torch.tensor(query_lon_high, dtype=torch.float32) - split_value))
#
#                     # 计算加权成本
#                     cost_kw = intersect_left * O1 + intersect_right * O2
#                     cost += cost_kw.squeeze()
#                     O1_TEST += (intersect_left * O1).item()
#                     O2_TEST += (intersect_right * O2).item()
#
#             total_cost = total_cost + cost
#             O1_TEST_sum += O1_TEST
#             O2_TEST_sum += O2_TEST
#         #print(f" epoch {epoch}:{total_cost}")
#         # print(f"SGD 01:{O1_TEST_sum},SGD 02:{O2_TEST_sum}")
#
#         # 反向传播前添加约束：限制 split_value 在合理范围内
#         if dim == 0:
#             split_value.data.clamp_(subspace['min_lat'], subspace['max_lat'])
#         else:
#             split_value.data.clamp_(subspace['min_lon'], subspace['max_lon'])
#
#         total_cost.backward()
#         torch.nn.utils.clip_grad_norm_([split_value], max_norm=0.5)  # 限制梯度范围
#         optimizer.step()
#
#         if total_cost.item() < best_cost:
#             best_cost = total_cost.item()
#             best_split = split_value.item()
#
#     return best_split, best_cost

def SGDLearn(subspace, dim, query_workload, cdf_models, epochs=10, lr=0.005):
    # 在范围中点初始化分割值
    if dim == 0:
        initial_value = subspace['min_lat'] + (subspace['max_lat'] - subspace['min_lat']) / 2
        lower_bound = subspace['min_lat']
        upper_bound = subspace['max_lat']
    else:
        initial_value = subspace['min_lon'] + (subspace['max_lon'] - subspace['min_lon']) / 2
        lower_bound = subspace['min_lon']
        upper_bound = subspace['max_lon']

    # 创建带梯度跟踪的张量
    split_value = torch.tensor([initial_value], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([split_value], lr=lr)

    best_cost = float('inf')
    best_split = initial_value

    # 获取子空间边界作为参考
    lat_min, lat_max = subspace['min_lat'], subspace['max_lat']
    lon_min, lon_max = subspace['min_lon'], subspace['max_lon']

    # 跟踪变量
    no_improvement_count = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_cost = torch.tensor(0.0, dtype=torch.float32)
        processed_queries = 0

        # 处理每个查询
        for query in query_workload:
            # 查询边界
            query_lat_low = query['area']['min_lat']
            query_lat_high = query['area']['max_lat']
            query_lon_low = query['area']['min_lon']
            query_lon_high = query['area']['max_lon']

            # 早期剪枝：如果没有空间交集则跳过
            spatial_intersect = (
                    lat_min <= query_lat_high and
                    lat_max >= query_lat_low and
                    lon_min <= query_lon_high and
                    lon_max >= query_lon_low
            )

            if not spatial_intersect:
                continue

            processed_queries += 1
            query_cost = torch.tensor(0.0, dtype=torch.float32)

            # 处理查询中的每个关键词
            for kw in query['keywords']:
                if kw not in cdf_models:
                    continue

                # 确保split_value在成本计算的边界内
                clamped_split = torch.clamp(split_value, min=lower_bound, max=upper_bound)

                # 获取两个维度的CDF值
                if cdf_models[kw]['gaussian']:
                    # 对于高斯模型

                    # 纬度(y)的CDF值
                    lat_mean = cdf_models[kw]['y']['mean']
                    lat_std = cdf_models[kw]['y']['std']
                    F_lat_min = torch.tensor(norm.cdf(lat_min, loc=lat_mean, scale=lat_std),
                                             dtype=torch.float32)
                    F_lat_max = torch.tensor(norm.cdf(lat_max, loc=lat_mean, scale=lat_std),
                                             dtype=torch.float32)

                    # 经度(x)的CDF值
                    lon_mean = cdf_models[kw]['x']['mean']
                    lon_std = cdf_models[kw]['x']['std']
                    F_lon_min = torch.tensor(norm.cdf(lon_min, loc=lon_mean, scale=lon_std),
                                             dtype=torch.float32)
                    F_lon_max = torch.tensor(norm.cdf(lon_max, loc=lon_mean, scale=lon_std),
                                             dtype=torch.float32)

                    # 分割点CDF - 梯度传播的关键部分
                    if dim == 0:  # 分割纬度
                        # 创建可微分的分割点CDF
                        # 这种方法使用autograd通过norm.cdf计算梯度
                        split_val = clamped_split.detach().item()
                        F_split_val = norm.cdf(split_val, loc=lat_mean, scale=lat_std)
                        F_split = torch.tensor(F_split_val, dtype=torch.float32)

                        # 将梯度连接到split_value
                        # 手动梯度近似
                        eps = 1e-5
                        if split_val + eps <= upper_bound:
                            F_split_plus = norm.cdf(split_val + eps, loc=lat_mean, scale=lat_std)
                            grad = (F_split_plus - F_split_val) / eps
                            F_split = F_split + (clamped_split - split_val) * grad
                    else:  # 分割经度
                        split_val = clamped_split.detach().item()
                        F_split_val = norm.cdf(split_val, loc=lon_mean, scale=lon_std)
                        F_split = torch.tensor(F_split_val, dtype=torch.float32)

                        # 连接梯度
                        eps = 1e-5
                        if split_val + eps <= upper_bound:
                            F_split_plus = norm.cdf(split_val + eps, loc=lon_mean, scale=lon_std)
                            grad = (F_split_plus - F_split_val) / eps
                            F_split = F_split + (clamped_split - split_val) * grad
                else:
                    # 高频关键词的神经网络模型
                    lat_model = cdf_models[kw]['y']
                    lon_model = cdf_models[kw]['x']

                    # 获取CDF值
                    F_lat_min = lat_model(torch.tensor([[lat_min]], dtype=torch.float32))
                    F_lat_max = lat_model(torch.tensor([[lat_max]], dtype=torch.float32))
                    F_lon_min = lon_model(torch.tensor([[lon_min]], dtype=torch.float32))
                    F_lon_max = lon_model(torch.tensor([[lon_max]], dtype=torch.float32))

                    # 分割点CDF
                    if dim == 0:
                        F_split = lat_model(clamped_split.view(1, 1))
                    else:
                        F_split = lon_model(clamped_split.view(1, 1))

                # 具有此关键词的总对象数
                total_kw = cdf_models[kw]['total_kw']

                # 计算每个分区中的预期对象
                if dim == 0:  # 分割纬度
                    # 左子空间: [lat_min, split] x [lon_min, lon_max]
                    O1 = torch.clamp((F_split - F_lat_min) * (F_lon_max - F_lon_min) * total_kw, min=0.0)
                    # 右子空间: [split, lat_max] x [lon_min, lon_max]
                    O2 = torch.clamp((F_lat_max - F_split) * (F_lon_max - F_lon_min) * total_kw, min=0.0)

                    # 查询交集指示器(sigmoid用于平滑梯度)
                    query_lat_low_t = torch.tensor(query_lat_low, dtype=torch.float32)
                    query_lat_high_t = torch.tensor(query_lat_high, dtype=torch.float32)

                    # 左子空间与查询相交的程度
                    left_intersect = torch.sigmoid(3 * (clamped_split - query_lat_low_t))
                    # 右子空间与查询相交的程度
                    right_intersect = torch.sigmoid(3 * (query_lat_high_t - clamped_split))

                else:  # 分割经度
                    # 左子空间: [lat_min, lat_max] x [lon_min, split]
                    O1 = torch.clamp((F_lat_max - F_lat_min) * (F_split - F_lon_min) * total_kw, min=0.0)
                    # 右子空间: [lat_min, lat_max] x [split, lon_max]
                    O2 = torch.clamp((F_lat_max - F_lat_min) * (F_lon_max - F_split) * total_kw, min=0.0)

                    # 查询交集指示器
                    query_lon_low_t = torch.tensor(query_lon_low, dtype=torch.float32)
                    query_lon_high_t = torch.tensor(query_lon_high, dtype=torch.float32)

                    # 左子空间与查询相交的程度
                    left_intersect = torch.sigmoid(3 * (clamped_split - query_lon_low_t))
                    # 右子空间与查询相交的程度
                    right_intersect = torch.sigmoid(3 * (query_lon_high_t - clamped_split))

                # 计算关键词成本(需要检查的预期对象数)
                kw_cost = left_intersect * O1 + right_intersect * O2
                query_cost += kw_cost.squeeze()

            # 将查询成本添加到总成本
            total_cost = total_cost + query_cost

        # 处理没有查询交集的情况
        if processed_queries == 0:
            # 使用默认分割
            split_value.data = torch.tensor([initial_value], dtype=torch.float32)
            continue

        # 添加软边界约束
        # 使用惩罚方法而不是硬裁剪以维持梯度流
        boundary_penalty = 100.0 * (
                torch.relu(lower_bound - split_value) +
                torch.relu(split_value - upper_bound)
        )
        penalty_cost = total_cost + boundary_penalty

        # 反向传播
        penalty_cost.backward()

        # 梯度裁剪以防止大更新
        if split_value.grad is not None:
            torch.nn.utils.clip_grad_norm_([split_value], max_norm=1.0)
            optimizer.step()

        # 保存最佳结果
        with torch.no_grad():
            # 硬裁剪以确保有效的分割值
            split_value.data.clamp_(lower_bound, upper_bound)
            current_cost = total_cost.item()

            if current_cost < best_cost:
                best_cost = current_cost
                best_split = split_value.item()
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # 如果没有改进则提前停止
            if no_improvement_count >= 5:
                break


        print(f"Epoch {epoch}, Cost: {current_cost:.4f}, Split: {split_value.item():.4f}")

    print(f"最终结果 - 最佳分割点: {best_split:.4f}, 最小成本: {best_cost:.4f}")
    return best_split, best_cost


def find_optimal_partition(subspace, dim, query_workload, cdf_models):
    optimal_val, predicted_cost = SGDLearn(subspace, dim, query_workload, cdf_models, epochs=25, lr=0.01)
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
                lat_mean = cdf_models[kw]['y']['mean']
                lat_std = cdf_models[kw]['y']['std']
                lon_mean = cdf_models[kw]['x']['mean']
                lon_std = cdf_models[kw]['x']['std']

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
                F_lat_low = lat_model(torch.tensor([mbr['min_lat']], dtype=torch.float32)).item()
                F_lat_high = lat_model(torch.tensor([mbr['max_lat']], dtype=torch.float32)).item()

                # Calculate CDF values for longitude
                F_lon_low = lon_model(torch.tensor([mbr['min_lon']], dtype=torch.float32)).item()
                F_lon_high = lon_model(torch.tensor([mbr['max_lon']], dtype=torch.float32)).item()

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
    正确计算子空间中与查询匹配的对象数量：
    1. 首先检查空间MBR是否相交
    2. 然后只统计关键词匹配的对象
    """
    # 首先检查空间相交性
    subspace_mbr = {
        'min_lat': subspace['min_lat'],
        'max_lat': subspace['max_lat'],
        'min_lon': subspace['min_lon'],
        'max_lon': subspace['max_lon']
    }
    query_mbr = query['area']

    # 检查空间相交
    spatial_intersect = (
            subspace_mbr['min_lat'] <= query_mbr['max_lat'] and
            subspace_mbr['max_lat'] >= query_mbr['min_lat'] and
            subspace_mbr['min_lon'] <= query_mbr['max_lon'] and
            subspace_mbr['max_lon'] >= query_mbr['min_lon']
    )

    # 如果不相交，直接返回0
    if not spatial_intersect:
        return 0

    # 如果相交，计算关键词匹配
    matching_count = 0
    query_keywords = set(query['keywords'])

    for obj in subspace['objects']:
        if any(kw in query_keywords for kw in obj['keywords']):
            matching_count += 1

    return matching_count


def test_cost_estimation(data_space, query_workload, cdf_models):
    """
    详细验证CDF成本估计的准确性，对比分割前后的估计成本与实际成本
    """
    if isinstance(query_workload, pd.DataFrame):
        query_workload = query_workload.to_dict('records')
    print("-" * 80)
    print("CDF成本估计验证测试")
    print("-" * 80)

    total_spaces_tested = 0

    for subspace in data_space:
        total_spaces_tested += 1
        print(f"\n测试子空间 #{total_spaces_tested}:")
        print(
            f"MBR: [lat: {subspace['min_lat']:.6f}-{subspace['max_lat']:.6f}, lon: {subspace['min_lon']:.6f}-{subspace['max_lon']:.6f}]")
        print(f"对象数量: {len(subspace['objects'])}")
        print(f"关键词数量: {len(subspace.get('labels', []))}")

        # 1. 测试分割前整体成本估计 (Cs)
        print("\n1. 分割前成本估计 (Cs):")

        # CDF估计的成本
        cdf_estimated_cost = 0.0
        for query in query_workload:
            query_cost = calculate_cost_with_cdf(subspace, query, cdf_models)
            cdf_estimated_cost += query_cost

        # 实际成本 (通过直接对象匹配)
        actual_cost = 0
        for query in query_workload:
            matching_count = count_matching_objects(subspace, query)
            actual_cost += matching_count

        print(f"CDF估计成本: {cdf_estimated_cost:.2f}")
        print(f"实际成本: {actual_cost}")
        print(f"估计/实际比率: {cdf_estimated_cost / actual_cost if actual_cost > 0 else 'N/A'}")

        # 2. 测试最佳分割和成本估计
        print("\n2. 分割后成本估计 (best_cost):")

        # 寻找最优分割
        opt_x = find_optimal_partition(subspace, dim=0, query_workload=query_workload, cdf_models=cdf_models)
        opt_y = find_optimal_partition(subspace, dim=1, query_workload=query_workload, cdf_models=cdf_models)

        # 选择成本更低的分割方式
        best = opt_x if opt_x['cost'] <= opt_y['cost'] else opt_y
        print(f"最佳分割维度: {'纬度(lat)' if best['dim'] == 0 else '经度(lon)'}")
        print(f"最佳分割值: {best['val']:.6f}")
        print(f"CDF估计分割后成本: {best['cost']:.2f}")

        # 生成实际分割的子空间
        s1, s2 = generate_subspace(subspace, dim=best['dim'], split_value=best['val'])

        if s1 and s2:
            # 计算实际分割后成本
            cost_s1 = 0
            cost_s2 = 0
            matching_count_s1 = 0
            matching_count_s2 = 0
            actual_cost_after_split = 0
            for query in query_workload:
                matching_count_s1 += count_matching_objects(s1, query)
                matching_count_s2 += count_matching_objects(s2, query)

                cost_s1 += calculate_cost_with_cdf(s1, query, cdf_models) #判别sigmoid函数是否正确
                cost_s2 += calculate_cost_with_cdf(s2, query, cdf_models)
            actual_cost_after_split += matching_count_s1 + matching_count_s2
            print(f"分割后子空间1对象数: {len(s1['objects'])},实际成本：{matching_count_s1},cdf估计成本：{cost_s1}")
            print(f"分割后子空间2对象数: {len(s2['objects'])},实际成本：{matching_count_s2},cdf估计成本：{cost_s2}")
            print(f"实际分割后成本: {actual_cost_after_split}")
            # 计算比率和误差
            cdf_error = abs(best['cost'] - actual_cost_after_split) / actual_cost_after_split * 100 if actual_cost_after_split > 0 else 0
            print(f"CDF估计误差: {cdf_error:.2f}%")


            # 分析是否应该分割
            w1, w2 = 0.1, 1  # 使用与bottom_clusters_generation相同的参数
            cluster_overhead = w1 * len(query_workload)
            actual_benefit = (actual_cost - actual_cost_after_split) * w2
            estimated_benefit = (cdf_estimated_cost - best['cost']) * w2

            print(f"\n3. 分割决策分析:")
            print(f"集群扫描开销 (w1 * |W|): {cluster_overhead:.2f}")
            print(f"估计的分割收益: {estimated_benefit:.2f}")
            print(f"实际的分割收益: {actual_benefit:.2f}")

            # 估计与实际分割决策是否一致
            estimated_decision = "分割" if estimated_benefit > cluster_overhead else "不分割"
            actual_decision = "分割" if actual_benefit > cluster_overhead else "不分割"

            decision_alignment = "一致" if estimated_decision == actual_decision else "不一致"
            print(f"基于CDF估计的决策: {estimated_decision}")
            print(f"基于实际成本的决策: {actual_decision}")
            print(f"决策一致性: {decision_alignment}")

            # 建议
            if decision_alignment == "不一致":
                print("\n检测到决策不一致！可能的原因:")

                if cdf_error > 20:
                    print(f"- CDF模型估计误差较大 ({cdf_error:.2f}%)")
                    print("  建议: 改进CDF模型或调整权重参数")

                if actual_benefit > 0 and estimated_benefit <= 0:
                    print("- CDF模型低估了分割收益")
                    print("  建议: 检查CDF模型的分布拟合")

                if actual_benefit <= 0 and estimated_benefit > 0:
                    print("- CDF模型高估了分割收益")
                    print("  建议: 调整SGDLearn参数或权重系数")

                if abs(estimated_benefit - actual_benefit) > cluster_overhead * 2:
                    print("- 收益估计与实际差异过大")
                    print("  建议: 减小w1值或增加w2值")
        else:
            print("无法生成有效分割，可能达到最小尺寸限制")

        print("-" * 80)

    # 总结
    print(f"\n总结: 测试了 {total_spaces_tested} 个子空间")
    print("建议采取的行动:")
    print("1. 如果CDF估计普遍与实际成本有较大差异，考虑重新训练CDF模型")
    print("2. 如果分割决策经常不一致，考虑调整w1和w2权重参数")
    print("3. 对于大簇的情况，可以添加强制分割机制")
    print("4. 减小最小尺寸限制，允许更细粒度的分割")
    print("-" * 80)

def bottom_clusters_generation(query_workload, data_space, cdf_models, w1, w2, MIN_OBJECTS=10):
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
            print(f"分割前对象验证成本: {Cs}, 分割后SGD对象验证成本: {best['cost']}, 增加簇扫描成本: {w1 * len(query_workload)}")
            cost_s1 = 0
            cost_s2 = 0
            matching_count_s1 = 0
            matching_count_s2 = 0
            actual_cost_after_split = 0
            cdf_cost_after_split = 0
            for query in query_workload:
                matching_count_s1 += count_matching_objects(s1, query)
                matching_count_s2 += count_matching_objects(s2, query)

                cost_s1 += calculate_cost_with_cdf(s1, query, cdf_models)  # 判别sigmoid函数是否正确
                cost_s2 += calculate_cost_with_cdf(s2, query, cdf_models)
            actual_cost_after_split += matching_count_s1 + matching_count_s2
            cdf_cost_after_split += cost_s1 + cost_s2
            print(f"分割后的实际成本：{actual_cost_after_split},cdf预测成本：{cdf_cost_after_split}")

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


def visualize_clusters(bottom_nodes, original_data, query_workload=None, output_file=None, max_points=1000, figsize=(12, 10)):
    """
    Visualize the CDF-based clusters, the original data points, and query regions.

    Parameters:
    -----------
    bottom_nodes : list
        List of leaf nodes (clusters) generated by bottom_clusters_generation
    original_data : list
        Original data points used for clustering
    query_workload : list, optional
        List of queries to visualize their regions
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

    # Plot query regions if provided
    if query_workload:
        for query in query_workload:
            # Extract query boundaries
            query_mbr = query['area']
            min_lat, max_lat = query_mbr['min_lat'], query_mbr['max_lat']
            min_lon, max_lon = query_mbr['min_lon'], query_mbr['max_lon']

            # Create a rectangle for the query region
            query_rect = patches.Rectangle(
                (min_lon, min_lat),
                max_lon - min_lon,
                max_lat - min_lat,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                linestyle='--',
                label='Query Region'
            )
            plt.gca().add_patch(query_rect)

    # Plot settings
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Spatial Clustering Results ({len(bottom_nodes)} clusters)')

    # Create a custom legend that doesn't show every cluster
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, alpha=0.3,
                   label='Original Data'),
        plt.Line2D([0], [0], marker='x', color=colors[0], markersize=8,
                   label=f'Cluster Centers ({len(bottom_nodes)} total)')
    ]
    if query_workload:
        legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Query Region'))

    plt.legend(handles=legend_elements, loc='upper right')

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    else:
        plt.show()


