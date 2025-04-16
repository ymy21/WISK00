import heapq
import itertools
import numpy as np
import torch
import pandas as pd
from scipy.stats import norm
from collections import defaultdict


class PriorityQueue:
    def __init__(self, query_workload, max_size=1000):
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

    def __init__(self, layer, mbr, objects, node_id=None):
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


# 优化的交叉查询计算函数
def calculate_intersecting_queries_gpu(subspace, query_workload):
    # 如果是GPU可用，则使用GPU加速
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取子空间信息
    if isinstance(subspace, dict):  # 旧格式
        if not subspace.get('labels'):
            return 0
        subspace_keywords = set(subspace.get('labels', []))
        mbr = {
            'min_lat': subspace['min_lat'],
            'max_lat': subspace['max_lat'],
            'min_lon': subspace['min_lon'],
            'max_lon': subspace['max_lon']
        }
    elif isinstance(subspace, LeafNode):  # 新LeafNode格式
        subspace_keywords = subspace.get_keywords()
        mbr = subspace.mbr
    else:
        return 0

    # 预计算所有查询边界
    query_mins = torch.tensor([[q['area']['min_lat'], q['area']['min_lon']]
                               for q in query_workload], device=device)
    query_maxs = torch.tensor([[q['area']['max_lat'], q['area']['max_lon']]
                               for q in query_workload], device=device)

    # 子空间边界转为tensor
    subspace_min = torch.tensor([mbr['min_lat'], mbr['min_lon']], device=device)
    subspace_max = torch.tensor([mbr['max_lat'], mbr['max_lon']], device=device)

    # 并行计算所有查询的空间交集
    spatial_intersect = (
            (subspace_min[0] <= query_maxs[:, 0]) &
            (subspace_max[0] >= query_mins[:, 0]) &
            (subspace_min[1] <= query_maxs[:, 1]) &
            (subspace_max[1] >= query_mins[:, 1])
    )

    # 统计同时具有空间交集和关键词交集的查询
    count = 0
    for i, query in enumerate(query_workload):
        if spatial_intersect[i].item() and bool(set(query['keywords']) & subspace_keywords):
            count += 1

    return count


# 保留原有函数作为兼容模式
def calculate_intersecting_queries(subspace, query_workload):
    return calculate_intersecting_queries_gpu(subspace, query_workload)


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
            return None, None
    else:
        if (split_value - subspace['min_lon'] < dynamic_min_size) or \
                (subspace['max_lon'] - split_value < dynamic_min_size):
            return None, None

    # 分割对象 - 使用NumPy向量化操作加速
    left_objects = []
    right_objects = []

    # 使用NumPy加速数组操作
    if dim == 0:
        coords = np.array([obj['latitude'] for obj in subspace['objects']])
        left_mask = coords < split_value
    else:
        coords = np.array([obj['longitude'] for obj in subspace['objects']])
        left_mask = coords < split_value

    # 根据掩码分配对象
    for i, obj in enumerate(subspace['objects']):
        if left_mask[i]:
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

        # 计算空间边界 - 使用NumPy加速
        if len(objects) > 0:
            lats = np.array([obj['latitude'] for obj in objects])
            lons = np.array([obj['longitude'] for obj in objects])

            return {
                'min_lat': np.min(lats),
                'max_lat': np.max(lats),
                'min_lon': np.min(lons),
                'max_lon': np.max(lons),
                'objects': objects,
                'labels': keywords
            }
        return None

    left_sub = create_subspace(left_objects)
    right_sub = create_subspace(right_objects)
    return left_sub, right_sub


# 优化的SGD学习函数
def SGDLearn_optimized(subspace, dim, query_workload, cdf_models, epochs=8, lr=0.02):
    # 使用GPU加速
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始分割点
    initial_value = subspace['min_lat'] + (subspace['max_lat'] - subspace['min_lat']) / 2 if dim == 0 \
        else subspace['min_lon'] + (subspace['max_lon'] - subspace['min_lon']) / 2
    split_value = torch.tensor([initial_value], dtype=torch.float32, device=device, requires_grad=True)

    # 使用Adam优化器
    optimizer = torch.optim.Adam([split_value], lr=lr)

    best_cost = float('inf')
    best_split = initial_value
    sigmoid = torch.nn.Sigmoid().to(device)

    # 预计算子空间边界
    if dim == 0:
        subspace_below = torch.tensor(subspace['min_lat'], dtype=torch.float32, device=device)
        subspace_up = torch.tensor(subspace['max_lat'], dtype=torch.float32, device=device)
    else:
        subspace_below = torch.tensor(subspace['min_lon'], dtype=torch.float32, device=device)
        subspace_up = torch.tensor(subspace['max_lon'], dtype=torch.float32, device=device)

    # 预处理查询数据，将查询的边界转换为GPU张量
    query_boundaries = []
    for query in query_workload:
        if dim == 0:
            query_boundaries.append([query['area']['min_lat'], query['area']['max_lat']])
        else:
            query_boundaries.append([query['area']['min_lon'], query['area']['max_lon']])

    query_boundaries = torch.tensor(query_boundaries, dtype=torch.float32, device=device)

    # 提前停止参数
    patience = 3
    patience_counter = 0
    prev_cost = float('inf')

    for epoch in range(epochs):
        optimizer.zero_grad()
        total_cost = torch.tensor(0.0, dtype=torch.float32, device=device)

        # 批量计算所有查询的成本
        for q_idx, query in enumerate(query_workload):
            cost = torch.tensor(0.0, dtype=torch.float32, device=device)
            boundary_low = query_boundaries[q_idx, 0]
            boundary_high = query_boundaries[q_idx, 1]

            # 批量处理查询中的所有关键词
            for kw in query['keywords']:
                if kw in cdf_models:
                    if cdf_models[kw]['gaussian']:
                        # 中频关键词：使用高斯估计
                        if dim == 0:
                            mean = cdf_models[kw]['y_mean']
                            std = cdf_models[kw]['y_std']
                        else:
                            mean = cdf_models[kw]['x_mean']
                            std = cdf_models[kw]['x_std']

                        # 使用GPU加速CDF计算
                        F_low = torch.tensor(norm.cdf(subspace_below.cpu().item(), loc=mean, scale=std),
                                             dtype=torch.float32, device=device)
                        F_split = torch.tensor(norm.cdf(split_value.detach().cpu().item(), loc=mean, scale=std),
                                               dtype=torch.float32, device=device)
                        F_high = torch.tensor(norm.cdf(subspace_up.cpu().item(), loc=mean, scale=std),
                                              dtype=torch.float32, device=device)
                    else:
                        # 高频关键词：使用NN模型
                        model = cdf_models[kw]['y'] if dim == 0 else cdf_models[kw]['x']

                        # 转移到相同设备
                        if hasattr(model, 'to'):
                            model = model.to(device)

                        F_low = model(subspace_below.view(1, 1))
                        F_split = model(split_value.view(1, 1))
                        F_high = model(subspace_up.view(1, 1))

                    total_kw = cdf_models[kw]['total_kw']
                    # 估计子区域内包含该关键词的对象数量
                    O1 = torch.clamp((F_split - F_low) * total_kw, min=0.0)
                    O2 = torch.clamp((F_high - F_split) * total_kw, min=0.0)

                    # 成本公式优化：使用更高效的sigmoid计算
                    bound_low_tensor = boundary_low.view(1)
                    bound_high_tensor = boundary_high.view(1)

                    cost_kw = sigmoid(3 * (split_value - bound_low_tensor)) * O1 + \
                              sigmoid(3 * (bound_high_tensor - split_value)) * O2
                    cost += cost_kw.squeeze()

            total_cost = total_cost + cost

        # 优化器步骤之前应用约束
        if dim == 0:
            split_value.data.clamp_(subspace['min_lat'], subspace['max_lat'])
        else:
            split_value.data.clamp_(subspace['min_lon'], subspace['max_lon'])

        # 反向传播和优化步骤
        total_cost.backward()
        optimizer.step()

        # 提前停止检查
        if abs(total_cost.item() - prev_cost) < 1e-3:
            patience_counter += 1
            if patience_counter >= patience:
                break
        else:
            patience_counter = 0

        prev_cost = total_cost.item()

        if total_cost.item() < best_cost:
            best_cost = total_cost.item()
            best_split = split_value.item()

    return best_split, best_cost


# 优化版本的最优分区查找函数
def find_optimal_partition(subspace, dim, query_workload, cdf_models, cache={}):
    # 创建缓存键
    cache_key = (subspace['min_lat'], subspace['max_lat'],
                 subspace['min_lon'], subspace['max_lon'], dim)

    # 检查缓存
    if cache_key in cache:
        return cache[cache_key]

    # 使用优化的SGD学习
    optimal_val, predicted_cost = SGDLearn_optimized(subspace, dim, query_workload, cdf_models)

    # 生成子空间
    s1, s2 = generate_subspace(subspace, dim, optimal_val)

    if s1 and s2:
        # 并行计算查询交集
        num_queries_s1 = calculate_intersecting_queries_gpu(s1, query_workload)
        num_queries_s2 = calculate_intersecting_queries_gpu(s2, query_workload)
        C_split = len(s1['objects']) * num_queries_s1 + len(s2['objects']) * num_queries_s2
    else:
        C_split = float('inf')

    result = {'dim': dim, 'cost': C_split, 'val': optimal_val}

    # 存入缓存
    cache[cache_key] = result
    return result


# 优化的底部簇生成函数
def bottom_clusters_generation(query_workload, data_space, cdf_models, w1=0.1, w2=1, MIN_OBJECTS=10):
    # 确保query_workload是字典列表格式
    if isinstance(query_workload, pd.DataFrame):
        query_workload = query_workload.to_dict('records')

    # 初始化优先队列
    Q = PriorityQueue(query_workload)
    G = []  # 存储最终生成的聚类
    node_id_counter = 0

    # 初始化缓存
    partition_cache = {}

    # 将整个数据空间添加到队列
    for item in data_space:
        Q.enqueue(item)

    # 将PyTorch设置为使用GPU
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.float32)
        print("使用GPU加速计算")
    else:
        print("GPU不可用，使用CPU计算")

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
            leaf_node = LeafNode(0, mbr, s['objects'], node_id=node_id_counter)
            node_id_counter += 1
            G.append(leaf_node)
            continue

        # 计算查询交集
        num_queries = calculate_intersecting_queries_gpu(s, query_workload)
        Cs = num_objects * num_queries

        # 并行查找两个维度的最优分割
        # 使用线程池并行化
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_x = executor.submit(find_optimal_partition, s, 0, query_workload, cdf_models, partition_cache)
            future_y = executor.submit(find_optimal_partition, s, 1, query_workload, cdf_models, partition_cache)

            opt_x = future_x.result()
            opt_y = future_y.result()

        # 选择成本更低的分割方式
        best = opt_x if opt_x['cost'] <= opt_y['cost'] else opt_y

        # 使用最佳分割生成子空间
        s1, s2 = generate_subspace(s, dim=best['dim'], split_value=best['val'])

        if s1 and s2:
            print(
                f"分割前对象验证成本: {Cs}, 分割后对象验证成本: {best['cost']}, 增加簇扫描成本: {w1 * len(query_workload)}")

            # 分割条件检查
            if (Cs - best['cost']) * w2 > (w1 * len(query_workload)):
                # 允许分割并加入队列
                print(f"分割前对象数: {len(s['objects'])}, 查询交集数: {num_queries}, Cs: {Cs}")
                print(
                    f"分割后对象数: {len(s1['objects'])} + {len(s2['objects'])} = {len(s1['objects']) + len(s2['objects'])}, C_split: {best['cost']}")

                # 预计算新子空间的优先级以减少后续计算
                priority_s1 = calculate_intersecting_queries_gpu(s1, query_workload)
                priority_s2 = calculate_intersecting_queries_gpu(s2, query_workload)

                Q.enqueue(s1, priority_s1)
                Q.enqueue(s2, priority_s2)
                print(f"成本缩小，可以分割")
            else:
                # 创建叶子节点并添加到结果
                mbr = {
                    'min_lat': s['min_lat'],
                    'max_lat': s['max_lat'],
                    'min_lon': s['min_lon'],
                    'max_lon': s['max_lon']
                }
                leaf_node = LeafNode(0, mbr, s['objects'], node_id=node_id_counter)
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
            leaf_node = LeafNode(0, mbr, s['objects'], node_id=node_id_counter)
            node_id_counter += 1
            G.append(leaf_node)

    print(f"已生成聚类数: {len(G)}")

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 转换为字典格式以兼容当前代码
    bottom_nodes = [leaf_node.to_dict() for leaf_node in G]
    return bottom_nodes