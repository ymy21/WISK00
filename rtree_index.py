import pandas as pd
import numpy as np
from collections import defaultdict

"""
R-tree 索引的构建、查询以及对比函数，
支持与WISK索引的公平比较：
  - 两种方法均可使用相同的底层簇作为叶子节点
  - 同时保留原有基于对象分组的方法进行比较
"""


class RTreeNode:
    def __init__(self, is_leaf):
        self.is_leaf = is_leaf
        self.entries = []  # 如果是叶节点，每个 entry 为 (objects_list, mbr)；如果是内部节点，每个 entry 为子节点
        self.mbr = None  # 当前节点的最小外接矩形
        self.keywords = []  # 节点包含的所有关键词


def merge_mbr(mbr1, mbr2):
    """合并两个 MBR，返回新的 MBR"""
    return {
        'min_lat': min(mbr1['min_lat'], mbr2['min_lat']),
        'max_lat': max(mbr1['max_lat'], mbr2['max_lat']),
        'min_lon': min(mbr1['min_lon'], mbr2['min_lon']),
        'max_lon': max(mbr1['max_lon'], mbr2['max_lon'])
    }


def extract_keywords_from_objects(objects):
    """从对象列表中提取所有唯一关键词"""
    all_keywords = set()
    for obj in objects:
        if 'keywords' in obj:
            all_keywords.update(obj['keywords'])
    return list(all_keywords)


def calculate_mbr_for_objects(objects):
    """计算一组对象的MBR"""
    if not objects:
        return None

    mbr = {
        'min_lat': float('inf'),
        'max_lat': float('-inf'),
        'min_lon': float('inf'),
        'max_lon': float('-inf')
    }

    for obj in objects:
        mbr['min_lat'] = min(mbr['min_lat'], obj['latitude'])
        mbr['max_lat'] = max(mbr['max_lat'], obj['latitude'])
        mbr['min_lon'] = min(mbr['min_lon'], obj['longitude'])
        mbr['max_lon'] = max(mbr['max_lon'], obj['longitude'])

    return mbr


def mbr_intersect(mbr1, mbr2):
    """判断两个 MBR 是否相交"""
    return not (mbr1['max_lat'] < mbr2['min_lat'] or
                mbr1['min_lat'] > mbr2['max_lat'] or
                mbr1['max_lon'] < mbr2['min_lon'] or
                mbr1['min_lon'] > mbr2['max_lon'])


def keyword_intersect(obj_keywords, query_keywords):
    """检查对象/聚类的关键词是否与查询关键词有交集"""
    return bool(set(obj_keywords) & set(query_keywords))


def group_objects_spatially(objects, avg_objects_per_leaf):
    """
    将对象按照空间邻近性分组，每组大约有avg_objects_per_leaf个对象
    使用简单的网格划分方法
    """
    if isinstance(objects, pd.DataFrame):
        objects = objects.to_dict('records')

    # 确定经纬度范围
    min_lat = min(obj['latitude'] for obj in objects)
    max_lat = max(obj['latitude'] for obj in objects)
    min_lon = min(obj['longitude'] for obj in objects)
    max_lon = max(obj['longitude'] for obj in objects)

    # 计算划分网格的数量
    total_cells = max(1, (len(objects) // avg_objects_per_leaf))
    grid_size = int(np.sqrt(total_cells))

    # 将对象分配到网格
    grid = defaultdict(list)
    for obj in objects:
        # 计算对象所在的网格坐标
        grid_x = min(grid_size - 1, int((obj['longitude'] - min_lon) / (max_lon - min_lon) * grid_size))
        grid_y = min(grid_size - 1, int((obj['latitude'] - min_lat) / (max_lat - min_lat) * grid_size))
        grid[(grid_x, grid_y)].append(obj)

    # 合并小网格，确保每个网格至少有一定数量的对象
    min_objects = max(1, avg_objects_per_leaf // 2)
    grouped_objects = []

    temp_group = []
    for cell_objects in grid.values():
        if not cell_objects:
            continue

        if len(temp_group) + len(cell_objects) <= avg_objects_per_leaf:
            temp_group.extend(cell_objects)
        else:
            if temp_group:
                grouped_objects.append(temp_group)
            temp_group = cell_objects

    if temp_group:
        grouped_objects.append(temp_group)

    # 处理特殊情况：如果某些组太小，合并它们
    final_groups = []
    temp_group = []

    for group in sorted(grouped_objects, key=len):
        if len(group) < min_objects and len(temp_group) + len(group) <= avg_objects_per_leaf * 1.5:
            temp_group.extend(group)
        else:
            if temp_group:
                final_groups.append(temp_group)
            temp_group = group

    if temp_group:
        final_groups.append(temp_group)

    return final_groups


def build_rtree_with_grouped_objects(avg_object, objects, wisk_index, max_entries=10):
    """
    根据对象组构建R-tree，每个叶子节点包含多个对象
    叶子节点的对象数量与WISK索引中聚类的平均对象数量相似
    """

    # 按空间邻近性将对象分组
    object_groups = group_objects_spatially(objects, avg_object)
    print(f"R-tree分组后的组数: {len(object_groups)}")
    print(f"每组平均对象数量: {sum(len(g) for g in object_groups) / len(object_groups)}")

    # 为每个组创建叶子节点
    leaves = []
    for group in object_groups:
        mbr = calculate_mbr_for_objects(group)
        leaf = RTreeNode(is_leaf=True)
        leaf.entries.append((group, mbr))
        leaf.mbr = mbr
        leaf.keywords = extract_keywords_from_objects(group)
        leaves.append(leaf)

    # 构建内部节点层次
    nodes = leaves
    while len(nodes) > 1:
        new_level = []
        for i in range(0, len(nodes), max_entries):
            group = nodes[i:i + max_entries]
            parent = RTreeNode(is_leaf=False)
            parent.entries = group
            parent_mbr = group[0].mbr
            for child in group[1:]:
                parent_mbr = merge_mbr(parent_mbr, child.mbr)
            parent.mbr = parent_mbr
            # 合并子节点的关键词
            all_keywords = set()
            for child in group:
                all_keywords.update(child.keywords)
            parent.keywords = list(all_keywords)
            new_level.append(parent)
        nodes = new_level

    return nodes[0]


# 新增：从已有的clusters构建Rtree
def build_rtree_from_clusters(clusters, max_entries=10):
    """
    使用已有的clusters作为R-tree的叶子节点
    Args:
        clusters: 底层簇列表，由LeafNode.to_dict()转换而来
        max_entries: 每个非叶子节点最多包含的子节点数
    Returns:
        R-tree的根节点
    """
    print("从已有的底层簇构建R-tree...")

    # 打印第一个簇的键值，帮助调试
    if clusters and len(clusters) > 0:
        print(f"簇示例键: {list(clusters[0].keys())}")

    # 为每个簇创建叶子节点
    leaves = []
    for i, cluster in enumerate(clusters):
        # 尝试查找MBR
        if 'MBR' in cluster:
            mbr = cluster['MBR']
        elif 'mbr' in cluster:
            mbr = cluster['mbr']
        else:
            # 如果没有找到MBR，跳过这个簇
            if i < 5:  # 只打印前几个警告，避免刷屏
                print(f"警告：簇 {i} 缺少MBR信息，已跳过。簇键：{list(cluster.keys())}")
            continue

        # 确保MBR有预期的结构
        if not all(key in mbr for key in ['min_lat', 'max_lat', 'min_lon', 'max_lon']):
            if i < 5:
                print(f"警告：簇 {i} 的MBR结构不正确。MBR键：{list(mbr.keys())}")
            continue

        leaf = RTreeNode(is_leaf=True)

        # 获取对象列表
        if 'objects' in cluster:
            leaf_objects = cluster['objects']
        elif 'leaf_objects' in cluster:
            leaf_objects = cluster['leaf_objects']
        else:
            leaf_objects = []
            if i < 5:
                print(f"警告：簇 {i} 中没有找到对象列表。簇键：{list(cluster.keys())}")

        # 将簇的对象和MBR作为叶节点entry
        leaf.entries.append((leaf_objects, mbr))
        leaf.mbr = mbr

        # 提取关键词
        if 'labels' in cluster and cluster['labels']:
            # 处理各种可能的labels类型
            if isinstance(cluster['labels'], (list, tuple)):
                leaf.keywords = list(cluster['labels'])
            elif isinstance(cluster['labels'], (set, frozenset)):
                leaf.keywords = list(cluster['labels'])
            else:
                leaf.keywords = [str(cluster['labels'])]
        else:
            # 如果没有labels，尝试从对象中提取关键词
            leaf.keywords = extract_keywords_from_objects(leaf_objects)

        leaves.append(leaf)

    if not leaves:
        print("警告：未能从簇中创建任何有效的叶子节点!")
        return None

    print(f"从簇创建的R-tree叶子节点数: {len(leaves)}")

    # 构建内部节点层次
    nodes = leaves
    level = 0
    while len(nodes) > 1:
        level += 1
        new_level = []
        for i in range(0, len(nodes), max_entries):
            group = nodes[i:i + max_entries]
            parent = RTreeNode(is_leaf=False)
            parent.entries = group

            # 计算合并的MBR
            parent_mbr = group[0].mbr
            for child in group[1:]:
                parent_mbr = merge_mbr(parent_mbr, child.mbr)
            parent.mbr = parent_mbr

            # 合并子节点的关键词
            all_keywords = set()
            for child in group:
                all_keywords.update(child.keywords)
            parent.keywords = list(all_keywords)
            new_level.append(parent)

        print(f"R-tree第{level}层节点数: {len(new_level)}")
        nodes = new_level

    if not nodes:
        return None
    return nodes[0]


def search_rtree(query_rect, query_keywords, node, node_counter, obj_counter=None):
    """
    在R-tree中搜索与query_rect相交且关键词匹配的对象
    node_counter：统计访问的节点数
    obj_counter：统计访问的对象数
    """
    node_counter[0] += 1  # 访问当前节点
    results = []

    # 首先检查当前节点的MBR与查询区域是否相交
    if not mbr_intersect(node.mbr, query_rect):
        return results

    # 检查当前节点的关键词与查询关键词是否有交集
    if not keyword_intersect(node.keywords, query_keywords):
        return results

    if node.is_leaf:
        for objects_list, obj_mbr in node.entries:
            # 使用关键词过滤，只处理相关对象
            relevant_objects = []
            for obj in objects_list:
                if keyword_intersect(obj.get('keywords', []), query_keywords):
                    relevant_objects.append(obj)

            # 只计数关键词匹配的对象
            if obj_counter is not None:
                obj_counter[0] += len(relevant_objects)

            # 在关键词匹配的对象中检查空间条件
            for obj in relevant_objects:
                if ('latitude' in obj and 'longitude' in obj and
                        query_rect['min_lat'] <= obj['latitude'] <= query_rect['max_lat'] and
                        query_rect['min_lon'] <= obj['longitude'] <= query_rect['max_lon']):
                    results.append(obj)
    else:
        # 非叶子节点，递归搜索子节点
        for child in node.entries:
            results.extend(search_rtree(query_rect, query_keywords, child, node_counter, obj_counter))

    return results


def search_wisk(query_rect, query_keywords, node, node_counter, obj_counter=None):
    """
    在WISK索引中搜索与query_rect相交且关键词匹配的对象
    node_counter：统计访问的节点数
    obj_counter：如果提供，统计访问的对象数
    """
    results = []

    def process_node(current_node):
        nonlocal results
        # 访问节点计数
        node_counter[0] += 1

        # 如果是列表，对每个节点递归调用
        if isinstance(current_node, list):
            for sub_node in current_node:
                process_node(sub_node)
            return

        # 确保当前节点是字典类型
        if not isinstance(current_node, dict):
            return

        # 获取MBR
        mbr = current_node.get('MBR')
        if mbr is None:
            return

        # 空间相交判断
        if not mbr_intersect(mbr, query_rect):
            return

        # 获取节点标签/关键词
        node_labels = set(current_node.get('labels', []))
        if not node_labels or not keyword_intersect(node_labels, query_keywords):
            return

        # 检查是否有子节点
        children = current_node.get('children', [])

        # 检查子节点类型，如果children是空列表或者不存在，可能是叶子节点
        if not children:
            # 检查是否有leaf_objects字段
            leaf_objects = current_node.get('leaf_objects', [])

            # 使用inverted_file索引过滤对象(如果存在)
            if 'inverted_file' in current_node and current_node['inverted_file']:
                inverted_file = current_node['inverted_file']
                relevant_indices = set()

                # 收集所有与查询关键词匹配的对象索引
                for kw in query_keywords:
                    if kw in inverted_file:
                        relevant_indices.update(inverted_file[kw])

                # 只计数关键词匹配的对象
                if obj_counter is not None:
                    obj_counter[0] += len(relevant_indices)

                # 检查这些对象是否在空间范围内
                for idx in relevant_indices:
                    if idx < len(leaf_objects):
                        obj = leaf_objects[idx]
                        if ('latitude' in obj and 'longitude' in obj and
                                query_rect['min_lat'] <= obj['latitude'] <= query_rect['max_lat'] and
                                query_rect['min_lon'] <= obj['longitude'] <= query_rect['max_lon']):
                            results.append(obj)
            else:
                # 没有inverted_file时的备用处理方式
                relevant_objects = []
                for obj in leaf_objects:
                    obj_keywords = obj.get('keywords', [])
                    if keyword_intersect(obj_keywords, query_keywords):
                        relevant_objects.append(obj)

                # 只计数关键词匹配的对象
                if obj_counter is not None:
                    obj_counter[0] += len(relevant_objects)

                # 检查这些对象是否在空间范围内
                for obj in relevant_objects:
                    if ('latitude' in obj and 'longitude' in obj and
                            query_rect['min_lat'] <= obj['latitude'] <= query_rect['max_lat'] and
                            query_rect['min_lon'] <= obj['longitude'] <= query_rect['max_lon']):
                        results.append(obj)
            return

        # 如果有子节点且是对象引用（不是索引），则处理子节点
        for child in children:
            process_node(child)

    # 开始处理节点
    process_node(node)
    return results


def compare_rtree_wisk(raw_objects, wisk_index, eval_queries, avg_object, clusters=None):
    """
    对比R-tree和WISK索引的性能，考虑两种方式：
    1. 传统方式：使用空间分组的对象作为R-tree叶子节点
    2. 新方式：使用与WISK相同的底层簇作为R-tree叶子节点
    """
    print("【R-tree vs WISK 性能比较】")

    # 方式1：传统方式构建R-tree
    print("\n===== 方式1：使用空间分组构建R-tree =====")
    rtree_traditional = build_rtree_with_grouped_objects(avg_object, raw_objects, wisk_index, max_entries=10)

    # 方式2：使用相同的底层簇构建R-tree
    if clusters:
        print("\n===== 方式2：使用相同的底层簇构建R-tree =====")
        rtree_from_clusters = build_rtree_from_clusters(clusters, max_entries=10)
    else:
        print("未提供底层簇，跳过方式2比较")
        rtree_from_clusters = None

    # 性能统计变量
    total_rtree_trad_nodes = 0
    total_rtree_trad_objects = 0
    total_rtree_cluster_nodes = 0
    total_rtree_cluster_objects = 0
    total_wisk_nodes = 0
    total_wisk_objects = 0

    print("\n===== 查询测试 =====")
    for i, query in enumerate(eval_queries):
        query_rect = query['area']
        query_keywords = query['keywords']

        # 1. 传统R-tree查询
        rtree_trad_node_counter = [0]
        rtree_trad_obj_counter = [0]
        rtree_trad_results = search_rtree(query_rect, query_keywords, rtree_traditional, rtree_trad_node_counter,
                                          rtree_trad_obj_counter)

        # 2. 基于相同簇的R-tree查询（如果有）
        if rtree_from_clusters:
            rtree_cluster_node_counter = [0]
            rtree_cluster_obj_counter = [0]
            rtree_cluster_results = search_rtree(query_rect, query_keywords, rtree_from_clusters,
                                                 rtree_cluster_node_counter, rtree_cluster_obj_counter)
        else:
            rtree_cluster_node_counter = [0]
            rtree_cluster_obj_counter = [0]
            rtree_cluster_results = []

        # 3. WISK查询
        wisk_node_counter = [0]
        wisk_obj_counter = [0]
        wisk_results = search_wisk(query_rect, query_keywords, wisk_index, wisk_node_counter, wisk_obj_counter)

        # 累加计数
        total_rtree_trad_nodes += rtree_trad_node_counter[0]
        total_rtree_trad_objects += rtree_trad_obj_counter[0]
        total_rtree_cluster_nodes += rtree_cluster_node_counter[0]
        total_rtree_cluster_objects += rtree_cluster_obj_counter[0]
        total_wisk_nodes += wisk_node_counter[0]
        total_wisk_objects += wisk_obj_counter[0]

        # 打印单次查询统计信息
        print(f"\n查询 {i + 1}（关键词：{query_keywords}）:")
        print(
            f"  传统R-tree: 访问节点数 = {rtree_trad_node_counter[0]}, 扫描对象数 = {rtree_trad_obj_counter[0]}, 结果数 = {len(rtree_trad_results)}")
        if rtree_from_clusters:
            print(
                f"  簇R-tree:   访问节点数 = {rtree_cluster_node_counter[0]}, 扫描对象数 = {rtree_cluster_obj_counter[0]}, 结果数 = {len(rtree_cluster_results)}")
        print(
            f"  WISK:       访问节点数 = {wisk_node_counter[0]}, 扫描对象数 = {wisk_obj_counter[0]}, 结果数 = {len(wisk_results)}")

    # 计算平均值
    num_queries = len(eval_queries)
    avg_rtree_trad_nodes = total_rtree_trad_nodes / num_queries
    avg_rtree_trad_objects = total_rtree_trad_objects / num_queries
    avg_rtree_cluster_nodes = total_rtree_cluster_nodes / num_queries if rtree_from_clusters else 0
    avg_rtree_cluster_objects = total_rtree_cluster_objects / num_queries if rtree_from_clusters else 0
    avg_wisk_nodes = total_wisk_nodes / num_queries
    avg_wisk_objects = total_wisk_objects / num_queries

    # 打印总体统计信息
    print("\n===== 总体统计 =====")
    print(f"查询数量: {num_queries}")
    print(f"传统R-tree: 平均访问节点数 = {avg_rtree_trad_nodes:.2f}, 平均扫描对象数 = {avg_rtree_trad_objects:.2f}")
    if rtree_from_clusters:
        print(
            f"簇R-tree:   平均访问节点数 = {avg_rtree_cluster_nodes:.2f}, 平均扫描对象数 = {avg_rtree_cluster_objects:.2f}")
    print(f"WISK:       平均访问节点数 = {avg_wisk_nodes:.2f}, 平均扫描对象数 = {avg_wisk_objects:.2f}")

    # 计算性能比值 - 传统R-tree vs WISK
    print("\n===== 传统R-tree vs WISK =====")
    trad_node_ratio = avg_wisk_nodes / avg_rtree_trad_nodes if avg_rtree_trad_nodes > 0 else float('inf')
    trad_object_ratio = avg_wisk_objects / avg_rtree_trad_objects if avg_rtree_trad_objects > 0 else float('inf')
    trad_total_cost_ratio = (avg_wisk_nodes + avg_wisk_objects) / (avg_rtree_trad_nodes + avg_rtree_trad_objects) if (avg_rtree_trad_nodes + avg_rtree_trad_objects) > 0 else float('inf')

    print(f"节点访问比值: {trad_node_ratio:.4f} {'(WISK更好)' if trad_node_ratio < 1 else '(传统R-tree更好)'}")
    print(f"对象扫描比值: {trad_object_ratio:.4f} {'(WISK更好)' if trad_object_ratio < 1 else '(传统R-tree更好)'}")
    print(
        f"总代价比值: {trad_total_cost_ratio:.4f} {'(WISK更好)' if trad_total_cost_ratio < 1 else '(传统R-tree更好)'}")

    # 计算性能比值 - 簇R-tree vs WISK
    if rtree_from_clusters:
        print("\n===== 簇R-tree vs WISK =====")
        cluster_node_ratio = avg_wisk_nodes / avg_rtree_cluster_nodes if avg_rtree_cluster_nodes > 0 else float('inf')
        cluster_object_ratio = avg_wisk_objects / avg_rtree_cluster_objects if avg_rtree_cluster_objects > 0 else float('inf')
        cluster_total_cost_ratio = (avg_wisk_nodes + avg_wisk_objects) / (avg_rtree_cluster_nodes + avg_rtree_cluster_objects) if (avg_rtree_cluster_nodes + avg_rtree_cluster_objects) > 0 else float('inf')

        print(f"节点访问比值: {cluster_node_ratio:.4f} {'(WISK更好)' if cluster_node_ratio < 1 else '(簇R-tree更好)'}")
        print(
            f"对象扫描比值: {cluster_object_ratio:.4f} {'(WISK更好)' if cluster_object_ratio < 1 else '(簇R-tree更好)'}")
        print(
            f"总代价比值: {cluster_total_cost_ratio:.4f} {'(WISK更好)' if cluster_total_cost_ratio < 1 else '(簇R-tree更好)'}")

    results = {
        'traditional': {
            'avg_rtree_nodes': avg_rtree_trad_nodes,
            'avg_rtree_objects': avg_rtree_trad_objects,
            'avg_wisk_nodes': avg_wisk_nodes,
            'avg_wisk_objects': avg_wisk_objects,
            'node_ratio': trad_node_ratio,
            'object_ratio': trad_object_ratio,
            'total_cost_ratio': trad_total_cost_ratio
        }
    }

    if rtree_from_clusters:
        results['cluster_based'] = {
            'avg_rtree_nodes': avg_rtree_cluster_nodes,
            'avg_rtree_objects': avg_rtree_cluster_objects,
            'avg_wisk_nodes': avg_wisk_nodes,
            'avg_wisk_objects': avg_wisk_objects,
            'node_ratio': cluster_node_ratio,
            'object_ratio': cluster_object_ratio,
            'total_cost_ratio': cluster_total_cost_ratio
        }

    return results