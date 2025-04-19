import pandas as pd
import numpy as np
from collections import defaultdict

"""
R-tree 索引的构建、查询以及对比函数，
支持与WISK索引的公平比较：
  - R-tree的叶子节点可以包含多个对象
  - 叶子节点的对象数量可以与WISK聚类中对象的平均数量保持一致
"""


class RTreeNode:
    def __init__(self, is_leaf):
        self.is_leaf = is_leaf
        self.entries = []  # 如果是叶节点，每个 entry 为 (objects_list, mbr)；如果是内部节点，每个 entry 为子节点
        self.mbr = None  # 当前节点的最小外接矩形
        self.keywords = [] # 节点包含的所有关键词

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
            for obj in objects_list:
                # 计数已扫描对象
                if obj_counter is not None:
                    obj_counter[0] += 1

                # 检查单个对象是否符合条件
                if (keyword_intersect(obj.get('keywords', []), query_keywords) and
                        'latitude' in obj and 'longitude' in obj and
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
            # 检查是否有leaf_objects字段（根据代码，这可能存储了叶子节点的对象）
            leaf_objects = current_node.get('leaf_objects', [])
            if leaf_objects:
                for obj in leaf_objects:
                    if obj_counter is not None:
                        obj_counter[0] += 1

                    # 获取对象关键词
                    obj_keywords = obj.get('keywords', [])

                    # 检查关键词匹配
                    if keyword_intersect(obj_keywords, query_keywords):
                        # 检查对象是否在查询区域内
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

def compare_rtree_wisk(raw_objects, wisk_index, eval_queries, avg_object):
    """
    对比R-tree和WISK索引的性能
    同时考虑节点访问数和对象扫描数
    """
    print("【R-tree vs WISK 性能比较】")
    # 构建R-tree，叶子节点包含多个对象
    rtree_root = build_rtree_with_grouped_objects(avg_object, raw_objects, wisk_index, max_entries=10)

    total_rtree_nodes = 0
    total_rtree_objects = 0
    total_wisk_nodes = 0
    total_wisk_objects = 0

    for i, query in enumerate(eval_queries):
        query_rect = query['area']
        query_keywords = query['keywords']

        # R-tree查询
        rtree_node_counter = [0]
        rtree_obj_counter = [0]
        rtree_results = search_rtree(query_rect, query_keywords, rtree_root, rtree_node_counter, rtree_obj_counter)

        # WISK查询
        wisk_node_counter = [0]
        wisk_obj_counter = [0]
        wisk_results = search_wisk(query_rect, query_keywords, wisk_index, wisk_node_counter, wisk_obj_counter)

        # 累加计数
        total_rtree_nodes += rtree_node_counter[0]
        total_rtree_objects += rtree_obj_counter[0]
        total_wisk_nodes += wisk_node_counter[0]
        total_wisk_objects += wisk_obj_counter[0]

        # 打印单次查询统计信息
        print(f"查询 {i + 1}（关键词：{query_keywords}）:")
        print(
            f"  R-tree: 访问节点数 = {rtree_node_counter[0]}, 扫描对象数 = {rtree_obj_counter[0]}, 结果数 = {len(rtree_results)}")
        print(
            f"  WISK:   访问节点数 = {wisk_node_counter[0]}, 扫描对象数 = {wisk_obj_counter[0]}, 结果数 = {len(wisk_results)}")

    # 计算平均值
    num_queries = len(eval_queries)
    avg_rtree_nodes = total_rtree_nodes / num_queries
    avg_rtree_objects = total_rtree_objects / num_queries
    avg_wisk_nodes = total_wisk_nodes / num_queries
    avg_wisk_objects = total_wisk_objects / num_queries

    # 打印总体统计信息
    print("\n总体统计:")
    print(f"查询数量: {num_queries}")
    print(f"R-tree: 平均访问节点数 = {avg_rtree_nodes:.2f}, 平均扫描对象数 = {avg_rtree_objects:.2f}")
    print(f"WISK:   平均访问节点数 = {avg_wisk_nodes:.2f}, 平均扫描对象数 = {avg_wisk_objects:.2f}")

    # 计算性能比值
    node_ratio = avg_wisk_nodes / avg_rtree_nodes if avg_rtree_nodes > 0 else float('inf')
    object_ratio = avg_wisk_objects / avg_rtree_objects if avg_rtree_objects > 0 else float('inf')
    total_cost_ratio = (avg_wisk_nodes + avg_wisk_objects) / (avg_rtree_nodes + avg_rtree_objects) if (
                                                                                                                  avg_rtree_nodes + avg_rtree_objects) > 0 else float(
        'inf')

    print("\n性能比较（WISK / R-tree）:")
    print(f"节点访问比值: {node_ratio:.4f} {'(WISK更好)' if node_ratio < 1 else '(R-tree更好)'}")
    print(f"对象扫描比值: {object_ratio:.4f} {'(WISK更好)' if object_ratio < 1 else '(R-tree更好)'}")
    print(f"总代价比值: {total_cost_ratio:.4f} {'(WISK更好)' if total_cost_ratio < 1 else '(R-tree更好)'}")

    return {
        'avg_rtree_nodes': avg_rtree_nodes,
        'avg_rtree_objects': avg_rtree_objects,
        'avg_wisk_nodes': avg_wisk_nodes,
        'avg_wisk_objects': avg_wisk_objects,
        'node_ratio': node_ratio,
        'object_ratio': object_ratio,
        'total_cost_ratio': total_cost_ratio
    }