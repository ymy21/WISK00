import pandas as pd
"""
R‑tree 索引的构建、查询以及对比函数，
支持两种方案：
  方案1：R‑tree 基于原始对象构建；查询代价包括内部节点访问和叶节点对象扫描代价。
  方案2：R‑tree 基于聚类构建；查询代价仅计算节点访问代价。
"""


class RTreeNode:
    def __init__(self, is_leaf):
        self.is_leaf = is_leaf
        self.entries = []  # 如果是叶节点，每个 entry 为 (object, mbr)；如果是内部节点，每个 entry 为子节点
        self.mbr = None  # 当前节点的最小外接矩形


def merge_mbr(mbr1, mbr2):
    """合并两个 MBR，返回新的 MBR"""
    return {
        'min_lat': min(mbr1['min_lat'], mbr2['min_lat']),
        'max_lat': max(mbr1['max_lat'], mbr2['max_lat']),
        'min_lon': min(mbr1['min_lon'], mbr2['min_lon']),
        'max_lon': max(mbr1['max_lon'], mbr2['max_lon'])
    }


def mbr_intersect(mbr1, mbr2):
    """判断两个 MBR 是否相交"""
    return not (mbr1['max_lat'] < mbr2['min_lat'] or
                mbr1['min_lat'] > mbr2['max_lat'] or
                mbr1['max_lon'] < mbr2['min_lon'] or
                mbr1['min_lon'] > mbr2['max_lon'])

def keyword_intersect(obj_keywords, query_keywords):
    """检查对象/聚类的关键词是否与查询关键词有交集"""
    return bool(set(obj_keywords) & set(query_keywords))

def calculate_intersecting_queries(subspace, query_workload):
    count = 0
    #print(f"subspace: {subspace}, type: {type(subspace)}")
    if not subspace.get('labels'):  # 无关键词的子空间直接过滤
        print(f"子空间无关键词labels。")
        return 0
    subspace_keywords = subspace.get('labels', set())
    for query in query_workload:
        # 检查空间交集
        spatial_intersect = (
                subspace['min_lat'] <= query['area']['max_lat'] and
                subspace['max_lat'] >= query['area']['min_lat'] and
                subspace['min_lon'] <= query['area']['max_lon'] and
                subspace['max_lon'] >= query['area']['min_lon']
        )
        # 检查关键词交集（使用集合操作）
        keyword_intersect = bool(set(query['keywords']) & subspace_keywords)

        if spatial_intersect and keyword_intersect:
            count += 1
    return count

def build_rtree_from_objects(objects, max_entries=10):
    """
    根据原始对象构建 R‑tree。
    每个对象需包含 'latitude' 和 'longitude' 字段，视为一个点，其 MBR 就是该点。
    """
    if isinstance(objects, pd.DataFrame):
        objects = objects.to_dict('records')
    leaves = []
    for obj in objects:
        mbr = {
            'min_lat': obj['latitude'],
            'max_lat': obj['latitude'],
            'min_lon': obj['longitude'],
            'max_lon': obj['longitude']
        }
        leaf = RTreeNode(is_leaf=True)
        leaf.entries.append((obj, mbr))
        leaf.mbr = mbr
        leaves.append(leaf)
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
            new_level.append(parent)
        nodes = new_level
    return nodes[0]


def build_rtree_from_clusters(clusters, max_entries=10):
    """
    根据聚类构建 R‑tree。
    每个聚类应包含 'MBR' 字段，视为一个整体。
    """
    leaves = []
    for cluster in clusters:
        mbr = cluster['MBR']
        leaf = RTreeNode(is_leaf=True)
        leaf.entries.append((cluster, mbr))
        leaf.mbr = mbr
        leaves.append(leaf)
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
            new_level.append(parent)
        nodes = new_level
    return nodes[0]


def search_rtree(query_rect, node, counter):
    """
    在 R‑tree 中搜索与 query_rect 相交的对象，同时统计访问的节点数。
    参数 counter 为列表（例如 [0]），用于累加访问节点数。
    """
    counter[0] += 1  # 访问当前节点
    results = []
    if node.is_leaf:
        for obj, obj_mbr in node.entries:
            if mbr_intersect(obj_mbr, query_rect):
                results.append(obj)
    else:
        for child in node.entries:
            if mbr_intersect(child.mbr, query_rect):
                results.extend(search_rtree(query_rect, child, counter))
    return results


def compare_scheme1(raw_objects, wisk_root, eval_queries):
    """
    方案1对比：
      - R‑tree：基于原始数据（对象作为点）构建，
        查询时计入节点访问及叶节点中对象扫描代价。
      - WISK：查询时计入内部节点访问和叶节点中扫描所有对象的代价。
    """
    print("【方案1】 R‑tree（原始数据） vs WISK（含对象扫描代价）")
    rtree_root = build_rtree_from_objects(raw_objects, max_entries=10)

    total_rtree_access = 0
    total_wisk_cost = 0
    for query in eval_queries:
        query_rect = query['area']
        counter_r = [0]
        _ = search_rtree(query_rect, rtree_root, counter_r)
        total_rtree_access += counter_r[0]

        counter_w = [0]
        # 假设 wisk_root 为 WISK 索引的根节点，其结构与底层节点保持一致，
        # 当查询到叶节点（聚类）时，每个对象的扫描代价为1。
        _ = search_wisk(query_rect, wisk_root, counter_w, count_objects=True)
        total_wisk_cost += counter_w[0]
        print(f"Query {query['keywords']} -> R‑tree访问: {counter_r[0]}, WISK总代价: {counter_w[0]}")

    avg_rtree = total_rtree_access / len(eval_queries)
    avg_wisk = total_wisk_cost / len(eval_queries)
    print(f"平均 R‑tree 访问节点数（原始数据）：{avg_rtree}")
    print(f"平均 WISK 查询总代价（含对象扫描）：{avg_wisk}")


def compare_scheme2(clusters, wisk_root, eval_queries):
    """
    方案2对比：
      - R‑tree：基于聚类构建（每个聚类作为一个整体），查询代价仅计节点访问数；
      - WISK：查询时不计聚类内对象扫描代价，仅计内部节点访问数。
    """
    print("【方案2】 R‑tree（基于聚类） vs WISK（仅节点访问）")
    rtree_root = build_rtree_from_clusters(clusters, max_entries=10)

    total_rtree_access = 0
    total_wisk_access = 0
    for query in eval_queries:
        query_rect = query['area']
        counter_r = [0]
        _ = search_rtree(query_rect, rtree_root, counter_r)
        total_rtree_access += counter_r[0]

        counter_w = [0]
        _ = search_wisk(query_rect, wisk_root, counter_w, count_objects=False)
        total_wisk_access += counter_w[0]
        print(f"Query {query['keywords']} -> R‑tree访问: {counter_r[0]}, WISK访问: {counter_w[0]}")

    avg_rtree = total_rtree_access / len(eval_queries)
    avg_wisk = total_wisk_access / len(eval_queries)
    print(f"平均 R‑tree 访问节点数（聚类）：{avg_rtree}")
    print(f"平均 WISK 节点访问数：{avg_wisk}")



def search_wisk(query_rect, node, counter, count_objects=True):
    """
    模拟 WISK 索引查询：
      - 若 node 为内部节点（其 children 中的元素均为字典且包含 'MBR'），则递归查询；
      - 否则认为 node 为叶节点（聚类），直接返回该聚类中的对象。
      当 count_objects 为 True 时，同时将聚类中每个对象的扫描代价计入 counter。
    """
    counter[0] += 1  # 访问当前节点
    results = []
    # 如果 node 是一个列表，则对列表中的每个节点递归调用 search_wisk
    if isinstance(node, list):
        for sub_node in node:
            results.extend(search_wisk(query_rect, sub_node, counter, count_objects))
        return results

    # 如果 node 的 children 存在且第一个子元素包含 'MBR'，则认为是内部节点
    if node.get('children') and len(node['children']) > 0 and 'MBR' in node['children'][0]:
        for child in node['children']:
            if mbr_intersect(child['MBR'], query_rect):
                results.extend(search_wisk(query_rect, child, counter, count_objects))
    else:
        # 叶节点：children 中存储的是原始对象
        results.extend(node.get('children', []))
        if count_objects:
            counter[0] += len(node.get('children', []))
    return results
