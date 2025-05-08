import time
import random
import pandas as pd

def measure_costs(data, query_workload, clusters=None, num_trials=100, sample_size=1000):
    """
    测量簇扫描和对象验证的相对成本，用于确定w1和w2参数

    参数:
    data - 原始数据点列表
    query_workload - 查询工作负载
    clusters - 已有的簇列表（如果没有，使用单个包含所有数据的簇）
    num_trials - 测试重复次数
    sample_size - 对象样本大小

    返回:
    (簇扫描成本, 对象验证成本, 建议的w1/w2比率)
    """
    if isinstance(query_workload, pd.DataFrame):
        queries = query_workload.to_dict('records')
    else:
        queries = query_workload

    # 如果没有提供簇，创建一个包含所有数据的单一簇
    if not clusters:
        # 计算边界
        min_lat = min(obj['latitude'] for obj in data)
        max_lat = max(obj['latitude'] for obj in data)
        min_lon = min(obj['longitude'] for obj in data)
        max_lon = max(obj['longitude'] for obj in data)

        # 收集所有关键词
        all_keywords = set()
        for obj in data:
            all_keywords.update(obj['keywords'])

        # 创建单一簇
        clusters = [{
            'MBR': {
                'min_lat': min_lat,
                'max_lat': max_lat,
                'min_lon': min_lon,
                'max_lon': max_lon
            },
            'labels': list(all_keywords),
            'objects': data
        }]

    # 随机抽样一部分对象用于测试
    if isinstance(data, list) and len(data) > sample_size:
        sample_objects = random.sample(data, sample_size)
    else:
        sample_objects = data

    print(f"正在测量簇扫描和对象验证成本 (trials={num_trials})...")

    # 测量簇扫描成本
    start_time = time.time()
    for _ in range(num_trials):
        for cluster in clusters:
            # 模拟簇扫描操作 - 检查MBR和标签
            mbr = cluster.get('MBR', {})
            labels = cluster.get('labels', [])

            # 进行一些必要的操作，确保编译器不会优化掉这些代码
            _ = (mbr.get('min_lat', 0) + mbr.get('max_lat', 0)) / 2
            _ = sum(1 for _ in labels)

    cluster_scan_time = (time.time() - start_time) / (num_trials * len(clusters))

    # 测量对象验证成本
    start_time = time.time()
    for _ in range(num_trials):
        for obj in sample_objects:
            # 模拟对象验证操作
            for query in queries:
                # 空间匹配检查
                spatial_match = (
                        obj['latitude'] >= query['area']['min_lat'] and
                        obj['latitude'] <= query['area']['max_lat'] and
                        obj['longitude'] >= query['area']['min_lon'] and
                        obj['longitude'] <= query['area']['max_lon']
                )

                # 关键词匹配检查
                obj_keywords = obj.get('keywords', [])
                query_keywords = query.get('keywords', [])
                keyword_match = any(kw in query_keywords for kw in obj_keywords)

                # 使用结果防止编译器优化
                _ = spatial_match and keyword_match

    object_verify_time = (time.time() - start_time) / (num_trials * len(sample_objects) * len(queries))

    # 计算比率
    if object_verify_time > 0:
        ratio = cluster_scan_time / object_verify_time
    else:
        ratio = 100  # 默认值

    # 根据比率建议w1和w2
    if ratio < 10:
        suggestion = "低比率: 建议w1=1, w2=1 (系统可能主要在内存中运行)"
    elif ratio < 50:
        suggestion = f"中等比率: 建议w1={max(1, round(ratio / 10))}, w2=1 (可能是SSD或混合系统)"
    else:
        suggestion = f"高比率: 建议w1={max(5, round(ratio / 20))}, w2=1 (可能是传统磁盘系统)"

    print(f"簇扫描平均时间: {cluster_scan_time * 1000:.5f} 毫秒")
    print(f"对象验证平均时间: {object_verify_time * 1000:.5f} 毫秒")
    print(f"比率 (簇扫描/对象验证): {ratio:.2f}")
    print(suggestion)

    return cluster_scan_time, object_verify_time, ratio