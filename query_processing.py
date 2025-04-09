def process_query(query, index):
    """
    query：包含 'area' 和 'keywords'
    index：可以是一个节点字典（具有 'MBR' 和 'children'）或者节点列表
    返回匹配的对象列表。
    """
    results = []
    # 如果 index 是列表，则对每个节点递归调用
    if isinstance(index, list):
        for node in index:
            results.extend(process_query(query, node))
        return results

    mbr = index.get('MBR')
    if mbr is None:
        return results

    # 判断查询区域与当前节点的 MBR 是否有交集
    if (mbr['min_lat'] <= query['area']['max_lat'] and
            mbr['max_lat'] >= query['area']['min_lat'] and
            mbr['min_lon'] <= query['area']['max_lon'] and
            mbr['max_lon'] >= query['area']['min_lon']):

        # 递归处理子节点
        for child in index.get('children', []):
            results.extend(process_query(query, child))

        # 处理叶节点对象
        for obj in index.get('leaf_objects', []):
            if (query['area']['min_lat'] <= obj['latitude'] <= query['area']['max_lat'] and
                    query['area']['min_lon'] <= obj['longitude'] <= query['area']['max_lon'] and
                    any(keyword in obj['keywords'] for keyword in query['keywords'])):
                results.append(obj)
    return results


