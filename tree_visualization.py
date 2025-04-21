import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_tree_layers(tree_structure, query_workload, original_data=None):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    """
    可视化树结构的每一层，显示：
    - 每个节点的空间MBR区域
    - 与每个节点匹配的查询数量（同时满足空间相交和关键词匹配）
    - 原始数据点

    参数:
        tree_structure: 列表的列表，每个内部列表代表树的一层节点
        query_workload: 用于匹配的查询对象列表
        original_data: 可选，原始数据点列表或DataFrame
    """
    # 获取层数
    num_layers = len(tree_structure)

    # 使用更鲜明的颜色方案
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    # 创建图形和坐标轴
    fig, axs = plt.subplots(1, num_layers, figsize=(6 * num_layers, 6), squeeze=False)
    fig.suptitle('层级树结构可视化', fontsize=16)

    # 存储全局最小/最大边界以保持一致的缩放
    global_min_lat = float('inf')
    global_max_lat = float('-inf')
    global_min_lon = float('inf')
    global_max_lon = float('-inf')

    # 首先找到全局边界(包括查询区域)
    for layer_idx, layer in enumerate(tree_structure):
        for node in layer:
            if node['MBR'] is not None:
                mbr = node['MBR']
                global_min_lat = min(global_min_lat, mbr['min_lat'])
                global_max_lat = max(global_max_lat, mbr['max_lat'])
                global_min_lon = min(global_min_lon, mbr['min_lon'])
                global_max_lon = max(global_max_lon, mbr['max_lon'])

    # 考虑查询MBR的边界
    for query in query_workload:
        if 'area' in query:
            area = query['area']
            global_min_lat = min(global_min_lat, area['min_lat'])
            global_max_lat = max(global_max_lat, area['max_lat'])
            global_min_lon = min(global_min_lon, area['min_lon'])
            global_max_lon = max(global_max_lon, area['max_lon'])

    # 添加边界缓冲区(5%)
    lat_buffer = (global_max_lat - global_min_lat) * 0.05
    lon_buffer = (global_max_lon - global_min_lon) * 0.05

    global_min_lat -= lat_buffer
    global_max_lat += lat_buffer
    global_min_lon -= lon_buffer
    global_max_lon += lon_buffer

    # 可视化每一层
    for layer_idx, layer in enumerate(tree_structure):
        ax = axs[0, layer_idx]
        ax.set_title(f'第 {layer_idx} 层', fontsize=14)
        ax.set_xlabel('经度', fontsize=12)
        ax.set_ylabel('纬度', fontsize=12)

        # 设置一致的坐标轴范围
        ax.set_xlim(global_min_lon, global_max_lon)
        ax.set_ylim(global_min_lat, global_max_lat)

        # 首先绘制查询MBR，使用统一的浅灰色
        for i, query in enumerate(query_workload):
            if 'area' in query:
                area = query['area']
                width = area['max_lon'] - area['min_lon']
                height = area['max_lat'] - area['min_lat']

                rect = Rectangle((area['min_lon'], area['min_lat']), width, height,
                                 linewidth=0.5, edgecolor='gray',
                                 facecolor='lightgray', alpha=0.1, linestyle='--')
                ax.add_patch(rect)

        # 然后绘制每个节点的MBR为矩形
        for node_idx, node in enumerate(layer):
            if node['MBR'] is None:
                continue

            # 为节点分配颜色
            node_color = distinct_colors[node_idx % len(distinct_colors)]

            mbr = node['MBR']
            width = mbr['max_lon'] - mbr['min_lon']
            height = mbr['max_lat'] - mbr['min_lat']

            # 计算完全匹配的查询数量（空间相交且关键词匹配）
            matching_queries = 0

            for query in query_workload:
                # 检查空间相交
                spatial_intersect = _mbr_intersect(mbr, query)

                # 检查关键词匹配
                keyword_intersect = any(kw in node['labels'] for kw in query['keywords'])

                # 只有同时满足空间和关键词匹配才计数
                if spatial_intersect and keyword_intersect:
                    matching_queries += 1

            # 创建带有节点颜色和一定透明度的矩形
            rect = Rectangle((mbr['min_lon'], mbr['min_lat']), width, height,
                             linewidth=1.5, edgecolor=node_color,
                             facecolor=node_color, alpha=0.5)
            ax.add_patch(rect)

            # 添加带有节点编号和匹配查询数的文本注释
            label_txt = f"节点 {node_idx}\n匹配查询: {matching_queries}"
            ax.text(mbr['min_lon'] + width / 2, mbr['min_lat'] + height / 2, label_txt,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

        # 最后绘制原始数据点(最上层)
        if original_data is not None:
            # 处理不同的数据格式
            if hasattr(original_data, 'iterrows'):  # DataFrame
                for _, row in original_data.iterrows():
                    ax.plot(row['longitude'], row['latitude'], 'k.', markersize=3, alpha=0.7)
            else:  # 假设是字典列表
                for obj in original_data:
                    if 'longitude' in obj and 'latitude' in obj:
                        ax.plot(obj['longitude'], obj['latitude'], 'k.', markersize=3, alpha=0.7)

    # 创建清晰的图例
    legend_elements = []

    # 为每层的每个节点添加图例
    for layer_idx, layer in enumerate(tree_structure):
        for node_idx, node in enumerate(layer):
            if node['MBR'] is None:
                continue

            color = distinct_colors[node_idx % len(distinct_colors)]
            label = f"第{layer_idx}层 节点{node_idx}"

            legend_elements.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.5,
                              edgecolor='black', label=label)
            )

            # 限制每层最多显示5个节点的图例
            if node_idx >= 4:
                legend_elements.append(
                    plt.Rectangle((0, 0), 0, 0, facecolor='white', alpha=0,
                                  label=f"... 等{len(layer) - 5}个节点")
                )
                break

    # 添加查询和数据点的图例
    legend_elements.append(
        plt.Rectangle((0, 0), 1, 1, facecolor='lightgray', alpha=0.3,
                      linestyle='--', edgecolor='gray', label="查询区域")
    )

    legend_elements.append(
        plt.Line2D([0], [0], marker='.', color='k', linestyle='',
                   markersize=8, label="原始数据点")
    )

    # 添加图例，使用多列排列
    if legend_elements:
        ncols = min(4, len(legend_elements))
        bbox_to_anchor = (0.5, -0.15) if num_layers <= 2 else (0.5, -0.25)
        fig.legend(handles=legend_elements, loc='upper center',
                   bbox_to_anchor=bbox_to_anchor, ncol=ncols,
                   fontsize=10, frameon=True, framealpha=0.9)

    # 调整布局并显示
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # 为图例留出更多空间
    plt.show()

    return fig


def _mbr_intersect(mbr, query):
    """
    辅助函数，检查MBR是否与查询区域相交
    """
    if 'area' not in query:
        return False

    area = query['area']
    return not (
            mbr['max_lat'] < area['min_lat'] or
            mbr['min_lat'] > area['max_lat'] or
            mbr['max_lon'] < area['min_lon'] or
            mbr['min_lon'] > area['max_lon']
    )


# 可视化最终树结构（层次视图）
def visualize_final_tree(root_node, query_workload, original_data=None):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    """
    可视化最终树结构，只展示树结构关系图

    参数:
        root_node: 最终树结构的根节点
        query_workload: 用于匹配的查询对象列表
        original_data: 可选，原始数据点列表或DataFrame
    """
    # 创建图形
    fig = plt.figure(figsize=(12, 9))
    # 创建树结构视图
    ax_tree = fig.add_subplot(111)

    fig.suptitle('最终树结构可视化', fontsize=18, fontweight='bold')

    # 提取所有节点及其连接，用于树结构图
    all_nodes = []
    node_connections = []

    def extract_nodes_and_connections(node, parent_idx=None, layer=0):
        current_idx = len(all_nodes)
        all_nodes.append({
            'node': node,
            'layer': layer,
            'index': current_idx
        })

        if parent_idx is not None:
            node_connections.append((parent_idx, current_idx))

        if 'children' in node and node['children']:
            for child in node['children']:
                if isinstance(child, dict) and child['MBR'] is not None:
                    extract_nodes_and_connections(child, current_idx, layer + 1)

    # 从根节点开始
    if isinstance(root_node, list):
        for node in root_node:
            extract_nodes_and_connections(node)
    else:
        extract_nodes_and_connections(root_node)

    # 树视图标题
    ax_tree.set_title('树结构关系', fontsize=16)

    # 计算节点位置
    max_layer = max(node['layer'] for node in all_nodes) if all_nodes else 0
    positions = {}

    # 使用分层布局策略
    for layer in range(max_layer + 1):
        layer_nodes = [n for n in all_nodes if n['layer'] == layer]

        for i, node_info in enumerate(layer_nodes):
            # 水平位置基于节点在当前层中的索引
            x = i / max(1, len(layer_nodes) - 1) if len(layer_nodes) > 1 else 0.5
            # 垂直位置基于层级
            y = 1 - (layer / max(1, max_layer))

            positions[node_info['index']] = (x, y)

    # 使用鲜明的颜色方案
    distinct_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # 首先绘制连接
    for parent_idx, child_idx in node_connections:
        parent_pos = positions[parent_idx]
        child_pos = positions[child_idx]
        ax_tree.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]], 'k-', alpha=0.7, linewidth=1.5)

    # 绘制节点
    for node_info in all_nodes:
        node = node_info['node']
        idx = node_info['index']
        pos = positions[idx]
        layer = node_info['layer']

        # 计算匹配的查询数量
        matching_queries = 0

        if node['MBR'] is not None:
            for query in query_workload:
                if 'area' in query:
                    spatial_intersect = _mbr_intersect(node['MBR'], query)
                    keyword_intersect = any(kw in node['labels'] for kw in query['keywords'])

                    if spatial_intersect and keyword_intersect:
                        matching_queries += 1

        # 基于子节点数量的节点大小
        num_children = len(node.get('children', []))
        node_size = 300 + (100 * num_children)

        # 节点颜色基于层级
        node_color = distinct_colors[layer % len(distinct_colors)]

        # 绘制节点
        ax_tree.scatter(pos[0], pos[1], s=node_size, alpha=0.8, c=[node_color],
                        edgecolor='black', linewidth=1.5)

        # 添加标签
        label = f"L{layer}_N{idx}\n匹配查询: {matching_queries}"
        ax_tree.text(pos[0], pos[1], label, ha='center', va='center', fontsize=9, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    # 移除树视图的坐标轴
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])
    ax_tree.set_xlim(-0.1, 1.1)
    ax_tree.set_ylim(-0.1, 1.1)

    # 创建图例
    # 层级图例
    layer_elements = []
    for layer in range(max_layer + 1):
        color = distinct_colors[layer % len(distinct_colors)]
        layer_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8,
                          edgecolor='black', linewidth=1.5, label=f"第 {layer} 层")
        )

    # 在右侧添加图例
    ax_tree.legend(handles=layer_elements,
                   loc='upper right', fontsize=10, frameon=True, title="层级图例",
                   title_fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为标题留出空间
    plt.show()

    return fig