from data_preparation import load_real_dataset, generate_query_workload
from cdf_model import train_cdf_models
from bottom_clusters import bottom_clusters_generation
from dqn_packing import (
    hierarchical_packing_training,
    final_tree_construction,
    build_nested_tree
)
from query_processing import process_query
import pandas as pd
import torch
# 导入修改后的R-tree比较函数
from rtree_index import compare_rtree_wisk
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 定义 device，若 GPU 可用则使用 cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. 准备数据
    print("Loading real dataset and generating query workload...")
    file_path = "dataset/dataset_TSMC2014_NYC.txt"  # 更新为真实数据集的路径
    num_object = 200
    data = load_real_dataset(file_path, num_objects=num_object)
    num_query = 100
    query_workload = generate_query_workload(data, num_queries=num_query, num_keywords=5, buffer=0.01)
    print("Dataset and query workload generated.")

    # 2. 训练 CDF 模型
    print("Training CDF models...")
    cdf_models = train_cdf_models(data, query_workload['train'])
    print("CDF models trained.")

    # 3. 定义数据空间（改为列表形式）
    # 初始化变量
    min_lat = float('inf')
    max_lat = -float('inf')
    min_lon = float('inf')
    max_lon = -float('inf')
    objects = []
    all_keywords = set()

    # 单次遍历数据
    for _, row in data.iterrows():
        lat = row['latitude']
        lon = row['longitude']

        # 更新地理坐标边界
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)

        # 处理关键词数据
        kw_list = list(row['keywords'])
        unique_kws = set(kw_list)
        all_keywords.update(unique_kws)

        # 构建对象集合
        objects.append({
            'latitude': lat,
            'longitude': lon,
            'keywords': kw_list
        })

    # 构建最终数据结构
    data_space = [{
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'objects': objects,
        'labels': frozenset(all_keywords)
    }]

    # 4. 生成底层聚类（clusters）
    print("Generating bottom clusters...")
    clusters = bottom_clusters_generation(query_workload['train'], data_space, cdf_models, w1=0.1, w2=1)
    print(f"Bottom clusters generated: {len(clusters)} clusters.")

    # 5. 构建底层节点
    print("Building bottom nodes...")
    #bottom_nodes = []
    train_queries = query_workload['train']
    if isinstance(train_queries, pd.DataFrame):
        train_queries = train_queries.to_dict('records')  # 转换为字典列表
    # # 对cluster做简单处理，变成dqn最底层节点
    # for cluster in clusters:
    #     labels = set()
    #     mbr = {
    #         'min_lat': cluster['min_lat'],
    #         'max_lat': cluster['max_lat'],
    #         'min_lon': cluster['min_lon'],
    #         'max_lon': cluster['max_lon']
    #     }
    #     # for obj in cluster['objects']:
    #     #     labels.update(obj['keywords'])
    #     bottom_node= {
    #         'layer': 0,
    #         'MBR': mbr,
    #         'labels': list(cluster['labels']), #上面底部簇生成也用了labels，将集合转化为列表
    #         'children': [],         # 下层节点引用（此处为空，因为叶节点不再重复存储原始对象）
    #         'leaf_objects': cluster['objects']  # 将原始对象存于 leaf_objects 中
    #     }
    #     bottom_nodes.append(bottom_node)
    print(f"Bottom nodes built: {len(clusters)} nodes.")
    avg_objects_per_leaf = num_object // len(clusters)
    # 6. 强化学习训练阶段
    print("Starting reinforcement learning training phase...")

    agent, training_upper_layers, total_levels = hierarchical_packing_training(clusters, train_queries, max_level=25)
    print(f"Reinforcement learning training completed. Total levels: {total_levels}")

    # 7. 重跑构造阶段
    print("Starting final tree construction phase...")
    final_tree = final_tree_construction(clusters, train_queries, agent)
    print("Final tree structure constructed.")

    # 8. 转换成树结构
    root_node = build_nested_tree(final_tree)
    print("Tree index structure built.")

    # 9. 处理查询
    print("Processing queries...")
    eval_queries = query_workload['eval'].to_dict('records')
    for query in eval_queries:
        result = process_query(query, root_node)
        print(f"Query: {query['keywords']}, Result: {len(result)} objects found.")
    print("Query processing completed.")

    # 10.比较wisk和r-tree
    comp_queries = query_workload['compare'].to_dict('records')

    print("\nComparing query costs between R‑tree and WISK index:")
    objects = []
    for _, row in data.iterrows():
        objects.append({
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'keywords': list(row['keywords'])
        })
    # 调用增强版的比较函数
    comparison_results = compare_rtree_wisk(objects, root_node, comp_queries,avg_objects_per_leaf)

    # # 可以选择绘制图表进行可视化
    # if sns is not None:  # 确保seaborn已安装
    #     plt.figure(figsize=(15, 5))
    #
    #     # 节点访问对比
    #     node_stats = {'R-tree': comparison_results['avg_rtree_nodes'], 'WISK': comparison_results['avg_wisk_nodes']}
    #     plt.subplot(131)
    #     sns.barplot(x=list(node_stats.keys()), y=list(node_stats.values()))
    #     plt.title('节点访问数对比')
    #     plt.ylabel('平均节点访问数')
    #
    #     # 对象扫描对比
    #     object_stats = {'R-tree': comparison_results['avg_rtree_objects'],
    #                     'WISK': comparison_results['avg_wisk_objects']}
    #     plt.subplot(132)
    #     sns.barplot(x=list(object_stats.keys()), y=list(object_stats.values()))
    #     plt.title('对象扫描数对比')
    #     plt.ylabel('平均对象扫描数')
    #
    #     # 总代价对比
    #     total_stats = {
    #         'R-tree': comparison_results['avg_rtree_nodes'] + comparison_results['avg_rtree_objects'],
    #         'WISK': comparison_results['avg_wisk_nodes'] + comparison_results['avg_wisk_objects']
    #     }
    #     plt.subplot(133)
    #     sns.barplot(x=list(total_stats.keys()), y=list(total_stats.values()))
    #     plt.title('总代价对比')
    #     plt.ylabel('平均总代价')
    #
    #     plt.tight_layout()
    #     plt.savefig('rtree_vs_wisk_performance.png')
    #     plt.close()

if __name__ == "__main__":
    main()
