from data_preparation import load_real_dataset, generate_query_workload
from cdf_model import train_cdf_models, visualize_cdf_model
from bottom_clusters import bottom_clusters_generation, visualize_clusters, test_cost_estimation
from dqn_packing import (
    hierarchical_packing_training,
    final_tree_construction,
    build_nested_tree
)
from query_processing import process_query
from tree_visualization import visualize_tree_layers, visualize_final_tree
import pandas as pd
import torch
# 导入修改后的R-tree比较函数
from rtree_index import compare_rtree_wisk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import os
from measure_costs import measure_costs
def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 设置随机种子以确保实验可重复性
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 定义 device，若 GPU 可用则使用 cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. 准备数据
    print("Loading real dataset and generating query workload...")
    file_path = "dataset/dataset_BEIJING.csv"  # 更新为真实数据集的路径
    num_object = 2000
    data = load_real_dataset(file_path, num_objects=num_object)
    num_query = 50
    query_workload = generate_query_workload(data, num_queries=num_query, num_keywords=3, buffer=0.01)
    print("Dataset and query workload generated.")

    # 2. 训练 CDF 模型
    print("Training CDF models...")
    cdf_models = train_cdf_models(data,query_workload['train'])
    print("CDF models trained.")
    #可视化几个关键词的CDF模型
    # important_keywords = list(cdf_models.keys())[:20]  # 选择前20个关键词
    # for keyword in important_keywords:
    #     visualize_cdf_model(keyword, cdf_models[keyword], data)

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
    # test_cost_estimation([data_space[0]], query_workload['train'], cdf_models)

    # # 测量w1、w2
    # print("\n===== 测量簇扫描和对象验证成本 =====")
    # cluster_scan_time, object_verify_time, cost_ratio = measure_costs(
    #     objects,  # 原始数据点
    #     query_workload['train'],  # 训练查询
    #     None,  # 还没有簇，将自动创建
    #     num_trials=50,  # 减少trials数量以加快测试
    #     sample_size=min(1000, len(objects))  # 最多使用1000个对象
    # )

    # # 根据测量结果设置w1和w2
    # recommended_w1 = max(1, round(cost_ratio / 10))  # 简单的启发式方法
    # w1 = recommended_w1
    # w2 = 1
    # print(f"基于测量结果，将使用 w1={w1}, w2={w2}")


    print("Generating bottom clusters...")
    clusters = bottom_clusters_generation(query_workload['train'], data_space, cdf_models, w1=0.5, w2=1)
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

    # 簇可视化
    visualize_clusters(clusters, objects, train_queries, output_file='clustering_result.png')

    avg_objects_per_leaf = num_object // len(clusters)
    # 6. 强化学习训练阶段
    print("Starting reinforcement learning training phase...")

    agent, training_upper_layers, total_levels = hierarchical_packing_training(clusters, train_queries, max_level=25)
    print(f"Reinforcement learning training completed. Total levels: {total_levels}")

    # 7. 重跑构造阶段
    print("Starting final tree construction phase...")
    final_tree = final_tree_construction(clusters, train_queries, agent)
    print("Final tree structure constructed.")
    # 新增：可视化最终构建的树结构（包含查询和原始数据点）
    print("Visualizing final tree structure...")
    visualize_tree_layers(final_tree, train_queries, objects)

    # 8. 转换成树结构
    root_node = build_nested_tree(final_tree)
    print("Tree index structure built.")
    # 新增：可视化最终的嵌套树结构（包含查询和原始数据点）
    print("Visualizing final nested tree structure...")
    if root_node is not None:
        if isinstance(root_node, list):
            for i, root in enumerate(root_node):
                print(f"Visualizing root node {i + 1}/{len(root_node)}")
                visualize_final_tree(root, train_queries, objects)
        else:
            visualize_final_tree(root_node, train_queries, objects)

    # 9. 处理查询
    print("Processing queries...")
    eval_queries = query_workload['eval'].to_dict('records')
    for query in train_queries:
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
    print("\ntrain和compare用的同一个query集:")
    comparison_results = compare_rtree_wisk(objects, root_node, train_queries, avg_objects_per_leaf, clusters)  #用同一个数据集试试
    print("\ntrain和compare用的不同query集:")
    comparison_results = compare_rtree_wisk(objects, root_node, comp_queries, avg_objects_per_leaf, clusters)

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


# x-->longitude-->dim=1
# y-->latitude-->dim=0