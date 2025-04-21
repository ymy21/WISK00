import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import matplotlib.patches as patches

# 全局关键词映射
KEYWORD_MAPPING = {}
GLOBAL_OBJECT_POOL = []
GLOBAL_OBJECT_INDEX = {}
# 这里直接用归一化后的经纬度，其值将始终在 [0, 1] 内
GLOBAL_GEO_BOUNDS = {
    'min_lat': 0.0,
    'max_lat': 1.0,
    'min_lon': 0.0,
    'max_lon': 1.0
}

def load_real_dataset(file_path, num_objects=None):
    """
    从 TSV 文件中加载真实数据集。文件包含 8 列：
      1. User ID
      2. Venue ID
      3. Venue category ID
      4. Venue category name (作为关键词，可能包含空格或特殊字符)
      5. Latitude
      6. Longitude
      7. Timezone offset
      8. UTC time

    返回 DataFrame，保留列：latitude, longitude, keywords，其中 keywords 为列表格式。
    此处直接对 latitude 与 longitude 进行归一化处理，归一化采用全局最小–最大方式。
    """
    global GLOBAL_OBJECT_POOL, GLOBAL_OBJECT_INDEX, KEYWORD_MAPPING

    # 修改1：添加正确的编码参数（尝试以下编码之一）
    df = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        names=[
            'user_id', 'venue_id', 'venue_category_id', 'venue_category_name',
            'latitude', 'longitude', 'timezone_offset', 'utc_time'
        ],
        encoding='ISO-8859-1'
    )

    # 修改2：检查实际数据行数是否足够
    if num_objects is not None and num_objects > len(df):
        print(f"Warning: Requested {num_objects} objects but only {len(df)} available")
        num_objects = len(df)

    if num_objects is not None:
        df = df.sample(n=num_objects, random_state=42).reset_index(drop=True)

    # 构建关键词映射，将 venue_category_name 编码为 int
    unique_keywords = df['venue_category_name'].str.strip().unique()
    KEYWORD_MAPPING = {kw: i for i, kw in enumerate(unique_keywords)}

    # 将每个 venue_category_name 转换为编码后的列表
    df['keywords'] = df['venue_category_name'].apply(lambda x: [KEYWORD_MAPPING[x.strip()]])
    df = df[['latitude', 'longitude', 'keywords']]

    # 计算全局经纬度边界（基于原始数据）
    min_lat = df['latitude'].min()
    max_lat = df['latitude'].max()
    min_lon = df['longitude'].min()
    max_lon = df['longitude'].max()

    # 对经纬度进行归一化：归一化到 [0, 1]
    df['latitude'] = df['latitude'].apply(lambda x: (x - min_lat) / (max_lat - min_lat + 1e-6))
    df['longitude'] = df['longitude'].apply(lambda x: (x - min_lon) / (max_lon - min_lon + 1e-6))

    # 更新全局对象池和索引（使用归一化后的经纬度）
    GLOBAL_OBJECT_POOL = df[['latitude', 'longitude', 'keywords']].to_dict('records')
    for idx, row in df.iterrows():
        GLOBAL_OBJECT_INDEX[idx] = row['keywords']  # 存储关键词索引

    # 更新全局经纬度边界为标准化后的值
    GLOBAL_GEO_BOUNDS['min_lat'] = 0.0
    GLOBAL_GEO_BOUNDS['max_lat'] = 1.0
    GLOBAL_GEO_BOUNDS['min_lon'] = 0.0
    GLOBAL_GEO_BOUNDS['max_lon'] = 1.0


#
# #观察原始数据分布
#     plt.figure(figsize=(12, 6))
#
#     # 绘制纬度的直方图
#     plt.subplot(1, 2, 1)
#     sns.histplot(df['latitude'], bins=50, kde=True)
#     plt.title('Latitude Distribution')
#
#     # 绘制经度的直方图
#     plt.subplot(1, 2, 2)
#     sns.histplot(df['longitude'], bins=50, kde=True)
#     plt.title('Longitude Distribution')
#
#     plt.tight_layout()
#     plt.show()
#     plt.figure(figsize=(6, 6))
#     stats.probplot(df['latitude'], dist="norm", plot=plt)
#     plt.title('Q-Q Plot of Latitude')
#     plt.show()
#
#     plt.figure(figsize=(6, 6))
#     stats.probplot(df['longitude'], dist="norm", plot=plt)
#     plt.title('Q-Q Plot of Longitude')
#     plt.show()

    return df


def sample_center(df, method='UNI', buffer=0.01):
    """
    从归一化后的数据集中采样一个中心点。
    buffer 的单位同样为归一化后的比例（范围[0,1]）。
    """
    if method == 'UNI':
        return df.sample(1).iloc[0]
    elif method == 'LAP':
        mu = df['latitude'].mean()
        b = df['latitude'].std() / 10
        lat = np.random.laplace(mu, b)
        lat = np.clip(lat, 0.0, 1.0)
        mu = df['longitude'].mean()
        b = df['longitude'].std() / 10
        lon = np.random.laplace(mu, b)
        lon = np.clip(lon, 0.0, 1.0)
    elif method == 'GAU':
        mu = df['latitude'].mean()
        sigma = df['latitude'].std() / 10
        lat = np.random.normal(mu, sigma)
        lat = np.clip(lat, 0.0, 1.0)
        mu = df['longitude'].mean()
        sigma = df['longitude'].std() / 10
        lon = np.random.normal(mu, sigma)
        lon = np.clip(lon, 0.0, 1.0)
    elif method == 'MIX':
        if random.random() < 0.5:
            return sample_center(df, 'UNI', buffer)
        else:
            return sample_center(df, 'LAP', buffer)
    else:
        raise ValueError("Unknown sampling method")

    filtered_df = df[(df['latitude'] >= lat - buffer) & (df['latitude'] <= lat + buffer) &
                     (df['longitude'] >= lon - buffer) & (df['longitude'] <= lon + buffer)]
    if filtered_df.empty:
        print("Haven't find center...")
        return df.iloc[df.shape[0] // 2]
    return filtered_df.sample(1).iloc[0]


def generate_query_workload(df, num_queries=500, num_keywords=5, buffer=0.01):
    """
    基于归一化后的数据生成查询工作负载：
      - 采样一个对象作为中心，构造归一化的矩形查询区域
      - 查询区域基于 buffer 参数（归一化值）
      - 生成的查询返回原始格式，且经纬度值为归一化后的值，
        保证与数据集和训练过程保持一致
    最后返回字典，划分为 'train'、'compare'、'eval' 三部分。
    """
    queries = []
    global_keywords = list(set(df['keywords'].explode().unique()))
    for _ in range(num_queries):
        center = sample_center(df, method='MIX', buffer=buffer)
        query_min_lat = center['latitude'] - buffer
        query_max_lat = center['latitude'] + buffer
        query_min_lon = center['longitude'] - buffer
        query_max_lon = center['longitude'] + buffer
        # 保证查询区域在 [0, 1] 内
        query_min_lat = max(query_min_lat, 0.0)
        query_max_lat = min(query_max_lat, 1.0)
        query_min_lon = max(query_min_lon, 0.0)
        query_max_lon = min(query_max_lon, 1.0)
        # 生成查询的关键词
        if len(center['keywords']) >= num_keywords:
            query_keywords = random.sample(center['keywords'], num_keywords)
        else:
            remaining_keywords = list(set(global_keywords) - set(center['keywords']))
            query_keywords = center['keywords'] + random.sample(remaining_keywords,
                                                                num_keywords - len(center['keywords']))
        queries.append({
            'area': {
                'min_lat': query_min_lat,
                'max_lat': query_max_lat,
                'min_lon': query_min_lon,
                'max_lon': query_max_lon
            },
            'keywords': query_keywords
        })
    num_train = int(num_queries * 0.6)
    num_build = int(num_queries * 0.2)
    workload = {
        'train': pd.DataFrame(queries[:num_train]),
        'compare': pd.DataFrame(queries[num_train:num_train + num_build]),
        'eval': pd.DataFrame(queries[num_train + num_build:])
    }

    # 假设 df 是你加载的原始数据，workload 是通过 generate_query_workload 得到的查询集合（例如取 train 部分）
    # 如果 workload 是一个字典，这里我们取 'train' 部分
    queries = workload['train'].to_dict('records')

    plt.figure(figsize=(10, 10))
    # 绘制所有原始数据点，蓝色散点图
    plt.scatter(df['longitude'], df['latitude'], s=10, c='blue', alpha=0.5, label='Data points')

    # 对于每个 query，绘制其矩形区域（红色边框）
    for query in queries:
        area = query['area']
        width = area['max_lon'] - area['min_lon']
        height = area['max_lat'] - area['min_lat']
        rect = patches.Rectangle((area['min_lon'], area['min_lat']), width, height, edgecolor='red', facecolor='none',
                                 lw=2)
        plt.gca().add_patch(rect)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Query Regions Overlay on Data Points')
    plt.legend()
    plt.show()
    return workload

# data 是一个 DataFrame，包含 latitude、longitude 和 keywords 列。keywords 列是一个列表，包含每个对象的关键词。
# query_workload 是一个 DataFrame，包含 area 和 keywords 列。area 是一个字典，包含 min_lat、max_lat、min_lon 和 max_lon。keywords 是一个列表，包含查询的关键词。

# # 1. 准备数据
#
# if __name__ == '__main__':
#
#     print("Loading real dataset and generating query workload...")
#     file_path = "dataset/dataset_TSMC2014_NYC.txt"  # 更新为真实数据集的路径
#     num_object = 2000
#     data = load_real_dataset(file_path, num_objects=num_object)
#     num_query = 500
#     query_workload = generate_query_workload(data, num_queries=num_query, num_keywords=5, buffer=0.01)
#     print("Dataset and query workload generated.")