import overpy
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import random
import time
import warnings

warnings.filterwarnings('ignore')

# 初始化 Overpass API
api = overpy.Overpass()

# 上海区域范围（扩大范围以获取更多数据）
SHANGHAI_REGION = {
    'name': 'shanghai',
    'min_lat': 30.65, 'min_lon': 121.05,  # 西南角
    'max_lat': 31.55, 'max_lon': 121.95  # 东北角
}

# 详细的POI类别到关键词映射（支持多关键词）
POI_TO_KEYWORDS = {
    # 餐饮类
    'restaurant': ['restaurant', 'dining', 'food'],
    'cafe': ['cafe', 'coffee', 'beverage'],
    'pub': ['pub', 'bar', 'drink'],
    'bar': ['bar', 'pub', 'drink'],
    'fast_food': ['fastfood', 'restaurant', 'food'],
    'food_court': ['foodcourt', 'restaurant', 'dining'],
    'bistro': ['bistro', 'restaurant', 'dining'],
    'chinese_restaurant': ['chinese', 'restaurant', 'dining'],

    # 购物类
    'supermarket': ['supermarket', 'grocery', 'shopping'],
    'mall': ['mall', 'shopping', 'center'],
    'department_store': ['department', 'shopping', 'retail'],
    'convenience': ['convenience', 'shop', 'store'],
    'shop': ['shop', 'retail', 'store'],
    'marketplace': ['market', 'shopping', 'bazaar'],
    'bookstore': ['bookshop', 'shop', 'bookstore'],
    'gift_shop': ['gift', 'shop', 'souvenir'],
    'clothes': ['clothing', 'fashion', 'shop'],
    'electronics': ['electronics', 'technology', 'shop'],
    'mobile_phone': ['mobile', 'phone', 'electronics'],

    # 服务类
    'bank': ['bank', 'finance', 'monetary'],
    'post_office': ['postal', 'mail', 'service'],
    'pharmacy': ['pharmacy', 'medicine', 'drug'],
    'dentist': ['dentist', 'dental', 'medical'],
    'clinic': ['clinic', 'medical', 'health'],
    'doctor': ['doctor', 'medical', 'physician'],
    'veterinary': ['veterinary', 'animal', 'medical'],
    'hairdresser': ['hairdresser', 'salon', 'beauty'],
    'beauty_salon': ['beauty', 'salon', 'spa'],
    'spa': ['spa', 'wellness', 'relaxation'],

    # 教育文化
    'school': ['school', 'education', 'academic'],
    'university': ['university', 'college', 'education'],
    'college': ['college', 'education', 'academic'],
    'kindergarten': ['kindergarten', 'preschool', 'education'],
    'library': ['library', 'book', 'reading'],
    'museum': ['museum', 'culture', 'history'],
    'gallery': ['gallery', 'art', 'exhibition'],
    'theatre': ['theatre', 'performance', 'entertainment'],
    'cinema': ['cinema', 'movie', 'entertainment'],
    'art_centre': ['art', 'culture', 'center'],
    'culture_centre': ['culture', 'community', 'center'],

    # 医疗健康
    'hospital': ['hospital', 'medical', 'healthcare'],
    'health_centre': ['health', 'medical', 'clinic'],
    'medical_centre': ['medical', 'health', 'clinic'],
    'pharmacy': ['pharmacy', 'medicine', 'health'],
    'emergency': ['emergency', 'hospital', 'medical'],
    'elderly_home': ['elderly', 'senior', 'care'],

    # 休闲娱乐
    'park': ['park', 'garden', 'recreation'],
    'garden': ['garden', 'park', 'nature'],
    'playground': ['playground', 'children', 'recreation'],
    'sports_centre': ['sports', 'fitness', 'exercise'],
    'gym': ['gym', 'fitness', 'exercise'],
    'stadium': ['stadium', 'sports', 'venue'],
    'swimming_pool': ['swimming', 'pool', 'sports'],
    'entertainment': ['entertainment', 'leisure', 'fun'],
    'karaoke': ['karaoke', 'entertainment', 'music'],
    'nightclub': ['nightclub', 'entertainment', 'music'],
    'arcade': ['arcade', 'game', 'entertainment'],

    # 交通设施
    'bus_station': ['bus', 'station', 'transport'],
    'metro_station': ['metro', 'subway', 'transport'],
    'train_station': ['train', 'station', 'transport'],
    'subway': ['subway', 'metro', 'transport'],
    'parking': ['parking', 'car', 'vehicle'],
    'bicycle_parking': ['bicycle', 'bike', 'parking'],
    'fuel': ['fuel', 'gas', 'petrol'],
    'taxi': ['taxi', 'cab', 'transport'],
    'charging_station': ['charging', 'electric', 'vehicle'],

    # 住宿类
    'hotel': ['hotel', 'accommodation', 'lodging'],
    'hostel': ['hostel', 'budget', 'accommodation'],
    'guest_house': ['guesthouse', 'accommodation', 'lodging'],
    'serviced_apartments': ['apartment', 'accommodation', 'residence'],

    # 宗教场所
    'place_of_worship': ['worship', 'religious', 'spiritual'],
    'church': ['church', 'christian', 'religious'],
    'temple': ['temple', 'religious', 'spiritual'],
    'mosque': ['mosque', 'islamic', 'religious'],
    'buddhist_temple': ['buddhist', 'temple', 'religious'],
    'taoist_temple': ['taoist', 'temple', 'religious'],

    # 公共设施
    'police': ['police', 'law', 'safety'],
    'fire_station': ['fire', 'emergency', 'safety'],
    'community_centre': ['community', 'center', 'social'],
    'public_building': ['public', 'government', 'building'],
    'townhall': ['townhall', 'government', 'municipal'],
    'public_toilet': ['toilet', 'bathroom', 'facility'],

    # 旅游景点
    'tourist_attraction': ['tourist', 'attraction', 'sightseeing'],
    'viewpoint': ['viewpoint', 'scenic', 'observation'],
    'information': ['information', 'tourist', 'service'],
    'tourist_info': ['tourist', 'information', 'service'],
    'monument': ['monument', 'memorial', 'historic'],
    'memorial': ['memorial', 'monument', 'historic'],
    'pagoda': ['pagoda', 'temple', 'historic'],
    'heritage_site': ['heritage', 'historic', 'culture'],

    # 历史文化
    'historic': ['historic', 'heritage', 'cultural'],
    'archaeological_site': ['archaeological', 'historic', 'heritage'],
    'ruins': ['ruins', 'historic', 'ancient'],
    'castle': ['castle', 'historic', 'palace'],
    'palace': ['palace', 'historic', 'royal'],
    'fort': ['fort', 'historic', 'military'],

    # 特色上海设施
    'tea_house': ['teahouse', 'tea', 'restaurant'],
    'hot_pot': ['hotpot', 'restaurant', 'dining'],
    'street_food': ['streetfood', 'food', 'local'],
    'traditional_medicine': ['traditional', 'medicine', 'health'],
    'pearl_tower': ['tower', 'landmark', 'tourist'],
    'bund': ['bund', 'waterfront', 'tourist']
}


def get_poi_keywords(tags):
    """从OSM标签提取多个关键词"""
    keywords = set()  # 使用集合避免重复

    # 按优先级检查标签类型
    tag_types = ['amenity', 'shop', 'leisure', 'tourism', 'historic', 'building',
                 'office', 'public_transport', 'healthcare', 'emergency', 'craft']

    for tag_type in tag_types:
        if tag_type in tags:
            value = tags[tag_type]
            # 处理多值标签（用分号分隔）
            values = value.split(';')
            for v in values:
                v = v.strip()
                # 查找对应的关键词列表
                if v in POI_TO_KEYWORDS:
                    keywords.update(POI_TO_KEYWORDS[v])
                else:
                    # 如果没有映射，使用原始值
                    keywords.add(v)

    # 确保至少有一个关键词
    if not keywords:
        keywords.add('poi')

    return list(keywords)


def get_objects_from_area(min_lat, min_lon, max_lat, max_lon, batch_size=0.03):  # 增大子区域大小
    """从大区域分批获取POI数据"""
    all_objects = []

    # 计算子区域数量
    lat_steps = int((max_lat - min_lat) / batch_size) + 1
    lon_steps = int((max_lon - min_lon) / batch_size) + 1

    print(f"准备从 {lat_steps * lon_steps} 个子区域收集数据...")

    for i in range(lat_steps):
        for j in range(lon_steps):
            sub_min_lat = min_lat + i * batch_size
            sub_max_lat = min(min_lat + (i + 1) * batch_size, max_lat)
            sub_min_lon = min_lon + j * batch_size
            sub_max_lon = min(min_lon + (j + 1) * batch_size, max_lon)

            print(f"正在查询子区域 ({i + 1}/{lat_steps}, {j + 1}/{lon_steps})...", end=' ')

            try:
                # 构建查询（获取所有类型的POI）
                query = f"""
                [out:json][timeout:90];
                (
                  node(
                    {sub_min_lat},{sub_min_lon},{sub_max_lat},{sub_max_lon}
                  )
                  [~"^(amenity|shop|leisure|tourism|historic|building|office|public_transport|healthcare|emergency|craft)$"~"."];

                  way(
                    {sub_min_lat},{sub_min_lon},{sub_max_lat},{sub_max_lon}
                  )
                  [~"^(amenity|shop|leisure|tourism|historic|building|office|public_transport|healthcare|emergency|craft)$"~"."];
                );
                out center;
                """

                result = api.query(query)
                sub_count = 0

                # 处理节点
                for node in result.nodes:
                    if hasattr(node, "lat") and hasattr(node, "lon"):
                        keywords = get_poi_keywords(node.tags)
                        all_objects.append({
                            "latitude": float(node.lat),
                            "longitude": float(node.lon),
                            "keywords": keywords
                        })
                        sub_count += 1

                # 处理路径
                for way in result.ways:
                    lat, lon = None, None

                    if hasattr(way, "center_lat") and hasattr(way, "center_lon"):
                        lat = float(way.center_lat)
                        lon = float(way.center_lon)
                    elif len(way.nodes) > 0:
                        # 计算多边形中心点
                        coords = []
                        for n in way.nodes:
                            if hasattr(n, "lat") and hasattr(n, "lon"):
                                coords.append((float(n.lat), float(n.lon)))

                        if len(coords) > 0:
                            lat = sum(c[0] for c in coords) / len(coords)
                            lon = sum(c[1] for c in coords) / len(coords)

                    if lat and lon:
                        keywords = get_poi_keywords(way.tags)
                        all_objects.append({
                            "latitude": lat,
                            "longitude": lon,
                            "keywords": keywords
                        })
                        sub_count += 1

                print(f"完成，收集 {sub_count} 个对象")
                time.sleep(0.5)  # 控制API访问频率

            except Exception as e:
                print(f"错误: {e}")
                time.sleep(2)
                continue

    return all_objects


def create_uniform_distribution(objects, grid_size=40, target_count=15000):  # 降低网格密度
    """创建均匀分布的数据集"""
    df = pd.DataFrame(objects)

    # 计算网格参数
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
    lat_step = (max_lat - min_lat) / grid_size
    lon_step = (max_lon - min_lon) / grid_size

    # 为每个数据点分配网格
    df['grid_x'] = ((df['longitude'] - min_lon) / lon_step).astype(int).clip(0, grid_size - 1)
    df['grid_y'] = ((df['latitude'] - min_lat) / lat_step).astype(int).clip(0, grid_size - 1)

    # 统计每个网格的数据点数量
    grid_counts = df.groupby(['grid_x', 'grid_y']).size()
    total_grids = grid_size * grid_size
    occupied_grids = len(grid_counts)

    print(f"网格使用情况：{occupied_grids}/{total_grids} ({occupied_grids / total_grids * 100:.1f}%)")

    # 每个网格的目标数据点数量
    target_per_grid = target_count // occupied_grids

    uniform_objects = []

    # 从每个网格采样数据点
    for (grid_x, grid_y), group in df.groupby(['grid_x', 'grid_y']):
        group_size = len(group)
        if group_size <= target_per_grid:
            selected = group
        else:
            selected = group.sample(n=target_per_grid, random_state=42)

        for _, row in selected.iterrows():
            uniform_objects.append({
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'keywords': row['keywords']
            })

    return uniform_objects


# 主程序
if __name__ == "__main__":
    region = SHANGHAI_REGION

    print(f"开始从{region['name']}收集POI数据...")
    print(f"区域范围：纬度 [{region['min_lat']}, {region['max_lat']}]，经度 [{region['min_lon']}, {region['max_lon']}]")

    # 收集数据（使用更大的子区域）
    raw_objects = get_objects_from_area(
        region['min_lat'], region['min_lon'],
        region['max_lat'], region['max_lon'],
        batch_size=0.03  # 扩大子区域大小
    )

    print(f"\n原始数据收集完成，共 {len(raw_objects)} 个对象")

    # 创建均匀分布数据集
    uniform_objects = create_uniform_distribution(raw_objects, grid_size=40, target_count=15000)

    print(f"均匀分布处理完成，保留 {len(uniform_objects)} 个对象")

    # 创建数据框
    df = pd.DataFrame(uniform_objects)

    # 统计信息
    print("\n数据集统计：")
    print(f"总数据点: {len(df)}")
    print(f"纬度范围: [{df['latitude'].min():.6f}, {df['latitude'].max():.6f}]")
    print(f"经度范围: [{df['longitude'].min():.6f}, {df['longitude'].max():.6f}]")

    # 关键词统计
    all_keywords = []
    for keywords in df['keywords']:
        all_keywords.extend(keywords)

    keyword_counts = pd.Series(all_keywords).value_counts()
    print(f"\n关键词种类数: {len(keyword_counts)}")
    print("\n最常见的20个关键词：")
    print(keyword_counts.head(20))

    # 平均关键词数
    avg_keywords = sum(len(kw) for kw in df['keywords']) / len(df)
    print(f"\n平均每个POI的关键词数: {avg_keywords:.2f}")

    # 可视化
    plt.figure(figsize=(18, 14))

    # 密度热力图
    plt.subplot(221)
    plt.hist2d(df['longitude'], df['latitude'], bins=30, cmap='viridis')
    plt.colorbar(label='密度')
    plt.title('上海POI密度分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')

    # 散点图
    plt.subplot(222)
    plt.scatter(df['longitude'], df['latitude'], s=3, alpha=0.5)
    plt.title('上海POI空间分布')
    plt.xlabel('经度')
    plt.ylabel('纬度')

    # 关键词分布条形图
    plt.subplot(223)
    keyword_counts.head(15).plot(kind='bar')
    plt.title('最常见的15个关键词')
    plt.xlabel('关键词')
    plt.ylabel('计数')
    plt.xticks(rotation=45)

    # 关键词数量分布
    plt.subplot(224)
    keyword_length_dist = pd.Series([len(kw) for kw in df['keywords']]).value_counts().sort_index()
    keyword_length_dist.plot(kind='bar')
    plt.title('每个POI的关键词数量分布')
    plt.xlabel('关键词数量')
    plt.ylabel('计数')

    plt.tight_layout()
    plt.savefig('shanghai_poi_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 保存数据
    csv_filename = "dataset/shanghai_poi_data.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8")
    print(f"\n数据已保存到 {csv_filename}")

    # 验证数据完整性
    print("\n数据验证：")
    print(f"空纬度: {df['latitude'].isnull().sum()}")
    print(f"空经度: {df['longitude'].isnull().sum()}")
    print(f"无关键词: {df[df['keywords'].apply(len) == 0].shape[0]}")

    # 显示样本数据
    print("\n样本数据（前5条）：")
    print(df.head())

    # 保存关键词统计
    keyword_stats = pd.DataFrame({
        'keyword': keyword_counts.index,
        'count': keyword_counts.values,
        'percentage': keyword_counts.values / len(all_keywords) * 100
    })
    keyword_stats.to_csv("dataset/shanghai_keyword_stats.csv", index=False)
    print(f"\n关键词统计已保存到 dataset/shanghai_keyword_stats.csv")