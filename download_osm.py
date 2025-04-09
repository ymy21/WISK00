import overpy
import pandas as pd
import random
from shapely.geometry import Polygon

# 初始化 Overpass API
api = overpy.Overpass()

# 定义查询语句，增加了 bbox 限制条件
query = """
[out:json][timeout:180];
(
  node["leisure"~"recreation_ground|park"](39.6826,115.9112,40.1492,116.8375);
  way["leisure"~"recreation_ground|park"](39.6826,115.9112,40.1492,116.8375);
  relation["leisure"~"recreation_ground|park"](39.6826,115.9112,40.1492,116.8375);
);
out center;
"""

print("正在向 Overpass API 发送请求，请耐心等待...")
result = api.query(query)
print("查询完成。")

# 初始化存放对象数据的列表
objects = []

# 处理节点数据
for node in result.nodes:
    lat = float(node.lat)
    lon = float(node.lon)
    keyword = node.tags.get("name")
    if keyword:  # 仅在 keyword 非空时添加
        objects.append({"latitude": lat, "longitude": lon, "keyword": keyword})

# 处理面数据（way 类型）
for way in result.ways:
    if hasattr(way, "center_lat") and hasattr(way, "center_lon"):
        lat = float(way.center_lat)
        lon = float(way.center_lon)
    else:
        coords = [(float(n.lat), float(n.lon)) for n in way.nodes]
        try:
            polygon = Polygon(coords)
            centroid = polygon.centroid
            lat, lon = centroid.y, centroid.x
        except Exception:
            lat, lon = None, None
    keyword = way.tags.get("name")
    if lat is not None and lon is not None and keyword:  # 仅在 keyword 非空时添加
        objects.append({"latitude": lat, "longitude": lon, "keyword": keyword})

# 处理关系数据
for rel in result.relations:
    if hasattr(rel, "center_lat") and hasattr(rel, "center_lon"):
        lat = float(rel.center_lat)
        lon = float(rel.center_lon)
    else:
        lat, lon = None, None
    keyword = rel.tags.get("name")
    if lat is not None and lon is not None and keyword:  # 仅在 keyword 非空时添加
        objects.append({"latitude": lat, "longitude": lon, "keyword": keyword})

print(f"原始查询共获取对象数：{len(objects)}")

# 如果对象数量超过 10000，则随机抽样 10000 个
if len(objects) > 10000:
    objects = random.sample(objects, 10000)
    print("已随机抽取 10000 个对象。")
else:
    print("对象数量未超过 10000 个，将全部保存。")

# 将结果转换为 Pandas DataFrame 并保存为 CSV 文件
df = pd.DataFrame(objects)
csv_filename = "spatial_textual_objects.csv"
df.to_csv(csv_filename, index=False, encoding="utf-8")
print(f"数据已保存到 {csv_filename}")



