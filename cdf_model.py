import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import QuantileTransformer  # 引入分位数变换工具
from scipy.stats import norm  # 用于计算标准正态CDF或高斯近似
import matplotlib.pyplot as plt
from torch.nn import Linear
from data_preparation import GLOBAL_GEO_BOUNDS

# 全局缓存字典，避免重复计算 CDF 模型的前向传播
cdf_cache = {}


def get_cdf_value(model, value, keyword, dimension):
    """
    获取CDF值，并使用缓存避免重复计算
    """
    # 统一键的格式，确保缓存一致性
    key = (int(keyword), float(value.item()) if isinstance(value, torch.Tensor) else float(value))

    if key not in cdf_cache:
        if model['gaussian']:
            # 高斯近似处理
            min_val = GLOBAL_GEO_BOUNDS['min_lon'] if dimension == 'lon' else GLOBAL_GEO_BOUNDS['min_lat']
            max_val = GLOBAL_GEO_BOUNDS['max_lon'] if dimension == 'lon' else GLOBAL_GEO_BOUNDS['max_lat']
            range_val = max_val - min_val
            normalized_value = (value - min_val) / range_val
            params = model['x'] if dimension == 'lon' else model['y']
            cdf = norm.cdf(normalized_value, loc=params['mean'], scale=params['std'])
            cdf_cache[key] = cdf
        else:
            # 神经网络模型处理
            model_nn = model['x'] if dimension == 'lon' else model['y']
            value_tensor = torch.tensor([[value]], dtype=torch.float32) if not isinstance(value,
                                                                                          torch.Tensor) else value.view(
                1, 1)
            with torch.no_grad():
                result = model_nn(value_tensor)
                cdf_cache[key] = result.item()
        return cdf_cache[key]


# 全局变量：保存查询工作负载中各关键词的频率
KEYWORD_QUERY_FREQ = {}


# 严格单调线性层，保证权重非负（通过绝对值实现）
class MonotonicLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MonotonicLinear, self).__init__()
        self.weight_raw = nn.Parameter(torch.randn(out_features, in_features))
        self.bias_raw = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        weight = torch.abs(self.weight_raw)  # 绝对值保证单调性
        return torch.matmul(x, weight.t()) + self.bias_raw


# 修改后的 CDFModel 使用 MonotonicLinear 层
class CDFModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_layers=4, dimension='lon'):
        super(CDFModel, self).__init__()
        self.dimension = dimension
        self.min_val = GLOBAL_GEO_BOUNDS['min_lon'] if dimension == 'lon' else GLOBAL_GEO_BOUNDS['min_lat']
        self.max_val = GLOBAL_GEO_BOUNDS['max_lon'] if dimension == 'lon' else GLOBAL_GEO_BOUNDS['max_lat']
        self.range_val = self.max_val - self.min_val
        if self.range_val <= 0:
            raise ValueError(f"Invalid range for {dimension}: min={self.min_val}, max={self.max_val}")
        self.layers = nn.ModuleList()
        # 第一层
        self.layers.append(Linear(input_dim, hidden_dim))
        # 中间隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(Linear(hidden_dim, hidden_dim))
        # 输出层
        self.layers.append(Linear(hidden_dim, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入x应为原始值，内部自动归一化
        x_normalized = (x - self.min_val) / self.range_val
        for layer in self.layers[:-1]:
            x_normalized = torch.relu(layer(x_normalized))
        # 输出层后接 sigmoid 确保输出在 [0,1] 范围内
        return self.sigmoid(self.layers[-1](x_normalized))


def compute_cdf_targets(data, sorted_indices=None):
    """
    计算基于排序的CDF目标值

    Args:
        data: 输入数据
        sorted_indices: 如果已经排序，可以直接传入排序索引

    Returns:
        对应的CDF目标值
    """
    n_samples = len(data)
    if sorted_indices is None:
        sorted_data, sorted_indices = torch.sort(data.view(-1))

    # 计算均匀分布的CDF值 (0到1)
    cdf_values = torch.linspace(0, 1, n_samples)
    target_cdf = torch.zeros(n_samples)
    for i in range(n_samples):
        target_cdf[sorted_indices[i]] = cdf_values[i]

    return target_cdf.view(-1, 1)


def train_cdf_models(data, query_workload, epochs=100, lr=0.06):
    """
    训练CDF模型，为每个关键词创建两个CDF模型（x和y方向）

    根据论文要求：
      - 低频关键词（频率 ≤ 0.001‰）：忽略训练
      - 中频关键词（0.001‰ < 频率 < 0.1‰）：采用高斯近似
      - 高频关键词（频率 ≥ 0.1‰）：采用NN拟合排序得到的CDF

    同时绘制训练结果图像，方便调试。

    Args:
        data: 数据集
        query_workload: 查询工作负载（DataFrame或字典列表）
        epochs: NN训练轮数（仅对高频关键词有效）
        lr: 学习率

    Returns:
        包含每个关键词对应模型或高斯参数的字典
    """
    print("开始CDF模型训练...")

    # 转换为字典列表格式
    query_workload = query_workload.to_dict('records') if isinstance(query_workload, pd.DataFrame) else query_workload
    global GLOBAL_GEO_BOUNDS
    # 遍历数据集中的每个对象，统计关键词的出现次数
    keyword_counts = {}
    total_keyword_num = 0
    for _, row in data.iterrows():
        for kw in row['keywords']:
            keyword_counts[kw] = keyword_counts.get(kw, 0) + 1
            total_keyword_num += 1
    # 统计关键词频率（计算每个关键词在数据集中的频率（单位‰））
    # total_data = len(data) #要改：不是所有的keyword的数量，可能不是1对1的关系
    keyword_freq = {kw: (count / total_keyword_num)  for kw, count in keyword_counts.items()}

    # 定义频率阈值
    low_threshold = 0.000001  # 低频关键词
    medium_threshold =0.0001  # 中频关键词

    # 提取所有关键词（从查询和数据集的并集）
    keywords = set(keyword_counts.keys())

    cdf_models = {}
    global cdf_cache
    cdf_cache = {}
    # 获取全局边界参数
    min_lon = GLOBAL_GEO_BOUNDS['min_lon']
    max_lon = GLOBAL_GEO_BOUNDS['max_lon']
    min_lat = GLOBAL_GEO_BOUNDS['min_lat']
    max_lat = GLOBAL_GEO_BOUNDS['max_lat']

    # 计算归一化范围（防止除零）
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    for keyword in keywords:
        freq = keyword_freq.get(keyword, 0)
        total_kw = keyword_counts.get(keyword, 0)  # 记录每个关键词的总频次
        # print(f"处理关键词 {keyword}, 频率: {freq:.6f}")

        # 低频关键词直接忽略
        if freq <= low_threshold:
            # print(f"关键词 {keyword} 是低频关键词 (freq={freq:.6f})，忽略。")
            continue

        # 获取包含该关键词的数据
        keyword_data = data[data['keywords'].apply(lambda x: keyword in x)]
        if len(keyword_data) < 10:
            # print(f"警告: 关键词 {keyword} 的数据量不足 ({len(keyword_data)} 条)，跳过训练。")
            continue

        # print(f"使用 {len(keyword_data)} 条数据训练关键词 {keyword} 的模型")

        # 使用原始经纬度数据
        x_vals_orig = keyword_data['longitude'].values.reshape(-1, 1)
        y_vals_orig = keyword_data['latitude'].values.reshape(-1, 1)

        # 计算归一化后的数据用于目标CDF
        x_vals_norm = (x_vals_orig - min_lon) / lon_range
        y_vals_norm = (y_vals_orig - min_lat) / lat_range

        if freq < medium_threshold:
            # 中频关键词：使用高斯近似
            # print(f"关键词 {keyword} 为中频关键词，采用高斯近似")
            # 直接使用原始数据计算均值和标准差
            x_mean = x_vals_orig.mean().item()
            x_std = max(x_vals_orig.std().item(), 1e-6)
            y_mean = y_vals_orig.mean().item()
            y_std = max(y_vals_orig.std().item(), 1e-6)


            # 将高斯参数保存，不训练NN
            cdf_models[keyword] = {
                'gaussian': True,
                'x_mean': x_mean, 'x_std': x_std,
                'y_mean': y_mean, 'y_std': y_std,
                'total_kw': total_kw  # 保存关键词总频数
            }
            # # 绘制高斯近似的CDF曲线
            # sort_idx_x = np.argsort(x_data.numpy().flatten())
            # sort_idx_y = np.argsort(y_data.numpy().flatten())
            # sorted_x = x_data.numpy().flatten()[sort_idx_x]
            # sorted_target_x = target_x.numpy().flatten()[sort_idx_x]
            # sorted_y = y_data.numpy().flatten()[sort_idx_y]
            # sorted_target_y = target_y.numpy().flatten()[sort_idx_y]
            #
            # plt.figure(figsize=(12, 5))
            # plt.subplot(1, 2, 1)
            # plt.plot(sorted_x, sorted_target_x, label='Gaussian CDF', marker='o', linestyle='-')
            # plt.title(f"Keyword: {keyword} - Longitude (X) Gaussian CDF")
            # plt.xlabel("Longitude")
            # plt.ylabel("CDF")
            # plt.legend()
            # plt.subplot(1, 2, 2)
            # plt.plot(sorted_y, sorted_target_y, label='Gaussian CDF', marker='o', linestyle='-')
            # plt.title(f"Keyword: {keyword} - Latitude (Y) Gaussian CDF")
            # plt.xlabel("Latitude")
            # plt.ylabel("CDF")
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

        else:
            # 高频关键词：采用NN预测CDF
            # print(f"关键词 {keyword} 为高频关键词，采用 NN 拟合排序的 CDF")
            # # 使用 QuantileTransformer 计算目标 CDF（基于排序）
            # qt_x = QuantileTransformer(output_distribution='uniform', random_state=0)
            # qt_y = QuantileTransformer(output_distribution='uniform', random_state=0)
            # x_vals = keyword_data['longitude'].values.reshape(-1, 1)
            # y_vals = keyword_data['latitude'].values.reshape(-1, 1)
            # target_x_np = qt_x.fit_transform(x_vals)
            # target_y_np = qt_y.fit_transform(y_vals)
            # target_x = torch.tensor(target_x_np, dtype=torch.float32)
            # target_y = torch.tensor(target_y_np, dtype=torch.float32)

            # 获取经度、纬度数据
            # x_vals = keyword_data['longitude'].values.reshape(-1, 1)
            # y_vals = keyword_data['latitude'].values.reshape(-1, 1)

            # 高频：直接使用原始坐标数据训练
            x_data = torch.tensor(x_vals_orig, dtype=torch.float32)
            y_data = torch.tensor(y_vals_orig, dtype=torch.float32)

            # 排序后的目标CDF（基于原始数据排序）
            sorted_x = np.sort(x_vals_orig.flatten())
            sorted_y = np.sort(y_vals_orig.flatten())
            target_x = torch.tensor(
                np.arange(1, len(sorted_x) + 1) / len(sorted_x),
                dtype=torch.float32
            ).view(-1, 1)
            target_y = torch.tensor(
                np.arange(1, len(sorted_y) + 1) / len(sorted_y),
                dtype=torch.float32
            ).view(-1, 1)

            # 初始化模型（自动处理归一化）
            model_x = CDFModel(dimension='lon')
            model_y = CDFModel(dimension='lat')
            criterion = nn.MSELoss()
            optimizer_x = optim.SGD(model_x.parameters(), lr=lr)
            optimizer_y = optim.SGD(model_y.parameters(), lr=lr)

            # 训练X方向
            model_x.train()
            for epoch in range(epochs):
                optimizer_x.zero_grad()
                outputs = model_x(x_data)
                loss = criterion(outputs, target_x)
                loss.backward()
                optimizer_x.step()
                # print(f"关键词: {keyword}, X方向, Epoch {epoch}, Loss: {loss.item():.6f}")

            # 训练Y方向
            model_y.train()
            for epoch in range(epochs):
                optimizer_y.zero_grad()
                outputs = model_y(y_data)
                loss = criterion(outputs, target_y)
                loss.backward()
                optimizer_y.step()
                # print(f"关键词: {keyword}, Y方向, Epoch {epoch}, Loss: {loss.item():.6f}")

            cdf_models[keyword] = {
                'gaussian': False,
                'x': model_x,
                'y': model_y,
                'total_kw': total_kw
            }

            # # 绘制 NN 预测的CDF与目标CDF对比图
            # with torch.no_grad():
            #     pred_x = model_x(x_data).detach().numpy().flatten()
            #     pred_y = model_y(y_data).detach().numpy().flatten()
            # x_vals = x_data.numpy().flatten()
            # y_vals = y_data.numpy().flatten()
            # target_x_np = target_x.numpy().flatten()
            # target_y_np = target_y.numpy().flatten()
            # sort_idx_x = np.argsort(x_vals)
            # sort_idx_y = np.argsort(y_vals)
            # sorted_x = x_vals[sort_idx_x]
            # sorted_pred_x = pred_x[sort_idx_x]
            # sorted_target_x = target_x_np[sort_idx_x]
            # sorted_y = y_vals[sort_idx_y]
            # sorted_pred_y = pred_y[sort_idx_y]
            # sorted_target_y = target_y_np[sort_idx_y]
            #
            # plt.figure(figsize=(12, 5))
            # plt.subplot(1, 2, 1)
            # plt.plot(sorted_x, sorted_pred_x, label='Predicted CDF', marker='o', linestyle='-')
            # plt.plot(sorted_x, sorted_target_x, label='Target CDF', marker='x', linestyle='--')
            # plt.title(f"Keyword: {keyword} - Longitude (X)")
            # plt.xlabel("Longitude")
            # plt.ylabel("CDF")
            # plt.legend()
            # plt.subplot(1, 2, 2)
            # plt.plot(sorted_y, sorted_pred_y, label='Predicted CDF', marker='o', linestyle='-')
            # plt.plot(sorted_y, sorted_target_y, label='Target CDF', marker='x', linestyle='--')
            # plt.title(f"Keyword: {keyword} - Latitude (Y)")
            # plt.xlabel("Latitude")
            # plt.ylabel("CDF")
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

    print(f"CDF模型训练完成，共 {len(cdf_models)} 个关键词")
    return cdf_models


