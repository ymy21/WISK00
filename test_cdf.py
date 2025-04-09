import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from cdf_model import train_cdf_models
from scipy.stats import norm

# 生成测试数据集
def generate_test_dataset(n_samples=500):
    np.random.seed(42)
    data = {
        'longitude': np.random.uniform(0, 100, n_samples),  # 均匀分布
        'latitude': np.random.normal(50, 15, n_samples),  # 正态分布
        'keywords': [np.random.choice(['A', 'B', 'C']) for _ in range(n_samples)]
    }
    df = pd.DataFrame(data)

    # 归一化：对经度和纬度进行归一化处理，确保它们的值在 [0, 1] 范围内
    df['longitude'] = (df['longitude'] - df['longitude'].min()) / (df['longitude'].max() - df['longitude'].min())
    # 对latitude采用z-score标准化，再利用标准正态CDF映射到[0,1]
    lat_mean = df['latitude'].mean()
    lat_std = df['latitude'].std()
    df['latitude'] = norm.cdf((df['latitude'] - lat_mean) / lat_std)

    return df


# 计算真实CDF
def compute_true_cdf(data):
    return data.rank(pct=True).values  # 计算排序百分比


# 计算并绘制 CDF 误差
def evaluate_cdf_model(cdf_models, test_data, keyword='A'):
    keyword_data = test_data[test_data['keywords'] == keyword]

    if keyword_data.empty:
        print(f"关键词 {keyword} 没有足够数据")
        return

    x_data = torch.tensor(keyword_data['longitude'].values, dtype=torch.float32).view(-1, 1)
    y_data = torch.tensor(keyword_data['latitude'].values, dtype=torch.float32).view(-1, 1)

    # 真实CDF
    true_cdf_x = compute_true_cdf(keyword_data['longitude'])
    true_cdf_y = compute_true_cdf(keyword_data['latitude'])

    # 预测CDF
    with torch.no_grad():
        pred_cdf_x = cdf_models[keyword]['x'](x_data).numpy().flatten()
        pred_cdf_y = cdf_models[keyword]['y'](y_data).numpy().flatten()

    # 绘图比较
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(true_cdf_x, pred_cdf_x, alpha=0.5, label='CDF Prediction')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal Line')
    plt.xlabel("True CDF")
    plt.ylabel("Predicted CDF")
    plt.title(f"CDF Evaluation for Longitude ({keyword})")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(true_cdf_y, pred_cdf_y, alpha=0.5, label='CDF Prediction')
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal Line')
    plt.xlabel("True CDF")
    plt.ylabel("Predicted CDF")
    plt.title(f"CDF Evaluation for Latitude ({keyword})")
    plt.legend()

    plt.show()

    # 计算MSE误差
    mse_x = np.mean((true_cdf_x - pred_cdf_x) ** 2)
    mse_y = np.mean((true_cdf_y - pred_cdf_y) ** 2)
    print(f"MSE (longitude): {mse_x:.6f}")
    print(f"MSE (latitude): {mse_y:.6f}")


# 生成数据集
test_data = generate_test_dataset()

# 训练模型
query_workload = pd.DataFrame([{'keywords': ['A', 'B']} for _ in range(300)])
cdf_models = train_cdf_models(test_data, query_workload)

# 评估训练效果
evaluate_cdf_model(cdf_models, test_data, keyword='A')

