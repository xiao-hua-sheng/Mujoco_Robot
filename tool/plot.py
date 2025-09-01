import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def gaussian_2d(x, y, mu, sigma):
    """计算二维高斯分布的概率密度"""
    x = np.array([x, y])
    mu = np.array(mu)

    # 确保协方差矩阵是正定的
    if isinstance(sigma, (int, float)):
        sigma = np.array([[sigma, 0], [0, sigma]])
    elif len(sigma) == 2:
        sigma = np.diag(sigma)

    n = mu.shape[0]
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)

    # 计算高斯公式
    exponent = -0.5 * (x - mu).T @ inv @ (x - mu)
    return (1.0 / (2 * np.pi * np.sqrt(det))) * np.exp(exponent)


def create_gaussian_field(x_range, y_range, num_gaussians=5):
    """创建由多个高斯分布叠加而成的二维场"""
    # 创建网格
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    # 初始化场
    field = np.zeros_like(X)

    # 随机生成多个高斯分布参数并叠加
    for _ in range(num_gaussians):
        # 随机均值和协方差
        mu = [np.random.uniform(x_range[0], x_range[1]),
              np.random.uniform(y_range[0], y_range[1])]

        sigma = np.random.uniform(0.5, 0.8)

        # 计算每个点的高斯值并叠加
        for i in range(len(x)):
            for j in range(len(y)):
                field[j, i] += gaussian_2d(X[j, i], Y[j, i], mu, sigma)

    return X, Y, field


def plot_gaussian_field(X, Y, field, points):
    """绘制二维高斯场并标记点"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制等高线图
    contours = ax.contourf(X, Y, field, levels=50, cmap='viridis', alpha=0.8)

    # 添加颜色条
    cbar = fig.colorbar(contours, ax=ax)
    cbar.set_label('Intensity')

    # 绘制随机点
    colors = ['red', 'green', 'blue']
    labels = ['Point 1', 'Point 2', 'Point 3']

    for i, point in enumerate(points):
        ax.scatter(point[0], point[1], c=colors[i], s=100, marker='o',
                   edgecolors='white', linewidth=2, label=labels[i])

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 添加图例
    ax.legend()

    # 设置标题
    ax.set_title('2D Gaussian Field with Random Points')

    # 设置等比例坐标轴
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_gaussian_field_3d(X, Y, field, points):
    """绘制三维视角的二维高斯场并标记点"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制表面图
    surf = ax.plot_surface(X, Y, field, cmap='viridis', alpha=0.8,
                           linewidth=0, antialiased=True)

    # 添加颜色条
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # 绘制随机点并计算它们的高度值
    colors = ['red', 'green', 'blue']
    labels = ['Point 1', 'Point 2', 'Point 3']

    for i, point in enumerate(points):
        # 找到最近的点来估计高度
        x_idx = np.argmin(np.abs(X[0, :] - point[0]))
        y_idx = np.argmin(np.abs(Y[:, 0] - point[1]))
        z_val = field[y_idx, x_idx]

        ax.scatter(point[0], point[1], z_val, c=colors[i], s=100, marker='o',
                   edgecolors='white', linewidth=2, label=labels[i])

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')

    # 添加图例
    ax.legend()

    # 设置标题
    ax.set_title('3D View of 2D Gaussian Field with Random Points')

    # 设置视角
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.show()


# 主程序
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    # np.random.seed(42)

    # 定义坐标范围
    x_range = (-5, 5)
    y_range = (-5, 5)

    # 创建高斯场
    X, Y, field = create_gaussian_field(x_range, y_range, num_gaussians=100)

    # 生成三个随机点
    points = []
    for _ in range(3):
        point = [np.random.uniform(x_range[0], x_range[1]),
                 np.random.uniform(y_range[0], y_range[1])]
        points.append(point)

    # 绘制二维等高线图
    plot_gaussian_field(X, Y, field, points)

    # 绘制三维视角图
    plot_gaussian_field_3d(X, Y, field, points)