import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def visualize_orientation(yaw=0, pitch=0, roll=0):
    """
    可视化人体在3D空间中的朝向，重点展示偏航角(Yaw)对方向的影响

    参数:
        yaw: 偏航角 (度)，控制左右转向
        pitch: 俯仰角 (度)，控制抬头低头
        roll: 翻滚角 (度)，控制左右倾斜
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. 创建表示人体的基本箭头 (初始方向为Y轴正方向)
    body_length = 2
    arrow_start = np.array([0, 0, 0])  # 起点在原点
    arrow_end = np.array([body_length, 0, 0])  # 初始指向x轴正方向

    # 2. 创建旋转矩阵 (ZYX顺序)
    rotation = R.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    rot_matrix = rotation.as_matrix()

    # 3. 应用旋转到箭头方向
    arrow_end_rotated = rot_matrix @ arrow_end  # 矩阵乘法旋转向量

    # 4. 绘制坐标系轴
    axis_length = 3
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, label='X-axis (Roll)')
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, label='Z-axis (Yaw)')

    # 5. 绘制人体朝向箭头
    ax.quiver(arrow_start[0], arrow_start[1], arrow_start[2],
              arrow_end_rotated[0], arrow_end_rotated[1], arrow_end_rotated[2],
              color='purple', linewidth=3, arrow_length_ratio=0.2, label='Human Facing Direction')

    # 6. 设置图表属性
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.set_xlabel('X (Roll - Tilt)')
    ax.set_ylabel('Y (Initial Forward)')
    ax.set_zlabel('Z (Yaw - Turning)')
    ax.set_title(f'Orientation Visualization\nYaw (Z): {yaw}°, Pitch (Y): {pitch}°, Roll (X): {roll}°')
    ax.legend()

    # 7. 添加文本说明
    explanation = f"""
    Rotation Order: ZYX
    - Z Rotation (Yaw): {yaw}° → Controls LEFT/RIGHT turning
    - Y Rotation (Pitch): {pitch}° → Controls UP/DOWN nodding
    - X Rotation (Roll): {roll}° → Controls LEFT/RIGHT leaning
    """
    plt.figtext(0.1, 0.02, explanation, fontsize=10, ha='left')

    plt.tight_layout()
    plt.show()


# 创建交互式滑块（如果在Jupyter Notebook中运行）
from ipywidgets import interact, IntSlider

print("""
人体朝向可视化说明：
- 紫色箭头表示人体的朝向
- Z轴（蓝色）：偏航角 (Yaw) - 控制左右转向
- X轴（红色）：翻滚角 (Roll) - 控制左右倾斜,初始的前进方向
- Y轴（绿色）：俯仰角（pitch）- 控制前后倾斜
""")

# 如果不在Jupyter中，可以取消注释下面一行来查看一个特定角度的示例
visualize_orientation(yaw=0, pitch=5, roll=0)

# 如果在Jupyter Notebook中，使用交互式滑块
# interact(visualize_orientation,
#          yaw=IntSlider(min=-180, max=180, step=5, value=0, description='Yaw (Z)'),
#          pitch=IntSlider(min=-90, max=90, step=5, value=0, description='Pitch (Y)'),
#          roll=IntSlider(min=-90, max=90, step=5, value=0, description='Roll (X)'))
