import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

# 创建环境
env = gym.make("Humanoid-v4")
obs, _ = env.reset()

# 获取 MuJoCo 数据
data = env.unwrapped.data
torso_id = data.body("torso").id
quat = data.xquat[torso_id]   # 四元数 (w, x, y, z)
pos = data.xpos[torso_id]     # 位置 (x, y, z)

# 转换为旋转矩阵
rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy: [x,y,z,w]
rot_matrix = rot.as_matrix()

# 局部坐标轴 (X=前方, Y=左侧, Z=上)
axes = np.eye(3)  # 单位向量
axes_rotated = rot_matrix @ axes.T  # 旋转后的坐标轴

# 绘制三维坐标系
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 画 torso 的位置
ax.scatter(pos[0], pos[1], pos[2], c="k", s=50, label="Torso")

# 绘制局部坐标轴
length = 0.5
colors = ["r", "g", "b"]  # X=红, Y=绿, Z=蓝
labels = ["X (forward)", "Y (left)", "Z (up)"]

for i in range(3):
    ax.quiver(
        pos[0], pos[1], pos[2],
        axes_rotated[0, i], axes_rotated[1, i], axes_rotated[2, i],
        length=length, color=colors[i], label=labels[i]
    )

# 设置图形范围
ax.set_xlim([pos[0]-1, pos[0]+1])
ax.set_ylim([pos[1]-1, pos[1]+1])
ax.set_zlim([pos[2]-1, pos[2]+1])

ax.set_xlabel("X axis (forward)")
ax.set_ylabel("Y axis (left)")
ax.set_zlabel("Z axis (up)")
ax.legend()
ax.set_title("Humanoid Torso Orientation (3D)")

plt.show()
env.close()
