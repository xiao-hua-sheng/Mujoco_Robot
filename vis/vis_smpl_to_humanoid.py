import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 设置随机种子确保可重现性
np.random.seed(42)


# 生成模拟SMPL姿态数据
def generate_sample_smpl_pose(num_frames=10):
    """生成随机SMPL姿态数据 (24个关节，每个关节3个旋转向量参数)"""
    poses = []
    for _ in range(num_frames):
        pose = np.zeros(72)  # 24个关节 * 3个参数

        # 根关节 (骨盆) 随机旋转
        pose[0:3] = np.random.uniform(-0.5, 0.5, 3)

        # 脊柱关节 (3,6,9)
        for idx in [3, 6, 9]:
            pose[3 * idx:3 * idx + 3] = np.random.uniform(-0.3, 0.3, 3)

        # 髋关节 (1,2)
        pose[3:6] = np.random.uniform(-0.8, 0.8, 3)  # 左髋
        pose[6:9] = np.random.uniform(-0.8, 0.8, 3)  # 右髋

        # 膝关节 (4,5)
        pose[12:15] = np.random.uniform(0, 1.5, 3)  # 左膝
        pose[15:18] = np.random.uniform(0, 1.5, 3)  # 右膝

        poses.append(pose)

    return np.array(poses)


# 将SMPL旋转向量转换为欧拉角
def smpl_to_euler(smpl_pose):
    eulers = []
    for i in range(24):
        rotvec = smpl_pose[3 * i:3 * i + 3]
        rot = R.from_rotvec(rotvec)
        euler = rot.as_euler('ZXY', degrees=False)
        eulers.append(euler)
    return np.array(eulers)


# 将SMPL姿态对齐到Humanoid结构
def align_to_humanoid(smpl_eulers):
    """将SMPL欧拉角映射到Humanoid的17个自由度"""
    humanoid_pose = np.zeros(17)

    # 脊柱 (取SMPL的3,6,9关节平均值)
    spine_eulers = np.mean(smpl_eulers[[3, 6, 9]], axis=0)
    humanoid_pose[0:3] = spine_eulers  # abdomen_yaw, pitch, roll

    # 左腿
    humanoid_pose[7:10] = smpl_eulers[1]  # left_hip (Z, X, Y)
    humanoid_pose[10] = smpl_eulers[4][1]  # left_knee (Y旋转)

    # 右腿
    humanoid_pose[11:14] = smpl_eulers[2]  # right_hip (Z, X, Y)
    humanoid_pose[14] = smpl_eulers[5][1]  # right_knee (Y旋转)

    # 肩膀 (简化处理)
    humanoid_pose[15] = smpl_eulers[16][0]  # right_shoulder
    humanoid_pose[16] = smpl_eulers[17][0]  # left_shoulder

    return humanoid_pose


# 创建Humanoid骨架模型
def create_humanoid_skeleton():
    """定义Humanoid骨架连接关系"""
    # 关节索引对应关系:
    # 0: root (骨盆)
    # 1: abdomen_yaw, 2: abdomen_pitch, 3: abdomen_roll
    # 4: left_hip_yaw, 5: left_hip_roll, 6: left_hip_pitch
    # 7: left_knee
    # 8: right_hip_yaw, 9: right_hip_roll, 10: right_hip_pitch
    # 11: right_knee
    # 12: right_shoulder, 13: left_shoulder

    # 骨架连接: (父节点, 子节点)
    connections = [
        (0, 1),  # 骨盆 -> 脊柱
        (1, 2),  # 脊柱 -> 上脊柱
        (2, 3),  # 上脊柱 -> 下脊柱

        # 左腿
        (0, 4),  # 骨盆 -> 左髋
        (4, 5),  # 左髋 -> 左大腿
        (5, 6),  # 左大腿 -> 左小腿
        (6, 7),  # 左小腿 -> 左脚

        # 右腿
        (0, 8),  # 骨盆 -> 右髋
        (8, 9),  # 右髋 -> 右大腿
        (9, 10),  # 右大腿 -> 右小腿
        (10, 11),  # 右小腿 -> 右脚

        # 手臂
        (3, 12),  # 上脊柱 -> 右肩
        (3, 13)  # 上脊柱 -> 左肩
    ]

    return connections


# 创建SMPL骨架模型
def create_smpl_skeleton():
    """定义SMPL标准骨架连接关系"""
    # SMPL关节连接关系 (父节点 -> 子节点)
    connections = [
        (0, 1), (0, 2), (0, 3),  # 骨盆 -> 左右髋和脊柱
        (1, 4), (4, 7), (7, 10),  # 左腿
        (2, 5), (5, 8), (8, 11),  # 右腿
        (3, 6), (6, 9), (9, 12), (12, 15),  # 脊柱
        (9, 13), (13, 16), (16, 18),  # 左臂
        (9, 14), (14, 17), (17, 19)  # 右臂
    ]
    return connections


# 计算关节位置 (简化正向运动学)
def calculate_joint_positions(angles, skeleton_type='smpl'):
    """根据欧拉角计算关节位置 (简化模型)"""
    if skeleton_type == 'smpl':
        num_joints = 24
        connections = create_smpl_skeleton()
    else:  # humanoid
        num_joints = 14  # 我们的简化Humanoid模型有14个关节
        connections = create_humanoid_skeleton()

    # 初始化关节位置
    positions = np.zeros((num_joints, 3))

    # 设置根关节位置
    positions[0] = [0, 0, 0]  # 骨盆位置

    # 计算每个关节的位置
    for connection in connections:
        parent, child = connection

        # 简化：假设每个骨骼长度为1
        bone_length = 1.0

        # 获取父关节的旋转
        if skeleton_type == 'smpl':
            rot = R.from_euler('ZXY', angles[parent])
        else:
            # Humanoid特殊处理
            if parent == 0:  # root
                rot = R.identity()
            elif parent == 1:  # abdomen_yaw
                rot = R.from_euler('Z', angles[1])
            elif parent == 2:  # abdomen_pitch
                rot = R.from_euler('X', angles[2])
            elif parent == 3:  # abdomen_roll
                rot = R.from_euler('Y', angles[3])
            else:
                rot = R.from_euler('ZXY', [0, 0, 0])  # 简化

        # 计算骨骼方向
        direction = rot.apply([0, bone_length, 0])

        # 计算子关节位置
        positions[child] = positions[parent] + direction

    return positions, connections


# 可视化函数
def visualize_pose_comparison(smpl_poses):
    fig = plt.figure(figsize=(15, 8))

    # 创建两个3D子图
    ax1 = fig.add_subplot(121, projection='3d')  # SMPL姿态
    ax2 = fig.add_subplot(122, projection='3d')  # Humanoid姿态

    # 设置坐标轴范围
    for ax in [ax1, ax2]:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=30)

    ax1.set_title('SMPL Pose (Original)')
    ax2.set_title('Humanoid Pose (Aligned)')

    # 初始化骨架线条
    smpl_lines = []
    humanoid_lines = []

    # 获取SMPL骨架连接
    smpl_connections = create_smpl_skeleton()
    humanoid_connections = create_humanoid_skeleton()

    # 创建初始骨架
    for _ in smpl_connections:
        line, = ax1.plot([], [], [], 'b-', lw=2)
        smpl_lines.append(line)

    for _ in humanoid_connections:
        line, = ax2.plot([], [], [], 'r-', lw=2)
        humanoid_lines.append(line)

    # 添加关节点
    smpl_joints = ax1.scatter([], [], [], c='blue', s=50)
    humanoid_joints = ax2.scatter([], [], [], c='red', s=50)

    # 更新函数
    def update(frame):
        # 处理SMPL姿态
        smpl_eulers = smpl_to_euler(smpl_poses[frame])
        smpl_positions, _ = calculate_joint_positions(smpl_eulers, 'smpl')

        # 对齐到Humanoid
        humanoid_pose = align_to_humanoid(smpl_eulers)
        humanoid_positions, _ = calculate_joint_positions(humanoid_pose, 'humanoid')

        # 更新SMPL骨架
        for i, (parent, child) in enumerate(smpl_connections):
            x = [smpl_positions[parent][0], smpl_positions[child][0]]
            y = [smpl_positions[parent][1], smpl_positions[child][1]]
            z = [smpl_positions[parent][2], smpl_positions[child][2]]
            smpl_lines[i].set_data(x, y)
            smpl_lines[i].set_3d_properties(z)

        # 更新Humanoid骨架
        for i, (parent, child) in enumerate(humanoid_connections):
            x = [humanoid_positions[parent][0], humanoid_positions[child][0]]
            y = [humanoid_positions[parent][1], humanoid_positions[child][1]]
            z = [humanoid_positions[parent][2], humanoid_positions[child][2]]
            humanoid_lines[i].set_data(x, y)
            humanoid_lines[i].set_3d_properties(z)

        # 更新关节点
        smpl_joints._offsets3d = (smpl_positions[:, 0], smpl_positions[:, 1], smpl_positions[:, 2])
        humanoid_joints._offsets3d = (humanoid_positions[:, 0], humanoid_positions[:, 1], humanoid_positions[:, 2])

        fig.suptitle(f'Pose Comparison - Frame: {frame + 1}/{len(smpl_poses)}', fontsize=16)
        return smpl_lines + humanoid_lines + [smpl_joints, humanoid_joints]

    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(smpl_poses),
                        interval=500, blit=True)

    plt.tight_layout()
    return ani


# 生成示例SMPL姿态数据
sample_poses = generate_sample_smpl_pose(num_frames=5)

# 创建可视化
ani = visualize_pose_comparison(sample_poses)

# 显示动画 (在Jupyter中)
HTML(ani.to_jshtml())

# 保存动画 (可选)
# ani.save('pose_comparison.mp4', writer='ffmpeg', fps=2)