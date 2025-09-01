import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.stats import norm
import os


def load_and_process_npz_data(npz_path):
    """
    加载并处理CLIFF输出的NPZ文件数据

    参数:
        npz_path: NPZ文件路径

    返回:
        处理后的关节角度数据，形状为 (帧数, 关节数)
    """
    # 加载数据
    data = np.load(npz_path)
    print("NPZ文件中的键:", list(data.keys()))
    poses = data['pose']

    # 将旋转矩阵转换为欧拉角
    num_frames = poses.shape[0]
    num_joints = poses.shape[1]

    # 存储所有关节的欧拉角
    euler_angles = np.zeros((num_frames, num_joints, 3))

    for frame in range(num_frames):
        for joint in range(num_joints):
            # 获取旋转矩阵
            rot_matrix = poses[frame, joint]

            # 将旋转矩阵转换为欧拉角 (ZYX顺序，即yaw-pitch-roll)
            rotation = R.from_matrix(rot_matrix)
            euler_angles[frame, joint] = rotation.as_euler('zyx', degrees=True)

    return euler_angles


def analyze_joint_angle_distributions(euler_angles, output_dir=None):
    """
    分析每个关节的角度分布并拟合正态分布

    参数:
        euler_angles: 欧拉角数据，形状为 (帧数, 关节数, 3)
        output_dir: 输出目录，用于保存图表
    """
    num_joints = euler_angles.shape[1]
    joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2',
        'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck',
        'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder',
        'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]

    # 为每个关节的每个欧拉角轴创建图表
    for axis_idx, axis_name in enumerate(['Yaw (Z)', 'Pitch (Y)', 'Roll (X)']):
        plt.figure(figsize=(15, 10))

        for joint in range(num_joints):
            # 获取当前关节当前轴的所有角度值
            angles = euler_angles[:, joint, axis_idx]

            # 跳过几乎没有变化的关节（例如，头部在行走中可能保持相对稳定）
            if np.std(angles) < 1.0:  # 标准差小于1度，视为基本不变
                continue

            # 拟合正态分布
            mu, std = norm.fit(angles)

            # 创建直方图
            plt.hist(angles, bins=30, density=True, alpha=0.6,
                     label=f'{joint_names[joint]} (μ={mu:.2f}, σ={std:.2f})')

            # 绘制拟合的正态分布曲线
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)

        plt.title(f'Joint Angle Distribution - {axis_name}')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if output_dir:
            plt.savefig(os.path.join(output_dir,
                                     f'joint_angle_distribution_{axis_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'),
                        bbox_inches='tight')

        plt.show()

    # 为每个关节创建详细的分析报告
    print("=" * 80)
    print("关节角度统计分析报告")
    print("=" * 80)

    for joint in range(num_joints):
        print(f"\n{joint_names[joint]} (关节 {joint}):")

        for axis_idx, axis_name in enumerate(['Yaw', 'Pitch', 'Roll']):
            angles = euler_angles[:, joint, axis_idx]
            mu, std = norm.fit(angles)

            print(f"  {axis_name}: μ={mu:.2f}°, σ={std:.2f}°, 范围=[{np.min(angles):.2f}°, {np.max(angles):.2f}°]")

            # 创建单个关节的详细图表
            if np.std(angles) > 1.0:  # 只对变化较大的关节创建详细图表
                plt.figure(figsize=(10, 6))
                plt.hist(angles, bins=30, density=True, alpha=0.6, color='blue')

                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, 'k', linewidth=2)

                plt.title(f'{joint_names[joint]} - {axis_name} Angle Distribution')
                plt.xlabel('Angle (degrees)')
                plt.ylabel('Density')

                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'{joint_names[joint]}_{axis_name}_distribution.png'),
                                bbox_inches='tight')

                plt.close()  # 关闭图形，避免在Notebook中显示所有图表


def main():
    # 设置NPZ文件路径
    npz_path = "../walk_data/walk_1_cliff_rotmat.npz"  # 替换为你的NPZ文件路径

    # 设置输出目录（可选）
    output_dir = "walk_data"
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载和处理数据
    euler_angles = load_and_process_npz_data(npz_path)

    # 分析角度分布
    analyze_joint_angle_distributions(euler_angles, output_dir)

    # 可选：保存处理后的欧拉角数据
    if output_dir:
        np.save(os.path.join(output_dir, "processed_euler_angles.npy"), euler_angles)
        print(f"处理后的欧拉角数据已保存到 {os.path.join(output_dir, 'processed_euler_angles.npy')}")


if __name__ == "__main__":
    main()