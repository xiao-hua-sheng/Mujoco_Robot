import numpy as np
from scipy.spatial.transform import Rotation as R

def smpl_to_euler(smpl_pose):
    """将SMPL的72维姿态向量转换为24个关节的欧拉角 (ZXY顺序)"""
    eulers = []
    for i in range(24):
        rotvec = smpl_pose[3*i : 3*i+3]
        # 旋转向量 → 旋转矩阵 → 欧拉角 (ZXY顺序)
        rot = R.from_rotvec(rotvec)
        euler = rot.as_euler('ZXY', degrees=False)  # 弧度制
        eulers.append(euler)
    return np.array(eulers)  # 形状: (24, 3)


def align_joints(smpl_eulers):
    """将SMPL欧拉角映射到Humanoid的关节结构"""
    # 初始化Humanoid姿态数组 (17个自由度)
    humanoid_pose = np.zeros(17)

    # 1. 脊柱 (SMPL: 3,6,9 → Humanoid: 0,1,2)
    spine_eulers = smpl_eulers[[3, 6, 9]].mean(axis=0)  # 平均脊柱
    humanoid_pose[0:3] = spine_eulers  # abdomen_yaw, pitch, roll

    # 2. 左腿 (SMPL 1 → Humanoid 7,8,9,10)
    humanoid_pose[7:10] = smpl_eulers[1]  # left_hip (Z, X, Y)
    humanoid_pose[10] = smpl_eulers[4][1]  # left_knee (取Y旋转)

    # 3. 右腿 (SMPL 2 → Humanoid 11,12,13,14)
    humanoid_pose[11:14] = smpl_eulers[2]  # right_hip (Z, X, Y)
    humanoid_pose[14] = smpl_eulers[5][1]  # right_knee (取Y旋转)

    # 4. 肩膀 (示例: 取SMPL 16,17的X旋转)
    humanoid_pose[15] = smpl_eulers[16][0]  # right_shoulder_z
    humanoid_pose[16] = smpl_eulers[17][0]  # left_shoulder_z

    return humanoid_pose


def reward_function(humanoid_state, smpl_pose):
    # 从Humanoid状态中提取当前关节角度 (qpos[7:24]是17个自由度)
    current_pose = humanoid_state[7:24]  # 形状: (17,)

    # 将SMPL姿态对齐到Humanoid结构
    smpl_eulers = smpl_to_euler(smpl_pose)
    target_pose = align_joints(smpl_eulers)

    # 计算欧氏距离作为奖励
    reward = -np.linalg.norm(current_pose - target_pose)
    return reward