"""
H3.6M 骨架到 Unitree G1 关节角映射脚本

功能: 将 VideoPose3D 输出的 H3.6M 3D 骨架坐标映射到 G1 机器人的 29 个关节角

注意: 由于模型训练错误，H3.6M 中 Y 负方向才是向上
      正确情况应该是 Y+ 向上

坐标系变换:
  1. H3.6M (Y- up) -> MuJoCo (Z+ up): R_Y_TO_Z = [[1,0,0],[0,0,1],[0,-1,0]]
  2. 正面方向对齐: f_infer = up_vec × shoulder_vec -> X+

三个主要函数:
  - io_handler(mode, result) : 读写数据、加载模型
  - process(pose_data, model, data) : 核心处理（坐标对齐、IK 求解）
  - visualize(result, model, data) : 可视化（matplotlib 动画）

依赖:
    pip install mujoco mink numpy matplotlib

需要的文件:
    1. test.npz                     — VideoPose3D 推理结果
    2. unitree_g1/g1_mocap_29dof.xml — G1 MuJoCo 模型
"""

import os
import sys
import time
import logging

import cv2
import numpy as np
import mujoco

# ============================================================================
# 全局路径配置
# ============================================================================
INPUT_NPZ = "taiji.npz"
G1_MJCF = "unitree_g1/g1_mocap_29dof.xml"
OUTPUT_NPZ = "g1_joint_angles.npz"
LOG_FILE = "mapping_debug.log"

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 常量定义
# ============================================================================

# H3.6M 17 关节名称
H36M_JOINT_NAMES = [
    'pelvis', 'r_hip', 'r_knee', 'r_ankle',
    'l_hip', 'l_knee', 'l_ankle',
    'spine', 'neck', 'head', 'head_top',
    'l_shoulder', 'l_elbow', 'l_wrist',
    'r_shoulder', 'r_elbow', 'r_wrist',
]

# H3.6M 关节索引
H36M_IDX = {name: i for i, name in enumerate(H36M_JOINT_NAMES)}

# H3.6M 骨架连线
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),      # 右腿
    (0, 4), (4, 5), (5, 6),      # 左腿
    (0, 7), (7, 8), (8, 9), (9, 10),  # 脊柱+头
    (8, 11), (11, 12), (12, 13),  # 左臂
    (8, 14), (14, 15), (15, 16),  # 右臂
]

# G1 关节定义（29 DOF）
G1_JOINT_NAMES = [
    # 腿部 (12)
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    # 腰部 (3)
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    # 手臂 (14)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# H3.6M -> G1 末端效应器映射
H36M_TO_G1_EE_MAP = {
    "l_ankle": "left_toe_link",
    "r_ankle": "right_toe_link",
    "l_wrist": "left_rubber_hand",
    "r_wrist": "right_rubber_hand",
}

# IK 参数
IK_GAIN = 0.5
IK_DAMPING = 1e-6
MAX_ITERATIONS = 100

# 坐标系变换矩阵: H3.6M (Y- up, 训练错误) -> MuJoCo (Z+ up)
# 注意: 由于模型训练错误，H3.6M 中 Y 负方向才是向上
# 正确应该是 Y+ 向上，变换矩阵为:
# R_CORRECT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
#
# 变换规则:
#   H3.6M (x, y, z) -> MuJoCo (x, -z, -y)
#   H3.6M Y- (up) -> MuJoCo Z+ (up)
#   H3.6M Z+ (toward viewer) -> MuJoCo Y- (right)
R_Y_TO_Z = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
], dtype=np.float64)

# 骨架颜色 (用于可视化)
SKELETON_COLORS = [
    (1, 0, 0), (1, 0, 0), (1, 0, 0),      # 右腿 - 红
    (0, 0, 1), (0, 0, 1), (0, 0, 1),      # 左腿 - 蓝
    (0, 1, 0), (0, 1, 0), (0, 0.8, 0), (0, 0.8, 0),  # 脊柱+头 - 绿
    (1, 0.5, 0), (1, 0.5, 0), (1, 0.5, 0),  # 左臂 - 橙
    (0.5, 0, 1), (0.5, 0, 1), (0.5, 0, 1),  # 右臂 - 紫
]

# ============================================================================
# 函数 1: io_handler
# ============================================================================

def io_handler(mode, result=None):
    """IO 处理函数。

    mode="read":        读取 INPUT_NPZ, 返回 poses_3d (T, 17, 3)
    mode="write":       将 result dict 保存为 OUTPUT_NPZ
    mode="load_model":  加载 MuJoCo 模型, 返回 (model, data)

    注意: 由于模型训练错误，H3.6M 中 Y 负方向才是向上

    正面方向计算 (右手定则):
    - shoulder_vec = l_shoulder - r_shoulder (从右肩指向左肩)
    - up_vec = neck - pelvis (从骨盆指向颈部)
    - f_infer = up_vec × shoulder_vec (得到 X+ 方向)
    """
    if mode == "read":
        if not os.path.isfile(INPUT_NPZ):
            logger.error(f"输入文件不存在: {INPUT_NPZ}")
            sys.exit(1)

        data = np.load(INPUT_NPZ)
        poses_3d = data['poses_3d']  # (T, 17, 3)
        logger.info(f"[io_handler] 读取 {INPUT_NPZ}: poses_3d shape={poses_3d.shape}")
        return poses_3d

    elif mode == "write":
        if result is None:
            raise ValueError("write 模式需要传入 result 参数")

        np.savez_compressed(
            OUTPUT_NPZ,
            joint_angles=result['joint_angles'],      # (T, 29)
            joint_names=np.array(G1_JOINT_NAMES),
            target_positions=result.get('target_positions'),
            errors=result.get('errors'),
        )
        logger.info(f"[io_handler] 结果已保存: {OUTPUT_NPZ}")

    elif mode == "load_model":
        if not os.path.isfile(G1_MJCF):
            logger.error(f"G1 模型文件不存在: {G1_MJCF}")
            sys.exit(1)

        model = mujoco.MjModel.from_xml_path(G1_MJCF)
        data = mujoco.MjData(model)
        logger.info(f"[io_handler] 加载 G1 模型: {model.nq} DOF, {model.nbody} bodies")
        return model, data

    else:
        raise ValueError(f"未知模式: {mode}, 请使用 'read', 'write' 或 'load_model'")


# ============================================================================
# 函数 2: process (将在后续实现)
# ============================================================================

def align_coordinates(pose_3d):
    """坐标系对齐: H3.6M (Y- up, 训练错误) -> MuJoCo (Z+ up, X+ forward)

    步骤:
    1. Y- -> Z+ 旋转 (由于训练错误，H3.6M 中 Y 负方向才是向上)
    2. 提取正面方向
    3. 旋转到 MuJoCo X+

    正确情况应该是 Y+ 向上，变换矩阵为:
    R_CORRECT = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    正面方向计算 (右手定则):
    - shoulder_vec = l_shoulder - r_shoulder (从右肩指向左肩)
    - up_vec = neck - pelvis (从骨盆指向颈部)
    - f_infer = up_vec × shoulder_vec (得到 X+ 方向)

    Args:
        pose_3d: (17, 3) H3.6M 格式的 3D 坐标

    Returns:
        pose_aligned: (17, 3) 对齐后的坐标
        R_align: (3, 3) 完整旋转矩阵（用于可视化）
    """
    logger.debug("[align_coordinates] 开始坐标系对齐")

    # 步骤 1: Y- -> Z+ 旋转
    # H3.6M: Y- = 上 (训练错误), MuJoCo: Z+ = 上
    # 正确情况: H3.6M: Y+ = 上, 变换矩阵 R_CORRECT = [[1,0,0],[0,0,-1],[0,1,0]]
    # 变换规则: H3.6M (x, y, z) -> MuJoCo (x, -z, -y)
    pose_temp = (R_Y_TO_Z @ pose_3d.T).T  # (17, 3)
    logger.debug(f"  步骤1 Y->Z 旋转完成: pelvis={pose_temp[0]}")

    # 步骤 2: 提取正面方向
    # 左右肩向量
    l_shoulder = pose_temp[H36M_IDX['l_shoulder']]
    r_shoulder = pose_temp[H36M_IDX['r_shoulder']]
    shoulder_vec = r_shoulder - l_shoulder  # 从左到右

    # 躯干上方向
    pelvis = pose_temp[H36M_IDX['pelvis']]
    neck = pose_temp[H36M_IDX['neck']]
    up_vec = neck - pelvis

    # 正面方向 = 叉积(上方向, 肩向量)
    f_infer = np.cross(up_vec, shoulder_vec)
    f_norm = np.linalg.norm(f_infer)
    if f_norm < 1e-6:
        logger.warning("  正面方向向量接近零，使用默认 X+")
        f_infer = np.array([1, 0, 0], dtype=np.float64)
    else:
        f_infer = f_infer / f_norm

    logger.debug(f"  步骤2 正面方向: {f_infer}")

    # 步骤 3: 旋转到 MuJoCo X+
    # 计算 f_infer 在 XY 平面的投影角度
    # 使用 atan2 计算角度，然后旋转使 f_infer 对齐 X+
    angle = np.arctan2(f_infer[1], f_infer[0])
    logger.debug(f"  步骤3 旋转角度: {np.degrees(angle):.2f}°")

    # 绕 Z 轴旋转，使 f_infer 对齐 X+
    # 使用 -angle 旋转，因为我们要将 f_infer 旋转到 X+ 方向
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R_z = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1],
    ], dtype=np.float64)

    pose_aligned = (R_z @ pose_temp.T).T  # (17, 3)

    # 完整旋转矩阵
    R_align = R_z @ R_Y_TO_Z

    # 验证: 正面方向应该接近 X+
    f_aligned = R_z @ f_infer
    logger.debug(f"  验证: 对齐后正面方向 = {f_aligned} (应接近 [1,0,0])")

    # 将原点移到骨盆位置
    pelvis_aligned = pose_aligned[H36M_IDX['pelvis']]
    pose_aligned = pose_aligned - pelvis_aligned
    logger.debug(f"  原点移到骨盆: pelvis={pose_aligned[0]}")

    return pose_aligned, R_align


def compute_reference_angles(pose_aligned):
    """计算参考弯曲角度（膝关节、肘关节）

    使用向量夹角公式：θ = π - arccos(v1·v2 / (|v1|·|v2|))

    弯曲角度 = π - 向量夹角:
    - 站立时: 向量夹角 ≈ 180°, 弯曲角度 ≈ 0°
    - 弯曲时: 向量夹角 < 180°, 弯曲角度 > 0°

    注意: 由于模型训练错误，H3.6M 中 Y 负方向才是向上

    Args:
        pose_aligned: (17, 3) 对齐后的坐标

    Returns:
        ref_angles: dict, 包含 knee_l, knee_r, elbow_l, elbow_r 的角度（弧度）
    """
    logger.debug("[compute_reference_angles] 计算参考弯曲角度")

    def angle_between(v1, v2):
        """计算两个向量之间的夹角（弧度）"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    ref_angles = {}

    # 左膝关节: hip -> knee -> ankle
    hip_l = pose_aligned[H36M_IDX['l_hip']]
    knee_l = pose_aligned[H36M_IDX['l_knee']]
    ankle_l = pose_aligned[H36M_IDX['l_ankle']]
    v_thigh_l = knee_l - hip_l
    v_shin_l = ankle_l - knee_l
    # 弯曲角度 = π - 向量夹角
    ref_angles['knee_l'] = np.pi - angle_between(v_thigh_l, v_shin_l)
    logger.debug(f"  左膝: {np.degrees(ref_angles['knee_l']):.2f}°")

    # 右膝关节
    hip_r = pose_aligned[H36M_IDX['r_hip']]
    knee_r = pose_aligned[H36M_IDX['r_knee']]
    ankle_r = pose_aligned[H36M_IDX['r_ankle']]
    v_thigh_r = knee_r - hip_r
    v_shin_r = ankle_r - knee_r
    ref_angles['knee_r'] = np.pi - angle_between(v_thigh_r, v_shin_r)
    logger.debug(f"  右膝: {np.degrees(ref_angles['knee_r']):.2f}°")

    # 左肘关节: shoulder -> elbow -> wrist
    shoulder_l = pose_aligned[H36M_IDX['l_shoulder']]
    elbow_l = pose_aligned[H36M_IDX['l_elbow']]
    wrist_l = pose_aligned[H36M_IDX['l_wrist']]
    v_upper_l = elbow_l - shoulder_l
    v_forearm_l = wrist_l - elbow_l
    ref_angles['elbow_l'] = np.pi - angle_between(v_upper_l, v_forearm_l)
    logger.debug(f"  左肘: {np.degrees(ref_angles['elbow_l']):.2f}°")

    # 右肘关节
    shoulder_r = pose_aligned[H36M_IDX['r_shoulder']]
    elbow_r = pose_aligned[H36M_IDX['r_elbow']]
    wrist_r = pose_aligned[H36M_IDX['r_wrist']]
    v_upper_r = elbow_r - shoulder_r
    v_forearm_r = wrist_r - elbow_r
    ref_angles['elbow_r'] = np.pi - angle_between(v_upper_r, v_forearm_r)
    logger.debug(f"  右肘: {np.degrees(ref_angles['elbow_r']):.2f}°")

    return ref_angles


def draw_skeleton_mpl(ax, pose, title, color='blue'):
    """绘制骨架 (matplotlib 3D)

    Args:
        ax: matplotlib 3D 坐标轴
        pose: (17, 3) 关节坐标
        title: 标题
        color: 颜色
    """
    # 绘制骨骼连线
    for i, j in H36M_SKELETON:
        ax.plot(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            [pose[i, 2], pose[j, 2]],
            color=color, linewidth=2
        )

    # 绘制关节点
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='black', s=30, zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # 设置坐标轴等比例
    max_range = np.max(np.abs(pose)) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])


def build_targets(pose_aligned, model, data):
    """构建 IK 目标位置

    用 G1 的运动学结构（段长）+ H3.6M 的方向，通过正运动学计算目标位置。
    这样目标位置天然就是 G1 可达的。

    Args:
        pose_aligned: (17, 3) 对齐后的坐标
        model: mujoco.MjModel
        data: mujoco.MjData

    Returns:
        targets: dict, G1 body 名称 -> 目标位置 (世界坐标)
    """
    logger.debug("[build_targets] 构建 IK 目标位置 (G1 运动学 + H3.6M 方向)")

    mujoco.mj_forward(model, data)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    g1_pelvis_pos = data.xpos[pelvis_id].copy()

    def get_g1_body_pos(body_name):
        """获取 G1 body 相对于骨盆的位置"""
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return data.xpos[body_id] - g1_pelvis_pos

    def fk_with_h36m_directions(h36m_chain, g1_chain, n_points):
        """用 G1 段长 + H3.6M 方向做正运动学

        h36m_chain: H3.6M 关节名称列表 (从近端到远端)
        g1_chain: G1 body 名称列表 (从近端到远端)
        n_points: 要计算的点数（不含近端）

        返回: list of (g1_body_name, target_position_world) for each point
        """
        results = []
        current_pos = get_g1_body_pos(g1_chain[0]).copy()  # 从 G1 近端开始

        for i in range(n_points):
            # H3.6M 该段的方向
            h36m_from = pose_aligned[H36M_IDX[h36m_chain[i]]]
            h36m_to = pose_aligned[H36M_IDX[h36m_chain[i + 1]]]
            h36m_dir = h36m_to - h36m_from
            h36m_len = np.linalg.norm(h36m_dir)

            # G1 该段的长度
            g1_from = get_g1_body_pos(g1_chain[i])
            g1_to = get_g1_body_pos(g1_chain[i + 1])
            g1_len = np.linalg.norm(g1_to - g1_from)

            # 用 H3.6M 方向 * G1 长度
            if h36m_len > 0:
                h36m_dir_unit = h36m_dir / h36m_len
            else:
                h36m_dir_unit = np.array([0, 0, -1])

            current_pos = current_pos + h36m_dir_unit * g1_len
            results.append((g1_chain[i + 1], current_pos + g1_pelvis_pos))

        return results

    # 定义链条
    l_leg_h36m = ['pelvis', 'l_hip', 'l_knee', 'l_ankle']
    l_leg_g1 = ['pelvis', 'left_hip_pitch_link', 'left_knee_link', 'left_toe_link']
    r_leg_h36m = ['pelvis', 'r_hip', 'r_knee', 'r_ankle']
    r_leg_g1 = ['pelvis', 'right_hip_pitch_link', 'right_knee_link', 'right_toe_link']
    l_arm_h36m = ['l_shoulder', 'l_elbow', 'l_wrist']
    l_arm_g1 = ['left_shoulder_pitch_link', 'left_elbow_link', 'left_rubber_hand']
    r_arm_h36m = ['r_shoulder', 'r_elbow', 'r_wrist']
    r_arm_g1 = ['right_shoulder_pitch_link', 'right_elbow_link', 'right_rubber_hand']
    spine_h36m = ['pelvis', 'spine', 'neck', 'head']
    spine_g1 = ['pelvis', 'torso_link', 'head_mocap', 'head_mocap']

    # 用正运动学计算所有目标位置
    targets = {}

    for name, pos in fk_with_h36m_directions(l_leg_h36m, l_leg_g1, 3):
        targets[name] = pos
    for name, pos in fk_with_h36m_directions(r_leg_h36m, r_leg_g1, 3):
        targets[name] = pos
    for name, pos in fk_with_h36m_directions(l_arm_h36m, l_arm_g1, 2):
        targets[name] = pos
    for name, pos in fk_with_h36m_directions(r_arm_h36m, r_arm_g1, 2):
        targets[name] = pos
    for name, pos in fk_with_h36m_directions(spine_h36m, spine_g1, 3):
        targets[name] = pos

    targets['_g1_pelvis_pos'] = g1_pelvis_pos

    logger.debug(f"  目标位置:")
    for name, pos in targets.items():
        if not name.startswith('_'):
            logger.debug(f"    {name}: {pos}")

    return targets


def solve_ik(model, data, targets, prev_angles=None):
    """使用 mink 求解微分 IK

    为所有关节目标位置创建 FrameTask，让 IK 求解器找到最优关节角度。
    支持防突变：通过 PostureTask 约束与上一帧的关节角差异。

    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
        targets: dict, G1 body 名称 -> 目标位置 (3,)
        prev_angles: (29,) 上一帧的关节角度，用于防突变

    Returns:
        joint_angles: (29,) 关节角度
        error: float, 平均位置误差
    """
    logger.debug("[solve_ik] 求解 IK")

    import mink

    # 如果有上一帧角度，先设置到配置中（作为初始值）
    if prev_angles is not None:
        for i, joint_name in enumerate(G1_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = prev_angles[i]

    # 创建配置对象
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)

    # 为每个关节目标创建 FrameTask
    tasks = []
    for body_name, target_pos in targets.items():
        if body_name.startswith('_'):
            continue

        task = mink.FrameTask(
            frame_name=body_name,
            frame_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        task.set_target(mink.SE3.from_translation(target_pos))
        tasks.append(task)

    # 防突变：PostureTask 约束与上一帧的关节角差异
    if prev_angles is not None:
        posture_task = mink.PostureTask(model, cost=0.1)  # 中等权重
        posture_task.set_target_from_configuration(configuration)
        tasks.append(posture_task)

    # 关节限位约束
    config_limit = mink.ConfigurationLimit(model)

    # 迭代求解
    total_error = 0.0
    dt = 0.5

    for iteration in range(MAX_ITERATIONS):
        # 计算误差
        error = 0.0
        n_tasks = 0
        for task in tasks:
            if isinstance(task, mink.FrameTask):
                task_error = task.compute_error(configuration)
                error += np.linalg.norm(task_error[:3])
                n_tasks += 1

        total_error = error / n_tasks if n_tasks > 0 else 0

        if total_error < 1e-3:
            logger.debug(f"  迭代 {iteration}: 误差 = {total_error:.6f} (收敛)")
            break

        velocity = mink.solve_ik(
            configuration=configuration,
            tasks=tasks,
            dt=dt,
            solver="daqp",
            damping=1e-2,
            limits=[config_limit],
        )

        configuration.integrate_inplace(velocity, dt)

        if iteration % 10 == 0:
            logger.debug(f"  迭代 {iteration}: 误差 = {total_error:.6f}")

    # 获取关节角度
    joint_angles = np.zeros(29)
    for i, joint_name in enumerate(G1_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_adr = model.jnt_qposadr[joint_id]
        joint_angles[i] = configuration.q[qpos_adr]

    logger.debug(f"  最终误差: {total_error:.6f}")

    return joint_angles, total_error


def smooth_poses(pose_data, window=5, polyorder=2):
    """对 H3.6M 坐标做 Savitzky-Golay 滤波平滑

    Args:
        pose_data: (T, 17, 3) 原始坐标
        window: 滤波窗口大小
        polyorder: 多项式阶数

    Returns:
        pose_smoothed: (T, 17, 3) 平滑后的坐标
    """
    from scipy.signal import savgol_filter
    T = len(pose_data)
    if T < window:
        return pose_data.copy()

    pose_smoothed = pose_data.copy()
    for j in range(17):
        for d in range(3):
            pose_smoothed[:, j, d] = savgol_filter(pose_data[:, j, d], window, polyorder)

    logger.info(f"[smooth_poses] 输入坐标平滑完成 (window={window}, polyorder={polyorder})")
    return pose_smoothed


def smooth_joint_angles(joint_angles, window=5, polyorder=2):
    """对关节角做 Savitzky-Golay 滤波平滑

    Args:
        joint_angles: (T, 29) 原始关节角
        window: 滤波窗口大小
        polyorder: 多项式阶数

    Returns:
        angles_smoothed: (T, 29) 平滑后的关节角
    """
    from scipy.signal import savgol_filter
    T = len(joint_angles)
    if T < window:
        return joint_angles.copy()

    angles_smoothed = joint_angles.copy()
    for j in range(29):
        angles_smoothed[:, j] = savgol_filter(joint_angles[:, j], window, polyorder)

    logger.info(f"[smooth_joint_angles] 关节角平滑完成 (window={window}, polyorder={polyorder})")
    return angles_smoothed


def process(pose_data, model, data):
    """核心处理函数: 坐标对齐 + IK 求解

    流程:
    1. 输入平滑: 对 H3.6M 坐标做 Savitzky-Golay 滤波
    2. 逐帧处理: 坐标对齐 -> 构建目标 -> IK 求解（带防突变）
    3. 输出平滑: 对关节角做 Savitzky-Golay 滤波

    Args:
        pose_data: (T, 17, 3) H3.6M 格式的 3D 坐标
        model: mujoco.MjModel
        data: mujoco.MjData

    Returns:
        result: dict, 包含 joint_angles, target_positions, errors
    """
    logger.info("=" * 60)
    logger.info("开始处理: H3.6M -> G1 关节角映射")
    logger.info("=" * 60)

    # 1. 输入平滑
    pose_smoothed = smooth_poses(pose_data, window=7, polyorder=3)

    n_frames = len(pose_smoothed)
    logger.info(f"总帧数: {n_frames}")

    # 存储结果
    all_joint_angles = []
    all_target_positions = []
    all_errors = []

    prev_angles = None  # 上一帧的关节角度，用于防突变

    t_start = time.time()

    for frame_idx in range(n_frames):
        logger.info(f"\n--- 处理帧 {frame_idx}/{n_frames} ---")

        # 2. 坐标系对齐
        pose_3d = pose_smoothed[frame_idx]
        pose_aligned, R_align = align_coordinates(pose_3d)

        # 3. 构建目标位置（全关节）
        targets = build_targets(pose_aligned, model, data)

        # 4. IK 求解（带防突变）
        joint_angles, error = solve_ik(model, data, targets, prev_angles=prev_angles)

        # 5. 记录结果
        all_joint_angles.append(joint_angles)
        all_target_positions.append(targets)
        all_errors.append(error)

        # 更新上一帧角度
        prev_angles = joint_angles.copy()

        # 重置 MuJoCo 数据
        mujoco.mj_resetData(model, data)

        # 进度显示
        if (frame_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            fps = (frame_idx + 1) / elapsed
            logger.info(f"进度: {frame_idx + 1}/{n_frames} ({elapsed:.1f}s, {fps:.1f} fps)")

    elapsed = time.time() - t_start
    logger.info(f"\n处理完成: {n_frames} 帧, {elapsed:.1f}s")

    # 6. 输出平滑
    joint_angles_raw = np.array(all_joint_angles)
    joint_angles_smooth = smooth_joint_angles(joint_angles_raw, window=7, polyorder=3)

    # 统计误差
    errors = np.array(all_errors)
    logger.info(f"误差统计: 平均={errors.mean():.6f}, 最大={errors.max():.6f}, 最小={errors.min():.6f}")

    result = {
        'joint_angles': joint_angles_smooth,              # (T, 29) 平滑后
        'joint_angles_raw': joint_angles_raw,              # (T, 29) 原始
        'target_positions': all_target_positions,         # list of dict
        'errors': errors,                                 # (T,)
        'pose_data': pose_data,                           # (T, 17, 3) 原始数据
    }

    return result


# ============================================================================
# 函数 3: visualize (将在后续实现)
# ============================================================================

def visualize(result, model, data):
    """可视化函数: matplotlib 动画

    左面板: 原始 H3.6M 3D 骨架
    右面板: 用关节角驱动的 G1 机器人 3D 骨架

    注意: 由于模型训练错误，H3.6M 中 Y 负方向才是向上

    正面方向计算 (右手定则):
    - shoulder_vec = l_shoulder - r_shoulder (从右肩指向左肩)
    - up_vec = neck - pelvis (从骨盆指向颈部)
    - f_infer = up_vec × shoulder_vec (得到 X+ 方向)

    Args:
        result: dict, process 函数的返回结果
        model: mujoco.MjModel
        data: mujoco.MjData
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    logger.info("=" * 60)
    logger.info("生成可视化动画")
    logger.info("=" * 60)

    joint_angles_all = result['joint_angles']  # (T, 29)
    n_frames = len(joint_angles_all)

    # 创建图形
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # 设置视频输出
    output_video = "g1_mapping_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15
    out_video = cv2.VideoWriter(output_video, fourcc, fps, (1400, 600))

    logger.info(f"渲染 {n_frames} 帧...")
    t_start = time.time()

    for frame_idx in range(n_frames):
        # 清除坐标轴
        ax1.clear()
        ax2.clear()

        # 左面板: 原始 H3.6M 骨架
        pose_3d = result['pose_data'][frame_idx]
        pose_aligned, _ = align_coordinates(pose_3d)

        # 绘制 H3.6M 骨架
        for i, j in H36M_SKELETON:
            ax1.plot(
                [pose_aligned[i, 0], pose_aligned[j, 0]],
                [pose_aligned[i, 1], pose_aligned[j, 1]],
                [pose_aligned[i, 2], pose_aligned[j, 2]],
                'b-', linewidth=2
            )
        ax1.scatter(pose_aligned[:, 0], pose_aligned[:, 1], pose_aligned[:, 2],
                    c='black', s=30, zorder=5)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'H3.6M 骨架 (帧 {frame_idx})')

        # 设置坐标轴范围
        max_range = np.max(np.abs(pose_aligned)) * 1.2
        ax1.set_xlim([-max_range, max_range])
        ax1.set_ylim([-max_range, max_range])
        ax1.set_zlim([-max_range, max_range])

        # 右面板: G1 机器人骨架
        # 设置关节角度
        joint_angles = joint_angles_all[frame_idx]
        for i, joint_name in enumerate(G1_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = joint_angles[i]

        # 前向运动学
        mujoco.mj_forward(model, data)

        # 获取身体部位位置
        body_positions = {}
        for body_name in ["pelvis", "left_hip_pitch_link", "right_hip_pitch_link",
                          "left_knee_link", "right_knee_link",
                          "left_toe_link", "right_toe_link",
                          "left_shoulder_pitch_link", "right_shoulder_pitch_link",
                          "left_elbow_link", "right_elbow_link",
                          "left_rubber_hand", "right_rubber_hand", "head_link"]:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            body_positions[body_name] = data.xpos[body_id].copy()

        # 绘制 G1 骨架连线
        connections = [
            ("pelvis", "left_hip_pitch_link"),
            ("left_hip_pitch_link", "left_knee_link"),
            ("left_knee_link", "left_toe_link"),
            ("pelvis", "right_hip_pitch_link"),
            ("right_hip_pitch_link", "right_knee_link"),
            ("right_knee_link", "right_toe_link"),
            ("pelvis", "left_shoulder_pitch_link"),
            ("left_shoulder_pitch_link", "left_elbow_link"),
            ("left_elbow_link", "left_rubber_hand"),
            ("pelvis", "right_shoulder_pitch_link"),
            ("right_shoulder_pitch_link", "right_elbow_link"),
            ("right_elbow_link", "right_rubber_hand"),
            ("pelvis", "head_link"),
        ]

        for start, end in connections:
            if start in body_positions and end in body_positions:
                p1 = body_positions[start]
                p2 = body_positions[end]
                ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                         'r-', linewidth=2)

        # 绘制关节点
        for name, pos in body_positions.items():
            ax2.scatter(pos[0], pos[1], pos[2], c='black', s=30, zorder=5)

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f'G1 机器人 (帧 {frame_idx})')

        # 设置坐标轴范围
        all_pos = np.array(list(body_positions.values()))
        max_range = np.max(np.abs(all_pos)) * 1.2
        ax2.set_xlim([-max_range, max_range])
        ax2.set_ylim([-max_range, max_range])
        ax2.set_zlim([-max_range, max_range])

        # 设置视角
        ax1.view_init(elev=20, azim=45)
        ax2.view_init(elev=20, azim=45)

        # 保存帧
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (1400, 600))
        out_video.write(img)

        # 进度显示
        if (frame_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            fps_render = (frame_idx + 1) / elapsed
            logger.info(f"  渲染: {frame_idx + 1}/{n_frames} ({elapsed:.1f}s, {fps_render:.1f} fps)")

    out_video.release()
    plt.close(fig)

    elapsed = time.time() - t_start
    logger.info(f"可视化视频已保存: {output_video} ({elapsed:.1f}s)")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数: 读取数据 -> 处理 -> 可视化

    注意: 由于模型训练错误，H3.6M 中 Y 负方向才是向上

    正面方向计算 (右手定则):
    - shoulder_vec = l_shoulder - r_shoulder (从右肩指向左肩)
    - up_vec = neck - pelvis (从骨盆指向颈部)
    - f_infer = up_vec × shoulder_vec (得到 X+ 方向)
    """
    logger.info("=" * 60)
    logger.info("H3.6M -> G1 关节角映射")
    logger.info("=" * 60)

    # 1. 读取数据
    pose_data = io_handler("read")

    # 2. 加载模型
    model, data = io_handler("load_model")

    # 3. 处理
    result = process(pose_data, model, data)

    # 4. 保存结果
    io_handler("write", result)

    # 5. 可视化
    visualize(result, model, data)

    logger.info("完成!")


if __name__ == '__main__':
    # 注意: 由于模型训练错误，H3.6M 中 Y 负方向才是向上
    # 正面方向计算 (右手定则):
    # - shoulder_vec = l_shoulder - r_shoulder (从右肩指向左肩)
    # - up_vec = neck - pelvis (从骨盆指向颈部)
    # - f_infer = up_vec × shoulder_vec (得到 X+ 方向)
    main()
