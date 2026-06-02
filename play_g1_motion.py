"""
播放 G1 关节角动画

功能: 读取 g1_joint_angles.npz，在 MuJoCo 查看器中播放 G1 机器人动作
      支持 root motion（骨盆位置 + 身体朝向）

使用方法:
    python play_g1_motion.py

需要的文件:
    1. g1_joint_angles.npz           — 关节角数据
    2. unitree_g1/g1_mocap_29dof.xml — G1 MuJoCo 模型
"""

import numpy as np
import mujoco
import mujoco.viewer
import time

# 配置
JOINT_ANGLES_NPZ = "g1_joint_angles.npz"
G1_MJCF = "unitree_g1/g1_mocap_29dof.xml"
FPS = 15


def main():
    # 加载关节角数据
    data = np.load(JOINT_ANGLES_NPZ, allow_pickle=True)
    joint_angles = data['joint_angles']  # (T, 29)
    joint_names = data['joint_names']    # (29,)
    n_frames = len(joint_angles)

    # 加载 root motion 数据（骨盆位置 + 朝向角）
    has_root_motion = 'pelvis_positions' in data and 'yaw_angles' in data
    if has_root_motion:
        pelvis_positions = data['pelvis_positions']  # (T, 3)
        yaw_angles = data['yaw_angles']               # (T,)
        print(f"Root motion 数据已加载: pelvis_positions shape={pelvis_positions.shape}")
    else:
        pelvis_positions = None
        yaw_angles = None
        print("警告: 未找到 root motion 数据，机器人将停留在原地")

    print(f"加载关节角数据: {n_frames} 帧, {n_frames / FPS:.1f} 秒")
    print(f"关节: {list(joint_names)}")

    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(G1_MJCF)
    d = mujoco.MjData(model)

    # 创建关节名到 qpos 索引的映射（仅 29 个活动关节）
    joint_qpos_map = {}
    for i, name in enumerate(joint_names):
        name_str = str(name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name_str)
        if joint_id >= 0:
            qpos_adr = model.jnt_qposadr[joint_id]
            joint_qpos_map[i] = qpos_adr

    print(f"关节数: {len(joint_qpos_map)}")

    # 查找 pelvis freejoint 的 qpos 地址
    free_qpos_adr = None
    pelvis_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')
    if pelvis_joint_id >= 0:
        free_qpos_adr = model.jnt_qposadr[pelvis_joint_id]
        # freejoint qpos 布局: [x, y, z, qw, qx, qy, qz]
    else:
        print("警告: 模型中未找到名为 'pelvis' 的 freejoint")

    # 启动查看器
    with mujoco.viewer.launch_passive(model, d) as viewer:
        # 设置相机
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        frame_idx = 0
        t_start = time.time()

        while viewer.is_running():
            t_now = time.time()
            frame_idx = int((t_now - t_start) * FPS) % n_frames

            # 设置 root motion（骨盆位置 + 朝向）
            if has_root_motion and free_qpos_adr is not None:
                px, py, pz = pelvis_positions[frame_idx]
                d.qpos[free_qpos_adr + 0] = px
                d.qpos[free_qpos_adr + 1] = py
                d.qpos[free_qpos_adr + 2] = max(pz, 0.793)  # 保持在地面以上

                # yaw 角转四元数: (qw, qx, qy, qz) = (cos(a/2), 0, 0, sin(a/2))
                half_a = yaw_angles[frame_idx] / 2.0
                d.qpos[free_qpos_adr + 3] = np.cos(half_a)  # qw
                d.qpos[free_qpos_adr + 4] = 0.0               # qx
                d.qpos[free_qpos_adr + 5] = 0.0               # qy
                d.qpos[free_qpos_adr + 6] = np.sin(half_a)    # qz

            # 设置 29 个活动关节角度
            for i, qpos_adr in joint_qpos_map.items():
                d.qpos[qpos_adr] = joint_angles[frame_idx, i]

            # 前向运动学
            mujoco.mj_forward(model, d)

            # 相机跟随机器人
            if has_root_motion:
                viewer.cam.lookat[0] = pelvis_positions[frame_idx, 0]
                viewer.cam.lookat[1] = pelvis_positions[frame_idx, 1]
                viewer.cam.lookat[2] = pelvis_positions[frame_idx, 2]

            # 同步到查看器
            viewer.sync()

            # 控制帧率
            time.sleep(1.0 / FPS)


if __name__ == '__main__':
    main()
