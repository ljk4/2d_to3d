"""
播放 G1 关节角动画

功能: 读取 g1_joint_angles.npz，在 MuJoCo 查看器中播放 G1 机器人动作

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
    data = np.load(JOINT_ANGLES_NPZ)
    joint_angles = data['joint_angles']  # (T, 29)
    joint_names = data['joint_names']    # (29,)
    n_frames = len(joint_angles)

    print(f"加载关节角数据: {n_frames} 帧, {n_frames / FPS:.1f} 秒")
    print(f"关节: {list(joint_names)}")

    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(G1_MJCF)
    d = mujoco.MjData(model)

    # 创建关节名到 qpos 索引的映射
    joint_qpos_map = {}
    for i, name in enumerate(joint_names):
        name_str = str(name)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name_str)
        if joint_id >= 0:
            qpos_adr = model.jnt_qposadr[joint_id]
            joint_qpos_map[i] = qpos_adr

    print(f"关节数: {len(joint_qpos_map)}")

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

            # 设置关节角度
            for i, qpos_adr in joint_qpos_map.items():
                d.qpos[qpos_adr] = joint_angles[frame_idx, i]

            # 前向运动学
            mujoco.mj_forward(model, d)

            # 同步到查看器
            viewer.sync()

            # 控制帧率
            time.sleep(1.0 / FPS)


if __name__ == '__main__':
    main()
