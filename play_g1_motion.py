"""
播放 G1 关节角动画

功能: 读取 g1_joint_angles.npz，在 MuJoCo 查看器中播放 G1 机器人动作
      支持 root motion（骨盆位置 + 身体朝向）
      支持保存为视频 (--save)

使用方法:
    python play_g1_motion.py                # 交互式查看器
    python play_g1_motion.py --save         # 保存为视频 (g1_motion.mp4)
    python play_g1_motion.py --save -o out.mp4  # 指定输出路径

需要的文件:
    1. g1_joint_angles.npz           — 关节角数据
    2. unitree_g1/g1_mocap_29dof.xml — G1 MuJoCo 模型
"""

import sys
import numpy as np
import mujoco
import mujoco.viewer
import time

# 配置
JOINT_ANGLES_NPZ = "g1_joint_angles.npz"
G1_MJCF = "unitree_g1/g1_mocap_29dof.xml"
FPS = 15
VIDEO_SIZE = (1280, 720)


def setup_frame(model, d, frame_idx, joint_qpos_map, joint_angles,
                has_root_motion, free_qpos_adr, pelvis_positions, yaw_angles):
    """设置一帧的关节角度和 root motion"""
    # 设置 root motion
    if has_root_motion and free_qpos_adr is not None:
        px, py, pz = pelvis_positions[frame_idx]
        d.qpos[free_qpos_adr + 0] = px
        d.qpos[free_qpos_adr + 1] = py
        d.qpos[free_qpos_adr + 2] = max(pz, 0.793)

        half_a = yaw_angles[frame_idx] / 2.0
        d.qpos[free_qpos_adr + 3] = np.cos(half_a)  # qw
        d.qpos[free_qpos_adr + 4] = 0.0
        d.qpos[free_qpos_adr + 5] = 0.0
        d.qpos[free_qpos_adr + 6] = np.sin(half_a)  # qz

    # 设置活动关节
    for i, qpos_adr in joint_qpos_map.items():
        d.qpos[qpos_adr] = joint_angles[frame_idx, i]

    mujoco.mj_forward(model, d)


def play_viewer(model, d, n_frames, joint_qpos_map, joint_angles,
                has_root_motion, free_qpos_adr, pelvis_positions, yaw_angles):
    """交互式查看器播放"""
    with mujoco.viewer.launch_passive(model, d) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 135

        frame_idx = 0
        t_start = time.time()

        while viewer.is_running():
            t_now = time.time()
            frame_idx = int((t_now - t_start) * FPS) % n_frames

            setup_frame(model, d, frame_idx, joint_qpos_map, joint_angles,
                        has_root_motion, free_qpos_adr, pelvis_positions, yaw_angles)

            if has_root_motion:
                viewer.cam.lookat[0] = pelvis_positions[frame_idx, 0]
                viewer.cam.lookat[1] = pelvis_positions[frame_idx, 1]
                viewer.cam.lookat[2] = pelvis_positions[frame_idx, 2]

            viewer.sync()
            time.sleep(1.0 / FPS)


def save_video(model, d, n_frames, joint_qpos_map, joint_angles,
               has_root_motion, free_qpos_adr, pelvis_positions, yaw_angles,
               output_path):
    """离线渲染并保存为视频"""
    import cv2

    renderer = mujoco.Renderer(model, height=VIDEO_SIZE[1], width=VIDEO_SIZE[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, VIDEO_SIZE)

    # 设置相机初始参数
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.elevation = -20
    cam.azimuth = 135

    print(f"渲染 {n_frames} 帧到 {output_path} ...")
    t_start = time.time()

    for frame_idx in range(n_frames):
        setup_frame(model, d, frame_idx, joint_qpos_map, joint_angles,
                    has_root_motion, free_qpos_adr, pelvis_positions, yaw_angles)

        # 相机跟随
        if has_root_motion:
            cam.lookat[0] = pelvis_positions[frame_idx, 0]
            cam.lookat[1] = pelvis_positions[frame_idx, 1]
            cam.lookat[2] = pelvis_positions[frame_idx, 2]

        renderer.update_scene(d, cam)
        img = renderer.render()  # (H, W, 3) RGB uint8
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img_bgr)

        if (frame_idx + 1) % 30 == 0:
            elapsed = time.time() - t_start
            fps_render = (frame_idx + 1) / elapsed
            print(f"  {frame_idx + 1}/{n_frames} ({elapsed:.1f}s, {fps_render:.1f} fps)")

    out.release()
    renderer.close()
    elapsed = time.time() - t_start
    print(f"视频已保存: {output_path} ({elapsed:.1f}s)")


def main():
    # 解析命令行参数
    args = sys.argv[1:]
    do_save = '--save' in args
    output_path = "g1_motion.mp4"
    if '-o' in args:
        idx = args.index('-o')
        if idx + 1 < len(args):
            output_path = args[idx + 1]

    # 加载数据
    data = np.load(JOINT_ANGLES_NPZ, allow_pickle=True)
    joint_angles = data['joint_angles']
    joint_names = data['joint_names']
    n_frames = len(joint_angles)

    has_root_motion = 'pelvis_positions' in data and 'yaw_angles' in data
    if has_root_motion:
        pelvis_positions = data['pelvis_positions']
        yaw_angles = data['yaw_angles']
        print(f"Root motion 数据已加载")
    else:
        pelvis_positions = None
        yaw_angles = None
        print("警告: 未找到 root motion 数据")

    print(f"帧数: {n_frames}, 时长: {n_frames / FPS:.1f}s")

    # 加载模型
    model = mujoco.MjModel.from_xml_path(G1_MJCF)
    d = mujoco.MjData(model)

    # 关节映射
    joint_qpos_map = {}
    for i, name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id >= 0:
            joint_qpos_map[i] = model.jnt_qposadr[joint_id]

    # freejoint 地址
    free_qpos_adr = None
    pelvis_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')
    if pelvis_joint_id >= 0:
        free_qpos_adr = model.jnt_qposadr[pelvis_joint_id]

    if do_save:
        save_video(model, d, n_frames, joint_qpos_map, joint_angles,
                   has_root_motion, free_qpos_adr, pelvis_positions, yaw_angles,
                   output_path)
    else:
        play_viewer(model, d, n_frames, joint_qpos_map, joint_angles,
                    has_root_motion, free_qpos_adr, pelvis_positions, yaw_angles)


if __name__ == '__main__':
    main()
