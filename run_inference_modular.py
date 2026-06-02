"""
VideoPose3D 模块化推理脚本

功能: 将端到端推理流程拆分为三个可组合的函数:
  - io_handler(mode)  : "read" 读取视频 / "write" 保存推理结果 (.npz)
  - process()         : 核心处理, MediaPipe 2D + VideoPose3D 3D 推理
  - visualize(result) : 根据结果生成 2D+3D 骨架可视化视频

依赖:
    pip install mediapipe onnxruntime numpy opencv-python matplotlib

需要的模型文件:
    1. pose_landmarker_heavy.task  — MediaPipe 姿态检测模型
    2. videopose3d.onnx            — VideoPose3D ONNX 模型
"""

import os
import sys
import time

import cv2
import numpy as np
from scipy.signal import savgol_filter

os.environ['MEDIAPIPE_DISABLE_ANALYTICS'] = '1'
os.environ['GLOG_minloglevel'] = '2'

# Debug 模式开关 (可通过 --debug 命令行参数启用)
DEBUG = '--debug' in sys.argv

# ============================================================================
# 全局路径配置
# ============================================================================
VIDEO_PATH = "taiji.mp4"                          # 输入视频路径
MEDIAPIPE_MODEL = "pose_landmarker_heavy.task"     # MediaPipe 模型路径
ONNX_MODEL = "videopose3d_finetuned_v3.onnx"           # VideoPose3D ONNX 模型路径 (微调V3: 预训练+全架构)
OUTPUT_NPZ = "taiji.npz"                         # 输出 .npz 文件路径
OUTPUT_VIDEO = "taiji_result.mp4"                      # 输出可视化视频路径
RECEPTIVE_FIELD = 243                                   # 模型感受野 (微调V3, filter_widths=[3,3,3,3,3])

# ============================================================================
# 常量定义 (骨架映射)
# ============================================================================

# H3.6M 17 关节名称
H36M_JOINT_NAMES = [
    'pelvis', 'r_hip', 'r_knee', 'r_ankle',
    'l_hip', 'l_knee', 'l_ankle',
    'spine', 'neck', 'head', 'head_top',
    'l_shoulder', 'l_elbow', 'l_wrist',
    'r_shoulder', 'r_elbow', 'r_wrist',
]

# MediaPipe 索引 -> H3.6M 17 关节索引
MP_PELVIS = [23, 24]
MP_SPINE = [11, 12]
MP_HEAD = [7, 8]
MP_DIRECT_MAP = {
    1: 24,   # right_hip
    2: 26,   # right_knee
    3: 28,   # right_ankle
    4: 23,   # left_hip
    5: 25,   # left_knee
    6: 27,   # left_ankle
    8: 0,    # neck ~ nose
    10: 0,   # head_top ~ nose
    11: 11,  # left_shoulder
    12: 13,  # left_elbow
    13: 15,  # left_wrist
    14: 12,  # right_shoulder
    15: 14,  # right_elbow
    16: 16,  # right_wrist
}

# H3.6M 骨架连线
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]

# 骨架颜色 (BGR)
SKELETON_COLORS = [
    (0, 0, 255), (0, 0, 255), (0, 0, 255),
    (255, 0, 0), (255, 0, 0), (255, 0, 0),
    (0, 255, 0), (0, 255, 0), (0, 200, 0), (0, 200, 0),
    (255, 0, 0), (255, 0, 0), (255, 0, 0),
    (0, 0, 255), (0, 0, 255), (0, 0, 255),
]

# 3D 骨架父关节
PARENTS_3D = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# ============================================================================
# 辅助函数
# ============================================================================

def init_mediapipe(model_path):
    """初始化 MediaPipe Pose Landmarker (IMAGE 模式, 每帧独立检测)。"""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.9,
        min_pose_presence_confidence=0.9,
    )
    return vision.PoseLandmarker.create_from_options(options)


def detect_keypoints_from_video(detector, video_path, max_frames=None):
    """从视频逐帧检测 2D 关键点。

    返回:
        keypoints: (T, 17, 2) H3.6M 格式像素坐标
        frames_bgr: list of BGR frames
        image_size: (height, width)
        fps: 帧率
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'无法打开视频: {video_path}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'视频: {width}x{height}, {fps:.1f}fps, {total_frames}帧')

    all_keypoints = []
    all_frames = []
    frame_idx = 0
    t_start = time.time()

    # Debug 统计变量
    per_frame_ms = []           # 每帧推理耗时 (ms)
    missing_frames = []         # 漏检帧索引
    prev_kps = None             # 上一帧关键点, 用于跳变检测

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        all_frames.append(frame.copy())

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        t_frame = time.time()
        result = detector.detect(mp_image)
        dt_ms = (time.time() - t_frame) * 1000
        per_frame_ms.append(dt_ms)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            # Debug: 打印置信度最低的 3 个关键点
            if DEBUG and frame_idx < 5:
                vis_scores = [(i, lm.visibility) for i, lm in enumerate(landmarks)]
                vis_scores.sort(key=lambda x: x[1])
                worst = vis_scores[:3]
                print(f'  [DEBUG] Frame {frame_idx} 最低可见度: '
                      f'{[(i, f"{v:.3f}") for i, v in worst]}')

            mp_kps = np.array(
                [[lm.x * width, lm.y * height] for lm in landmarks],
                dtype=np.float32,
            )
            kps_h36m = np.zeros((17, 2), dtype=np.float32)
            for h36m_idx, mp_idx in MP_DIRECT_MAP.items():
                kps_h36m[h36m_idx] = mp_kps[mp_idx]
            kps_h36m[0] = (mp_kps[MP_PELVIS[0]] + mp_kps[MP_PELVIS[1]]) / 2
            kps_h36m[7] = (mp_kps[MP_SPINE[0]] + mp_kps[MP_SPINE[1]]) / 2
            kps_h36m[9] = (mp_kps[MP_HEAD[0]] + mp_kps[MP_HEAD[1]]) / 2

            # Debug: 检测帧间跳变
            if DEBUG and prev_kps is not None:
                jump = np.nanmax(np.abs(kps_h36m - prev_kps))
                if jump > 50:  # 像素级阈值
                    worst_joint = np.unravel_index(
                        np.argmax(np.abs(kps_h36m - prev_kps)), kps_h36m.shape)
                    print(f'  [DEBUG] Frame {frame_idx} 大跳变: {jump:.1f}px '
                          f'(joint={worst_joint[0]}, {H36M_JOINT_NAMES[worst_joint[0]]})')
            prev_kps = kps_h36m.copy()
        else:
            kps_h36m = np.full((17, 2), np.nan, dtype=np.float32)
            missing_frames.append(frame_idx)

        all_keypoints.append(kps_h36m)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            print(f'  MediaPipe: {frame_idx} 帧 ({elapsed:.1f}s)')

    cap.release()

    # ========== Debug: MediaPipe 逐帧统计 ==========
    per_frame_ms = np.array(per_frame_ms)
    print(f'\n  --- MediaPipe 逐帧耗时统计 ---')
    print(f'  平均: {per_frame_ms.mean():.1f}ms | 中位: {np.median(per_frame_ms):.1f}ms | '
          f'最大: {per_frame_ms.max():.1f}ms | 最小: {per_frame_ms.min():.1f}ms')
    n_realtime = np.sum(per_frame_ms < 33.3)
    print(f'  实时达标 (<33ms): {n_realtime}/{len(per_frame_ms)} 帧 ({n_realtime/len(per_frame_ms)*100:.1f}%)')

    n_total = len(all_keypoints)
    n_missing = len(missing_frames)
    print(f'\n  --- 漏检统计 ---')
    print(f'  总帧数: {n_total} | 漏检帧: {n_missing} ({n_missing/n_total*100:.1f}%)')
    if missing_frames and DEBUG:
        # 显示漏检帧的连续区间
        gaps = np.diff(missing_frames)
        starts = [missing_frames[0]]
        lengths = [1]
        for g in gaps:
            if g == 1:
                lengths[-1] += 1
            else:
                starts.append(starts[-1] + lengths[-1] + int(g) - 1)
                lengths.append(1)
        print(f'  [DEBUG] 漏检区间: {list(zip(starts, lengths))}')

    keypoints = np.array(all_keypoints)

    valid_mask = ~np.isnan(keypoints[:, 0, 0])
    if not np.all(valid_mask):
        n_missing = np.sum(~valid_mask)
        indices = np.arange(len(keypoints))
        for j in range(17):
            for d in range(2):
                keypoints[:, j, d] = np.interp(
                    indices, indices[valid_mask], keypoints[valid_mask, j, d],
                )
        print(f'  插值填补了 {n_missing} 个缺失帧')

    # 时序后处理: 异常值剔除 + Savitzky-Golay 平滑
    n_frames = len(keypoints)
    if n_frames >= 5:
        # Step 1: 异常值剔除 — 滑动中值检测, 偏差 > 阈值的点替换为中值
        med_win = 11  # 中值滤波窗口
        outlier_thresh = 40.0  # 像素阈值
        n_outliers = 0
        half = med_win // 2
        for j in range(17):
            for d in range(2):
                signal = keypoints[:, j, d].copy()
                for i in range(n_frames):
                    lo = max(0, i - half)
                    hi = min(n_frames, i + half + 1)
                    med = np.median(signal[lo:hi])
                    if abs(signal[i] - med) > outlier_thresh:
                        keypoints[i, j, d] = med
                        n_outliers += 1
        if n_outliers > 0:
            print(f'  异常值剔除: 修正了 {n_outliers} 个关节-帧 (阈值={outlier_thresh}px)')

        # Step 2: Savitzky-Golay 时序平滑
        win = min(21, n_frames if n_frames % 2 == 1 else n_frames - 1)
        if win >= 5:
            for j in range(17):
                for d in range(2):
                    keypoints[:, j, d] = savgol_filter(keypoints[:, j, d], win, polyorder=3)
            print(f'  时序平滑完成 (窗口={win}, polyorder=3)')

    # ========== Debug: 2D 关键点质量分析 ==========
    if DEBUG and len(keypoints) > 1:
        diffs = np.abs(np.diff(keypoints, axis=0))  # (T-1, 17, 2)
        jump_per_frame = np.nanmax(diffs.reshape(diffs.shape[0], -1), axis=1)
        print(f'\n  --- 2D 关键点帧间跳变分析 (平滑后) ---')
        print(f'  平均最大跳变: {jump_per_frame.mean():.2f}px | '
              f'P95: {np.percentile(jump_per_frame, 95):.2f}px | '
              f'最大: {jump_per_frame.max():.2f}px')
        big_jumps = np.where(jump_per_frame > 30)[0]
        if len(big_jumps) > 0:
            print(f'  大跳变帧 (>30px): {len(big_jumps)} 个, 索引: {big_jumps[:10].tolist()}...')

        # 关键点是否在图像范围内
        oob = (keypoints < 0) | (keypoints > [width, height])
        oob_count = np.sum(oob.any(axis=2))
        if oob_count > 0:
            print(f'  超出图像范围的关键点: {oob_count} 个')

    elapsed = time.time() - t_start
    print(f'MediaPipe 完成: {len(keypoints)} 帧, {elapsed:.1f}s')

    return keypoints, all_frames, (height, width), fps


def normalize_screen_coordinates(X, w, h):
    """像素坐标 -> 屏幕坐标, 与 VideoPose3D common/camera.py 一致。"""
    return X / w * 2 - [1, h / w]


def run_videopose3d_onnx(keypoints_2d, onnx_path, receptive_field=RECEPTIVE_FIELD):
    """ONNX 推理。输入 (1, T, 17, 2) 归一化坐标, 输出 (1, T, 17, 3)。"""
    import onnxruntime as ort

    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
    session = ort.InferenceSession(onnx_path, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    half_pad = (receptive_field - 1) // 2
    padded = np.pad(keypoints_2d, ((0, 0), (half_pad, half_pad), (0, 0), (0, 0)), mode='edge')

    output = session.run([output_name], {input_name: padded.astype(np.float32)})[0]
    return output


def draw_2d_skeleton(frame, keypoints):
    """在视频帧上绘制 H3.6M 17 关节骨架。"""
    overlay = frame.copy()

    for idx, (i, j) in enumerate(H36M_SKELETON):
        x1, y1 = int(keypoints[i, 0]), int(keypoints[i, 1])
        x2, y2 = int(keypoints[j, 0]), int(keypoints[j, 1])
        color = SKELETON_COLORS[idx % len(SKELETON_COLORS)]
        cv2.line(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    for i in range(17):
        x, y = int(keypoints[i, 0]), int(keypoints[i, 1])
        cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)

    return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)


def render_3d_skeleton(pose_3d, ax, title='3D Pose', elev=10, azim=-90):
    """在 matplotlib 3D 坐标轴上绘制骨架。"""
    ax.clear()

    pelvis = pose_3d[0]
    p = pose_3d - pelvis

    for i in range(17):
        parent = PARENTS_3D[i]
        if parent >= 0:
            ax.plot(
                [p[i, 0], p[parent, 0]],
                [p[i, 2], p[parent, 2]],
                [-p[i, 1], -p[parent, 1]],
                'b-', linewidth=2.5, solid_capstyle='round',
            )

    ax.scatter(p[:, 0], p[:, 2], -p[:, 1], c='red', s=10, zorder=5)

    ax.set_xlabel('X (left-right)', fontsize=9)
    ax.set_ylabel('Z (depth)', fontsize=9)
    ax.set_zlabel('Height', fontsize=9)
    ax.set_title(title, fontsize=10)

    limit = 0.8
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)

    ax.view_init(elev=elev, azim=azim)


# ============================================================================
# 三个主要函数
# ============================================================================

def io_handler(mode, result=None):
    """IO 处理函数。

    mode="read":  读取全局 VIDEO_PATH, 返回 (frames_bgr, image_size, fps)
    mode="write": 将 result dict 保存为全局 OUTPUT_NPZ (.npz)
    """
    if mode == "read":
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise FileNotFoundError(f'无法打开视频: {VIDEO_PATH}')

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'[io_handler] 读取视频: {VIDEO_PATH}')
        print(f'  {width}x{height}, {fps:.1f}fps, {total_frames}帧')

        frames_bgr = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
        cap.release()

        return frames_bgr, (height, width), fps

    elif mode == "write":
        if result is None:
            raise ValueError("write 模式需要传入 result 参数")

        np.savez_compressed(
            OUTPUT_NPZ,
            poses_3d=result['poses_3d'],
            keypoints_2d=result['keypoints_2d'],
            image_size=np.array(result['image_size']),
            fps=result['fps'],
        )
        print(f'[io_handler] 结果已保存: {OUTPUT_NPZ}')

    else:
        raise ValueError(f"未知模式: {mode}, 请使用 'read' 或 'write'")


def process(max_frames=None):
    """核心处理函数: MediaPipe 2D 检测 + VideoPose3D 3D 推理。

    内部调用 io_handler("read") 验证视频可读, 再由 detect_keypoints_from_video 逐帧检测。

    返回 result dict:
        poses_3d:     (T, 17, 3) 3D 姿态
        keypoints_2d: (T, 17, 2) 2D 关键点 (像素坐标)
        frames_bgr:   list of BGR frames
        image_size:   (height, width)
        fps:          float
    """
    # 检查文件存在性
    for path, name in [(VIDEO_PATH, '视频'), (MEDIAPIPE_MODEL, 'MediaPipe'),
                       (ONNX_MODEL, 'ONNX')]:
        if not os.path.isfile(path):
            print(f'错误: {name} 不存在: {path}')
            sys.exit(1)

    # 1. 读取视频 (通过 io_handler)
    print('=' * 50)
    print('[1/3] 读取视频')
    print('=' * 50)
    _, (img_h, img_w), video_fps = io_handler("read")

    # 2. MediaPipe 2D 检测
    print('\n' + '=' * 50)
    print('[2/3] MediaPipe 2D 关键点检测')
    print('=' * 50)
    detector = init_mediapipe(MEDIAPIPE_MODEL)
    keypoints_2d_px, frames_bgr, (img_h, img_w), video_fps = detect_keypoints_from_video(
        detector, VIDEO_PATH, max_frames,
    )
    n_frames = len(keypoints_2d_px)
    print(f'检测到 {n_frames} 帧, 图像 {img_w}x{img_h}')

    # 3. 归一化 + ONNX 推理
    print('\n' + '=' * 50)
    print('[3/3] VideoPose3D ONNX 3D 预测')
    print('=' * 50)
    kps_norm = normalize_screen_coordinates(keypoints_2d_px.copy(), w=img_w, h=img_h)
    kps_batch = kps_norm[np.newaxis, ...]

    t0 = time.time()
    poses_3d = run_videopose3d_onnx(kps_batch, ONNX_MODEL, RECEPTIVE_FIELD)
    t_infer = time.time() - t0
    poses_3d = poses_3d[0]
    print(f'输出: {poses_3d.shape}, 耗时 {t_infer:.2f}s ({n_frames / t_infer:.0f} fps)')

    # ========== Debug: 3D 姿态质量分析 ==========
    if DEBUG:
        print(f'\n  --- 3D 姿态统计 ---')
        print(f'  形状: poses_3d={poses_3d.shape}, keypoints_2d={keypoints_2d_px.shape}, '
              f'frames_bgr={len(frames_bgr)}')
        print(f'  3D 值范围: X=[{poses_3d[:,:,0].min():.3f}, {poses_3d[:,:,0].max():.3f}] '
              f'Y=[{poses_3d[:,:,1].min():.3f}, {poses_3d[:,:,1].max():.3f}] '
              f'Z=[{poses_3d[:,:,2].min():.3f}, {poses_3d[:,:,2].max():.3f}]')
        n_nan_3d = np.isnan(poses_3d).any(axis=(1,2)).sum()
        if n_nan_3d > 0:
            print(f'  含 NaN 的 3D 帧: {n_nan_3d}')

        # 帧对齐验证
        n_frames_2d = len(keypoints_2d_px)
        n_frames_3d = len(poses_3d)
        n_frames_bgr = len(frames_bgr)
        if n_frames_2d == n_frames_3d == n_frames_bgr:
            print(f'  帧对齐: OK (2D={n_frames_2d}, 3D={n_frames_3d}, BGR={n_frames_bgr})')
        else:
            print(f'  帧对齐: ** MISMATCH ** (2D={n_frames_2d}, 3D={n_frames_3d}, BGR={n_frames_bgr})')

    result = {
        'poses_3d': poses_3d,
        'keypoints_2d': keypoints_2d_px,
        'frames_bgr': frames_bgr,
        'image_size': (img_h, img_w),
        'fps': video_fps,
    }

    return result


def visualize(result):
    """可视化函数: 根据 result 生成 2D+3D 骨架视频。

    布局为 2x2 网格:
      左上: 2D MediaPipe 骨架
      右上: 3D 侧面视角 (azim=-90)
      左下: 3D 正面视角 (azim=0)
      右下: 3D 俯视视角 (elev=90, azim=-90)

    输出到全局 OUTPUT_VIDEO 路径。
    """
    poses_3d = result['poses_3d']
    keypoints_2d = result['keypoints_2d']
    frames_bgr = result['frames_bgr']
    img_h, img_w = result['image_size']
    video_fps = result['fps']
    n_frames = len(poses_3d)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print('需要 matplotlib: pip install matplotlib')
        return

    print('\n' + '=' * 50)
    print('生成可视化视频 (2D + 3个3D视角)')
    print('=' * 50)

    # 用独立 figure 分别渲染三个视角, 避免 subplot 3D 共享冲突
    fig_side = plt.figure(figsize=(3, 3))
    ax_side = fig_side.add_subplot(111, projection='3d')
    fig_front = plt.figure(figsize=(3, 3))
    ax_front = fig_front.add_subplot(111, projection='3d')
    fig_top = plt.figure(figsize=(3, 3))
    ax_top = fig_top.add_subplot(111, projection='3d')

    # 2x2 网格: 每个单元格 img_w x img_h
    canvas_w = img_w * 2
    canvas_h = img_h * 2
    half_w, half_h = img_w, img_h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, video_fps, (canvas_w, canvas_h))

    def render_view(ax, fig_obj, pose_3d, title, elev=10, azim=-90):
        render_3d_skeleton(pose_3d, ax, title=title, elev=elev, azim=azim)
        fig_obj.canvas.draw()
        buf = fig_obj.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return cv2.resize(img, (half_w, half_h))

    print(f'渲染 {n_frames} 帧...')
    t_start = time.time()

    for i in range(n_frames):
        frame_2d = draw_2d_skeleton(frames_bgr[i], keypoints_2d[i])
        frame_2d = cv2.resize(frame_2d, (half_w, half_h))

        # 渲染三个 3D 视角
        img_side = render_view(ax_side, fig_side, poses_3d[i], f'Side - F{i}', azim=-90)
        img_front = render_view(ax_front, fig_front, poses_3d[i], f'Front - F{i}', azim=0)
        img_top = render_view(ax_top, fig_top, poses_3d[i], f'Top - F{i}', elev=90, azim=-90)

        # 组装 2x2 网格
        top_row = np.hstack([frame_2d, img_side])
        bottom_row = np.hstack([img_front, img_top])
        canvas = np.vstack([top_row, bottom_row])

        # 网格分割线
        cv2.line(canvas, (half_w, 0), (half_w, canvas_h), (255, 255, 255), 2)
        cv2.line(canvas, (0, half_h), (canvas_w, half_h), (255, 255, 255), 2)

        # 标签
        cv2.putText(canvas, '2D (MediaPipe)', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, '3D Side', (half_w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, '3D Front', (10, half_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, '3D Top', (half_w + 10, half_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Debug: 帧号
        if DEBUG:
            cv2.putText(canvas, f'Frame {i}/{n_frames}', (10, half_h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        out_video.write(canvas)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f'  {i + 1}/{n_frames} 帧 ({elapsed:.1f}s)')

    out_video.release()
    plt.close(fig_side)
    plt.close(fig_front)
    plt.close(fig_top)

    elapsed = time.time() - t_start
    print(f'可视化视频已保存: {OUTPUT_VIDEO} ({elapsed:.1f}s)')


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 完整流程: 推理 -> 保存 -> 可视化
    result = process()
    io_handler("write", result=result)
    visualize(result)

if __name__ == '__main__':
    main()
