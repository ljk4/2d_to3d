"""
将两个视频按帧对齐，合成为左右并排的视频

使用方法:
    python merge_videos.py left.mp4 right.mp4 -o merged.mp4
    python merge_videos.py left.mp4 right.mp4 -o merged.mp4 --height 720
"""

import sys
import argparse
import cv2
import numpy as np


def merge_side_by_side(left_path, right_path, output_path, height=None, fps=None):
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)

    if not cap_l.isOpened():
        print(f"无法打开: {left_path}")
        return
    if not cap_r.isOpened():
        print(f"无法打开: {right_path}")
        return

    # 视频信息
    fps_l = cap_l.get(cv2.CAP_PROP_FPS)
    fps_r = cap_r.get(cv2.CAP_PROP_FPS)
    frames_l = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = fps or fps_l
    n_frames = min(frames_l, frames_r)

    print(f"左: {left_path} ({frames_l}帧, {fps_l:.1f}fps)")
    print(f"右: {right_path} ({frames_r}帧, {fps_r:.1f}fps)")
    print(f"输出: {output_path} ({n_frames}帧, {out_fps:.1f}fps)")

    out = None
    written = 0

    for i in range(n_frames):
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break

        # 统一高度
        if height is not None:
            h_l, w_l = frame_l.shape[:2]
            h_r, w_r = frame_r.shape[:2]
            frame_l = cv2.resize(frame_l, (int(w_l * height / h_l), height))
            frame_r = cv2.resize(frame_r, (int(w_r * height / h_r), height))

        # 高度不一致时补齐
        h_l, w_l = frame_l.shape[:2]
        h_r, w_r = frame_r.shape[:2]
        h = max(h_l, h_r)
        if h_l < h:
            pad = np.zeros((h - h_l, w_l, 3), dtype=np.uint8)
            frame_l = np.vstack([frame_l, pad])
        if h_r < h:
            pad = np.zeros((h - h_r, w_r, 3), dtype=np.uint8)
            frame_r = np.vstack([frame_r, pad])

        merged = np.hstack([frame_l, frame_r])

        if out is None:
            oh, ow = merged.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, out_fps, (ow, oh))

        out.write(merged)
        written += 1

        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{n_frames}")

    cap_l.release()
    cap_r.release()
    if out:
        out.release()
    print(f"完成: {written} 帧 -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="将两个视频左右合并")
    parser.add_argument("left", help="左侧视频路径")
    parser.add_argument("right", help="右侧视频路径")
    parser.add_argument("-o", "--output", default="merged.mp4", help="输出路径")
    parser.add_argument("--height", type=int, default=None, help="统一高度（自动缩放）")
    parser.add_argument("--fps", type=float, default=None, help="输出帧率（默认同左侧视频）")
    args = parser.parse_args()

    merge_side_by_side(args.left, args.right, args.output, args.height, args.fps)


if __name__ == '__main__':
    main()
