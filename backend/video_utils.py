# video_utils.py
import cv2
import numpy as np
import os
from typing import Optional

def extract_frames_from_video(video_path, max_frames=None, resize_width=None):
    """
    Returns list of grayscale frames (np.uint8).
    Optionally resize to width while preserving aspect ratio.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: " + video_path)

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_width is not None:
            h, w = frame.shape[:2]
            new_w = resize_width
            new_h = int(h * (new_w / w))
            frame = cv2.resize(frame, (new_w, new_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    cap.release()
    return frames

def extract_motion_features_from_video(video_path,
                                       max_frames=None,
                                       resize_width=320,
                                       node_features=4,
                                       num_nodes=19) -> Optional[np.ndarray]:
    """
    Extract simple optical-flow based features per frame and convert to (T, node_features*num_nodes)
    Heuristic:
      per frame compute: mean_flow_x, mean_flow_y, mean_magnitude, std_magnitude -> 4 dims
      tile to num_nodes to make 76 dims (19*4)
    """
    frames = extract_frames_from_video(video_path, max_frames=max_frames, resize_width=resize_width)
    if len(frames) < 2:
        return None

    T = len(frames) - 1
    feats = np.zeros((T, node_features), dtype=np.float32)

    prev = frames[0]
    for i in range(1, len(frames)):
        curr = frames[i]
        # calculate dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev, curr,
                                            None,
                                            pyr_scale=0.5, levels=2, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        # flow shape (H,W,2) -> (dx, dy)
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag = np.sqrt(fx**2 + fy**2)

        mean_fx = float(np.mean(fx))
        mean_fy = float(np.mean(fy))
        mean_mag = float(np.mean(mag))
        std_mag = float(np.std(mag))

        feats[i-1, 0] = mean_fx
        feats[i-1, 1] = mean_fy
        feats[i-1, 2] = mean_mag
        feats[i-1, 3] = std_mag

        prev = curr

    # Normalize each column to zero mean, unit std (safe)
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mu) / sd

    # tile to match num_nodes*node_features
    tiled = np.tile(feats, (1, num_nodes))  # shape (T, 4*num_nodes)
    if tiled.shape[1] != node_features * num_nodes:
        # fallback pad/truncate
        wanted = node_features * num_nodes
        if tiled.shape[1] < wanted:
            pad = np.zeros((tiled.shape[0], wanted - tiled.shape[1]), dtype=np.float32)
            tiled = np.concatenate([tiled, pad], axis=1)
        else:
            tiled = tiled[:, :wanted]

    return tiled  # shape (T, 76) by default
