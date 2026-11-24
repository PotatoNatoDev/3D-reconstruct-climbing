#!/usr/bin/env python3
"""Extract per-frame 3D pose landmarks from a video using MediaPipe PoseLandmarker.

Usage:
    python pose_extract.py climb.mp4
    python pose_extract.py climb.mp4 -o climb_pose.json --model heavy --preview
    python pose_extract.py climb.mp4 --stride 2        # process every 2nd frame

Output JSON layout:
{
  "meta": {video, fps, width, height, frame_count, stride, model,
           landmark_names, connections, coordinate_notes},
  "frames": [
    {"i": 0, "t": 0.0, "detected": true,
     "world": [[x, y, z, visibility], ...33],   # meters, origin at hip center
     "image": [[x, y, z, visibility], ...33]},  # x,y normalized 0..1 in the frame
    ...
  ]
}
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{name}/float16/latest/pose_landmarker_{name}.task"
)

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# Body-only skeleton (index pairs into LANDMARK_NAMES); face points are skipped.
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def ensure_model(name: str, cache_dir: Path) -> Path:
    """Download the .task model file once and reuse it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pose_landmarker_{name}.task"
    if not path.exists():
        url = MODEL_URL.format(name=name)
        print(f"Downloading {name} model (one time)...", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return path


def r5(v: float) -> float:
    return round(float(v), 5)


def draw_preview(frame, image_lms):
    h, w = frame.shape[:2]
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in image_lms]
    for a, b in CONNECTIONS:
        # Dim a bone if either end is low confidence
        low = image_lms[a][3] < 0.5 or image_lms[b][3] < 0.5
        color = (0, 140, 255) if low else (0, 255, 0)
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if i >= 11:
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None, help="output JSON path (default: <video>_pose.json)")
    ap.add_argument("--model", choices=["lite", "full", "heavy"], default="heavy",
                    help="heavy is most accurate and fine for offline use (default)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (default 1 = all)")
    ap.add_argument("--preview", action="store_true", help="also write an mp4 with the 2D skeleton drawn on it")
    ap.add_argument("--model-dir", type=Path, default=Path.home() / ".cache" / "mediapipe_models")
    args = ap.parse_args()

    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")
    out_path = args.output or args.video.with_name(args.video.stem + "_pose.json")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model_path = ensure_model(args.model, args.model_dir)
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,  # enables temporal tracking between frames
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    writer = None
    if args.preview:
        preview_path = out_path.with_name(out_path.stem + "_preview.mp4")
        writer = cv2.VideoWriter(
            str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (width, height)
        )

    frames = []
    detected_count = 0
    idx = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(idx * 1000 / fps)  # must increase monotonically
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                entry = {"i": idx, "t": round(idx / fps, 4), "detected": False}
                if result.pose_landmarks and result.pose_world_landmarks:
                    img_lms = [[r5(l.x), r5(l.y), r5(l.z), r5(l.visibility)] for l in result.pose_landmarks[0]]
                    world_lms = [[r5(l.x), r5(l.y), r5(l.z), r5(l.visibility)] for l in result.pose_world_landmarks[0]]
                    entry.update(detected=True, world=world_lms, image=img_lms)
                    detected_count += 1
                    if writer:
                        draw_preview(bgr, img_lms)
                frames.append(entry)
                if writer:
                    writer.write(bgr)

                if len(frames) % 50 == 0:
                    print(f"  processed {len(frames)} frames (video frame {idx}/{total})", file=sys.stderr)
            idx += 1

    cap.release()
    if writer:
        writer.release()

    data = {
        "meta": {
            "video": args.video.name,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": idx,
            "stride": args.stride,
            "model": args.model,
            "landmark_names": LANDMARK_NAMES,
            "connections": CONNECTIONS,
            "coordinate_notes": (
                "world: meters, origin at midpoint of hips, +x toward the person's left in image space, "
                "+y DOWN, +z away from camera (flip y for Three.js). "
                "image: x,y normalized to frame width/height, y down."
            ),
        },
        "frames": frames,
    }
    out_path.write_text(json.dumps(data, separators=(",", ":")))

    n = len(frames)
    rate = 100 * detected_count / n if n else 0
    print(f"Done: {detected_count}/{n} frames with a pose ({rate:.0f}%) -> {out_path}")
    if args.preview:
        print(f"Preview video -> {preview_path}")


if __name__ == "__main__":
    main()