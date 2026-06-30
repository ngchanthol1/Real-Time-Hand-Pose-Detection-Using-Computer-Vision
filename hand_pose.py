
import os, sys, time, urllib.request, argparse, pathlib
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


#  21 LANDMARK INDICES  (identical to TF.js handpose schema)

WRIST        = 0
THUMB_CMC, THUMB_MCP, THUMB_IP,   THUMB_TIP   = 1,  2,  3,  4
INDEX_MCP,  INDEX_PIP, INDEX_DIP,  INDEX_TIP   = 5,  6,  7,  8
MIDDLE_MCP, MIDDLE_PIP,MIDDLE_DIP, MIDDLE_TIP  = 9,  10, 11, 12
RING_MCP,   RING_PIP,  RING_DIP,   RING_TIP    = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP, PINKY_DIP,  PINKY_TIP   = 17, 18, 19, 20

# Same 21 bone connections as TF.js handpose annotations
HAND_CONNECTIONS = [
    (WRIST, THUMB_CMC),(WRIST, INDEX_MCP),(WRIST, PINKY_MCP),
    (INDEX_MCP,MIDDLE_MCP),(MIDDLE_MCP,RING_MCP),(RING_MCP,PINKY_MCP),
    (THUMB_CMC,THUMB_MCP),(THUMB_MCP,THUMB_IP),(THUMB_IP,THUMB_TIP),
    (INDEX_MCP,INDEX_PIP),(INDEX_PIP,INDEX_DIP),(INDEX_DIP,INDEX_TIP),
    (MIDDLE_MCP,MIDDLE_PIP),(MIDDLE_PIP,MIDDLE_DIP),(MIDDLE_DIP,MIDDLE_TIP),
    (RING_MCP,RING_PIP),(RING_PIP,RING_DIP),(RING_DIP,RING_TIP),
    (PINKY_MCP,PINKY_PIP),(PINKY_PIP,PINKY_DIP),(PINKY_DIP,PINKY_TIP),
]

FINGER_COLORS = {
    "thumb" :(  0,215,255), "index" :(  0,255,  0),
    "middle":(255,165,  0), "ring"  :(147, 20,255),
    "pinky" :(  0,  0,255), "palm"  :(255,255,255),
}
FINGER_IDS = {
    "thumb" : {THUMB_CMC,THUMB_MCP,THUMB_IP,THUMB_TIP},
    "index" : {INDEX_MCP,INDEX_PIP,INDEX_DIP,INDEX_TIP},
    "middle": {MIDDLE_MCP,MIDDLE_PIP,MIDDLE_DIP,MIDDLE_TIP},
    "ring"  : {RING_MCP,RING_PIP,RING_DIP,RING_TIP},
    "pinky" : {PINKY_MCP,PINKY_PIP,PINKY_DIP,PINKY_TIP},
}

LANDMARK_NAMES = [
    "Wrist",
    "Thumb-CMC","Thumb-MCP","Thumb-IP","Thumb-Tip",
    "Index-MCP","Index-PIP","Index-DIP","Index-Tip",
    "Middle-MCP","Middle-PIP","Middle-DIP","Middle-Tip",
    "Ring-MCP","Ring-PIP","Ring-DIP","Ring-Tip",
    "Pinky-MCP","Pinky-PIP","Pinky-DIP","Pinky-Tip",
]


#  MODEL DOWNLOAD

MODEL_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
def _safe_model_cache() -> pathlib.Path:
    """
    Return a model cache path that contains only ASCII characters.
    On Windows the home directory may contain non-ASCII characters (e.g. Korean)
    which MediaPipe's C++ backend cannot open.  Fall back to C:/handpose_model
    in that case.
    """
    candidate = pathlib.Path.home() / ".cache" / "handpose" / "hand_landmarker.task"
    try:
        candidate.as_posix().encode("ascii")   # raises if non-ASCII
        return candidate
    except UnicodeEncodeError:
        # Home path contains non-ASCII chars — use a safe system-level path
        import sys
        if sys.platform == "win32":
            safe = pathlib.Path("C:/Workspace/Rogic/python-library/rogic-camera/RoboCam/models/hand_landmarker.task")
            # safe = pathlib.Path("C:/handpose_model/hand_landmarker.task")
        else:
            safe = pathlib.Path("/tmp/handpose/hand_landmarker.task")
        print(f"[WARN] Home path contains non-ASCII characters.")
        print(f"       Using safe model path: {safe}")
        return safe

MODEL_CACHE = _safe_model_cache()


def download_model():
    """Download the MediaPipe hand landmarker model if not already cached."""
    if MODEL_CACHE.exists():
        print(f"[OK] Model found: {MODEL_CACHE}")
        return str(MODEL_CACHE)
    print(f"[..] Downloading hand landmarker model (~8 MB)...")
    print(f"     {MODEL_URL}")
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE)
        size_kb = MODEL_CACHE.stat().st_size // 1024
        print(f"[OK] Model saved  {MODEL_CACHE}  ({size_kb} KB)")
        return str(MODEL_CACHE)
    except Exception as exc:
        print(f"[ERR] Download failed: {exc}")
        print(f"   Please download manually:\n   {MODEL_URL}")
        print(f"   and save it to: {MODEL_CACHE}")
        sys.exit(1)



#  TENSORFLOW 2.20.0 — joint-angle computation (called every frame)

@tf.function(input_signature=[
    tf.TensorSpec(shape=(2,), dtype=tf.float32),
    tf.TensorSpec(shape=(2,), dtype=tf.float32),
    tf.TensorSpec(shape=(2,), dtype=tf.float32),
])
def tf_joint_angle(tip, joint, base):
    """
    Angle (degrees) at `joint` formed by the vectors base->joint and tip->joint.
    Pure TF 2.20.0 computation.
    """
    v1 = base - joint
    v2 = tip  - joint
    cos_a = tf.reduce_sum(v1 * v2) / (tf.norm(v1) * tf.norm(v2) + 1e-8)
    return tf.math.acos(tf.clip_by_value(cos_a, -1.0, 1.0)) * (180.0 / np.pi)


def lm_to_tf(lm_list, idx, w, h):
    lm = lm_list[idx]
    return tf.constant([lm.x * w, lm.y * h], dtype=tf.float32)



#  GESTURE RECOGNITION  (rule-based on landmark geometry)

def _tip_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def _thumb_out(lm):
    return abs(lm[THUMB_TIP].x - lm[THUMB_IP].x) > 0.04

def recognize_gesture(lm):
    thumb  = _thumb_out(lm)
    index  = _tip_up(lm, INDEX_TIP,  INDEX_PIP)
    middle = _tip_up(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring   = _tip_up(lm, RING_TIP,   RING_PIP)
    pinky  = _tip_up(lm, PINKY_TIP,  PINKY_PIP)
    n = sum([index, middle, ring, pinky])

    if n == 4 and thumb:                                    return "Open Hand",     "[5]"
    if n == 0 and not thumb:                                return "Fist",          "[0]"
    if index and middle and not ring and not pinky:         return "Peace / V",     "[2]"
    if index and not middle and not ring and not pinky:     return "Pointing",      "[1]"
    if thumb and not index and not middle and not ring and not pinky:
                                                            return "Thumbs Up",     "[+]"
    if pinky and not index and not middle and not ring:     return "Pinky",         "[~]"
    if n == 3 and index and middle and ring:                return "Three Fingers",  "[3]"
    if n == 4:                                              return "Four Fingers",  "[4]"
    return "Unknown", "[?]"



#  DRAWING UTILITIES

def _bone_color(s, e):
    for name, ids in FINGER_IDS.items():
        if s in ids or e in ids:
            return FINGER_COLORS[name]
    return FINGER_COLORS["palm"]


def draw_skeleton(frame, lm_list, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], _bone_color(s, e), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        if i == WRIST:
            cv2.circle(frame, (x, y), 8, (255, 230, 0), -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)
        elif i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP):
            cv2.circle(frame, (x, y), 6, (0, 0, 220), -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)
        else:
            cv2.circle(frame, (x, y), 4, (200, 200, 200), -1)


def draw_bbox(frame, lm_list, h, w, label, color):
    xs = [lm.x * w for lm in lm_list]
    ys = [lm.y * h for lm in lm_list]
    pad = 20
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def draw_info_panel(frame, lm_list, h, w, hand_idx):
    px = 8 if hand_idx == 0 else frame.shape[1] - 218
    py = 165
    ov = frame.copy()
    cv2.rectangle(ov, (px - 4, py - 18), (px + 210, py + 21 * 14 + 4), (10,10,10), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    cv2.putText(frame, f"Hand {hand_idx+1} — Landmarks (x, y, z)",
                (px, py - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180,220,255), 1)
    for i, (lm, name) in enumerate(zip(lm_list, LANDMARK_NAMES)):
        row = f"{i:2d} {name:<12s} {lm.x:.2f} {lm.y:.2f} {lm.z:+.3f}"
        cv2.putText(frame, row, (px, py + 12 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (160, 255, 160), 1)


def draw_hud(frame, fps, n_hands):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.60, frame, 0.40, 0, frame)
    cv2.putText(frame, "HAND POSE DETECTION", (10, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"TensorFlow {tf.__version__}", (w - 215, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (130, 130, 130), 1)
    fps_col = (0, 255, 0) if fps >= 20 else (0, 160, 255)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Hands: {n_hands}",
                (w - 215, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_col, 1)
    cv2.putText(frame, "Q:Quit  I:Info  M:Mirror  S:Screenshot",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 120, 120), 1)



#  PROCESS ONE FRAME

def process_frame(frame, landmarker, mirror, show_info, hand_colors):
    if mirror:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
    )
    result = landmarker.detect(mp_img)

    lm_groups = result.hand_landmarks or []
    handed    = result.handedness     or []
    n_hands   = len(lm_groups)

    for idx, (lm_list, handedness) in enumerate(zip(lm_groups, handed)):
        color   = hand_colors[idx % 2]
        side    = handedness[0].display_name   # "Left" or "Right"
        conf    = handedness[0].score

        draw_skeleton(frame, lm_list, h, w)

        gesture, glyph = recognize_gesture(lm_list)

        # TF 2.20.0 — compute index-finger MCP joint angle
        angle = tf_joint_angle(
            lm_to_tf(lm_list, INDEX_TIP, w, h),
            lm_to_tf(lm_list, INDEX_MCP, w, h),
            lm_to_tf(lm_list, WRIST,     w, h),
        ).numpy()

        draw_bbox(frame, lm_list, h, w,
                  f"{side}  {gesture}  {angle:.0f}deg", color)

        by = 80 + idx * 62
        cv2.putText(frame, f"{glyph}  {gesture}", (10, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"{side}  conf:{conf:.2f}  idx-angle:{angle:.0f}deg",
                    (10, by + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 170, 170), 1)

        if show_info:
            draw_info_panel(frame, lm_list, h, w, idx)

    return frame, n_hands



#  WEBCAM MODE

def run_webcam(model_path, camera_idx, max_hands, det_conf, track_conf):
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"[ERR] Cannot open camera {camera_idx}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[OK] Camera {camera_idx} ready  ({fw}x{fh})")
    print("     Controls:  Q=Quit  I=Toggle info panel  M=Mirror  S=Screenshot")

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options                   = mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode                   = mp.tasks.vision.RunningMode.IMAGE,
        num_hands                      = max_hands,
        min_hand_detection_confidence  = det_conf,
        min_hand_presence_confidence   = det_conf,
        min_tracking_confidence        = track_conf,
    )
    landmarker  = mp.tasks.vision.HandLandmarker.create_from_options(opts)
    hand_colors = [(0, 210, 140), (255, 100, 0)]
    show_info   = False
    mirror      = True
    t_prev      = time.time()
    fps         = 0.0
    shot_n      = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Frame read failed — retrying…")
            continue

        frame, n = process_frame(frame, landmarker, mirror, show_info, hand_colors)

        t_now  = time.time()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        draw_hud(frame, fps, n)

        cv2.imshow("Hand Pose Detection  (TF 2.20.0 + MediaPipe)", frame)
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'):  break
        elif key == ord('i'):  show_info = not show_info
        elif key == ord('m'):  mirror    = not mirror
        elif key == ord('s'):
            fname = f"handpose_{shot_n:04d}.png"
            cv2.imwrite(fname, frame)
            print(f"[OK] Screenshot saved: {fname}")
            shot_n += 1

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[OK] Session ended.")



#  IMAGE MODE  (static photo)

def run_image(model_path, image_path, max_hands, det_conf):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERR] Cannot read: {image_path}")
        return

    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options                  = mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode                  = mp.tasks.vision.RunningMode.IMAGE,
        num_hands                     = max_hands,
        min_hand_detection_confidence = det_conf,
        min_hand_presence_confidence  = det_conf,
        min_tracking_confidence       = 0.5,
    )
    landmarker  = mp.tasks.vision.HandLandmarker.create_from_options(opts)
    hand_colors = [(0, 210, 140), (255, 100, 0)]

    out, n = process_frame(frame.copy(), landmarker, mirror=False,
                           show_info=True, hand_colors=hand_colors)
    draw_hud(out, fps=0.0, n_hands=n)
    print(f"[OK] Detected {n} hand(s) in {image_path}")

    result_path = "handpose_result.png"
    cv2.imwrite(result_path, out)
    print(f"[OK] Result saved: {result_path}")

    cv2.imshow(f"Hand Pose — {image_path}  (any key to close)", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    landmarker.close()



#  ENTRY POINT

def main():
    ap = argparse.ArgumentParser(
        description="Hand Pose Detection — TensorFlow 2.20.0 + MediaPipe"
    )
    ap.add_argument("--camera",      type=int,   default=0,
                    help="Webcam device index (default: 0)")
    ap.add_argument("--image",       type=str,   default=None,
                    help="Path to a static image (skips webcam)")
    ap.add_argument("--max-hands",   type=int,   default=2,
                    help="Max simultaneous hands to track (default: 2)")
    ap.add_argument("--detect-conf", type=float, default=0.6,
                    help="Min detection confidence (default: 0.6)")
    ap.add_argument("--track-conf",  type=float, default=0.5,
                    help="Min tracking confidence (default: 0.5)")
    ap.add_argument("--model",       type=str,   default=None,
                    help="Path to hand_landmarker.task (auto-downloaded if omitted)")
    args = ap.parse_args()

    print("=" * 62)
    print(f"  Hand Pose Detection")
    print(f"  TensorFlow {tf.__version__}  |  MediaPipe {mp.__version__}  |  OpenCV {cv2.__version__}")
    print("=" * 62)

    model_path = args.model or download_model()

    if args.image:
        run_image(model_path, args.image,
                  args.max_hands, args.detect_conf)
    else:
        run_webcam(model_path, args.camera,
                   args.max_hands, args.detect_conf, args.track_conf)


if __name__ == "__main__":
    main()
