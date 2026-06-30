"""
Detect hands on streams.

Fixes applied
─────────────
1. mp.solutions.* replaced with direct mediapipe.python.solutions imports
   → works with mediapipe ≥ 0.10.21 + protobuf ≥ 5.28 (tensorflow 2.20 compat)
2. draw_landmarks() now draws ALL 21 joint circles + ALL connection lines
   using an explicit, hand-crafted connection list so nothing is ever skipped.

Usage:
    $ python hand.py --max_hands 2
"""

import argparse
import cv2
import time
import numpy as np

# ── mediapipe: explicit submodule imports (no mp.solutions.* needed) ──────────
import mediapipe as mp
try:
    from mediapipe.python.solutions import hands          as _mp_hands
    from mediapipe.python.solutions import drawing_utils  as _mp_drawing
    from mediapipe.python.solutions import drawing_styles as _mp_draw_sty
except ImportError:
    # older mediapipe fallback
    _mp_hands     = mp.solutions.hands
    _mp_drawing   = mp.solutions.drawing_utils
    _mp_draw_sty  = mp.solutions.drawing_styles

from utils.utils import check_hand_direction, find_boundary_lm
from utils.utils import calculate_angle, display_hand_info


CAM_W      = 1280
CAM_H      = 720
TEXT_COLOR = (243, 236, 27)
LM_COLOR   = (102, 255, 255)   # cyan  – joint dots
LINE_COLOR = ( 51,  51,  51)   # dark  – bones

# ── Full 21-joint connection list (MediaPipe hand topology) ───────────────────
#   Thumb  : 0-1-2-3-4
#   Index  : 0-5-6-7-8
#   Middle : 0-9-10-11-12
#   Ring   : 0-13-14-15-16
#   Pinky  : 0-17-18-19-20
#   Palm   : 5-9, 9-13, 13-17, 5-17 (closing the palm ring)
HAND_CONNECTIONS = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # palm cross-bar
    (5, 9), (9, 13), (13, 17), (5, 17),
]

# Fingertip indices (drawn larger)
FINGERTIPS = {4, 8, 12, 16, 20}
# MCP knuckle indices
MCPS      = {1, 5, 9, 13, 17}


def draw_hand_landmarks(img, lm_array, thickness_scale=1):
    """
    Draw all 21 landmarks and all bone connections on *img* (BGR, in-place).

    lm_array : np.ndarray shape (21, 3)  — pixel coords [cx, cy, cz]
    """
    if lm_array is None or len(lm_array) < 21:
        return

    w = img.shape[1]
    t = max(1, int(w / 500) * thickness_scale)

    # 1. Draw connection lines first (so dots render on top)
    for a, b in HAND_CONNECTIONS:
        pt_a = (int(lm_array[a][0]), int(lm_array[a][1]))
        pt_b = (int(lm_array[b][0]), int(lm_array[b][1]))
        cv2.line(img, pt_a, pt_b, LINE_COLOR, t + 1, cv2.LINE_AA)
        cv2.line(img, pt_a, pt_b, (180, 180, 180), t,   cv2.LINE_AA)

    # 2. Draw every joint as a filled circle with an outline ring
    for idx, lm in enumerate(lm_array):
        cx, cy = int(lm[0]), int(lm[1])

        if idx in FINGERTIPS:
            r_outer, r_inner = 4 * t + 2, 3 * t
            dot_col = (0, 255, 180)     # teal – fingertips
        elif idx in MCPS:
            r_outer, r_inner = 3 * t + 1, 2 * t
            dot_col = (80, 180, 255)    # orange-ish – knuckles
        elif idx == 0:
            r_outer, r_inner = 4 * t + 2, 3 * t
            dot_col = (200, 80, 255)    # purple – wrist
        else:
            r_outer, r_inner = 3 * t,   2 * t - 1
            dot_col = LM_COLOR          # cyan – mid joints

        cv2.circle(img, (cx, cy), r_outer, (0, 0, 0),   -1, cv2.LINE_AA)   # black ring
        cv2.circle(img, (cx, cy), r_inner, dot_col,      -1, cv2.LINE_AA)   # colour fill


# ─────────────────────────────────────────────────────────────────────────────
class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=2,
                 min_detection_confidence=0.8, min_tracking_confidence=0.5):

        self.static_image_mode        = static_image_mode
        self.max_num_hands            = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence

        # ── patched: use submodule references directly ────────────────
        self.mp_hands    = _mp_hands
        self.mp_drawing  = _mp_drawing
        self.mp_draw_sty = _mp_draw_sty

        self.hands = self.mp_hands.Hands(
            static_image_mode        = self.static_image_mode,
            max_num_hands            = self.max_num_hands,
            min_detection_confidence = self.min_detection_confidence,
            min_tracking_confidence  = self.min_tracking_confidence,
        )

        self.decoded_hands = None
        self.results       = None

    # ── detect ────────────────────────────────────────────────────────
    def detect_hands(self, img):
        self.decoded_hands = None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            h, w, _ = img.shape
            num_hands = len(self.results.multi_hand_landmarks)
            self.decoded_hands = [None] * num_hands

            for i in range(num_hands):
                self.decoded_hands[i] = {}
                lm_list    = []
                handedness = self.results.multi_handedness[i]
                hand_lms   = self.results.multi_hand_landmarks[i]
                wrist_z    = hand_lms.landmark[0].z

                for lm in hand_lms.landmark:
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cz = int((lm.z - wrist_z) * w)
                    lm_list.append([cx, cy, cz])

                label       = handedness.classification[0].label.lower()
                lm_array    = np.array(lm_list, dtype=int)
                direction, facing = check_hand_direction(lm_array, label)
                boundary    = find_boundary_lm(lm_array)
                wrist_angle = calculate_angle(lm_array[[5, 0, 17]])

                self.decoded_hands[i]['label']       = label
                self.decoded_hands[i]['landmarks']   = lm_array
                self.decoded_hands[i]['wrist_angle'] = wrist_angle
                self.decoded_hands[i]['direction']   = direction
                self.decoded_hands[i]['facing']      = facing
                self.decoded_hands[i]['boundary']    = boundary

        return self.decoded_hands

    # ── draw ──────────────────────────────────────────────────────────
    def draw_landmarks(self, img):
        """Draw ALL 21 joints + ALL bones for every detected hand."""
        if not self.decoded_hands:
            return
        for hand in self.decoded_hands:
            if hand is not None:
                draw_hand_landmarks(img, hand['landmarks'])


# ─────────────────────────────────────────────────────────────────────────────
def main(max_hands=2):
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    detector = HandDetector(max_num_hands=max_hands)
    ptime = 0

    while True:
        ret, img = cap.read()
        if not ret or img is None:
            continue
        img = cv2.flip(img, 1)
        detector.detect_hands(img)
        detector.draw_landmarks(img)

        if detector.decoded_hands:
            for hand in detector.decoded_hands:
                if hand:
                    display_hand_info(img, hand)

        ctime = time.time()
        fps   = 1.0 / max(ctime - ptime, 1e-6)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (50, 50), 0, 0.8,
                    TEXT_COLOR, 2, lineType=cv2.LINE_AA)

        cv2.imshow('Hand detection', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_hands', type=int, default=2,
                        help='max number of hands (default: 2)')
    opt = parser.parse_args()
    main(**vars(opt))
