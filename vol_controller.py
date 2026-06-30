
import argparse
import cv2
import numpy as np
import time
import sys
import platform

from gesture import GestureDetector
from utils.utils import two_landmark_distance, draw_vol_bar, draw_landmarks
from utils.utils import update_trajectory, check_trajectory


CAM_W      = 1280
CAM_H      = 720
TEXT_COLOR = (102,  51,   0)
ACTI_COLOR = (  0, 255,   0)
VOL_RANGE  = [0, 100]
BAR_X_RANGE = [50, CAM_W // 5]

# ── Cross-platform volume setter ──────────────────────────────────────────────
_OS = platform.system()

if _OS == 'Windows':
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        import math as _math

        _devices    = AudioUtilities.GetSpeakers()
        _interface  = _devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _volume_com = cast(_interface, POINTER(IAudioEndpointVolume))
        _vol_range  = _volume_com.GetVolumeRange()   # (min_dB, max_dB, step_dB)

        def set_volume(percent):
            """Set system volume 0-100 via pycaw (Windows)."""
            percent = max(0.0, min(100.0, float(percent)))
            min_dB, max_dB, _ = _vol_range
            # linear percent → dB
            if percent == 0:
                db = min_dB
            else:
                db = min_dB + (max_dB - min_dB) * (percent / 100.0)
            _volume_com.SetMasterVolumeLevel(db, None)

        print("[INFO] Volume backend: pycaw (Windows)")

    except Exception as _e:
        print(f"[WARN] pycaw not available ({_e}).  "
              f"Run:  pip install pycaw comtypes\n"
              f"      Volume control disabled – camera+gesture will still work.")

        def set_volume(percent):
            pass   # no-op if pycaw missing

elif _OS == 'Darwin':
    import subprocess

    def set_volume(percent):
        """Set system volume 0-100 via osascript (macOS)."""
        percent = int(max(0, min(100, percent)))
        subprocess.run(
            ['osascript', '-e', f'set volume output volume {percent}'],
            capture_output=True)

    print("[INFO] Volume backend: osascript (macOS)")

else:
    import subprocess

    def set_volume(percent):
        """Set system volume 0-100 via amixer (Linux)."""
        percent = int(max(0, min(100, percent)))
        subprocess.run(
            ['amixer', '-q', 'sset', 'Master', f'{percent}%'],
            capture_output=True)

    print("[INFO] Volume backend: amixer (Linux)")


# ─────────────────────────────────────────────────────────────────────────────
def vol_control(control='continuous', vol_step=10, traj_size=10):
    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)

    ges_detector = GestureDetector(max_num_hands=1)

    vol     = (VOL_RANGE[0] + VOL_RANGE[1]) // 2
    vol_bar = (BAR_X_RANGE[0] + BAR_X_RANGE[1]) // 2
    set_volume(vol)

    ptime      = 0
    trajectory = []
    target_gestures = ['Pinch', 'C shape']
    wrist, thumb_tip, index_tip = 0, 4, 8
    activated  = False
    len_range  = None
    step_threshold = None

    while True:
        ret, img = cap.read()
        if not ret or img is None:
            continue
        img = cv2.flip(img, 1)

        gesture = ges_detector.detect_gesture(img, 'single')
        hands   = ges_detector.hand_detector.decoded_hands

        if gesture:
            hand      = hands[-1]
            landmarks = hand['landmarks']
            if gesture in target_gestures:
                ges_detector.draw_gesture_box(img)
            if gesture == target_gestures[0]:
                if not activated:
                    base_len       = two_landmark_distance(
                                        landmarks[wrist], landmarks[thumb_tip])
                    len_range      = [0.1 * base_len, 0.6 * base_len]
                    step_threshold = [0.2 * base_len, 0.9 * base_len]
                activated = True
            if activated and gesture == target_gestures[1]:
                activated = False

        if activated:
            if hands:
                hand      = hands[-1]
                landmarks = hand['landmarks']
                pt1       = landmarks[thumb_tip][:2]
                pt2       = landmarks[index_tip][:2]
                length    = two_landmark_distance(pt1, pt2)

                # continuous mode
                if control == 'continuous':
                    draw_landmarks(img, pt1, pt2)
                    finger_states = ges_detector.check_finger_states(hand)
                    if finger_states[4] > 2:
                        vol     = np.interp(length, len_range, VOL_RANGE)
                        vol_bar = np.interp(length, len_range, BAR_X_RANGE)
                        set_volume(vol)

                # step mode
                if control == 'step':
                    draw_landmarks(img, pt1, pt2)
                    trajectory = update_trajectory(length, trajectory, traj_size)
                    up = down = False
                    if (len(trajectory) == traj_size
                            and step_threshold
                            and length > step_threshold[1]):
                        up = check_trajectory(trajectory, direction=1)
                        if up:
                            vol = min(vol + vol_step, VOL_RANGE[1])
                            set_volume(vol)
                    if (len(trajectory) == traj_size
                            and step_threshold
                            and length < step_threshold[0]):
                        down = check_trajectory(trajectory, direction=-1)
                        if down:
                            vol = max(vol - vol_step, VOL_RANGE[0])
                            set_volume(vol)
                    if up or down:
                        vol_bar    = np.interp(vol, VOL_RANGE, BAR_X_RANGE)
                        trajectory = []

        ctime = time.time()
        fps   = 1.0 / max(ctime - ptime, 1e-6)
        ptime = ctime

        pt1 = (30, 20)
        pt2 = (BAR_X_RANGE[1] + 80, 150)
        draw_vol_bar(img, pt1, pt2, vol_bar, vol, fps, BAR_X_RANGE, activated)

        cv2.imshow('Volume controller', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control',   type=str, default='continuous',
                        help='volume control mode (default: continuous)')
    parser.add_argument('--vol_step',  type=int, default=10,
                        help='volume step for step control (default: 10)')
    parser.add_argument('--traj_size', type=int, default=10,
                        help='trajectory size (default: 10)')
    opt = parser.parse_args()
    try:
        vol_control(**vars(opt))
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C.")
    except Exception as e:
        import traceback
        print(f"\n[CRASH] {e}")
        traceback.print_exc()
        input("\nPress ENTER to close …")
