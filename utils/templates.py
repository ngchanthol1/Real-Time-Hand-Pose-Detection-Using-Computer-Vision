
import numpy as np


class Gesture:
    def __init__(self, label):
        self.label    = label
        self.gestures = self._define_gestures()

    def _define_gestures(self):
        gestures = {}

        # ── FIST ──────────────────────────────────────────────────────
        gestures['fist'] = {
            'finger states': [[0, 1, 2], [3, 4], [3, 4], [3, 4], [3, 4]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── PALM ──────────────────────────────────────────────────────
        gestures['palm'] = {
            'finger states': [[0, 1], [0], [0], [0], [0]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── THUMBS UP ─────────────────────────────────────────────────
        gestures['thumbs_up'] = {
            'finger states': [[0], [3, 4], [3, 4], [3, 4], [3, 4]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── THUMBS DOWN ───────────────────────────────────────────────
        gestures['thumbs_down'] = {
            'finger states': [[0], [3, 4], [3, 4], [3, 4], [3, 4]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'down',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── PEACE / VICTORY ───────────────────────────────────────────
        gestures['peace'] = {
            'finger states': [[1, 2], [0], [0], [3, 4], [3, 4]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── OK ────────────────────────────────────────────────────────
        gestures['ok'] = {
            'finger states': [[0, 1, 2], [2, 3], [0], [0], [0]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : [[4, 8]],
            'boundary'     : None,
        }

        # ── POINT ─────────────────────────────────────────────────────
        gestures['point'] = {
            'finger states': [[1, 2], [0], [3, 4], [3, 4], [3, 4]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── CALL ME ───────────────────────────────────────────────────
        gestures['call'] = {
            'finger states': [[0], [3, 4], [3, 4], [3, 4], [0]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── ROCK ON ───────────────────────────────────────────────────
        gestures['rock'] = {
            'finger states': [[0, 1, 2], [0], [3, 4], [3, 4], [0]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── GUN ───────────────────────────────────────────────────────
        gestures['gun'] = {
            'finger states': [[0], [0], [3, 4], [3, 4], [3, 4]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── THREE ─────────────────────────────────────────────────────
        gestures['three'] = {
            'finger states': [[1, 2], [0], [0], [0], [3, 4]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        # ── FOUR ──────────────────────────────────────────────────────
        gestures['four'] = {
            'finger states': [[1, 2], [0], [0], [0], [0]],
            'wrist angle'  : (0.0, np.pi),
            'direction'    : 'up',
            'overlap'      : None,
            'boundary'     : None,
        }

        return gestures
