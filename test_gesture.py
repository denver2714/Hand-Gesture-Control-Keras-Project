"""
test_gesture.py — Unit tests for gesture_utils and gesture_control logic.
Run: pytest test_gesture.py -v
"""

import math
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


# ─── Landmark Mock ────────────────────────────────────────────────────────────

class Lm:
    """Minimal stand-in for mediapipe NormalizedLandmark."""
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(overrides: dict = None) -> list:
    """
    Build a list of 21 default landmarks (all at origin).
    overrides: {index: Lm(...)} to set specific joints.
    """
    lms = [Lm() for _ in range(21)]
    if overrides:
        for idx, lm in overrides.items():
            lms[idx] = lm
    return lms


def make_fist_landmarks() -> list:
    """All fingers curled — tips below PIPs, thumb not extended."""
    lms = make_landmarks()
    # wrist at origin
    lms[0] = Lm(0.5, 0.9)
    # thumb: tip not far from wrist x
    lms[3] = Lm(0.55, 0.75)  # thumb IP
    lms[4] = Lm(0.56, 0.72)  # thumb tip — barely past IP, not extended
    # index: tip BELOW pip (curled)
    lms[6]  = Lm(0.5, 0.5)   # index PIP
    lms[8]  = Lm(0.5, 0.6)   # index tip (y > pip → curled)
    # middle
    lms[10] = Lm(0.5, 0.5)
    lms[12] = Lm(0.5, 0.6)
    # ring
    lms[14] = Lm(0.5, 0.5)
    lms[16] = Lm(0.5, 0.6)
    # pinky
    lms[18] = Lm(0.5, 0.5)
    lms[20] = Lm(0.5, 0.6)
    return lms


def make_open_hand_landmarks() -> list:
    """All fingers extended — tips above PIPs."""
    lms = make_landmarks()
    lms[0]  = Lm(0.5, 0.9)
    lms[3]  = Lm(0.3, 0.75)   # thumb IP far from wrist → thumb extended
    lms[4]  = Lm(0.25, 0.7)   # thumb tip even farther
    lms[6]  = Lm(0.5, 0.5);  lms[8]  = Lm(0.5, 0.2)   # index up
    lms[10] = Lm(0.5, 0.5);  lms[12] = Lm(0.5, 0.2)   # middle up
    lms[14] = Lm(0.5, 0.5);  lms[16] = Lm(0.5, 0.2)   # ring up
    lms[18] = Lm(0.5, 0.5);  lms[20] = Lm(0.5, 0.2)   # pinky up
    return lms


def make_point_landmarks() -> list:
    """Index finger up only."""
    lms = make_fist_landmarks()
    lms[6] = Lm(0.5, 0.5)
    lms[8] = Lm(0.5, 0.2)   # index tip above PIP → up
    return lms


def make_peace_landmarks() -> list:
    """Index + middle up, others curled."""
    lms = make_fist_landmarks()
    lms[6]  = Lm(0.5, 0.5);  lms[8]  = Lm(0.5, 0.2)   # index up
    lms[10] = Lm(0.5, 0.5);  lms[12] = Lm(0.5, 0.2)   # middle up
    return lms


def make_pinch_landmarks() -> list:
    """Thumb tip (4) and index tip (8) very close together."""
    lms = make_fist_landmarks()
    lms[4] = Lm(0.5, 0.5, 0.0)   # thumb tip
    lms[8] = Lm(0.5, 0.5, 0.0)   # index tip — same position → distance ≈ 0
    return lms


# ─── gesture_utils tests ─────────────────────────────────────────────────────

from gesture_utils import (
    calculate_distance,
    calculate_distance_xy,
    normalize_landmarks,
    map_to_screen,
    detect_gesture_simple,
    _is_finger_up,
    _is_thumb_up,
    GESTURE_LABELS,
)


class TestCalculateDistance:
    def test_same_point_returns_zero(self):
        p = Lm(0.5, 0.5, 0.5)
        assert calculate_distance(p, p) == 0.0

    def test_known_distance(self):
        p1 = Lm(0.0, 0.0, 0.0)
        p2 = Lm(1.0, 0.0, 0.0)
        assert calculate_distance(p1, p2) == pytest.approx(1.0)

    def test_3d_distance(self):
        p1 = Lm(0.0, 0.0, 0.0)
        p2 = Lm(1.0, 1.0, 1.0)
        assert calculate_distance(p1, p2) == pytest.approx(math.sqrt(3))

    def test_symmetry(self):
        p1 = Lm(0.1, 0.2, 0.3)
        p2 = Lm(0.4, 0.5, 0.6)
        assert calculate_distance(p1, p2) == pytest.approx(calculate_distance(p2, p1))

    def test_returns_non_negative(self):
        p1 = Lm(-1.0, -1.0, -1.0)
        p2 = Lm(1.0, 1.0, 1.0)
        assert calculate_distance(p1, p2) >= 0


class TestCalculateDistanceXY:
    def test_same_point_returns_zero(self):
        assert calculate_distance_xy((0.0, 0.0), (0.0, 0.0)) == 0.0

    def test_horizontal_distance(self):
        assert calculate_distance_xy((0.0, 0.0), (3.0, 0.0)) == pytest.approx(3.0)

    def test_pythagorean_triple(self):
        assert calculate_distance_xy((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


class TestNormalizeLandmarks:
    def test_output_shape_is_63(self):
        lms = make_landmarks()
        result = normalize_landmarks(lms)
        assert result.shape == (63,)

    def test_dtype_is_float32(self):
        lms = make_landmarks()
        result = normalize_landmarks(lms)
        assert result.dtype == np.float32

    def test_wrist_offset_is_zero(self):
        lms = make_landmarks(overrides={0: Lm(0.3, 0.7, 0.1)})
        result = normalize_landmarks(lms)
        # First 3 values are wrist - wrist = (0, 0, 0)
        np.testing.assert_array_almost_equal(result[:3], [0.0, 0.0, 0.0])

    def test_landmark_relative_to_wrist(self):
        wrist = Lm(0.1, 0.2, 0.3)
        tip   = Lm(0.4, 0.5, 0.6)
        lms   = make_landmarks(overrides={0: wrist, 1: tip})
        result = normalize_landmarks(lms)
        np.testing.assert_array_almost_equal(
            result[3:6],
            [tip.x - wrist.x, tip.y - wrist.y, tip.z - wrist.z],
        )

    def test_all_zeros_when_landmarks_at_origin(self):
        lms = make_landmarks()
        result = normalize_landmarks(lms)
        np.testing.assert_array_equal(result, np.zeros(63, dtype=np.float32))


class TestMapToScreen:
    def test_center_maps_to_center(self):
        sx, sy = map_to_screen(0.5, 0.5, 640, 480, 1920, 1080)
        assert sx == pytest.approx(960, abs=5)
        assert sy == pytest.approx(540, abs=5)

    def test_returns_tuple_of_ints(self):
        sx, sy = map_to_screen(0.5, 0.5, 640, 480, 1920, 1080)
        assert isinstance(sx, int)
        assert isinstance(sy, int)

    def test_clamps_below_margin(self):
        # x=0 should be clamped to left edge → screen x = 0
        sx, _ = map_to_screen(0.0, 0.5, 640, 480, 1920, 1080)
        assert sx == 0

    def test_clamps_above_margin(self):
        # x=1 should be clamped to right edge → screen x = screen_w
        sx, _ = map_to_screen(1.0, 0.5, 640, 480, 1920, 1080)
        assert sx == 1920

    def test_output_within_screen_bounds(self):
        for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for y in [0.0, 0.25, 0.5, 0.75, 1.0]:
                sx, sy = map_to_screen(x, y, 640, 480, 1920, 1080)
                assert 0 <= sx <= 1920
                assert 0 <= sy <= 1080


class TestIsFingerUp:
    def test_tip_above_pip_is_up(self):
        lms = make_landmarks(overrides={8: Lm(0.5, 0.2), 6: Lm(0.5, 0.5)})
        assert _is_finger_up(lms, 8, 6) is True

    def test_tip_below_pip_is_down(self):
        lms = make_landmarks(overrides={8: Lm(0.5, 0.7), 6: Lm(0.5, 0.5)})
        assert _is_finger_up(lms, 8, 6) is False


class TestIsThumbUp:
    def test_thumb_extended_sideways(self):
        # thumb tip far from wrist x compared to thumb IP
        lms = make_landmarks(overrides={
            0: Lm(0.5, 0.9),
            3: Lm(0.55, 0.75),   # IP close to wrist x
            4: Lm(0.75, 0.7),    # tip far from wrist x
        })
        assert _is_thumb_up(lms) is True

    def test_thumb_not_extended(self):
        lms = make_landmarks(overrides={
            0: Lm(0.5, 0.9),
            3: Lm(0.55, 0.75),   # IP: abs(0.55-0.5) = 0.05
            4: Lm(0.52, 0.72),   # tip: abs(0.52-0.5) = 0.02 < 0.05 → not extended
        })
        assert _is_thumb_up(lms) is False


class TestDetectGestureSimple:
    def test_fist_detected(self):
        assert detect_gesture_simple(make_fist_landmarks()) == 'fist'

    def test_open_hand_detected(self):
        assert detect_gesture_simple(make_open_hand_landmarks()) == 'open_hand'

    def test_point_detected(self):
        assert detect_gesture_simple(make_point_landmarks()) == 'point'

    def test_peace_detected(self):
        assert detect_gesture_simple(make_peace_landmarks()) == 'peace'

    def test_pinch_detected(self):
        assert detect_gesture_simple(make_pinch_landmarks()) == 'pinch'

    def test_pinch_takes_priority_over_other_gestures(self):
        # Even if fingers are extended, pinch distance < 0.05 wins
        lms = make_open_hand_landmarks()
        lms[4] = Lm(0.5, 0.5, 0.0)
        lms[8] = Lm(0.5, 0.5, 0.0)
        assert detect_gesture_simple(lms) == 'pinch'

    def test_returns_string(self):
        result = detect_gesture_simple(make_fist_landmarks())
        assert isinstance(result, str)

    def test_result_in_gesture_labels(self):
        for make_fn in [make_fist_landmarks, make_open_hand_landmarks,
                        make_point_landmarks, make_peace_landmarks,
                        make_pinch_landmarks]:
            result = detect_gesture_simple(make_fn())
            assert result in GESTURE_LABELS


class TestGestureLabels:
    def test_has_five_gestures(self):
        assert len(GESTURE_LABELS) == 5

    def test_contains_expected_labels(self):
        for label in ['open_hand', 'fist', 'point', 'pinch', 'peace']:
            assert label in GESTURE_LABELS

    def test_no_duplicates(self):
        assert len(GESTURE_LABELS) == len(set(GESTURE_LABELS))


# ─── gesture_control logic tests ─────────────────────────────────────────────
# Import only the pure functions — avoids triggering webcam/model loading.

import importlib, sys, types

def _import_control_functions():
    """
    Import gesture_to_mode and set_volume without running the module-level
    side effects (MediaPipe init, Keras load, cv2 imshow).
    """
    import gesture_control as gc
    return gc.gesture_to_mode, gc.set_volume, gc.get_volume


# Lazy import so pytest collection doesn't fail if mediapipe is absent.
try:
    gesture_to_mode, set_volume, get_volume = _import_control_functions()
    _control_available = True
except Exception:
    _control_available = False

skip_if_no_control = pytest.mark.skipif(
    not _control_available,
    reason="gesture_control could not be imported (missing deps or webcam)",
)


class TestGestureToMode:
    @skip_if_no_control
    def test_open_hand_returns_draw(self):
        assert gesture_to_mode('open_hand', 'CLICK') == 'DRAW'

    @skip_if_no_control
    def test_peace_returns_volume(self):
        assert gesture_to_mode('peace', 'DRAW') == 'VOLUME'

    @skip_if_no_control
    def test_fist_returns_click(self):
        assert gesture_to_mode('fist', 'DRAW') == 'CLICK'

    @skip_if_no_control
    def test_same_mode_returns_current(self):
        assert gesture_to_mode('open_hand', 'DRAW') == 'DRAW'

    @skip_if_no_control
    def test_unknown_gesture_returns_current_mode(self):
        assert gesture_to_mode('point', 'VOLUME') == 'VOLUME'

    @skip_if_no_control
    def test_none_gesture_returns_current_mode(self):
        assert gesture_to_mode('none', 'CLICK') == 'CLICK'


class TestSetVolume:
    @skip_if_no_control
    def test_clamps_above_100(self):
        with patch('subprocess.run'):
            assert set_volume(150) == 100

    @skip_if_no_control
    def test_clamps_below_zero(self):
        with patch('subprocess.run'):
            assert set_volume(-10) == 0

    @skip_if_no_control
    def test_normal_value_returned(self):
        with patch('subprocess.run'):
            assert set_volume(72) == 72

    @skip_if_no_control
    def test_zero_allowed(self):
        with patch('subprocess.run'):
            assert set_volume(0) == 0

    @skip_if_no_control
    def test_100_allowed(self):
        with patch('subprocess.run'):
            assert set_volume(100) == 100

    @skip_if_no_control
    def test_float_truncated(self):
        with patch('subprocess.run'):
            assert set_volume(49.9) == 49


class TestGetVolume:
    @skip_if_no_control
    def test_returns_int_on_success(self):
        mock_result = MagicMock()
        mock_result.stdout = "65\n"
        with patch('subprocess.run', return_value=mock_result):
            assert get_volume() == 65

    @skip_if_no_control
    def test_returns_50_on_failure(self):
        with patch('subprocess.run', side_effect=Exception("osascript not found")):
            assert get_volume() == 50
