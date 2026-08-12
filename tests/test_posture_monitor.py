import pytest
from reachy_mini.utils import create_head_pose

from reachy_mini_openclaw.moves import AntennaFlutterMove
from reachy_mini_openclaw.vision.posture_monitor import Landmark, assess_forward_head


def _landmarks(ear_x: float):
    return {
        "left_ear": Landmark(ear_x, 0.25),
        "right_ear": Landmark(ear_x, 0.25),
        "left_shoulder": Landmark(0.5, 0.5),
        "right_shoulder": Landmark(0.5, 0.5),
        "left_hip": Landmark(0.5, 0.9),
        "right_hip": Landmark(0.5, 0.9),
    }


def test_forward_head_assessment_detects_large_ear_offset():
    assessment = assess_forward_head(_landmarks(0.62), threshold=0.20)
    assert assessment is not None
    assert assessment.turtle_neck_suspected is True
    assert assessment.forward_head_ratio == pytest.approx(0.30)


def test_forward_head_assessment_ignores_neutral_alignment():
    assessment = assess_forward_head(_landmarks(0.53), threshold=0.20)
    assert assessment is not None
    assert assessment.turtle_neck_suspected is False


def test_forward_head_assessment_needs_visible_torso():
    points = _landmarks(0.62)
    points["left_hip"] = Landmark(0.5, 0.9, visibility=0.2)
    assert assess_forward_head(points, threshold=0.20) is None


def test_posture_alert_move_flutters_antennas_without_moving_head():
    start_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
    move = AntennaFlutterMove(start_pose, (0.0, 0.0))
    head, antennas, body_yaw = move.evaluate(0.35)
    assert head.shape == start_pose.shape
    assert antennas[0] == pytest.approx(-antennas[1])
    assert antennas[0] != 0.0
    assert body_yaw == 0.0
