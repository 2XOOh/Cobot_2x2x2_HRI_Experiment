from __future__ import annotations

import math

import cv2

try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import pose as mp_pose
except ImportError:
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.pose as mp_pose

from experiment_data import PostureSample


MEASURED_SIDE = "right"
VISIBILITY_THRESHOLD = 0.6

SELECTED_LANDMARKS = {
    "shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    "elbow": mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    "wrist": mp_pose.PoseLandmark.RIGHT_WRIST.value,
    "hip": mp_pose.PoseLandmark.RIGHT_HIP.value,
}


class PostureEstimator:
    """MediaPipe 프레임에서 어깨/팔꿈치 각도와 간이 RULA 값을 계산한다."""

    def __init__(self) -> None:
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, frame) -> PostureSample:
        """BGR 카메라 프레임을 분석하고, 화면 표시용 skeleton은 frame에 바로 그린다."""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(image_rgb)

        if not result.pose_landmarks:
            return empty_posture_sample()

        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = result.pose_landmarks.landmark

        if not landmarks_visible(landmarks):
            return empty_posture_sample()

        height, width, _ = frame.shape
        shoulder_pt = landmark_to_pixel(landmarks[SELECTED_LANDMARKS["shoulder"]], width, height)
        elbow_pt = landmark_to_pixel(landmarks[SELECTED_LANDMARKS["elbow"]], width, height)
        wrist_pt = landmark_to_pixel(landmarks[SELECTED_LANDMARKS["wrist"]], width, height)
        hip_pt = landmark_to_pixel(landmarks[SELECTED_LANDMARKS["hip"]], width, height)

        shoulder_angle_deg = calculate_angle(hip_pt, shoulder_pt, elbow_pt)
        elbow_angle_deg = calculate_angle(shoulder_pt, elbow_pt, wrist_pt)
        rula_proxy = estimate_rula_score(shoulder_angle_deg, elbow_angle_deg)

        return PostureSample(
            shoulder_angle_deg=shoulder_angle_deg,
            elbow_angle_deg=elbow_angle_deg,
            rula_proxy=rula_proxy,
            visibility_ok=True,
            side=MEASURED_SIDE,
        )

    def close(self) -> None:
        self.pose.close()


def empty_posture_sample() -> PostureSample:
    return PostureSample(
        shoulder_angle_deg=None,
        elbow_angle_deg=None,
        rula_proxy=None,
        visibility_ok=False,
        side=MEASURED_SIDE,
    )


def landmarks_visible(landmarks) -> bool:
    """각도 계산에 필요한 관절 4개가 충분히 보이는지 확인한다."""
    return (
        landmarks[SELECTED_LANDMARKS["shoulder"]].visibility > VISIBILITY_THRESHOLD
        and landmarks[SELECTED_LANDMARKS["elbow"]].visibility > VISIBILITY_THRESHOLD
        and landmarks[SELECTED_LANDMARKS["wrist"]].visibility > VISIBILITY_THRESHOLD
        and landmarks[SELECTED_LANDMARKS["hip"]].visibility > VISIBILITY_THRESHOLD
    )


def landmark_to_pixel(landmark, width: int, height: int) -> list[int]:
    return [int(landmark.x * width), int(landmark.y * height)]


def calculate_angle(a: list[int], b: list[int], c: list[int]) -> float:
    """세 점 a-b-c가 이루는 2D 각도를 degree 단위로 계산한다."""
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if mag_ba == 0 or mag_bc == 0:
        return 0.0

    # cos(theta) = (BA dot BC) / (|BA| * |BC|)
    cosine_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine_angle))


def estimate_rula_score(shoulder_angle_deg: float, elbow_angle_deg: float) -> int:
    """어깨/팔꿈치 각도 기반의 간이 RULA 점수를 계산한다."""
    score = 1

    if shoulder_angle_deg > 45:
        score += 2
    elif shoulder_angle_deg > 20:
        score += 1

    if elbow_angle_deg < 60 or elbow_angle_deg > 100:
        score += 1

    return min(score, 7)
