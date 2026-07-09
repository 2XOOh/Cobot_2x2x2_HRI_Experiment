from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


# 손목에서 드릴 끝/TCP까지의 보정 길이는 사용자별 입력을 받지 않고 27 cm로 고정한다.
DRILL_TCP_OFFSET_CM = 21.0

# pose_generator_model_summary.txt의 4차 측정 기준 모델 파라미터다.
BASE_HEIGHT_M = 0.6612
SHOULDER_X_LINK0_M = 1.281
PASS_Y_LINK0_M = -0.103

# 로봇 link0 기준 TCP z 사용 가능 범위다.
MIN_LINK0_Z_M = 0.30
MAX_LINK0_Z_M = 1.00

# 4차 측정 quaternion 샘플이다. 각 항목은 (바닥 기준 높이 H, quaternion xyzw) 형식이다.
# position은 수식 모델로 만들고, orientation은 이 샘플 사이를 높이 기준으로 slerp 보간한다.
QUATERNION_SAMPLES_4TH: tuple[tuple[float, tuple[float, float, float, float]], ...] = (
    (1.625, (0.161, 0.748, 0.362, 0.533)),
    (1.554, (0.217, 0.755, 0.410, 0.463)),
    (1.484, (0.285, 0.741, 0.446, 0.413)),
    (1.408, (0.368, 0.662, 0.523, 0.391)),
    (1.332, (0.457, 0.493, 0.680, 0.293)),
    (1.232, (0.520, 0.526, 0.637, 0.218)),
    (1.119, (0.478, 0.415, 0.753, 0.182)),
    (1.038, (0.523, 0.347, 0.769, 0.123)),
    (0.967, (0.428, 0.308, 0.841, 0.121)),
    (0.860, (0.383, 0.351, 0.829, 0.209)),
)


@dataclass(frozen=True)
class HumanArmProfile:
    """수동 입력으로 받은 작업자 신체 치수를 저장한다."""

    user_height_cm: float
    shoulder_height_cm: float
    upper_arm_cm: float
    forearm_cm: float
    drill_tcp_offset_cm: float = DRILL_TCP_OFFSET_CM

    @property
    def shoulder_height_m(self) -> float:
        return self.shoulder_height_cm / 100.0

    @property
    def upper_arm_m(self) -> float:
        return self.upper_arm_cm / 100.0

    @property
    def forearm_m(self) -> float:
        return self.forearm_cm / 100.0

    @property
    def drill_tcp_offset_m(self) -> float:
        return self.drill_tcp_offset_cm / 100.0

    @property
    def total_arm_length_m(self) -> float:
        # 이 모델에서 전체 팔 길이 L은 상완 + 하완 + 손목~드릴/TCP 보정 길이다.
        return self.upper_arm_m + self.forearm_m + self.drill_tcp_offset_m


@dataclass(frozen=True)
class TcpPoseResult:
    """생성된 link0 기준 TCP pose 결과."""

    requested_floor_height_m: float
    target_floor_height_m: float
    frame_id: str
    x_m: float
    y_m: float
    z_m: float
    qx: float
    qy: float
    qz: float
    qw: float
    shoulder_height_m: float
    total_arm_length_m: float
    was_height_clamped: bool

    def to_pose_dict(self) -> dict[str, Any]:
        """로봇 수신부가 읽을 TCP pose dict를 만든다."""
        return {
            "frame_id": self.frame_id,
            "position": {
                "x": self.x_m,
                "y": self.y_m,
                "z": self.z_m,
            },
            "orientation": {
                "x": self.qx,
                "y": self.qy,
                "z": self.qz,
                "w": self.qw,
            },
        }

    def to_pass_goal_dict(self, msg: str = "") -> dict[str, Any]:
        """HTTP sender에 넘길 JSON-ready dict를 만든다."""
        return {
            "msg": msg,
            "target_type": "tcp_pose",
            "target_floor_z_m": self.target_floor_height_m,
            "target_floor_z_mm": self.target_floor_height_m * 1000.0,
            "target_pose": self.to_pose_dict(),
            "model_info": {
                "model": "human_aware_tcp_pose_generator",
                "requested_floor_z_m": self.requested_floor_height_m,
                "was_height_clamped": self.was_height_clamped,
                "shoulder_height_m": self.shoulder_height_m,
                "total_arm_length_m": self.total_arm_length_m,
                "drill_tcp_offset_cm": DRILL_TCP_OFFSET_CM,
            },
        }

    def to_json_string(self, msg: str = "") -> str:
        """저장이나 디버그 출력이 필요할 때 JSON 문자열로 변환한다."""
        return json.dumps(self.to_pass_goal_dict(msg=msg), ensure_ascii=False)


class HumanAwareTcpPoseGenerator:
    """
    사람 기준 목표 높이 H를 link0 기준 TCP pose로 바꾸는 생성기.

    모델은 두 단계다.
    1. position: x_R = S_x_R - sqrt(L^2 - (H - H_s)^2), y_R 고정, z_R = H - h_base
    2. orientation: 높이 H 기준 측정 quaternion slerp 보간
    """

    def __init__(
        self,
        base_height_m: float = BASE_HEIGHT_M,
        shoulder_x_link0_m: float = SHOULDER_X_LINK0_M,
        pass_y_link0_m: float = PASS_Y_LINK0_M,
        min_link0_z_m: float = MIN_LINK0_Z_M,
        max_link0_z_m: float = MAX_LINK0_Z_M,
        quaternion_samples: tuple[
            tuple[float, tuple[float, float, float, float]], ...
        ] = QUATERNION_SAMPLES_4TH,
    ) -> None:
        self.base_height_m = base_height_m
        self.shoulder_x_link0_m = shoulder_x_link0_m
        self.pass_y_link0_m = pass_y_link0_m
        self.min_link0_z_m = min_link0_z_m
        self.max_link0_z_m = max_link0_z_m
        self.quaternion_samples = tuple(sorted(quaternion_samples, key=lambda sample: sample[0]))
        if len(self.quaternion_samples) < 2:
            raise ValueError("quaternion 보간에는 최소 2개 이상의 측정 샘플이 필요합니다.")

    @property
    def min_floor_height_m(self) -> float:
        return self.base_height_m + self.min_link0_z_m

    @property
    def max_floor_height_m(self) -> float:
        return self.base_height_m + self.max_link0_z_m

    def floor_height_from_shoulder_angle(
        self,
        profile: HumanArmProfile,
        target_shoulder_angle_deg: float,
    ) -> float:
        """
        목표 어깨각을 바닥 기준 목표 높이 H로 변환한다.

        L은 항상 상완 + 하완 + 손목~드릴/TCP 보정 길이 전체를 사용한다.
        여기서 반환되는 H가 이후 TCP pose 생성의 입력 높이다.
        """
        # H = H_s - L * cos(theta)
        # H_s: 어깨 높이, L: 전체 팔 길이, theta: 목표 어깨각
        return profile.shoulder_height_m - (
            profile.total_arm_length_m * math.cos(math.radians(target_shoulder_angle_deg))
        )

    def generate_pose_from_shoulder_angle(
        self,
        target_shoulder_angle_deg: float,
        profile: HumanArmProfile,
        frame_id: str = "link0",
        clamp_to_robot_range: bool = True,
    ) -> TcpPoseResult:
        """목표 어깨각에서 H를 계산한 뒤 TCP pose까지 생성한다."""
        target_floor_height_m = self.floor_height_from_shoulder_angle(
            profile=profile,
            target_shoulder_angle_deg=target_shoulder_angle_deg,
        )
        return self.generate_pose_from_floor_height(
            target_floor_height_m=target_floor_height_m,
            profile=profile,
            frame_id=frame_id,
            clamp_to_robot_range=clamp_to_robot_range,
        )

    def generate_pose_from_floor_height(
        self,
        target_floor_height_m: float,
        profile: HumanArmProfile,
        frame_id: str = "link0",
        clamp_to_robot_range: bool = True,
    ) -> TcpPoseResult:
        """바닥 기준 목표 높이 H를 link0 기준 TCP pose로 변환한다."""
        requested_height_m = float(target_floor_height_m)
        height_m = (
            self._clamp_floor_height(requested_height_m)
            if clamp_to_robot_range
            else requested_height_m
        )
        was_height_clamped = abs(height_m - requested_height_m) > 1e-9

        total_arm_length_m = profile.total_arm_length_m
        shoulder_height_m = profile.shoulder_height_m
        # Position model:
        # x_R = S_x_R - sqrt(L^2 - (H - H_s)^2)
        # y_R = y_fixed_R
        # z_R = H - h_base
        vertical_delta_m = height_m - shoulder_height_m
        radicand = total_arm_length_m**2 - vertical_delta_m**2
        if radicand < -1e-9:
            raise ValueError(
                "목표 높이가 전체 팔 길이 반경 밖에 있습니다. "
                f"H={height_m:.3f}m, H_s={shoulder_height_m:.3f}m, L={total_arm_length_m:.3f}m"
            )

        x_m = self.shoulder_x_link0_m - math.sqrt(max(0.0, radicand))
        y_m = self.pass_y_link0_m
        z_m = height_m - self.base_height_m
        qx, qy, qz, qw = self.interpolate_quaternion(height_m)

        return TcpPoseResult(
            requested_floor_height_m=requested_height_m,
            target_floor_height_m=height_m,
            frame_id=frame_id,
            x_m=x_m,
            y_m=y_m,
            z_m=z_m,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
            shoulder_height_m=shoulder_height_m,
            total_arm_length_m=total_arm_length_m,
            was_height_clamped=was_height_clamped,
        )

    def build_pass_goal_dict_from_shoulder_angle(
        self,
        target_shoulder_angle_deg: float,
        profile: HumanArmProfile,
        msg: str = "",
        frame_id: str = "link0",
    ) -> dict[str, Any]:
        """
        목표 어깨각에서 바로 HTTP 전송용 dict를 만든다.

        메인 통합 파일에서는 기본적으로 이 함수를 쓰면 된다.
        LLM/rule이 어깨 목표각을 정하면, 이 함수가 H 계산과 TCP pose 생성을 이어서 처리한다.
        """
        pose = self.generate_pose_from_shoulder_angle(
            target_shoulder_angle_deg=target_shoulder_angle_deg,
            profile=profile,
            frame_id=frame_id,
        )
        return pose.to_pass_goal_dict(msg=msg)

    def build_pass_goal_dict_from_floor_height(
        self,
        target_floor_height_m: float,
        profile: HumanArmProfile,
        msg: str = "",
        frame_id: str = "link0",
    ) -> dict[str, Any]:
        """
        바닥 기준 목표 높이 H에서 HTTP 전송용 dict를 만든다.

        예비용 함수다. 초기 위치처럼 목표 높이 H가 이미 정해져 있을 때만 사용한다.
        """
        pose = self.generate_pose_from_floor_height(
            target_floor_height_m=target_floor_height_m,
            profile=profile,
            frame_id=frame_id,
        )
        return pose.to_pass_goal_dict(msg=msg)

    def build_pass_goal_dict_from_floor_height_mm(
        self,
        target_floor_height_mm: float,
        profile: HumanArmProfile,
        msg: str = "",
        frame_id: str = "link0",
    ) -> dict[str, Any]:
        """
        현재 main의 mm 단위 높이와 연결하기 위한 예비용 편의 함수.

        최종 통합에서 main의 높이 단위를 m로 정리하면 없어져도 된다.
        """
        return self.build_pass_goal_dict_from_floor_height(
            target_floor_height_m=target_floor_height_mm / 1000.0,
            profile=profile,
            msg=msg,
            frame_id=frame_id,
        )

    def interpolate_quaternion(self, target_floor_height_m: float) -> tuple[float, float, float, float]:
        """목표 높이 H에 가장 가까운 두 측정 quaternion을 slerp로 보간한다."""
        height_m = float(target_floor_height_m)
        samples = self.quaternion_samples

        if height_m <= samples[0][0]:
            return _normalize_quaternion(samples[0][1])
        if height_m >= samples[-1][0]:
            return _normalize_quaternion(samples[-1][1])

        lower = samples[0]
        upper = samples[-1]
        for left, right in zip(samples, samples[1:]):
            if left[0] <= height_m <= right[0]:
                lower = left
                upper = right
                break

        interval = upper[0] - lower[0]
        # q_R(H) = slerp(q_i, q_j, alpha)
        # alpha = (H - H_i) / (H_j - H_i)
        alpha = 0.0 if interval == 0 else (height_m - lower[0]) / interval
        return _slerp(lower[1], upper[1], alpha)

    def _clamp_floor_height(self, target_floor_height_m: float) -> float:
        return max(self.min_floor_height_m, min(self.max_floor_height_m, target_floor_height_m))


def prompt_human_arm_profile() -> HumanArmProfile:
    """
    기존 main_integrated.py의 신체 치수 입력 로직을 옮겨오기 위한 함수.

    손목~드릴/TCP 보정 길이는 사용자에게 묻지 않고 27 cm로 고정한다.
    """
    print("\n" + "=" * 60)
    print(" 실험자 신체 정보 입력 (Enter를 누르면 기본값 적용)")
    print("=" * 60)

    user_height_cm = _prompt_float(" 1. 작업자 키(cm) [기본: 175.0]: ", 175.0)
    default_shoulder_height_cm = user_height_cm - 30.0
    shoulder_height_cm = _prompt_float(
        f" 2. 어깨 높이(cm) [기본: {default_shoulder_height_cm:.1f}]: ",
        default_shoulder_height_cm,
    )
    upper_arm_cm = _prompt_float(" 3. 상완 길이(어깨~팔꿈치, cm) [기본: 30.0]: ", 30.0)
    forearm_cm = _prompt_float(" 4. 하완 길이(팔꿈치~손목, cm) [기본: 25.0]: ", 25.0)

    profile = HumanArmProfile(
        user_height_cm=user_height_cm,
        shoulder_height_cm=shoulder_height_cm,
        upper_arm_cm=upper_arm_cm,
        forearm_cm=forearm_cm,
    )
    print(
        "\n [적용 완료] "
        f"키: {profile.user_height_cm:.1f}cm | "
        f"어깨 높이: {profile.shoulder_height_cm:.1f}cm | "
        f"상완: {profile.upper_arm_cm:.1f}cm | "
        f"하완: {profile.forearm_cm:.1f}cm | "
        f"손목~드릴/TCP: {profile.drill_tcp_offset_cm:.1f}cm 고정"
    )
    return profile


def make_human_arm_profile(
    shoulder_height_cm: float,
    upper_arm_cm: float,
    forearm_cm: float,
    user_height_cm: float = 175.0,
) -> HumanArmProfile:
    """main에서 이미 입력받은 값으로 HumanArmProfile을 만들 때 쓴다."""
    return HumanArmProfile(
        user_height_cm=user_height_cm,
        shoulder_height_cm=shoulder_height_cm,
        upper_arm_cm=upper_arm_cm,
        forearm_cm=forearm_cm,
    )


def _prompt_float(prompt: str, default_value: float) -> float:
    try:
        raw_value = input(prompt)
        return float(raw_value) if raw_value.strip() else default_value
    except Exception:
        return default_value


def _normalize_quaternion(
    quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion_xyzw))
    if norm == 0:
        raise ValueError("0 길이 quaternion은 사용할 수 없습니다.")
    return tuple(value / norm for value in quaternion_xyzw)


def _slerp(
    q1_xyzw: tuple[float, float, float, float],
    q2_xyzw: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    """두 quaternion 사이를 구면 선형 보간한다."""
    q1 = _normalize_quaternion(q1_xyzw)
    q2 = _normalize_quaternion(q2_xyzw)
    alpha = max(0.0, min(1.0, alpha))

    dot = sum(a * b for a, b in zip(q1, q2))
    if dot < 0.0:
        # 같은 회전을 더 짧은 경로로 보간하기 위한 처리다.
        q2 = tuple(-value for value in q2)
        dot = -dot

    if dot > 0.9995:
        # 두 quaternion이 거의 같으면 slerp 대신 일반 선형 보간을 쓴다.
        blended = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(q1, q2))
        return _normalize_quaternion(blended)

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)

    scale_1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    scale_2 = sin_theta / sin_theta_0
    return tuple(scale_1 * a + scale_2 * b for a, b in zip(q1, q2))
