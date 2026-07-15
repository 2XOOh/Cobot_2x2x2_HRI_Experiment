from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


DRILL_TCP_OFFSET_M = 0.21
DRILL_TCP_OFFSET_CM = DRILL_TCP_OFFSET_M * 100.0

BASE_HEIGHT_M = 0.6612

MeasuredPoseSample = tuple[
    float,
    tuple[float, float, float],
    tuple[float, float, float, float],
]

# 5th measurement IK-feasible TCP pose lookup.
# Each sample is (floor height H, link0 position xyz, orientation quaternion xyzw).
MEASURED_POSES_5TH: tuple[MeasuredPoseSample, ...] = (
    (0.7122, (0.766, -0.103, 0.051), (0.490, 0.327, 0.793, 0.152)),
    (0.7942, (0.777, -0.090, 0.133), (0.455, 0.252, 0.839, 0.161)),
    (0.8882, (0.707, -0.100, 0.227), (0.502, 0.357, 0.770, 0.164)),
    (0.9552, (0.613, -0.090, 0.294), (0.562, 0.346, 0.731, 0.171)),
    (1.1062, (0.618, -0.159, 0.445), (0.351, 0.684, 0.518, 0.376)),
    (1.1722, (0.564, -0.110, 0.511), (0.294, 0.731, 0.403, 0.465)),
    (1.2692, (0.518, -0.103, 0.608), (0.184, 0.752, 0.342, 0.533)),
    (1.4362, (0.458, -0.139, 0.775), (0.253, 0.766, 0.364, 0.466)),
    (1.5242, (0.511, -0.148, 0.863), (0.273, 0.789, 0.373, 0.406)),
    (1.6382, (0.446, -0.144, 0.977), (0.190, 0.802, 0.295, 0.483)),
    (1.6802, (0.337, -0.094, 1.019), (0.126, 0.804, 0.305, 0.495)),
)

MIN_LINK0_Z_M = MEASURED_POSES_5TH[0][1][2]
MAX_LINK0_Z_M = MEASURED_POSES_5TH[-1][1][2]


@dataclass(frozen=True)
class HumanArmProfile:
    """Worker body dimensions used for shoulder-angle to floor-height conversion."""

    user_height_cm: float
    shoulder_height_cm: float
    upper_arm_cm: float
    forearm_cm: float
    drill_tcp_offset_cm: float = DRILL_TCP_OFFSET_CM

    @property
    def user_height_m(self) -> float:
        return self.user_height_cm / 100.0

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
        return self.upper_arm_m + self.forearm_m + self.drill_tcp_offset_m


@dataclass(frozen=True)
class TcpPoseResult:
    """Generated link0-frame TCP pose."""

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
        return {
            "msg": msg,
            "target_type": "tcp_pose",
            "target_floor_z_m": self.target_floor_height_m,
            "target_floor_z_mm": self.target_floor_height_m * 1000.0,
            "target_pose": self.to_pose_dict(),
            "model_info": {
                "model": "human_aware_ik_feasible_pose_generator_5th",
                "requested_floor_z_m": self.requested_floor_height_m,
                "was_height_clamped": self.was_height_clamped,
                "shoulder_height_m": self.shoulder_height_m,
                "total_arm_length_m": self.total_arm_length_m,
                "drill_tcp_offset_m": DRILL_TCP_OFFSET_M,
                "drill_tcp_offset_cm": DRILL_TCP_OFFSET_CM,
            },
        }

    def to_json_string(self, msg: str = "") -> str:
        return json.dumps(self.to_pass_goal_dict(msg=msg), ensure_ascii=False)


class HumanAwareTcpPoseGenerator:
    """
    Converts a human target height H into a link0-frame TCP pose.

    Human model: target shoulder angle + body dimensions -> floor height H.
    Robot model: interpolate 5th measurement IK-feasible TCP pose samples by H.
    """

    def __init__(
        self,
        base_height_m: float = BASE_HEIGHT_M,
        measured_pose_samples: tuple[MeasuredPoseSample, ...] = MEASURED_POSES_5TH,
    ) -> None:
        self.base_height_m = base_height_m
        self.measured_pose_samples = tuple(
            sorted(measured_pose_samples, key=lambda sample: sample[0])
        )
        if len(self.measured_pose_samples) < 2:
            raise ValueError("At least two measured pose samples are required.")

        self.min_link0_z_m = self.measured_pose_samples[0][1][2]
        self.max_link0_z_m = self.measured_pose_samples[-1][1][2]

    @property
    def min_floor_height_m(self) -> float:
        return self.measured_pose_samples[0][0]

    @property
    def max_floor_height_m(self) -> float:
        return self.measured_pose_samples[-1][0]

    def floor_height_from_shoulder_angle(
        self,
        profile: HumanArmProfile,
        target_shoulder_angle_deg: float,
    ) -> float:
        # H = H_s - L * cos(theta)
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
        requested_height_m = float(target_floor_height_m)
        height_m = (
            self._clamp_floor_height(requested_height_m)
            if clamp_to_robot_range
            else requested_height_m
        )
        was_height_clamped = abs(height_m - requested_height_m) > 1e-9

        x_m, y_m, z_m, qx, qy, qz, qw = self.interpolate_measured_pose(height_m)

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
            shoulder_height_m=profile.shoulder_height_m,
            total_arm_length_m=profile.total_arm_length_m,
            was_height_clamped=was_height_clamped,
        )

    def build_pass_goal_dict_from_shoulder_angle(
        self,
        target_shoulder_angle_deg: float,
        profile: HumanArmProfile,
        msg: str = "",
        frame_id: str = "link0",
    ) -> dict[str, Any]:
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
        return self.build_pass_goal_dict_from_floor_height(
            target_floor_height_m=target_floor_height_mm / 1000.0,
            profile=profile,
            msg=msg,
            frame_id=frame_id,
        )

    def interpolate_quaternion(
        self,
        target_floor_height_m: float,
    ) -> tuple[float, float, float, float]:
        lower, upper, alpha = self._find_measured_pose_interval(target_floor_height_m)
        return _slerp(lower[2], upper[2], alpha)

    def interpolate_measured_pose(
        self,
        target_floor_height_m: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        height_m = float(target_floor_height_m)
        lower, upper, alpha = self._find_measured_pose_interval(height_m)

        x_m = _lerp(lower[1][0], upper[1][0], alpha)
        y_m = _lerp(lower[1][1], upper[1][1], alpha)
        z_m = height_m - self.base_height_m
        qx, qy, qz, qw = _slerp(lower[2], upper[2], alpha)
        return x_m, y_m, z_m, qx, qy, qz, qw

    def _find_measured_pose_interval(
        self,
        target_floor_height_m: float,
    ) -> tuple[MeasuredPoseSample, MeasuredPoseSample, float]:
        height_m = float(target_floor_height_m)
        samples = self.measured_pose_samples

        if height_m <= samples[0][0]:
            return samples[0], samples[0], 0.0
        if height_m >= samples[-1][0]:
            return samples[-1], samples[-1], 0.0

        for lower, upper in zip(samples, samples[1:]):
            if lower[0] <= height_m <= upper[0]:
                interval = upper[0] - lower[0]
                alpha = 0.0 if interval == 0 else (height_m - lower[0]) / interval
                return lower, upper, alpha

        return samples[-1], samples[-1], 0.0

    def _clamp_floor_height(self, target_floor_height_m: float) -> float:
        return max(self.min_floor_height_m, min(self.max_floor_height_m, target_floor_height_m))


def prompt_human_arm_profile() -> HumanArmProfile:
    print("\n" + "=" * 60)
    print(" Experiment body-dimension input (press Enter for defaults)")
    print("=" * 60)

    user_height_cm = _prompt_float(" 1. Worker height (cm) [default: 175.0]: ", 175.0)
    default_shoulder_height_cm = user_height_cm - 30.0
    shoulder_height_cm = _prompt_float(
        f" 2. Shoulder height (cm) [default: {default_shoulder_height_cm:.1f}]: ",
        default_shoulder_height_cm,
    )
    upper_arm_cm = _prompt_float(" 3. Upper arm length (cm) [default: 30.0]: ", 30.0)
    forearm_cm = _prompt_float(" 4. Forearm length (cm) [default: 25.0]: ", 25.0)

    profile = HumanArmProfile(
        user_height_cm=user_height_cm,
        shoulder_height_cm=shoulder_height_cm,
        upper_arm_cm=upper_arm_cm,
        forearm_cm=forearm_cm,
    )
    print(
        "\n [Applied] "
        f"height: {profile.user_height_cm:.1f}cm | "
        f"shoulder: {profile.shoulder_height_cm:.1f}cm | "
        f"upper arm: {profile.upper_arm_cm:.1f}cm | "
        f"forearm: {profile.forearm_cm:.1f}cm | "
        f"drill/TCP offset: {profile.drill_tcp_offset_cm:.1f}cm fixed"
    )
    return profile


def make_human_arm_profile(
    shoulder_height_cm: float,
    upper_arm_cm: float,
    forearm_cm: float,
    user_height_cm: float = 175.0,
) -> HumanArmProfile:
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


def _lerp(start: float, end: float, alpha: float) -> float:
    return (1.0 - alpha) * start + alpha * end


def _normalize_quaternion(
    quaternion_xyzw: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion_xyzw))
    if norm == 0:
        raise ValueError("Cannot use a zero-length quaternion.")
    return tuple(value / norm for value in quaternion_xyzw)


def _slerp(
    q1_xyzw: tuple[float, float, float, float],
    q2_xyzw: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    q1 = _normalize_quaternion(q1_xyzw)
    q2 = _normalize_quaternion(q2_xyzw)
    alpha = max(0.0, min(1.0, alpha))

    dot = sum(a * b for a, b in zip(q1, q2))
    if dot < 0.0:
        q2 = tuple(-value for value in q2)
        dot = -dot

    if dot > 0.9995:
        blended = tuple((1.0 - alpha) * a + alpha * b for a, b in zip(q1, q2))
        return _normalize_quaternion(blended)

    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)

    scale_1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    scale_2 = sin_theta / sin_theta_0
    return tuple(scale_1 * a + scale_2 * b for a, b in zip(q1, q2))
