from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class PostureSample:
    """자세 측정 파일이 한 프레임에서 계산한 현재 자세값."""

    shoulder_angle_deg: float | None
    elbow_angle_deg: float | None
    rula_proxy: float | None
    visibility_ok: bool
    side: str


@dataclass
class CycleResult:
    """한 작업 cycle이 끝났을 때 metrics가 계산한 자세/위험 결과."""

    task_time_s: float
    risky_time_s: float
    risky_ratio: float
    is_risky_cycle: bool
    visibility_ok_time_s: float
    visibility_ok_ratio: float
    representative_shoulder_angle_deg: float
    avg_elbow_angle_deg: float
    avg_rula_proxy: float
    max_rula_proxy: float
    rula_high_ratio: float


@dataclass
class TrialRecord:
    """raw CSV 한 줄에 저장할 trial 단위 기록."""

    timestamp: str
    condition_name: str
    trial_num: int
    lead_type: str
    control_type: str
    measured_side: str
    risk_shoulder_threshold_deg: float
    risky_cycle_ratio_threshold: float
    user_height_cm: float
    shoulder_height_cm: float
    upper_arm_cm: float
    forearm_cm: float
    drill_tcp_offset_cm: float
    cycle: CycleResult
    target_shoulder_angle_deg: float | None
    angle_adjustment_deg: float | None
    target_angle_source: str
    response_action: str
    response_source: str
    llm_confidence: float
    decision_reason: str
    llm_fallback: bool
    prev_z_mm: float
    final_z_mm: float
    adjustment_z_mm: float
    user_voice: str
    final_z_m: float
    pose_height_clamped: bool
    robot_command_sent: bool
    is_approved: bool
    llm_latency_s: float
    is_invalid: bool
    pose_x_m: float
    pose_y_m: float
    pose_z_m: float
    pose_qx: float
    pose_qy: float
    pose_qz: float
    pose_qw: float


@dataclass
class SummaryRecord:
    """summary CSV 한 줄에 저장할 실험 실행 단위 요약."""

    condition_name: str
    completed_transfers: int
    measured_side: str
    experiment_duration_s: float
    avg_cycle_task_time_s: float
    risky_time_s: float
    risky_cycle_count: int
    risky_cycle_ratio_total: float
    system_interventions: int
    adjust_count: int
    avg_adj_mm: float
    correction_cmds: int
    invalid_cmds: int
    worker_approve_count: int
    worker_reject_count: int
    llm_call_count: int
    llm_fallback_count: int
    avg_llm_latency_s: float
    avg_representative_shoulder_angle_deg: float
    avg_rula_proxy: float
    risk_shoulder_threshold_deg: float
    risky_cycle_ratio_threshold: float
    user_height_cm: float
    shoulder_height_cm: float
    upper_arm_cm: float
    forearm_cm: float
    drill_tcp_offset_cm: float
    early_stop_flag: int


class ExperimentMetrics:
    """cycle 중 자세값을 누적하고, 실험 전체 summary 값을 계산한다."""

    def __init__(
        self,
        risk_shoulder_deg: float,
        risky_cycle_ratio_threshold: float,
        rula_high_score_threshold: float,
    ) -> None:
        self.risk_shoulder_deg = risk_shoulder_deg
        self.risky_cycle_ratio_threshold = risky_cycle_ratio_threshold
        self.rula_high_score_threshold = rula_high_score_threshold

        self.completed_transfers = 0
        self.risky_posture_time_s = 0.0
        self.risky_cycle_count = 0
        self.system_intervention_count = 0
        self.robot_adjustment_count = 0
        self.total_adjustment_magnitude_mm = 0.0
        self.correction_commands_count = 0
        self.invalid_cmds = 0
        self.worker_approve_count = 0
        self.worker_reject_count = 0
        self.llm_call_count = 0
        self.llm_fallback_count = 0
        self.early_stop_flag = 0

        self.llm_latencies: list[float] = []
        self.cycle_durations: list[float] = []
        self.cycle_representative_shoulder_angles: list[float] = []
        self.cycle_avg_rula_scores: list[float] = []

        self.start_cycle()

    def start_cycle(self) -> None:
        """새 AT_TASK cycle의 자세 누적값을 초기화한다."""
        self.cycle_task_time_s = 0.0
        self.cycle_risky_time_s = 0.0
        self.cycle_shoulder_angles: list[float] = []
        self.cycle_elbow_weighted_sum = 0.0
        self.cycle_rula_weighted_sum = 0.0
        self.cycle_max_rula = 1.0
        self.cycle_rula_high_time_s = 0.0
        self.cycle_visibility_ok_time_s = 0.0

    def add_posture_sample(self, posture: PostureSample, dt: float) -> None:
        """AT_TASK 중 들어온 posture sample을 cycle 지표에 누적한다."""
        if dt <= 0:
            return

        self.cycle_task_time_s += dt

        if not posture.visibility_ok:
            return
        if posture.shoulder_angle_deg is None or posture.elbow_angle_deg is None or posture.rula_proxy is None:
            return

        self.cycle_visibility_ok_time_s += dt
        self.cycle_shoulder_angles.append(float(posture.shoulder_angle_deg))
        self.cycle_elbow_weighted_sum += posture.elbow_angle_deg * dt
        self.cycle_rula_weighted_sum += posture.rula_proxy * dt
        self.cycle_max_rula = max(self.cycle_max_rula, float(posture.rula_proxy))

        if posture.rula_proxy >= self.rula_high_score_threshold:
            self.cycle_rula_high_time_s += dt
        if posture.shoulder_angle_deg >= self.risk_shoulder_deg:
            self.cycle_risky_time_s += dt

    def finish_cycle(self) -> CycleResult:
        """현재 cycle 누적값을 평균/비율로 정리한다."""
        task_time = self.cycle_task_time_s
        visible_time = self.cycle_visibility_ok_time_s
        risky_ratio = self.cycle_risky_time_s / visible_time if visible_time > 0 else 0.0
        visibility_ratio = self.cycle_visibility_ok_time_s / task_time if task_time > 0 else 0.0
        rula_high_ratio = self.cycle_rula_high_time_s / visible_time if visible_time > 0 else 0.0
        representative_shoulder_angle = _mode_angle_by_bin(self.cycle_shoulder_angles)

        return CycleResult(
            task_time_s=task_time,
            risky_time_s=self.cycle_risky_time_s,
            risky_ratio=max(0.0, min(1.0, risky_ratio)),
            is_risky_cycle=risky_ratio >= self.risky_cycle_ratio_threshold,
            visibility_ok_time_s=self.cycle_visibility_ok_time_s,
            visibility_ok_ratio=max(0.0, min(1.0, visibility_ratio)),
            representative_shoulder_angle_deg=representative_shoulder_angle,
            avg_elbow_angle_deg=self.cycle_elbow_weighted_sum / visible_time if visible_time > 0 else 80.0,
            avg_rula_proxy=self.cycle_rula_weighted_sum / visible_time if visible_time > 0 else 1.0,
            max_rula_proxy=self.cycle_max_rula,
            rula_high_ratio=max(0.0, min(1.0, rula_high_ratio)),
        )

    def record_completed_trial(
        self,
        cycle: CycleResult,
        adjustment_z_mm: float,
        is_invalid: bool,
        is_correction: bool,
    ) -> None:
        """trial 확정 후 summary용 누적 지표를 갱신한다."""
        self.completed_transfers += 1
        self.risky_posture_time_s += cycle.risky_time_s
        self.cycle_durations.append(cycle.task_time_s)
        self.cycle_representative_shoulder_angles.append(cycle.representative_shoulder_angle_deg)
        self.cycle_avg_rula_scores.append(cycle.avg_rula_proxy)

        if cycle.is_risky_cycle:
            self.risky_cycle_count += 1
        if abs(adjustment_z_mm) > 10.0:
            self.robot_adjustment_count += 1
        self.total_adjustment_magnitude_mm += abs(adjustment_z_mm)
        if is_invalid:
            self.invalid_cmds += 1
        if is_correction:
            self.correction_commands_count += 1

    def record_llm_call(self, latency_s: float) -> None:
        self.llm_call_count += 1
        if latency_s > 0:
            self.llm_latencies.append(latency_s)

    def record_llm_fallback(self) -> None:
        self.llm_fallback_count += 1

    def record_worker_response(self, action: str) -> None:
        if action in ("approve", "adjust"):
            self.worker_approve_count += 1
        elif action == "reject":
            self.worker_reject_count += 1

    def record_system_intervention(self) -> None:
        self.system_intervention_count += 1

    def record_invalid_command(self) -> None:
        self.invalid_cmds += 1

    def mark_early_stop(self) -> None:
        self.early_stop_flag = 1

    def build_summary(
        self,
        condition_name: str,
        measured_side: str,
        experiment_duration_s: float,
        user_height_cm: float,
        shoulder_height_cm: float,
        upper_arm_cm: float,
        forearm_cm: float,
        drill_tcp_offset_cm: float,
    ) -> SummaryRecord:
        """실험 종료 시 summary CSV에 쓸 값을 만든다."""
        completed = self.completed_transfers
        avg_cycle_time = sum(self.cycle_durations) / len(self.cycle_durations) if self.cycle_durations else 0.0
        avg_adj_mm = self.total_adjustment_magnitude_mm / completed if completed > 0 else 0.0
        risky_cycle_ratio_total = self.risky_cycle_count / completed if completed > 0 else 0.0
        avg_llm_latency = sum(self.llm_latencies) / len(self.llm_latencies) if self.llm_latencies else 0.0
        avg_representative_shoulder = (
            sum(self.cycle_representative_shoulder_angles) / len(self.cycle_representative_shoulder_angles)
            if self.cycle_representative_shoulder_angles
            else 0.0
        )
        avg_rula = (
            sum(self.cycle_avg_rula_scores) / len(self.cycle_avg_rula_scores)
            if self.cycle_avg_rula_scores
            else 0.0
        )

        return SummaryRecord(
            condition_name=condition_name,
            completed_transfers=completed,
            measured_side=measured_side,
            experiment_duration_s=experiment_duration_s,
            avg_cycle_task_time_s=avg_cycle_time,
            risky_time_s=self.risky_posture_time_s,
            risky_cycle_count=self.risky_cycle_count,
            risky_cycle_ratio_total=risky_cycle_ratio_total,
            system_interventions=self.system_intervention_count,
            adjust_count=self.robot_adjustment_count,
            avg_adj_mm=avg_adj_mm,
            correction_cmds=self.correction_commands_count,
            invalid_cmds=self.invalid_cmds,
            worker_approve_count=self.worker_approve_count,
            worker_reject_count=self.worker_reject_count,
            llm_call_count=self.llm_call_count,
            llm_fallback_count=self.llm_fallback_count,
            avg_llm_latency_s=avg_llm_latency,
            avg_representative_shoulder_angle_deg=avg_representative_shoulder,
            avg_rula_proxy=avg_rula,
            risk_shoulder_threshold_deg=self.risk_shoulder_deg,
            risky_cycle_ratio_threshold=self.risky_cycle_ratio_threshold,
            user_height_cm=user_height_cm,
            shoulder_height_cm=shoulder_height_cm,
            upper_arm_cm=upper_arm_cm,
            forearm_cm=forearm_cm,
            drill_tcp_offset_cm=drill_tcp_offset_cm,
            early_stop_flag=self.early_stop_flag,
        )


class ExperimentDataLogger:
    """TrialRecord와 SummaryRecord를 results CSV 파일에 저장한다."""

    # raw CSV는 trial/cycle 하나가 확정될 때 한 줄씩 저장한다.
    RAW_HEADER = [
        # 실행 조건
        "Time", "Condition", "Trial_Num", "Lead_Type", "Control_Type", "Measured_Side",
        # 실험 기준값
        "Risk_Shoulder_Threshold_deg", "Risky_Cycle_Ratio_Threshold",
        # 피험자 치수
        "User_Height_cm", "Shoulder_Height_cm", "Upper_Arm_cm", "Forearm_cm", "Drill_TCP_Offset_cm",
        # cycle 자세 결과
        "Task_Time_s", "Risky_Time_s", "Risky_Ratio", "Is_Risky_Cycle",
        "Visibility_OK_Time_s", "Visibility_OK_Ratio",
        "Representative_Shoulder_Angle_deg", "Avg_Elbow_Angle_deg",
        "Avg_RULA_Proxy", "Max_RULA_Proxy", "RULA_High_Ratio",
        # 의사결정 결과
        "Target_Shoulder_Angle_deg", "Angle_Adjustment_deg", "Target_Angle_Source",
        "Response_Action", "Response_Source", "LLM_Confidence", "Decision_Reason", "LLM_Fallback",
        # 로봇 목표와 전송 결과
        "Prev_Z_mm", "Final_Z_mm", "Adjustment_Z_mm", "User_Voice", "Final_Z_m",
        "Pose_Height_Clamped", "Robot_Command_Sent", "Is_Approved", "LLM_Latency_s", "Is_Invalid",
        # TCP pose
        "Pose_X_m", "Pose_Y_m", "Pose_Z_m", "Pose_QX", "Pose_QY", "Pose_QZ", "Pose_QW",
    ]

    # summary CSV는 실험 조건 1회 실행이 끝났을 때 한 줄 저장한다.
    SUMMARY_HEADER = [
        # 실행 조건
        "Condition", "Completed_Transfers", "Measured_Side",
        # 시간/위험 요약
        "Experiment_Duration_s", "Avg_Cycle_Task_Time_s",
        "Risky_Time_s", "Risky_Cycle_Count", "Risky_Cycle_Ratio_Total",
        # 개입/명령 요약
        "System_Interventions", "Adjust_Count", "Avg_Adj_mm", "Correction_Cmds", "Invalid_Cmds",
        # 작업자/LLM 응답 요약
        "Worker_Approve_Count", "Worker_Reject_Count",
        "LLM_Call_Count", "LLM_Fallback_Count", "Avg_LLM_Latency_s",
        # 자세 요약
        "Avg_Representative_Shoulder_Angle_deg", "Avg_RULA_Proxy",
        # 분석 기준값과 피험자 치수
        "Risk_Shoulder_Threshold_deg", "Risky_Cycle_Ratio_Threshold",
        "User_Height_cm", "Shoulder_Height_cm", "Upper_Arm_cm", "Forearm_cm", "Drill_TCP_Offset_cm",
        # 종료 상태
        "Early_Stop_Flag",
    ]

    def __init__(self, result_dir: str) -> None:
        os.makedirs(result_dir, exist_ok=True)
        self.raw_path = os.path.join(result_dir, "experiment_raw_data_per_trial.csv")
        self.summary_path = os.path.join(result_dir, "experiment_summary_matrix.csv")

    def write_trial(self, record: TrialRecord) -> None:
        self._append_row(self.raw_path, self.RAW_HEADER, self._trial_to_row(record))

    def write_summary(self, record: SummaryRecord) -> None:
        self._append_row(self.summary_path, self.SUMMARY_HEADER, self._summary_to_row(record))

    def _append_row(self, filename: str, header: list[str], row: list[Any]) -> None:
        header_needed = not os.path.isfile(filename) or os.path.getsize(filename) == 0
        with open(filename, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if header_needed:
                writer.writerow(header)
            writer.writerow(row)

    def _trial_to_row(self, record: TrialRecord) -> list[Any]:
        cycle = record.cycle
        return [
            record.timestamp,
            record.condition_name,
            record.trial_num,
            record.lead_type,
            record.control_type,
            record.measured_side,
            record.risk_shoulder_threshold_deg,
            record.risky_cycle_ratio_threshold,
            _round(record.user_height_cm, 1),
            _round(record.shoulder_height_cm, 1),
            _round(record.upper_arm_cm, 1),
            _round(record.forearm_cm, 1),
            _round(record.drill_tcp_offset_cm, 1),
            _round(cycle.task_time_s, 2),
            _round(cycle.risky_time_s, 2),
            _round(cycle.risky_ratio, 3),
            cycle.is_risky_cycle,
            _round(cycle.visibility_ok_time_s, 2),
            _round(cycle.visibility_ok_ratio, 3),
            _round(cycle.representative_shoulder_angle_deg, 2),
            _round(cycle.avg_elbow_angle_deg, 2),
            _round(cycle.avg_rula_proxy, 2),
            _round(cycle.max_rula_proxy, 2),
            _round(cycle.rula_high_ratio, 3),
            _round(record.target_shoulder_angle_deg, 2),
            _round(record.angle_adjustment_deg, 2),
            record.target_angle_source,
            record.response_action,
            record.response_source,
            _round(record.llm_confidence, 3),
            record.decision_reason,
            record.llm_fallback,
            _round(record.prev_z_mm, 1),
            _round(record.final_z_mm, 1),
            _round(record.adjustment_z_mm, 1),
            record.user_voice,
            _round(record.final_z_m, 3),
            record.pose_height_clamped,
            record.robot_command_sent,
            record.is_approved,
            _round(record.llm_latency_s, 2),
            record.is_invalid,
            _round(record.pose_x_m, 4),
            _round(record.pose_y_m, 4),
            _round(record.pose_z_m, 4),
            _round(record.pose_qx, 5),
            _round(record.pose_qy, 5),
            _round(record.pose_qz, 5),
            _round(record.pose_qw, 5),
        ]

    def _summary_to_row(self, record: SummaryRecord) -> list[Any]:
        return [
            record.condition_name,
            record.completed_transfers,
            record.measured_side,
            _round(record.experiment_duration_s, 2),
            _round(record.avg_cycle_task_time_s, 2),
            _round(record.risky_time_s, 2),
            record.risky_cycle_count,
            _round(record.risky_cycle_ratio_total, 3),
            record.system_interventions,
            record.adjust_count,
            _round(record.avg_adj_mm, 1),
            record.correction_cmds,
            record.invalid_cmds,
            record.worker_approve_count,
            record.worker_reject_count,
            record.llm_call_count,
            record.llm_fallback_count,
            _round(record.avg_llm_latency_s, 2),
            _round(record.avg_representative_shoulder_angle_deg, 2),
            _round(record.avg_rula_proxy, 2),
            record.risk_shoulder_threshold_deg,
            record.risky_cycle_ratio_threshold,
            _round(record.user_height_cm, 1),
            _round(record.shoulder_height_cm, 1),
            _round(record.upper_arm_cm, 1),
            _round(record.forearm_cm, 1),
            _round(record.drill_tcp_offset_cm, 1),
            record.early_stop_flag,
        ]

def _mode_angle_by_bin(angles: list[float], bin_size_deg: float = 5.0, default: float = 0.0) -> float:
    """5도 단위로 묶어 가장 오래 머문 어깨각 구간의 대표값을 계산한다."""
    if not angles:
        return default

    bins: dict[int, list[float]] = {}
    for angle in angles:
        bin_key = int(float(angle) // bin_size_deg)
        bins.setdefault(bin_key, []).append(float(angle))

    mode_bin = max(bins.values(), key=len)
    return sum(mode_bin) / len(mode_bin)


def _round(value: float | None, digits: int) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)
