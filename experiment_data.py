from __future__ import annotations

import csv
import json
import math
import os
import time
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
    avg_shoulder_angle_deg: float
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
    target_angle_clamped: bool
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
    avg_representative_shoulder_angle_deg: float
    avg_shoulder_flexion_angle_deg: float
    avg_elbow_angle_deg: float
    risky_posture_time_s: float
    risky_posture_ratio_total: float
    risky_cycle_count: int
    risky_cycle_ratio_total: float
    visibility_ok_ratio_total: float
    avg_rula_proxy: float
    safe_posture_attainment_rate: float | None
    avg_post_adjustment_shoulder_improvement_deg: float | None
    risk_recurrence_rate: float | None
    completed_transfers: int
    avg_cycle_task_time_s: float
    task_time_sd_s: float
    throughput_transfers_per_min: float
    early_stop_flag: int
    system_interventions: int
    worker_requested_adjustment_count: int
    adjust_count: int
    avg_adjustment_per_cycle_mm: float
    avg_adjustment_per_adjustment_mm: float
    total_adjustment_magnitude_mm: float
    correction_cmds: int
    direction_reversal_count: int
    avg_adjustments_to_safe_posture: float | None
    invalid_cmds: int
    worker_approve_count: int
    worker_reject_count: int
    worker_approval_rate: float
    worker_rejection_rate: float
    llm_call_count: int
    avg_llm_latency_s: float
    llm_fallback_rate: float
    avg_command_to_action_latency_s: float | None
    avg_adjustment_completion_time_s: float | None
    robot_target_reach_success_rate: float | None
    avg_target_height_error_mm: float | None
    height_limit_hit_rate: float


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
        self.worker_requested_adjustment_count = 0
        self.direction_reversal_count = 0
        self.height_limit_hit_count = 0
        self.total_task_time_s = 0.0
        self.total_visibility_ok_time_s = 0.0
        self._last_adjustment_direction = 0

        self.llm_latencies: list[float] = []
        self.cycle_durations: list[float] = []
        self.cycle_representative_shoulder_angles: list[float] = []
        self.cycle_avg_shoulder_angles: list[float] = []
        self.cycle_avg_elbow_angles: list[float] = []
        self.cycle_avg_rula_scores: list[float] = []

        self.start_cycle()

    def start_cycle(self) -> None:
        """새 AT_TASK cycle의 자세 누적값을 초기화한다."""
        self.cycle_task_time_s = 0.0
        self.cycle_risky_time_s = 0.0
        self.cycle_shoulder_angles: list[float] = []
        self.cycle_shoulder_weighted_sum = 0.0
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
        self.cycle_shoulder_weighted_sum += posture.shoulder_angle_deg * dt
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
            avg_shoulder_angle_deg=self.cycle_shoulder_weighted_sum / visible_time if visible_time > 0 else 0.0,
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
        pose_height_clamped: bool = False,
        target_angle_clamped: bool = False,
        is_worker_requested_adjustment: bool = False,
    ) -> None:
        """trial 확정 후 summary용 누적 지표를 갱신한다."""
        self.completed_transfers += 1
        self.risky_posture_time_s += cycle.risky_time_s
        self.total_task_time_s += cycle.task_time_s
        self.total_visibility_ok_time_s += cycle.visibility_ok_time_s
        self.cycle_durations.append(cycle.task_time_s)
        self.cycle_representative_shoulder_angles.append(cycle.representative_shoulder_angle_deg)
        self.cycle_avg_shoulder_angles.append(cycle.avg_shoulder_angle_deg)
        self.cycle_avg_elbow_angles.append(cycle.avg_elbow_angle_deg)
        self.cycle_avg_rula_scores.append(cycle.avg_rula_proxy)

        if cycle.is_risky_cycle:
            self.risky_cycle_count += 1
        adjustment_magnitude_mm = abs(adjustment_z_mm)
        is_adjusted = adjustment_magnitude_mm > 10.0
        if is_adjusted:
            self.robot_adjustment_count += 1
            adjustment_direction = 1 if adjustment_z_mm > 0 else -1
            if self._last_adjustment_direction and adjustment_direction != self._last_adjustment_direction:
                self.direction_reversal_count += 1
            self._last_adjustment_direction = adjustment_direction
        if is_worker_requested_adjustment and is_adjusted:
            self.worker_requested_adjustment_count += 1
        if pose_height_clamped or target_angle_clamped:
            self.height_limit_hit_count += 1
        self.total_adjustment_magnitude_mm += adjustment_magnitude_mm
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
        avg_shoulder = (
            sum(self.cycle_avg_shoulder_angles) / len(self.cycle_avg_shoulder_angles)
            if self.cycle_avg_shoulder_angles
            else 0.0
        )
        avg_rula = (
            sum(self.cycle_avg_rula_scores) / len(self.cycle_avg_rula_scores)
            if self.cycle_avg_rula_scores
            else 0.0
        )
        avg_elbow = (
            sum(self.cycle_avg_elbow_angles) / len(self.cycle_avg_elbow_angles)
            if self.cycle_avg_elbow_angles
            else 0.0
        )
        task_time_sd = _population_sd(self.cycle_durations)
        throughput_transfers_per_min = (
            completed / self.total_task_time_s * 60.0
            if self.total_task_time_s > 0
            else 0.0
        )
        risky_posture_ratio_total = (
            self.risky_posture_time_s / self.total_visibility_ok_time_s
            if self.total_visibility_ok_time_s > 0
            else 0.0
        )
        visibility_ok_ratio_total = (
            self.total_visibility_ok_time_s / self.total_task_time_s
            if self.total_task_time_s > 0
            else 0.0
        )
        avg_adjustment_per_adjustment = (
            self.total_adjustment_magnitude_mm / self.robot_adjustment_count
            if self.robot_adjustment_count > 0
            else 0.0
        )
        worker_response_total = self.worker_approve_count + self.worker_reject_count
        worker_approval_rate = self.worker_approve_count / worker_response_total if worker_response_total > 0 else 0.0
        worker_rejection_rate = self.worker_reject_count / worker_response_total if worker_response_total > 0 else 0.0
        llm_fallback_rate = self.llm_fallback_count / self.llm_call_count if self.llm_call_count > 0 else 0.0
        height_limit_hit_rate = self.height_limit_hit_count / completed if completed > 0 else 0.0

        return SummaryRecord(
            condition_name=condition_name,
            avg_representative_shoulder_angle_deg=avg_representative_shoulder,
            avg_shoulder_flexion_angle_deg=avg_shoulder,
            avg_elbow_angle_deg=avg_elbow,
            risky_posture_time_s=self.risky_posture_time_s,
            risky_posture_ratio_total=risky_posture_ratio_total,
            risky_cycle_count=self.risky_cycle_count,
            risky_cycle_ratio_total=risky_cycle_ratio_total,
            visibility_ok_ratio_total=visibility_ok_ratio_total,
            avg_rula_proxy=avg_rula,
            safe_posture_attainment_rate=None,
            avg_post_adjustment_shoulder_improvement_deg=None,
            risk_recurrence_rate=None,
            completed_transfers=completed,
            avg_cycle_task_time_s=avg_cycle_time,
            task_time_sd_s=task_time_sd,
            throughput_transfers_per_min=throughput_transfers_per_min,
            early_stop_flag=self.early_stop_flag,
            system_interventions=self.system_intervention_count,
            worker_requested_adjustment_count=self.worker_requested_adjustment_count,
            adjust_count=self.robot_adjustment_count,
            avg_adjustment_per_cycle_mm=avg_adj_mm,
            avg_adjustment_per_adjustment_mm=avg_adjustment_per_adjustment,
            total_adjustment_magnitude_mm=self.total_adjustment_magnitude_mm,
            correction_cmds=self.correction_commands_count,
            direction_reversal_count=self.direction_reversal_count,
            avg_adjustments_to_safe_posture=None,
            invalid_cmds=self.invalid_cmds,
            worker_approve_count=self.worker_approve_count,
            worker_reject_count=self.worker_reject_count,
            worker_approval_rate=worker_approval_rate,
            worker_rejection_rate=worker_rejection_rate,
            llm_call_count=self.llm_call_count,
            avg_llm_latency_s=avg_llm_latency,
            llm_fallback_rate=llm_fallback_rate,
            avg_command_to_action_latency_s=None,
            avg_adjustment_completion_time_s=None,
            robot_target_reach_success_rate=None,
            avg_target_height_error_mm=None,
            height_limit_hit_rate=height_limit_hit_rate,
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
        "Representative_Shoulder_Angle_deg", "Avg_Shoulder_Angle_deg", "Avg_Elbow_Angle_deg",
        "Avg_RULA_Proxy", "Max_RULA_Proxy", "RULA_High_Ratio",
        # 의사결정 결과
        "Target_Shoulder_Angle_deg", "Target_Angle_Clamped", "Angle_Adjustment_deg", "Target_Angle_Source",
        "Response_Action", "Response_Source", "LLM_Confidence", "Decision_Reason", "LLM_Fallback",
        # 로봇 목표와 전송 결과
        "Prev_Z_mm", "Final_Z_mm", "Adjustment_Z_mm", "User_Voice", "Final_Z_m",
        "Pose_Height_Clamped", "Robot_Command_Sent", "Is_Approved", "LLM_Latency_s", "Is_Invalid",
        # TCP pose
        "Pose_X_m", "Pose_Y_m", "Pose_Z_m", "Pose_QX", "Pose_QY", "Pose_QZ", "Pose_QW",
    ]

    # summary CSV는 실험 조건 1회 실행이 끝났을 때 한 줄 저장한다.
        # 실행 조건
        # 시간/위험 요약
        # 개입/명령 요약
        # 작업자/LLM 응답 요약
        # 자세 요약
        # 분석 기준값과 피험자 치수
        # 종료 상태

    SUMMARY_CATEGORY_HEADER = (
        ["Posture-based Indicators"] * 12
        + ["Performance"] * 5
        + ["Robot Control"] * 22
    )

    SUMMARY_HEADER = [
        "Avg_Representative_Shoulder_Angle_deg",
        "Avg_Shoulder_Flexion_Angle_deg",
        "Avg_Elbow_Angle_deg",
        "Risky_Posture_Time_s",
        "Risky_Posture_Ratio_Total",
        "Risky_Cycle_Count",
        "Risky_Cycle_Ratio_Total",
        "Visibility_OK_Ratio_Total",
        "Avg_RULA_Proxy",
        "Safe_Posture_Attainment_Rate",
        "Avg_Post_Adjustment_Shoulder_Improvement_deg",
        "Risk_Recurrence_Rate",
        "Completed_Transfers",
        "Avg_Cycle_Task_Time_s",
        "Task_Time_SD_s",
        "Throughput_Transfers_Per_Min",
        "Early_Stop_Flag",
        "System_Interventions",
        "Worker_Requested_Adjustment_Count",
        "Adjust_Count",
        "Avg_Adjustment_Per_Cycle_mm",
        "Avg_Adjustment_Per_Adjustment_mm",
        "Total_Adjustment_Magnitude_mm",
        "Correction_Cmds",
        "Direction_Reversal_Count",
        "Avg_Adjustments_To_Safe_Posture",
        "Invalid_Cmds",
        "Worker_Approve_Count",
        "Worker_Reject_Count",
        "Worker_Approval_Rate",
        "Worker_Rejection_Rate",
        "LLM_Call_Count",
        "Avg_LLM_Latency_s",
        "LLM_Fallback_Rate",
        "Avg_Command_To_Action_Latency_s",
        "Avg_Adjustment_Completion_Time_s",
        "Robot_Target_Reach_Success_Rate",
        "Avg_Target_Height_Error_mm",
        "Height_Limit_Hit_Rate",
    ]

    def __init__(self, result_dir: str, run_label: str | None = None) -> None:
        os.makedirs(result_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_run_label = _safe_filename_part(run_label or "experiment")
        self.run_id = f"{timestamp}_{safe_run_label}"

        self.per_run_dir = os.path.join(result_dir, "per_condition_runs")
        os.makedirs(self.per_run_dir, exist_ok=True)
        self.raw_path = os.path.join(
            self.per_run_dir,
            f"{self.run_id}_raw_data_per_trial.csv",
        )
        self.summary_path = os.path.join(
            self.per_run_dir,
            f"{self.run_id}_condition_level_metrics.csv",
        )

        self.pass_goal_dir = os.path.join(result_dir, "pass_goal_json", self.run_id)
        os.makedirs(self.pass_goal_dir, exist_ok=True)
        self._pass_goal_json_index = 0

    def write_trial(self, record: TrialRecord) -> None:
        self._append_row(self.raw_path, self.RAW_HEADER, self._trial_to_row(record))

    def write_summary(self, record: SummaryRecord) -> None:
        self._append_row(
            self.summary_path,
            [self.SUMMARY_CATEGORY_HEADER, self.SUMMARY_HEADER],
            self._summary_to_row(record),
        )

    def write_pass_goal_json(self, payload: dict[str, Any], label: str) -> str:
        self._pass_goal_json_index += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_label = _safe_filename_part(label)
        filename = f"{timestamp}_{self._pass_goal_json_index:03d}_{safe_label}.json"
        path = os.path.join(self.pass_goal_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

        return path

    def _append_row(self, filename: str, header: list[str] | list[list[str]], row: list[Any]) -> None:
        header_needed = not os.path.isfile(filename) or os.path.getsize(filename) == 0
        with open(filename, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if header_needed:
                if header and isinstance(header[0], list):
                    writer.writerows(header)
                else:
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
            _round(cycle.avg_shoulder_angle_deg, 2),
            _round(cycle.avg_elbow_angle_deg, 2),
            _round(cycle.avg_rula_proxy, 2),
            _round(cycle.max_rula_proxy, 2),
            _round(cycle.rula_high_ratio, 3),
            _round(record.target_shoulder_angle_deg, 2),
            record.target_angle_clamped,
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
            _round(record.avg_representative_shoulder_angle_deg, 2),
            _round(record.avg_shoulder_flexion_angle_deg, 2),
            _round(record.avg_elbow_angle_deg, 2),
            _round(record.risky_posture_time_s, 2),
            _round(record.risky_posture_ratio_total, 3),
            record.risky_cycle_count,
            _round(record.risky_cycle_ratio_total, 3),
            _round(record.visibility_ok_ratio_total, 3),
            _round(record.avg_rula_proxy, 2),
            _round(record.safe_posture_attainment_rate, 3),
            _round(record.avg_post_adjustment_shoulder_improvement_deg, 2),
            _round(record.risk_recurrence_rate, 3),
            record.completed_transfers,
            _round(record.avg_cycle_task_time_s, 2),
            _round(record.task_time_sd_s, 2),
            _round(record.throughput_transfers_per_min, 2),
            record.early_stop_flag,
            record.system_interventions,
            record.worker_requested_adjustment_count,
            record.adjust_count,
            _round(record.avg_adjustment_per_cycle_mm, 1),
            _round(record.avg_adjustment_per_adjustment_mm, 1),
            _round(record.total_adjustment_magnitude_mm, 1),
            record.correction_cmds,
            record.direction_reversal_count,
            _round(record.avg_adjustments_to_safe_posture, 2),
            record.invalid_cmds,
            record.worker_approve_count,
            record.worker_reject_count,
            _round(record.worker_approval_rate, 3),
            _round(record.worker_rejection_rate, 3),
            record.llm_call_count,
            _round(record.avg_llm_latency_s, 2),
            _round(record.llm_fallback_rate, 3),
            _round(record.avg_command_to_action_latency_s, 2),
            _round(record.avg_adjustment_completion_time_s, 2),
            _round(record.robot_target_reach_success_rate, 3),
            _round(record.avg_target_height_error_mm, 1),
            _round(record.height_limit_hit_rate, 3),
        ]


def _mode_angle_by_bin(angles: list[float], bin_size_deg: float = 2.0, default: float = 0.0) -> float:
    """5도 단위로 묶어 가장 오래 머문 어깨각 구간의 대표값을 계산한다."""
    if not angles:
        return default

    bins: dict[int, list[float]] = {}
    for angle in angles:
        bin_key = int(float(angle) // bin_size_deg)
        bins.setdefault(bin_key, []).append(float(angle))

    mode_bin = max(bins.values(), key=len)
    return sum(mode_bin) / len(mode_bin)


def _population_sd(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _round(value: float | None, digits: int) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)


def _safe_filename_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    safe = safe.strip("_")
    return safe or "pass_goal"
