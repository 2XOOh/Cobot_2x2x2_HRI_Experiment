# main_integrated.py
import math
import os
import time

import cv2

from experiment_data import ExperimentDataLogger, ExperimentMetrics, TrialRecord
from hri_http_sender import GetRobotState, SendHoldFinished, SendPassGoal, SetReviewPending
from pose_generator import (
    DRILL_TCP_OFFSET_CM,
    HumanAwareTcpPoseGenerator,
    make_human_arm_profile,
)
from posture_estimator import MEASURED_SIDE, PostureEstimator
from voice_intent_interface import (
    ContinuousSpeechRecognizer,
    LlmIntentParser,
    QueuedTtsSpeaker,
    RuleIntentParser,
    is_task_completion_input,
    parse_worker_adjustment_input,
)


# =========================================================
# API 및 환경 설정
# =========================================================
OPENAI_API_KEY = ""
LLAMA_BASE_URL = "https://api.groq.com/openai/v1"

INITIAL_SHOULDER_ANGLE_DEG = 130.0
MAX_EXPERIMENT_TIME_SEC = 480.0
PILOT_FUNCTIONAL_MIN_SHOULDER_DEG = 60.0
PILOT_FUNCTIONAL_MAX_SHOULDER_DEG = 80.0
RULE_WORKER_UP_STEP_DEG = 10.0
RULE_WORKER_DOWN_STEP_DEG = 10.0
ROBOT_STATE_POLL_SEC = 0.2
RISK_SHOULDER_DEG = 110.0
RISKY_CYCLE_RATIO_THRESHOLD = 0.60
RULA_HIGH_SCORE_THRESHOLD = 3.0

CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
DISPLAY_WINDOW_NAME = "HRI Ergonomic Bolt Fastening Task"
DISPLAY_WINDOW_WIDTH = 1280
DISPLAY_WINDOW_HEIGHT = 720

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

CONDITIONS = {
    1: {"intervention": "Intervention", "lead": "System", "control": "LLM", "name": "Cond1_Sys_LLM"},
    2: {"intervention": "Intervention", "lead": "System", "control": "Rule", "name": "Cond2_Sys_Rule"},
    3: {"intervention": "Intervention", "lead": "Worker", "control": "LLM", "name": "Cond3_Worker_LLM"},
    4: {"intervention": "Intervention", "lead": "Worker", "control": "Rule", "name": "Cond4_Worker_Rule"},
    5: {"intervention": "Non-Intervention", "lead": "System", "control": "None", "name": "Cond5_Control_NoInterv"},
}

tts_speaker = QueuedTtsSpeaker()


def shoulder_angle_from_floor_height_deg(
    shoulder_height_m: float,
    total_arm_length_m: float,
    floor_height_m: float,
) -> float:
    if total_arm_length_m <= 0:
        return PILOT_FUNCTIONAL_MIN_SHOULDER_DEG

    value = (shoulder_height_m - floor_height_m) / total_arm_length_m
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def robot_min_reachable_shoulder_angle_deg(
    shoulder_height_m: float,
    total_arm_length_m: float,
    robot_min_floor_height_m: float,
) -> float:
    return shoulder_angle_from_floor_height_deg(
        shoulder_height_m,
        total_arm_length_m,
        robot_min_floor_height_m,
    )


def speak(text):
    # TTS 문장을 콘솔에 찍고 비동기 음성 출력 큐에 넣는다.
    print(f"[TTS] {text}")
    tts_speaker.speak(text)


def speak_and_wait(text, timeout_sec=None):
    speak(text)
    tts_speaker.wait_until_done(timeout_sec=timeout_sec)


def decide_returning_policy(condition, is_risky_cycle):
    # 조건별 주도권과 위험 cycle 여부에 따라 유지/자동보정/작업자질문을 결정한다.
    lead_type = condition["lead"]
    control_type = condition["control"]

    if control_type == "None":
        return {
            "mode": "auto",
            "message": "비개입 조건이므로 기존 높이를 그대로 유지합니다.",
            "should_adjust": False,
            "is_risky": is_risky_cycle,
        }

    if lead_type == "Worker":
        if control_type == "LLM":
            message = (
                "불편자세를 감지했습니다. 작업 높이를 변경해드릴까요?"
                if is_risky_cycle
                else "안전자세가 감지되었으나 작업높이를 변경해드릴까요?"
            )
        else:
            message = (
                "자세 부담이 감지되었습니다. 작업높이를 변경할까요?"
                if is_risky_cycle
                else "작업높이를 변경할까요?"
            )
        return {
            "mode": "ask_worker",
            "message": message,
            "should_adjust": False,
            "is_risky": is_risky_cycle,
        }

    if not is_risky_cycle:
        return {
            "mode": "auto",
            "message": "안전자세가 감지되어 유지합니다.",
            "should_adjust": False,
            "is_risky": False,
        }

    if lead_type == "System":
        return {
            "mode": "auto",
            "message": "불편자세가 감지되어 조정합니다.",
            "should_adjust": True,
            "is_risky": True,
        }

    return {
        "mode": "ask_worker",
        "message": "방금 전 자세 불편이 감지되었습니다. 높이 조정을 진행할까요?",
        "should_adjust": False,
        "is_risky": True,
    }


def main():
    # 전체 HRI 실험 루프를 실행한다.
    print("\n" + "=" * 60)
    print(" 피실험자 신체 정보 입력 (엔터키를 누르면 괄호 안의 기본값 적용)")
    print("=" * 60)

    try:
        in_h = input(" 1. 작업자 키 (cm) [기본: 175.0]: ")
        user_height_cm = float(in_h) if in_h.strip() else 175.0
    except Exception:
        user_height_cm = 175.0

    try:
        default_sh = user_height_cm - 30.0
        in_sh = input(f" 2. 어깨까지의 높이 (cm) [기본: {default_sh}]: ")
        user_shoulder_height_cm = float(in_sh) if in_sh.strip() else default_sh
    except Exception:
        user_shoulder_height_cm = user_height_cm - 30.0

    try:
        in_l1 = input(" 3. 상완 길이 (어깨~팔꿈치, cm) [기본: 30.0]: ")
        l1_cm = float(in_l1) if in_l1.strip() else 30.0
    except Exception:
        l1_cm = 30.0

    try:
        in_l2 = input(" 4. 하완 길이 (팔꿈치~손목, cm) [기본: 25.0]: ")
        l2_cm = float(in_l2) if in_l2.strip() else 25.0
    except Exception:
        l2_cm = 25.0

    print(
        f"\n [적용 완료] 키: {user_height_cm}cm | 어깨 높이: {user_shoulder_height_cm}cm | "
        f"상완: {l1_cm}cm | 하완: {l2_cm}cm"
    )

    # 신체 치수는 pose generator가 쓸 profile로 한 번만 묶는다.
    human_profile = make_human_arm_profile(
        user_height_cm=user_height_cm,
        shoulder_height_cm=user_shoulder_height_cm,
        upper_arm_cm=l1_cm,
        forearm_cm=l2_cm,
    )
    pose_generator = HumanAwareTcpPoseGenerator()
    robot_min_shoulder_deg = robot_min_reachable_shoulder_angle_deg(
        shoulder_height_m=human_profile.shoulder_height_m,
        total_arm_length_m=human_profile.total_arm_length_m,
        robot_min_floor_height_m=pose_generator.min_floor_height_m,
    )
    effective_min_shoulder_deg = max(
        PILOT_FUNCTIONAL_MIN_SHOULDER_DEG,
        robot_min_shoulder_deg,
    )
    robot_max_shoulder_deg = shoulder_angle_from_floor_height_deg(
        shoulder_height_m=human_profile.shoulder_height_m,
        total_arm_length_m=human_profile.total_arm_length_m,
        floor_height_m=pose_generator.max_floor_height_m,
    )
    safe_range_upper_shoulder_deg = min(
        robot_max_shoulder_deg,
        max(effective_min_shoulder_deg, PILOT_FUNCTIONAL_MAX_SHOULDER_DEG),
    )
    print(
        "[RULE 높이 정책] "
        f"로봇 최저 도달 어깨각={robot_min_shoulder_deg:.2f}도 | "
        f"개인별 effective_min={effective_min_shoulder_deg:.2f}도 | "
        f"로봇 최고 도달 어깨각={robot_max_shoulder_deg:.2f}도"
    )

    print("\n" + "=" * 60)
    for k, v in CONDITIONS.items():
        print(f" [{k}] {v['name']}")
    print("=" * 60)
    try:
        choice = int(input("수행할 실험 조건 번호를 입력하세요 (1~5): "))
    except Exception:
        choice = 1
    current_condition = CONDITIONS.get(choice, CONDITIONS[1])

    # 음성 입력은 완료 감지, rule 응답, LLM 조정 의도 해석으로 나누어 처리한다.
    rule_intent_parser = RuleIntentParser()
    llm_intent_parser = LlmIntentParser(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL) if OPENAI_API_KEY else None
    speech_recognizer = ContinuousSpeechRecognizer(
        on_text=lambda text: print(f"🗣️ [음성 인식]: '{text}'")
    )
    speech_recognizer.start()

    # 카메라 파일은 프레임마다 자세값을 만들고 화면용 skeleton을 그린다.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CAMERA ERROR] 카메라를 열 수 없어 실험을 시작하지 않습니다.")
        cap.release()
        speech_recognizer.stop()
        tts_speaker.stop()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_WINDOW_NAME, DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT)
    posture_estimator = PostureEstimator()

    # metrics는 cycle 누적/위험 판정/summary 계산을 담당하고 logger는 CSV 저장만 담당한다.
    metrics = ExperimentMetrics(
        risk_shoulder_deg=RISK_SHOULDER_DEG,
        risky_cycle_ratio_threshold=RISKY_CYCLE_RATIO_THRESHOLD,
        rula_high_score_threshold=RULA_HIGH_SCORE_THRESHOLD,
    )
    data_logger = ExperimentDataLogger(RESULT_DIR)

    def save_pass_goal_json(payload, label):
        try:
            json_path = data_logger.write_pass_goal_json(payload, label)
            print(f"[PASS_GOAL JSON 저장] {json_path}")
        except Exception as e:
            print(f"[PASS_GOAL JSON 저장 실패] {e}")

    trial_count = 0
    wait_start_time = 0.0
    current_tighten_z_mm = 0.0
    last_task_sample_time = 0.0
    completion_sent = False
    awaiting_worker_answer = False
    current_cycle_result = None
    key = -1

    robot_state = "UNKNOWN"
    previous_robot_state = None
    next_robot_state_poll_time = 0.0
    experiment_start_time = 0.0

    def build_adjustment_metadata():
        # LLM과 rule parser가 볼 cycle 결과를 한 dict로 정리한다.
        cycle = current_cycle_result
        if cycle is None:
            raise RuntimeError("cycle 결과가 만들어지기 전에 조정 metadata를 요청했습니다.")

        current_robot_shoulder_deg = shoulder_angle_from_floor_height_deg(
            shoulder_height_m=human_profile.shoulder_height_m,
            total_arm_length_m=human_profile.total_arm_length_m,
            floor_height_m=current_tighten_z_mm / 1000.0,
        )

        return {
            "condition": current_condition,
            "cycle_task_time_sec": cycle.task_time_s,
            "cycle_risky_time_sec": cycle.risky_time_s,
            "cycle_risky_ratio": cycle.risky_ratio,
            "cycle_is_risky": cycle.is_risky_cycle,
            "cycle_representative_shoulder_angle_deg": cycle.representative_shoulder_angle_deg,
            "cycle_avg_elbow_angle_deg": cycle.avg_elbow_angle_deg,
            "cycle_avg_rula_proxy_score": cycle.avg_rula_proxy,
            "cycle_max_rula_proxy_score": cycle.max_rula_proxy,
            "cycle_rula_high_ratio": cycle.rula_high_ratio,
            "current_work_z_mm": current_tighten_z_mm,
            "pilot_functional_min_shoulder_deg": PILOT_FUNCTIONAL_MIN_SHOULDER_DEG,
            "pilot_functional_max_shoulder_deg": PILOT_FUNCTIONAL_MAX_SHOULDER_DEG,
            "robot_min_reachable_shoulder_deg": robot_min_shoulder_deg,
            "robot_max_reachable_shoulder_deg": robot_max_shoulder_deg,
            "effective_min_shoulder_deg": effective_min_shoulder_deg,
            "current_robot_shoulder_angle_deg": current_robot_shoulder_deg,
            "rule_worker_up_step_deg": RULE_WORKER_UP_STEP_DEG,
            "rule_worker_down_step_deg": RULE_WORKER_DOWN_STEP_DEG,
            "robot_min_floor_height_m": pose_generator.min_floor_height_m,
            "risk_trigger_deg": RISK_SHOULDER_DEG,
            "user_shoulder_height_mm": user_shoulder_height_cm * 10,
            "upper_arm_mm": l1_cm * 10,
            "forearm_mm": l2_cm * 10,
            "drill_tcp_offset_mm": DRILL_TCP_OFFSET_CM * 10,
            "total_arm_length_mm": human_profile.total_arm_length_m * 1000.0,
        }

    def build_system_llm_metadata():
        metadata = build_adjustment_metadata()
        return {
            "condition": metadata["condition"],
            "cycle_is_risky": metadata["cycle_is_risky"],
            "cycle_representative_shoulder_angle_deg": metadata[
                "cycle_representative_shoulder_angle_deg"
            ],
            "effective_min_shoulder_deg": metadata["effective_min_shoulder_deg"],
            "pilot_functional_min_shoulder_deg": metadata[
                "pilot_functional_min_shoulder_deg"
            ],
            "pilot_functional_max_shoulder_deg": metadata[
                "pilot_functional_max_shoulder_deg"
            ],
            "robot_min_reachable_shoulder_deg": metadata[
                "robot_min_reachable_shoulder_deg"
            ],
            "robot_max_reachable_shoulder_deg": metadata[
                "robot_max_reachable_shoulder_deg"
            ],
        }

    def worker_target_limit_message(direction, target_shoulder_angle_deg):
        if target_shoulder_angle_deg is None:
            return None

        preview = pose_generator.generate_pose_from_shoulder_angle(
            float(target_shoulder_angle_deg),
            human_profile,
        )
        current_floor_height_m = current_tighten_z_mm / 1000.0
        if abs(preview.target_floor_height_m - current_floor_height_m) > 1e-6:
            return None

        if direction == "down" and current_floor_height_m <= pose_generator.min_floor_height_m + 1e-6:
            return (
                "현재 로봇이 전달할 수 있는 최저 높이입니다. "
                "더 낮춰서 전달할 수 없습니다."
            )
        if direction == "up" and current_floor_height_m >= pose_generator.max_floor_height_m - 1e-6:
            return (
                "현재 로봇이 전달할 수 있는 최고 높이입니다. "
                "더 높여서 전달할 수 없습니다."
            )
        return None

    def apply_next_target(
        user_response_text,
        should_adjust,
        target_shoulder_angle_deg=None,
        llm_latency=0.0,
        is_invalid=False,
        response_action="",
        response_source="",
        response_direction="",
        llm_confidence=0.0,
        decision_reason="",
        llm_fallback=False,
    ):
        # RETURNING 평가 뒤 다음 cycle TCP pose를 계산하고 trial 한 줄을 저장한다.
        nonlocal trial_count, current_tighten_z_mm, awaiting_worker_answer, completion_sent

        cycle = current_cycle_result
        if cycle is None:
            raise RuntimeError("cycle 결과가 만들어지기 전에 다음 target을 계산했습니다.")

        trial_count += 1
        previous_target_z_mm = current_tighten_z_mm
        target_angle_deg = None
        angle_adjustment_deg = None
        target_angle_source = "none"
        is_correction = False

        if not user_response_text:
            user_response_text = f"Task completed; risky time {cycle.risky_time_s:.2f}s"

        if should_adjust:
            if target_shoulder_angle_deg is None:
                raise ValueError("높이 조정에는 명시적인 목표 어깨각이 필요합니다.")

            safe_guidance_required = (
                current_condition["lead"] == "System"
                or (
                    current_condition["lead"] == "Worker"
                    and cycle.is_risky_cycle
                    and response_direction == "down"
                )
            )
            target_lower_bound = (
                effective_min_shoulder_deg
                if safe_guidance_required
                else robot_min_shoulder_deg
            )
            target_upper_bound = (
                safe_range_upper_shoulder_deg
                if safe_guidance_required
                else robot_max_shoulder_deg
            )
            target_angle_deg = max(
                target_lower_bound,
                min(target_upper_bound, float(target_shoulder_angle_deg)),
            )
            if response_source == "rule_llm":
                if response_direction == "up":
                    target_angle_source = "rule_current_plus_10deg"
                elif cycle.is_risky_cycle:
                    target_angle_source = "rule_safe_range_upper"
                else:
                    target_angle_source = "rule_current_minus_10deg"
            elif response_source in ("system_rule", "system_rule_fallback"):
                target_angle_source = "rule_safe_range_upper"
            else:
                target_angle_source = "llm"

            angle_adjustment_deg = target_angle_deg - cycle.representative_shoulder_angle_deg
            pose_result = pose_generator.generate_pose_from_shoulder_angle(
                target_angle_deg,
                human_profile,
            )
            is_approved = True
        else:
            # 조정하지 않는 경우에는 현재 pass 높이를 그대로 pose로 다시 만든다.
            pose_result = pose_generator.generate_pose_from_floor_height(
                current_tighten_z_mm / 1000.0,
                human_profile,
            )
            is_approved = False

        next_target_z_mm = pose_result.target_floor_height_m * 1000.0
        next_target_floor_z_m = next_target_z_mm / 1000.0
        adjustment_z_mm = next_target_z_mm - previous_target_z_mm

        should_send_next_goal = abs(adjustment_z_mm) > 1e-6
        current_tighten_z_mm = next_target_z_mm

        robot_command_sent = False
        if should_send_next_goal:
            payload = pose_result.to_pass_goal_dict(msg=f"Trial {trial_count} Setup")
            save_pass_goal_json(payload, f"trial_{trial_count:03d}_pass_goal")
            robot_command_sent = SendPassGoal(payload)
            if not robot_command_sent:
                print("[PASS_GOAL 실패] 다음 목표가 로봇으로 전송되지 않았습니다.")
        else:
            print("[PASS_GOAL 생략] 이전 목표를 그대로 유지합니다.")

        if llm_fallback:
            metrics.record_llm_fallback()
        if current_condition["lead"] == "Worker":
            metrics.record_worker_response(response_action)

        metrics.record_completed_trial(
            cycle=cycle,
            adjustment_z_mm=adjustment_z_mm,
            is_invalid=is_invalid,
            is_correction=is_correction,
        )

        response_action_for_row = response_action or ("adjust" if should_adjust else "maintain")
        response_source_for_row = response_source or "policy"

        trial_record = TrialRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            condition_name=current_condition["name"],
            trial_num=trial_count,
            lead_type=current_condition["lead"],
            control_type=current_condition["control"],
            measured_side=MEASURED_SIDE,
            risk_shoulder_threshold_deg=RISK_SHOULDER_DEG,
            risky_cycle_ratio_threshold=RISKY_CYCLE_RATIO_THRESHOLD,
            user_height_cm=user_height_cm,
            shoulder_height_cm=user_shoulder_height_cm,
            upper_arm_cm=l1_cm,
            forearm_cm=l2_cm,
            drill_tcp_offset_cm=DRILL_TCP_OFFSET_CM,
            cycle=cycle,
            target_shoulder_angle_deg=target_angle_deg,
            angle_adjustment_deg=angle_adjustment_deg,
            target_angle_source=target_angle_source,
            response_action=response_action_for_row,
            response_source=response_source_for_row,
            llm_confidence=llm_confidence,
            decision_reason=decision_reason,
            llm_fallback=llm_fallback,
            prev_z_mm=previous_target_z_mm,
            final_z_mm=next_target_z_mm,
            adjustment_z_mm=adjustment_z_mm,
            user_voice=user_response_text,
            final_z_m=next_target_floor_z_m,
            pose_height_clamped=pose_result.was_height_clamped,
            robot_command_sent=robot_command_sent,
            is_approved=is_approved,
            llm_latency_s=llm_latency,
            is_invalid=is_invalid,
            pose_x_m=pose_result.x_m,
            pose_y_m=pose_result.y_m,
            pose_z_m=pose_result.z_m,
            pose_qx=pose_result.qx,
            pose_qy=pose_result.qy,
            pose_qz=pose_result.qz,
            pose_qw=pose_result.qw,
        )
        data_logger.write_trial(trial_record)

        SetReviewPending(False)
        awaiting_worker_answer = False
        completion_sent = False

    # 초기 pass goal은 목표 어깨각 130도에서 미리 계산한다.
    print(f"[INITIAL GOAL READY] 목표 어깨각 {INITIAL_SHOULDER_ANGLE_DEG:.1f}도 초기 위치를 계산합니다.")
    initial_pose = pose_generator.generate_pose_from_shoulder_angle(
        INITIAL_SHOULDER_ANGLE_DEG,
        human_profile,
    )
    current_tighten_z_mm = initial_pose.target_floor_height_m * 1000.0

    # 카메라 화면에서 자세를 확인하고 S를 누르면 초기 goal 전송과 8분 타이머를 시작한다.
    print("[START READY] 카메라 화면을 확인한 뒤 S를 누르면 실험을 시작합니다. Q/ESC는 종료입니다.")
    started = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[CAMERA ERROR] 시작 대기 중 프레임을 읽지 못해 실험을 시작하지 않습니다.")
            break

        posture_sample = posture_estimator.process_frame(frame)
        shoulder_ang = posture_sample.shoulder_angle_deg if posture_sample.shoulder_angle_deg is not None else 0.0
        elbow_ang = posture_sample.elbow_angle_deg if posture_sample.elbow_angle_deg is not None else 0.0
        current_rula = posture_sample.rula_proxy if posture_sample.rula_proxy is not None else 1.0

        cv2.putText(frame, "Press S to start experiment", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Q/ESC: Stop", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)
        cv2.putText(frame, f"Shoulder Angle: {shoulder_ang:.1f} deg", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Elbow Angle: {elbow_ang:.1f} deg", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Live RULA: {current_rula:.1f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(DISPLAY_WINDOW_NAME, frame)

        start_key = cv2.waitKey(10) & 0xFF
        if start_key in (ord("s"), ord("S")):
            started = True
            break
        if start_key in (27, ord("q"), ord("Q")):
            posture_estimator.close()
            cap.release()
            cv2.destroyAllWindows()
            speech_recognizer.stop()
            tts_speaker.stop()
            return

    if not started:
        posture_estimator.close()
        cap.release()
        cv2.destroyAllWindows()
        speech_recognizer.stop()
        tts_speaker.stop()
        return

    initial_payload = initial_pose.to_pass_goal_dict(msg="Initial Trial Setup")
    save_pass_goal_json(initial_payload, "initial_pass_goal")
    speak_and_wait("실험을 시작하겠습니다. 바른 자세로 로봇을 바라보고 앉아주세요.")
    SendPassGoal(initial_payload)
    SetReviewPending(False)
    experiment_start_time = time.time()
    key = -1

    # 실시간 HRI 제어 루프는 상태 변화에 맞춰 필요한 객체만 호출한다.
    while cap.isOpened():
        elapsed_time = time.time() - experiment_start_time
        manual_stop_requested = key in (27, ord("q"), ord("Q"))
        if elapsed_time >= MAX_EXPERIMENT_TIME_SEC or manual_stop_requested:
            if manual_stop_requested:
                print("[수동 조작 감지]: 실험 중단 키 입력")
                speak("실험 중단 키가 입력되어 실험을 종료합니다.")
                metrics.mark_early_stop()
            else:
                print(f"[시간 종료] 8분({MAX_EXPERIMENT_TIME_SEC}초)이 경과되어 실험을 자동 종료합니다.")
                speak("제한 시간 8분이 경과하여 실험을 종료합니다.")
            break

        current_voice = speech_recognizer.get_and_clear()

        ret, frame = cap.read()
        if not ret:
            break

        # HTTP bridge가 주는 최신 로봇 상태를 주기적으로 읽는다.
        now = time.time()
        if now >= next_robot_state_poll_time:
            fetched_robot_state = GetRobotState()
            if fetched_robot_state:
                robot_state = fetched_robot_state
            next_robot_state_poll_time = now + ROBOT_STATE_POLL_SEC

        # 자세 추정 파일이 한 프레임의 각도/RULA/visibility를 계산한다.
        posture_sample = posture_estimator.process_frame(frame)
        shoulder_ang = posture_sample.shoulder_angle_deg if posture_sample.shoulder_angle_deg is not None else 0.0
        elbow_ang = posture_sample.elbow_angle_deg if posture_sample.elbow_angle_deg is not None else 0.0
        current_rula = posture_sample.rula_proxy if posture_sample.rula_proxy is not None else 1.0

        entered_at_task = robot_state == "AT_TASK" and previous_robot_state != "AT_TASK"
        entered_returning = robot_state == "RETURNING" and previous_robot_state != "RETURNING"
        entered_picking = robot_state == "PICKING" and previous_robot_state != "PICKING"
        entered_idle = robot_state == "IDLE" and previous_robot_state != "IDLE"

        # PICKING에서는 이전 검토 상태를 정리하고 다음 AT_TASK를 기다린다.
        if entered_picking:
            awaiting_worker_answer = False
            completion_sent = False
            SetReviewPending(False)

        # AT_TASK 진입 시 새 cycle 자세 누적을 시작한다.
        if entered_at_task:
            speech_recognizer.get_and_clear()
            if previous_robot_state != "AT_TASK":
                speak("블록의 네 개 볼트에 있는 너트를 드릴로 빼주세요.")
            metrics.start_cycle()
            current_cycle_result = None
            last_task_sample_time = 0.0
            awaiting_worker_answer = False
            completion_sent = False
            key = -1

        # AT_TASK 중에는 프레임 자세값과 dt만 metrics에 넘긴다.
        if robot_state == "AT_TASK":
            sample_time = time.time()
            dt = max(0.0, sample_time - last_task_sample_time) if last_task_sample_time else 0.0
            last_task_sample_time = sample_time
            metrics.add_posture_sample(posture_sample, dt)

        # 작업 완료 입력이 오면 hold_finished를 보내고 cycle 결과를 확정한다.
        if robot_state == "AT_TASK" and not completion_sent:
            is_done = is_task_completion_input(
                key,
                current_voice,
                llm_parser=llm_intent_parser,
                metadata={"condition": current_condition},
            )

            if is_done:
                if key == ord(" "):
                    print("[수동 조작 감지]: 스페이스바(완료) 눌림")
                elif current_voice:
                    print(f"[작업 완료 음성 감지]: '{current_voice}'")
                speak("조립 완료블록을 내려놓습니다.")
                SendHoldFinished()
                current_cycle_result = metrics.finish_cycle()
                completion_sent = True

        # RETURNING 진입 시 cycle 결과를 보고 자동 유지/보정/작업자 질문을 결정한다.
        if entered_returning and completion_sent:
            SetReviewPending(True)
            user_response_text = ""
            policy = decide_returning_policy(current_condition, current_cycle_result.is_risky_cycle)
            speak(policy["message"])
            if policy["mode"] == "ask_worker":
                tts_speaker.wait_until_done()

            if policy["mode"] == "auto":
                if policy["is_risky"] and current_condition["lead"] == "System":
                    metrics.record_system_intervention()

                if (
                    policy["should_adjust"]
                    and current_condition["lead"] == "System"
                    and current_condition["control"] == "LLM"
                    and llm_intent_parser is not None
                ):
                    llm_start_time = time.time()
                    llm_decision = llm_intent_parser.parse(
                        "",
                        context="system_adjustment",
                        metadata=build_system_llm_metadata(),
                    )
                    llm_latency = time.time() - llm_start_time
                    metrics.record_llm_call(llm_latency)

                    print(
                        f"[System+LLM 판단]: action={llm_decision.action}, "
                        f"target_angle={llm_decision.target_shoulder_angle_deg}, "
                        f"reason={llm_decision.reason}"
                    )

                    if (
                        llm_decision.action in ("approve", "adjust")
                        and llm_decision.target_shoulder_angle_deg is not None
                    ):
                        apply_next_target(
                            "[System+LLM] automatic adjustment",
                            True,
                            target_shoulder_angle_deg=llm_decision.target_shoulder_angle_deg,
                            llm_latency=llm_latency,
                            is_invalid=llm_decision.is_invalid,
                            response_action=llm_decision.action,
                            response_source=llm_decision.source,
                            response_direction=llm_decision.direction,
                            llm_confidence=llm_decision.confidence,
                            decision_reason=llm_decision.reason,
                        )
                    else:
                        apply_next_target(
                            "[System+LLM fallback] rule adjustment",
                            policy["should_adjust"],
                            target_shoulder_angle_deg=safe_range_upper_shoulder_deg,
                            llm_latency=llm_latency,
                            is_invalid=llm_decision.is_invalid,
                            response_action=llm_decision.action,
                            response_source="system_rule_fallback",
                            response_direction="down",
                            llm_confidence=llm_decision.confidence,
                            decision_reason=llm_decision.reason,
                            llm_fallback=True,
                        )
                else:
                    system_rule_target = (
                        safe_range_upper_shoulder_deg
                        if policy["should_adjust"]
                        and current_condition["lead"] == "System"
                        and current_condition["control"] == "Rule"
                        else None
                    )
                    apply_next_target(
                        user_response_text,
                        policy["should_adjust"],
                        target_shoulder_angle_deg=system_rule_target,
                        response_action="adjust" if policy["should_adjust"] else "maintain",
                        response_source="system_rule" if system_rule_target is not None else "policy",
                        response_direction="down" if system_rule_target is not None else "maintain",
                    )
            else:
                speech_recognizer.get_and_clear()
                wait_start_time = time.time()
                awaiting_worker_answer = True

        # Worker 주도 조건에서는 RETURNING 중 작업자 답변을 받아 다음 target을 만든다.
        if robot_state == "RETURNING" and awaiting_worker_answer:
            worker_response = parse_worker_adjustment_input(
                wait_start_time=wait_start_time,
                key=key,
                voice_text=current_voice,
                control_type=current_condition["control"],
                rule_parser=rule_intent_parser,
                llm_parser=llm_intent_parser,
                metadata=build_adjustment_metadata(),
            )
            elapsed_wait = worker_response["elapsed_wait"]
            cv2.putText(
                frame,
                f"Waiting Answer... {elapsed_wait:.1f}s",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),
                2,
            )

            if (
                worker_response["action"] != "ignored"
                and (current_voice or worker_response["source"] == "manual")
            ):
                print(
                    f"[작업자 답변]: '{worker_response['text']}' -> "
                    f"{worker_response['action']} ({worker_response['source']}) | "
                    f"direction={worker_response['direction']} | "
                    f"target={worker_response['target_shoulder_angle_deg']} | "
                    f"confidence={worker_response['confidence']:.2f} | "
                    f"reason={worker_response['reason']}"
                )

            if current_voice and worker_response["source"] in ("llm", "llm_error", "rule_llm"):
                metrics.record_llm_call(worker_response["latency"])

            if worker_response["action"] == "ask_clarification" and worker_response["clarification_question"]:
                speak_and_wait(worker_response["clarification_question"])
                speech_recognizer.get_and_clear()

            if worker_response["action"] == "limit_reached":
                speak_and_wait(
                    "현재 로봇이 전달할 수 있는 최저 높이입니다. "
                    "더 낮춰서 전달할 수 없습니다."
                )
                speech_recognizer.get_and_clear()
                print("[최저 높이 도달] 작업자 답변을 계속 기다립니다.")
            elif worker_response["answered"]:
                should_adjust = worker_response["action"] in ("approve", "adjust")
                limit_message = (
                    worker_target_limit_message(
                        worker_response["direction"],
                        worker_response["target_shoulder_angle_deg"],
                    )
                    if should_adjust
                    else None
                )
                if limit_message:
                    speak_and_wait(limit_message)
                    speech_recognizer.get_and_clear()
                    print("[로봇 높이 한계 도달] 작업자 답변을 계속 기다립니다.")
                else:
                    apply_next_target(
                        worker_response["text"],
                        should_adjust,
                        target_shoulder_angle_deg=worker_response["target_shoulder_angle_deg"],
                        llm_latency=worker_response["latency"],
                        is_invalid=worker_response["is_invalid"],
                        response_action=worker_response["action"],
                        response_source=worker_response["source"],
                        response_direction=worker_response["direction"],
                        llm_confidence=worker_response["confidence"],
                        decision_reason=worker_response["reason"],
                    )
            elif current_voice and worker_response["action"] not in ("ask_clarification", "ignored"):
                if worker_response["is_invalid"]:
                    metrics.record_invalid_command()
                speak_and_wait("잘 인식하지 못했습니다. 다시 말씀해 주세요.")
                speech_recognizer.get_and_clear()
                print("[작업자 답변 해석 실패] 다시 답변을 기다립니다.")

        # IDLE에서는 한 cycle 흐름을 마치고 대기 상태로 정리한다.
        if entered_idle:
            awaiting_worker_answer = False
            completion_sent = False

        if robot_state == "AT_TASK":
            hri_phase_label = "AT_TASK_MEASURING"
        elif robot_state == "RETURNING" and awaiting_worker_answer:
            hri_phase_label = "RETURNING_WAIT_ANSWER"
        elif robot_state == "RETURNING":
            hri_phase_label = "RETURNING_REVIEW"
        else:
            hri_phase_label = robot_state

        cv2.putText(frame, f"HRI State: {hri_phase_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Robot State: {robot_state}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
        cv2.putText(frame, f"Trial: {trial_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Live RULA: {current_rula:.1f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Shoulder Angle: {shoulder_ang:.1f} deg", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        current_elapsed_sec = time.time() - experiment_start_time
        elapsed_mins = int(current_elapsed_sec // 60)
        elapsed_secs = int(current_elapsed_sec % 60)
        timer_text = f"Time: {elapsed_mins:02d}:{elapsed_secs:02d} / 08:00"
        cv2.putText(frame, timer_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "[Manual Override] SPACE: Done | Y: Yes | N: No | Q/ESC: Stop",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 200),
            2,
        )

        cv2.imshow(DISPLAY_WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        previous_robot_state = robot_state

    # 종료 시 장치 정리 후 summary 한 줄을 저장한다.
    speech_recognizer.stop()
    posture_estimator.close()
    cap.release()
    cv2.destroyAllWindows()

    actual_experiment_duration = time.time() - experiment_start_time if experiment_start_time else 0.0
    print("\n" + "=" * 60)
    print(f"📊 [{current_condition['name']}] 매트릭스 추출 완료 (실제 진행 시간: {actual_experiment_duration:.1f}초)")
    print("=" * 60)

    summary_record = metrics.build_summary(
        condition_name=current_condition["name"],
        measured_side=MEASURED_SIDE,
        experiment_duration_s=actual_experiment_duration,
        user_height_cm=user_height_cm,
        shoulder_height_cm=user_shoulder_height_cm,
        upper_arm_cm=l1_cm,
        forearm_cm=l2_cm,
        drill_tcp_offset_cm=DRILL_TCP_OFFSET_CM,
    )
    data_logger.write_summary(summary_record)

    speak_and_wait("수고하셨습니다. 실험이 종료되었습니다.")
    tts_speaker.stop()


if __name__ == "__main__":
    main()
