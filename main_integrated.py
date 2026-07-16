from __future__ import annotations

import os
import time

import cv2

from experiment_data import ExperimentDataLogger, ExperimentMetrics, TrialRecord

from hri_http_sender import GetRobotState, SendHoldFinished, SendPassGoal, SetReviewPending
from pose_generator import (
    DRILL_TCP_OFFSET_M,
    HumanArmProfile,
    HumanAwareTcpPoseGenerator,
)
from posture_estimator import MEASURED_SIDE, PostureEstimator
from voice_intent_interface import (
    ACTION_COMPLETE,
    ContinuousSpeechRecognizer,
    LlmIntentInterpreter,
    QueuedTtsSpeaker,
    manual_task_completion_from_key,
)


OPENAI_API_KEY = ""
LLAMA_BASE_URL = "https://api.groq.com/openai/v1"

INITIAL_SHOULDER_ANGLE_DEG = 130.0
MAX_EXPERIMENT_TIME_SEC = 480.0
PILOT_FUNCTIONAL_MIN_SHOULDER_DEG = 60.0
PILOT_FUNCTIONAL_MAX_SHOULDER_DEG = 80.0
LLM_DEFAULT_SAFE_TARGET_DEG = 70.0
RULE_Z_STEP_M = 0.05
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

def speak(text: str) -> None:
    print(f"[TTS] {text}")
    tts_speaker.speak(text)


def speak_and_wait(text: str, timeout_sec: float | None = None) -> None:
    speak(text)
    tts_speaker.wait_until_done(timeout_sec=timeout_sec)


def main() -> None:
    print("\n" + "=" * 60)
    print("작업자 신체 정보 입력")
    print("=" * 60)

    try:
        raw = input(" 1. 작업자 키(cm) [기본: 175.0]: ")
        user_height_cm = float(raw) if raw.strip() else 175.0
    except Exception:
        user_height_cm = 175.0

    try:
        default_shoulder = user_height_cm - 30.0
        raw = input(f" 2. 어깨 높이(cm) [기본: {default_shoulder:.1f}]: ")
        user_shoulder_height_cm = float(raw) if raw.strip() else default_shoulder
    except Exception:
        user_shoulder_height_cm = user_height_cm - 30.0

    try:
        raw = input(" 3. 상완 길이(cm) [기본: 30.0]: ")
        l1_cm = float(raw) if raw.strip() else 30.0
    except Exception:
        l1_cm = 30.0

    try:
        raw = input(" 4. 하완 길이(cm) [기본: 25.0]: ")
        l2_cm = float(raw) if raw.strip() else 25.0
    except Exception:
        l2_cm = 25.0

    print(
        f"\n[신체 정보] 키={user_height_cm:.1f}cm | 어깨={user_shoulder_height_cm:.1f}cm | "
        f"상완={l1_cm:.1f}cm | 하완={l2_cm:.1f}cm"
    )

    human_profile = HumanArmProfile(
        user_height_m=user_height_cm / 100.0,
        shoulder_height_m=user_shoulder_height_cm / 100.0,
        upper_arm_m=l1_cm / 100.0,
        forearm_m=l2_cm / 100.0,
    )
    pose_generator = HumanAwareTcpPoseGenerator()

    effective_safe_min_shoulder_angle_deg = PILOT_FUNCTIONAL_MIN_SHOULDER_DEG
    effective_safe_max_shoulder_angle_deg = PILOT_FUNCTIONAL_MAX_SHOULDER_DEG
    default_safe_shoulder_angle_deg = LLM_DEFAULT_SAFE_TARGET_DEG


    print("\n" + "=" * 60)
    for key_num, condition in CONDITIONS.items():
        print(f" [{key_num}] {condition['name']}")
    print("=" * 60)
    try:
        choice = int(input("실험 조건 번호를 입력하세요 (1~5): "))
    except Exception:
        choice = 1
    current_condition = CONDITIONS.get(choice, CONDITIONS[1])
    print(f"[CONDITION] {current_condition['name']}")

    intent_interpreter = (
        LlmIntentInterpreter(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)
        if OPENAI_API_KEY
        else None
    )
    if intent_interpreter is None:
        print("[LLM DISABLED] OPENAI_API_KEY is empty. Voice intent requires manual keys.")

    speech_recognizer = ContinuousSpeechRecognizer(
        on_text=lambda text: print(f"[STT] '{text}'")
    )
    speech_recognizer.start()

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
    metrics = ExperimentMetrics(
        risk_shoulder_deg=RISK_SHOULDER_DEG,
        risky_cycle_ratio_threshold=RISKY_CYCLE_RATIO_THRESHOLD,
        rula_high_score_threshold=RULA_HIGH_SCORE_THRESHOLD,
    )
    data_logger = ExperimentDataLogger(RESULT_DIR)
    
    review_cycle_result = None
    last_task_sample_time = 0.0
    current_target_floor_height_m = 0.0
    trial_count = 0

    # Save pass-goal payloads for trial traceability.
    def save_pass_goal_json(payload: dict, label: str) -> None:
        try:
            json_path = data_logger.write_pass_goal_json(payload, label)
            print(f"[PASS_GOAL JSON saved] {json_path}")
        except Exception as exc:
            print(f"[PASS_GOAL JSON save failed] {exc}")

    # LLM 요청 payload와 모델 원문 응답을 별도 JSON으로 남긴다.
    def save_llm_response_json(label: str) -> None:
        if intent_interpreter is None:
            return
        payload = {
            "condition": current_condition,
            "trial_num": trial_count + 1,
            "label": label,
            "model": intent_interpreter.model,
            "latency_s": intent_interpreter.last_latency_s,
            "request_payload": intent_interpreter.last_request_payload,
            "raw_response": intent_interpreter.last_raw_response,
            "parsed_response": intent_interpreter.last_parsed_response,
            "error": intent_interpreter.last_error,
        }
        try:
            json_path = data_logger.write_llm_response_json(payload, label)
            print(f"[LLM JSON saved] {json_path}")
        except Exception as exc:
            print(f"[LLM JSON save failed] {exc}")

 
    # pose_generator가 만든 목표 pose를 로봇에 보내고 높이 변화량을 반환한다.
    def send_pose_goal(
        pose_result,
        trial_number: int,
        previous_target_floor_height_m: float,
    ) -> tuple[float, float, bool]:
        next_target_floor_height_m = pose_result.target_floor_height_m
        adjustment_m = next_target_floor_height_m - previous_target_floor_height_m
        if abs(adjustment_m) <= 1e-9:
            print("[PASS_GOAL skipped] Target height unchanged.")
            return next_target_floor_height_m, adjustment_m, False

        payload = pose_result.to_pass_goal_dict(msg=f"Trial {trial_number} Setup")
        save_pass_goal_json(payload, f"trial_{trial_number:03d}_pass_goal")
        robot_command_sent = SendPassGoal(payload)
        if not robot_command_sent:
            print("[PASS_GOAL failed] Next target was not sent to robot.")
        return next_target_floor_height_m, adjustment_m, robot_command_sent

    # resolve_returning_pose_target()의 결과와 실제 전송 pose를 raw trial로 기록한다.
    def write_target_trial_record(
        target_info: dict,
        cycle,
        pose_result,
        trial_number: int,
        previous_target_floor_height_m: float,
        next_target_floor_height_m: float,
        adjustment_m: float,
        robot_command_sent: bool,
    ) -> None:
        action = str(target_info.get("action", "keep"))
        target_type = str(target_info.get("target_type", "none"))
        response_source = str(target_info.get("response_source", target_type))
        llm_confidence = float(target_info.get("confidence") or 0.0)
        decision_reason = str(target_info.get("reason", ""))
        llm_fallback = bool(target_info.get("llm_fallback", False))
        llm_latency_s = float(target_info.get("llm_latency_s") or 0.0)
        user_voice = str(target_info.get("user_voice", ""))
        is_invalid = bool(target_info.get("is_invalid", action == "clarify"))
        should_adjust = action == "adjust"

        if llm_fallback:
            metrics.record_llm_fallback()
        if (
            should_adjust
            and current_condition["lead"] == "System"
            and cycle.is_risky_cycle
        ):
            metrics.record_system_intervention()

        metrics.record_completed_trial(
            cycle=cycle,
            adjustment_z_mm=adjustment_m * 1000.0,
            is_invalid=is_invalid,
            is_correction=should_adjust,
        )
        if current_condition["lead"] == "Worker":
            metrics.record_worker_response(action)

        target_shoulder_angle_deg = target_info.get("target_shoulder_angle_deg")
        angle_adjustment_deg = (
            float(target_shoulder_angle_deg) - cycle.representative_shoulder_angle_deg
            if target_shoulder_angle_deg is not None
            else None
        )

        trial_record = TrialRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            condition_name=current_condition["name"],
            trial_num=trial_number,
            lead_type=current_condition["lead"],
            control_type=current_condition["control"],
            measured_side=MEASURED_SIDE,
            risk_shoulder_threshold_deg=RISK_SHOULDER_DEG,
            risky_cycle_ratio_threshold=RISKY_CYCLE_RATIO_THRESHOLD,
            user_height_cm=user_height_cm,
            shoulder_height_cm=user_shoulder_height_cm,
            upper_arm_cm=l1_cm,
            forearm_cm=l2_cm,
            drill_tcp_offset_cm=DRILL_TCP_OFFSET_M * 100.0,
            cycle=cycle,
            target_shoulder_angle_deg=target_shoulder_angle_deg,
            angle_adjustment_deg=angle_adjustment_deg,
            target_angle_source=target_type,
            response_action=action,
            response_source=response_source,
            llm_confidence=llm_confidence,
            decision_reason=decision_reason,
            llm_fallback=llm_fallback,
            prev_z_mm=previous_target_floor_height_m * 1000.0,
            final_z_mm=next_target_floor_height_m * 1000.0,
            adjustment_z_mm=adjustment_m * 1000.0,
            user_voice=user_voice,
            final_z_m=pose_result.target_floor_height_m,
            pose_height_clamped=pose_result.was_height_clamped,
            robot_command_sent=robot_command_sent,
            is_approved=should_adjust,
            llm_latency_s=llm_latency_s,
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

    def resolve_returning_pose_target(cycle_result) -> dict:
        # RETURNING에서 cycle_result와 current_condition을 보고 pose_generator 입력을 만든다.
        # 컨디션별 실험 로직은 유지한다.
        # - System+LLM, Worker+LLM: LLM이 만든 target_shoulder_angle_deg를 사용한다.
        # - System+Rule, Worker+Rule: target_shoulder_angle_deg를 쓰지 않고 floor height를 5cm step으로 바꾼다.
        # - Control: 위험 여부와 상관없이 유지한다.
        #
        # 위험 판단은 무조건 cycle_result.is_risky_cycle 하나만 쓴다.
        # 대표 어깨각도나 RULA proxy로 여기서 다시 위험 여부를 만들지 않는다.
        is_risky_cycle = bool(cycle_result.is_risky_cycle)
        condition_name = current_condition["name"]
        worker_clarify_text = "잘 이해하지 못했습니다. 올려드릴까요 내려드릴까요?"

        def keep_target() -> dict:
            return {
                "action": "keep",
                "target_type": "none",
                "target_floor_height_m": None,
                "target_shoulder_angle_deg": None,
            }

        def clarify_target() -> dict:
            return {
                "action": "clarify",
                "target_type": "none",
                "target_floor_height_m": None,
                "target_shoulder_angle_deg": None,
            }

        def floor_target(target_floor_height_m: float) -> dict:
            clamped_floor_height_m = max(
                pose_generator.min_floor_height_m,
                min(pose_generator.max_floor_height_m, float(target_floor_height_m)),
            )
            return {
                "action": "adjust",
                "target_type": "floor_height",
                "target_floor_height_m": clamped_floor_height_m,
                "target_shoulder_angle_deg": None,
            }

        def shoulder_target(target_shoulder_angle_deg: float) -> dict:
            return {
                "action": "adjust",
                "target_type": "shoulder_angle",
                "target_floor_height_m": None,
                "target_shoulder_angle_deg": float(target_shoulder_angle_deg),
            }

        def call_adjustment_llm(worker_text: str | None, is_system_review: bool):
            if intent_interpreter is None:
                return None
            intent = intent_interpreter.interpret_adjustment(
                worker_text,
                cycle_result,
                effective_safe_min_shoulder_angle_deg,
                default_safe_shoulder_angle_deg,
                effective_safe_max_shoulder_angle_deg,
                is_system_review=is_system_review,
            )
            metrics.record_llm_call(intent_interpreter.last_latency_s)
            save_llm_response_json(
                f"trial_{trial_count + 1:03d}_{'system_adjustment' if is_system_review else 'worker_adjustment'}"
            )
            return intent

        def wait_for_worker_voice(question_text: str, timeout_sec: float = 8.0) -> str | None:
            # 작업자 주도 조건에서는 이 함수 안에서 질문하고 답변을 받는 것이 최종 의도다.
            # 지금은 background STT의 최신 텍스트를 짧게 기다리는 초안이다.
            speech_recognizer.get_and_clear()
            speak_and_wait(question_text)
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                worker_voice = speech_recognizer.get_and_clear()
                if worker_voice:
                    return worker_voice
                time.sleep(0.05)
            return None

        match condition_name:
            case "Cond1_Sys_LLM":
                # 시스템 주도 + LLM:
                # 위험하지 않으면 유지한다.
                # 위험하면 빈 utterance로 조정 LLM을 호출하고 target_shoulder_angle_deg만 사용한다.
                if not is_risky_cycle:
                    return keep_target()

                intent = call_adjustment_llm(worker_text="", is_system_review=True)
                if intent is None:
                    return clarify_target()

                # Cond1 System+LLM:
                # 의도해석기 출력 중 action과 target_shoulder_angle_deg만 사용한다.
                # direction, amount_ratio는 이 조건에서 쓰지 않으므로 None 처리한 것과 같다.
                condition_intent = {
                    "action": intent.action,
                    "direction": None,
                    "amount_ratio": None,
                    "target_shoulder_angle_deg": intent.target_shoulder_angle_deg,
                    "confidence": intent.confidence,
                    "reason": intent.reason,
                }

                if condition_intent["action"] == "keep":
                    return keep_target()
                if condition_intent["action"] == "adjust" and condition_intent["target_shoulder_angle_deg"] is not None:
                    return shoulder_target(condition_intent["target_shoulder_angle_deg"])
                return clarify_target()

            case "Cond2_Sys_Rule":
                # 시스템 주도 + Rule:
                # 위험하지 않으면 유지한다.
                # 위험하면 현재 floor height에서 5cm 내린다.
                if not is_risky_cycle:
                    return keep_target()

                return floor_target(current_target_floor_height_m - RULE_Z_STEP_M)

            case "Cond3_Worker_LLM":
                # 작업자 주도 + LLM:
                # 위험/비위험에 따라 질문 문구를 다르게 말한다.
                # 작업자 답변은 조정 LLM으로 해석하고, target_shoulder_angle_deg만 실행에 사용한다.
                # 답변이 비었거나 애매하면 같은 자리에서 다시 질문한다.
                question_text = (
                    "방금 작업 자세에서 부담이 감지됐습니다. 작업 높이를 변경할까요?"
                    if is_risky_cycle
                    else "현재 작업 자세는 위험하지 않았습니다. 그래도 작업 높이를 변경할까요?"
                )
                while True:
                    worker_voice = wait_for_worker_voice(question_text)
                    question_text = worker_clarify_text
                    if not worker_voice:
                        continue

                    intent = call_adjustment_llm(worker_text=worker_voice, is_system_review=False)
                    if intent is None:
                        continue

                    # Cond3 Worker+LLM:
                    # 의도해석기 출력의 action, direction, amount_ratio, target_shoulder_angle_deg를 모두 사용한다.
                    # LLM은 direction/amount_ratio를 바탕으로 target_shoulder_angle_deg까지 계산해줘야 한다.
                    condition_intent = {
                        "action": intent.action,
                        "direction": intent.direction,
                        "amount_ratio": intent.amount_ratio,
                        "target_shoulder_angle_deg": intent.target_shoulder_angle_deg,
                        "confidence": intent.confidence,
                        "reason": intent.reason,
                    }

                    if condition_intent["action"] == "keep":
                        return keep_target()
                    if (
                        condition_intent["action"] == "adjust"
                        and condition_intent["target_shoulder_angle_deg"] is not None
                    ):
                        return shoulder_target(condition_intent["target_shoulder_angle_deg"])

            case "Cond4_Worker_Rule":
                # 작업자 주도 + Rule:
                # 작업자 답변은 조정 LLM으로 해석하되 target_shoulder_angle_deg는 무조건 무시한다.
                # 실행에는 direction만 사용한다. up이면 +5cm, down이면 -5cm다.
                # 답변이 비었거나 방향이 애매하면 같은 자리에서 다시 질문한다.
                question_text = (
                    "방금 작업 자세에서 부담이 감지됐습니다. 높이를 올릴까요, 내릴까요?"
                    if is_risky_cycle
                    else "현재 작업 자세는 위험하지 않았습니다. 그래도 높이를 조정할까요?"
                )
                while True:
                    worker_voice = wait_for_worker_voice(question_text)
                    question_text = worker_clarify_text
                    if not worker_voice:
                        continue

                    intent = call_adjustment_llm(worker_text=worker_voice, is_system_review=False)
                    if intent is None:
                        continue

                    # Cond4 Worker+Rule:
                    # 의도해석기 출력 중 action과 direction만 사용한다.
                    # amount_ratio와 target_shoulder_angle_deg는 이 조건에서 쓰지 않으므로 None 처리한다.
                    condition_intent = {
                        "action": intent.action,
                        "direction": intent.direction,
                        "amount_ratio": None,
                        "target_shoulder_angle_deg": None,
                        "confidence": intent.confidence,
                        "reason": intent.reason,
                    }

                    if condition_intent["action"] == "keep":
                        return keep_target()
                    if condition_intent["action"] != "adjust":
                        continue
                    if condition_intent["direction"] == "up":
                        return floor_target(current_target_floor_height_m + RULE_Z_STEP_M)
                    if condition_intent["direction"] == "down":
                        return floor_target(current_target_floor_height_m - RULE_Z_STEP_M)

            case "Cond5_Control_NoInterv":
                # 비개입 조건:
                # 위험 사이클이어도 실험 조건상 아무 조정도 하지 않는다.
                return keep_target()

            case _:
                # 알 수 없는 조건:
                # 잘못된 condition이 들어와도 로봇을 움직이지 않는 쪽으로 둔다.
                return keep_target()

    print(f"[INITIAL GOAL READY] 목표 어깨각 {INITIAL_SHOULDER_ANGLE_DEG:.1f}도로 초기 위치를 계산합니다.")
    initial_pose = pose_generator.generate_pose_from_shoulder_angle(INITIAL_SHOULDER_ANGLE_DEG, human_profile)
    current_target_floor_height_m = initial_pose.target_floor_height_m

    print("[START READY] 카메라 화면을 확인하고 S를 누르면 실험을 시작합니다. Q/ESC는 종료입니다.")
    started = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[CAMERA ERROR] 시작 대기 중 프레임을 읽지 못해 실험을 시작하지 않습니다.")
            break

        posture_sample = posture_estimator.process_frame(frame)
        posture_angle = posture_sample.shoulder_angle_deg if posture_sample.shoulder_angle_deg is not None else 0.0

        cv2.putText(frame, "Press S to start experiment", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Q/ESC: Stop", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 200), 2)
        cv2.putText(frame, f"Posture Angle: {posture_angle:.1f} deg", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
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
    robot_state = "UNKNOWN"
    previous_robot_state = None
    next_robot_state_poll_time = 0.0

    while cap.isOpened():
        elapsed_time = time.time() - experiment_start_time
        manual_stop_requested = key in (27, ord("q"), ord("Q"))
        if elapsed_time >= MAX_EXPERIMENT_TIME_SEC or manual_stop_requested:
            if manual_stop_requested:
                print("[MANUAL STOP] Q/ESC input detected.")
                speak("실험 중단 키가 입력되어 실험을 종료합니다.")
                metrics.mark_early_stop()
            else:
                print("[TIMEOUT] Experiment time limit reached.")
                speak("제한 시간이 경과하여 실험을 종료합니다.")
            break

        current_voice = speech_recognizer.get_and_clear()

        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now >= next_robot_state_poll_time:
            fetched_robot_state = GetRobotState()
            if fetched_robot_state:
                robot_state = fetched_robot_state
            next_robot_state_poll_time = now + ROBOT_STATE_POLL_SEC

        posture_sample = posture_estimator.process_frame(frame)
        entered_phase = robot_state != previous_robot_state
        posture_angle = posture_sample.shoulder_angle_deg if posture_sample.shoulder_angle_deg is not None else 0.0

        match robot_state:
            case "PICKING" | "IDLE" | "UNKNOWN":
                # 준비/대기 상태:
                # 이전 사이클에서 남은 리뷰 결과를 정리하고 review pending 상태를 꺼둔다.
                if entered_phase:
                    review_cycle_result = None
                    SetReviewPending(False)

            case "AT_TASK":
                # 작업 상태:
                # 자세 샘플을 계속 누적하고, SPACE 또는 음성 완료 의도만 처리한다.
                # 완료 의도가 들어오면 cycle 측정을 끝내 review_cycle_result에 저장한 뒤
                # SendHoldFinished()로 로봇이 RETURNING으로 넘어가게 알린다.
                # AT_TASK records posture samples and listens only for completion intent.
                if entered_phase:
                    speech_recognizer.get_and_clear()
                    speak("블록의 네 개 볼트에 있는 너트를 드릴로 빼주세요.")
                    metrics.start_cycle()
                    review_cycle_result = None
                    last_task_sample_time = 0.0
                    key = -1

                if review_cycle_result is None:
                    sample_time = time.time()
                    dt = max(0.0, sample_time - last_task_sample_time) if last_task_sample_time else 0.0
                    last_task_sample_time = sample_time
                    metrics.add_posture_sample(posture_sample, dt)

                    intent = manual_task_completion_from_key(key)
                    if intent is None and current_voice and intent_interpreter is not None:
                        intent = intent_interpreter.interpret_task_completion(current_voice)
                        metrics.record_llm_call(intent_interpreter.last_latency_s)
                        save_llm_response_json(f"trial_{trial_count + 1:03d}_task_completion")

                    if intent is not None and intent.action == ACTION_COMPLETE:
                        print(f"[TASK COMPLETE] reason={intent.reason}")
                        speak("작업 완료를 확인했습니다.")
                        review_cycle_result = metrics.finish_cycle()
                        SendHoldFinished()

            case "RETURNING":
                # 리뷰 상태:
                # 확정된 cycle_result를 조건별 target_info로 바꾸고,
                # 그 target_info를 pose_generator와 SendPassGoal/CSV 기록으로 연결한다.
                if review_cycle_result is not None:
                    SetReviewPending(True)
                    target_info = resolve_returning_pose_target(review_cycle_result)
                    action = str(target_info.get("action", "keep"))
                    target_type = str(target_info.get("target_type", "none"))
                    previous_target_floor_height_m = current_target_floor_height_m
                    trial_number = trial_count + 1

                    if action == "adjust" and target_type == "shoulder_angle":
                        pose_result = pose_generator.generate_pose_from_shoulder_angle(
                            target_info["target_shoulder_angle_deg"],
                            human_profile,
                        )
                        next_target_floor_height_m, adjustment_m, robot_command_sent = send_pose_goal(
                            pose_result,
                            trial_number,
                            previous_target_floor_height_m,
                        )
                    elif action == "adjust" and target_type == "floor_height":
                        pose_result = pose_generator.generate_pose_from_floor_height(
                            target_info["target_floor_height_m"],
                            human_profile,
                        )
                        next_target_floor_height_m, adjustment_m, robot_command_sent = send_pose_goal(
                            pose_result,
                            trial_number,
                            previous_target_floor_height_m,
                        )
                    else:
                        pose_result = pose_generator.generate_pose_from_floor_height(
                            current_target_floor_height_m,
                            human_profile,
                        )
                        next_target_floor_height_m = current_target_floor_height_m
                        adjustment_m = 0.0
                        robot_command_sent = False

                    write_target_trial_record(
                        target_info=target_info,
                        cycle=review_cycle_result,
                        pose_result=pose_result,
                        trial_number=trial_number,
                        previous_target_floor_height_m=previous_target_floor_height_m,
                        next_target_floor_height_m=next_target_floor_height_m,
                        adjustment_m=adjustment_m,
                        robot_command_sent=robot_command_sent,
                    )
                    current_target_floor_height_m = next_target_floor_height_m
                    trial_count = trial_number
                    review_cycle_result = None
                    SetReviewPending(False)

            case _:
                # 그 외 상태:
                # 아직 실험 판단에는 쓰지 않는 로봇 상태이므로 화면 표시만 유지한다.
                # Other robot states only keep the display alive.
                pass

        phase_label = robot_state

        cv2.putText(frame, f"HRI State: {phase_label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Robot State: {robot_state}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
        cv2.putText(frame, f"Trial: {trial_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Posture Angle: {posture_angle:.1f} deg", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        current_elapsed_sec = time.time() - experiment_start_time
        elapsed_mins = int(current_elapsed_sec // 60)
        elapsed_secs = int(current_elapsed_sec % 60)
        timer_text = f"Time: {elapsed_mins:02d}:{elapsed_secs:02d} / 08:00"
        cv2.putText(frame, timer_text, (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            "[Manual Override] SPACE: Done | Q/ESC: Stop",
            (20, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 200, 200),
            2,
        )

        cv2.imshow(DISPLAY_WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        previous_robot_state = robot_state

    speech_recognizer.stop()
    posture_estimator.close()
    cap.release()
    cv2.destroyAllWindows()

    actual_experiment_duration = time.time() - experiment_start_time if experiment_start_time else 0.0
    print("\n" + "=" * 60)
    print(f"[SUMMARY] {current_condition['name']} complete ({actual_experiment_duration:.1f}s)")
    print("=" * 60)

    summary_record = metrics.build_summary(
        condition_name=current_condition["name"],
        measured_side=MEASURED_SIDE,
        experiment_duration_s=actual_experiment_duration,
        user_height_cm=user_height_cm,
        shoulder_height_cm=user_shoulder_height_cm,
        upper_arm_cm=l1_cm,
        forearm_cm=l2_cm,
        drill_tcp_offset_cm=DRILL_TCP_OFFSET_M * 100.0,
    )
    data_logger.write_summary(summary_record)

    speak_and_wait("수고하셨습니다. 실험이 종료되었습니다.")
    tts_speaker.stop()


if __name__ == "__main__":
    main()

