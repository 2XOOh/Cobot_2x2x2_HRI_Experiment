# main_integrated.py
import cv2
import math
import os
import time

try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import pose as mp_pose
except (ImportError, AttributeError):
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.pose as mp_pose

from experiment_controller import PickAndPlaceExperiment
from hri_http_sender import GetRobotState, SendHoldFinished, SendPassGoal, SetReviewPending
from voice_intent_interface import ContinuousSpeechRecognizer, QueuedTtsSpeaker, RuleIntentParser

# =========================================================
# API 및 환경 설정
# =========================================================
OPENAI_API_KEY = ""  # ⚠️ 여기에 실제 Groq API 키를 입력하세요.
LLAMA_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_PASS_FLOOR_Z_CM = 135.5
ROBOT_STATE_POLL_SEC = 0.2
RISK_SHOULDER_DEG = 110.0
RISKY_CYCLE_RATIO_THRESHOLD = 0.80
LINK0_HEIGHT_MM = 634.0
MIN_Z_M = 0.634
MAX_Z_M = 1.200
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
DISPLAY_WINDOW_NAME = "HRI Ergonomic Bolt Fastening Task"
DISPLAY_WINDOW_WIDTH = 1280
DISPLAY_WINDOW_HEIGHT = 720

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SUMMARY_FILENAME = os.path.join(RESULT_DIR, "experiment_summary_matrix.csv")
RAW_CSV_FILENAME = os.path.join(RESULT_DIR, "experiment_raw_data_per_trial.csv")

CONDITIONS = {
    1: {"intervention": "Intervention", "lead": "System", "control": "LLM", "name": "Cond1_Sys_LLM"},
    2: {"intervention": "Intervention", "lead": "System", "control": "Rule", "name": "Cond2_Sys_Rule"},
    3: {"intervention": "Intervention", "lead": "Worker", "control": "LLM", "name": "Cond3_Worker_LLM"},
    4: {"intervention": "Intervention", "lead": "Worker", "control": "Rule", "name": "Cond4_Worker_Rule"},
    5: {"intervention": "Non-Intervention", "lead": "System", "control": "None", "name": "Cond5_Control_NoInterv"},
}

tts_speaker = QueuedTtsSpeaker()


def speak(text):
    # TTS 문장을 콘솔에 찍고 비동기 음성 출력 큐에 넣는다.
    print(f"[TTS] {text}")
    tts_speaker.speak(text)


def calculate_angle(a, b, c):
    # 세 점 a-b-c가 이루는 2D 각도를 degree 단위로 계산한다.
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    cosine_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine_angle))


def estimate_rula_score(shoulder_angle, elbow_angle):
    # 어깨/팔꿈치 각도 기반의 간이 RULA 점수를 계산한다.
    score = 1
    if shoulder_angle > 45:
        score += 2
    elif shoulder_angle > 20:
        score += 1
    if elbow_angle < 60 or elbow_angle > 100:
        score += 1
    return min(score, 7)


def detect_experiment_stop(key, voice_text, intent_parser):
    # Q/ESC 또는 조기 종료 음성 명령으로 전체 실험 중단 여부를 판별한다.
    if key in (27, ord("q"), ord("Q")):
        print("[수동 조작 감지]: 실험 중단 키 입력")
        return True

    if not voice_text:
        return False

    intent = intent_parser.parse(voice_text, context="any")
    if intent.action == "early_stop":
        print(f"[🚨 조기 종료 감지]: '{voice_text}'")
        return True

    return False


def detect_task_completion(key, voice_text, intent_parser):
    # AT_TASK 중 SPACE 또는 완료 음성 명령으로 현재 작업 완료 여부만 판별한다.
    if key == ord(" "):
        print("[수동 조작 감지]: 스페이스바(완료) 눌림")
        return True

    if not voice_text:
        return False

    intent = intent_parser.parse(voice_text, context="task_completion")
    if intent.action == "complete":
        print(f"[작업 완료 음성 감지]: '{voice_text}'")
        return True

    return False


def poll_worker_adjust_answer(wait_start_time, key, voice_text, intent_parser):
    # Worker 주도 조건에서 시간 제한 없이 Y/N 또는 음성 응답을 기다려 해석한다.
    elapsed_wait = time.time() - wait_start_time
    user_response_text = voice_text or ""

    manual_yes = key == ord("y") or key == ord("Y")
    manual_no = key == ord("n") or key == ord("N")

    if manual_yes:
        user_response_text = "[Manual] approve adjustment"
        print("[수동 조작 감지]: Y 키 (조정 승인)")
        return True, user_response_text, "approve", elapsed_wait
    elif manual_no:
        user_response_text = "[Manual] reject adjustment"
        print("[수동 조작 감지]: N 키 (조정 거절)")
        return True, user_response_text, "reject", elapsed_wait

    if not user_response_text:
        return False, "", "unknown", elapsed_wait

    intent = intent_parser.parse(user_response_text, context="adjustment_response")
    print(f"[작업자 답변]: '{user_response_text}' -> {intent.action}")

    if intent.action in ("approve", "reject", "adjust"):
        return True, user_response_text, intent.action, elapsed_wait

    print("[작업자 답변 해석 실패] 다시 답변을 기다립니다.")
    return False, "", "unknown", elapsed_wait


def is_risky_shoulder_angle(shoulder_angle_deg, threshold_deg=RISK_SHOULDER_DEG):
    # 현재 어깨 각도가 위험 기준 이상인지 판별한다.
    return shoulder_angle_deg >= threshold_deg


def evaluate_risky_cycle(task_time_sec, risky_time_sec, ratio_threshold=RISKY_CYCLE_RATIO_THRESHOLD):
    # AT_TASK 중 위험 자세 시간이 전체 유효 측정 시간의 기준 비율 이상인지 판별한다.
    if task_time_sec <= 0:
        return False, 0.0
    risky_ratio = max(0.0, min(1.0, risky_time_sec / task_time_sec))
    return risky_ratio >= ratio_threshold, risky_ratio


def compute_recommended_floor_z_mm(
    shoulder_height_mm,
    upper_arm_mm,
    forearm_mm,
    target_shoulder_angle_deg,
):
    # 신체 치수와 목표 어깨각도로 다음 pass 높이를 바닥 기준 mm 단위로 계산한다.
    shoulder_rad = math.radians(target_shoulder_angle_deg)
    recommended_floor_z_mm = (
        shoulder_height_mm
        - (
            upper_arm_mm * math.cos(shoulder_rad)
            + forearm_mm * math.cos(math.radians(0.0))
        )
    )
    recommended_floor_z_m = recommended_floor_z_mm / 1000.0
    return max(MIN_Z_M, min(MAX_Z_M, recommended_floor_z_m)) * 1000.0


def floor_z_mm_to_link0_m(floor_z_mm):
    # HTTP 전송 직전에 바닥 기준 mm 높이를 link0 기준 m 높이로 변환한다.
    return (floor_z_mm - LINK0_HEIGHT_MM) / 1000.0


def decide_returning_policy(condition, is_risky_cycle):
    # 조건별 주도권과 위험 cycle 여부에 따라 유지/자동보정/작업자질문을 결정한다.
    lead_type = condition["lead"]
    control_type = condition["control"]

    if control_type == "None":
        return {
            "mode": "auto",
            "message": "비개입 조건이므로 기존 높이를 그대로 유지합니다.",
            "is_approved_rule": False,
            "is_risky": is_risky_cycle,
        }

    if not is_risky_cycle:
        return {
            "mode": "auto",
            "message": "안전한 자세입니다. 동일한 높이로 다음 사이클을 진행합니다.",
            "is_approved_rule": False,
            "is_risky": False,
        }

    if lead_type == "System":
        return {
            "mode": "auto",
            "message": "방금 전 위험 자세가 감지되어 시스템이 다음 높이를 보정합니다.",
            "is_approved_rule": True,
            "is_risky": True,
        }

    return {
        "mode": "ask_worker",
        "message": "방금 전 자세 불편이 감지되었습니다. 높이 조정을 진행할까요?",
        "is_approved_rule": False,
        "is_risky": True,
    }


def main():
    # 전체 HRI 실험 루프를 실행한다.
    def run_llm_interface(user_response_text, recommended_floor_z_mm):
        # TODO: 구버전 LLM 높이 결정 경로. 이후 LLM은 의도 분석만 맡기고 이 함수는 제거한다.
        llm_start_time = time.time()
        llm_shoulder_context_deg = 130.0 if cycle_is_risky else 0.0
        llm_result = experiment_controller.run_task(
            condition=current_condition,
            sh_angle=llm_shoulder_context_deg,
            avg_sh_angle=llm_shoulder_context_deg,
            elb_angle=0.0,
            target_pass_floor_z_mm=recommended_floor_z_mm,
            adj_mm=0.0,
            current_pass_floor_z_mm=current_tighten_z_mm,
            h_sh=user_shoulder_height_cm * 10,
            l1=l1_cm * 10,
            l2=l2_cm * 10,
            user_voice_text=user_response_text,
            is_approved_rule=False,
        )
        return llm_result, time.time() - llm_start_time

    def apply_next_target(user_response_text, is_approved_rule):
        # RETURNING 평가가 끝난 뒤 다음 cycle에 쓸 target_z를 계산하고 HTTP로 전송한다.
        nonlocal trial_count, current_tighten_z_mm, awaiting_worker_answer, completion_sent

        trial_count += 1
        metrics["completed_transfers"] = trial_count
        metrics["risky_posture_time_sec"] += cycle_risky_time_sec
        cycle_durations.append(cycle_task_time_sec)

        if not user_response_text:
            user_response_text = f"Task completed; risky time {cycle_risky_time_sec:.2f}s"

        control_type = current_condition["control"]
        recommended_floor_z_mm = compute_recommended_floor_z_mm(
            user_shoulder_height_cm * 10,
            l1_cm * 10,
            l2_cm * 10,
            20.0,
        )

        if control_type == "LLM" and is_approved_rule:
            # LLM 조건에서는 승인된 작업자 응답이 있을 때만 LLM 제어기를 호출한다.
            llm_result, latency = run_llm_interface(user_response_text, recommended_floor_z_mm)
            llm_latencies.append(latency)
            final_floor_z_m = llm_result.get("final_z_m", current_tighten_z_mm / 1000.0)
            final_floor_z_m = max(MIN_Z_M, min(MAX_Z_M, final_floor_z_m))
            next_target_z_mm = final_floor_z_m * 1000.0
            is_approved = llm_result.get("is_approved", is_approved_rule)
            is_correction = llm_result.get("is_correction", False)
            is_invalid = llm_result.get("is_invalid", False)
        elif control_type == "Rule" and is_approved_rule:
            # Rule 조건에서는 계산식이 제한 범위까지 반영한 추천 높이를 그대로 사용한다.
            latency = 0.0
            next_target_z_mm = recommended_floor_z_mm
            is_approved = True
            is_correction = False
            is_invalid = False
        else:
            # 비개입, 안전 자세, 거절 응답에서는 현재 pass 높이를 유지한다.
            latency = 0.0
            next_target_z_mm = current_tighten_z_mm
            is_approved = False
            is_correction = False
            is_invalid = False

        next_target_floor_z_m = next_target_z_mm / 1000.0

        adj_mag = abs(next_target_z_mm - current_tighten_z_mm)
        metrics["total_adjustment_magnitude_mm"] += adj_mag
        if adj_mag > 10.0:
            metrics["robot_adjustment_count"] += 1
        if is_correction:
            metrics["correction_commands_count"] += 1
        if is_invalid:
            metrics["invalid_cmds"] += 1

        should_send_next_goal = abs(next_target_z_mm - current_tighten_z_mm) > 1e-6
        current_tighten_z_mm = next_target_z_mm

        if should_send_next_goal:
            # HTTP 전송 직전에만 바닥 기준 mm를 link0 기준 m로 변환한다.
            target_z_link0_m = floor_z_mm_to_link0_m(next_target_z_mm)
            SendPassGoal({"target_z_mm": target_z_link0_m, "msg": f"Trial {trial_count} Setup"})
        else:
            print("[PASS_GOAL 생략] 이전 목표를 그대로 유지합니다.")

        raw_file_exists = os.path.isfile(RAW_CSV_FILENAME)
        with open(RAW_CSV_FILENAME, "a", encoding="utf-8-sig") as f:
            # cycle별 원자료는 trial이 확정되는 RETURNING 단계에서 한 줄씩 저장한다.
            if not raw_file_exists:
                f.write("Time,Condition,Trial_Num,Lead_Type,Control_Type,Task_Time_s,Risky_Time_s,Is_Risky_Cycle,User_Voice,Final_Z_m,Is_Approved,LLM_Latency_s,Is_Invalid\n")
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')},{current_condition['name']},{trial_count},"
                f"{current_condition['lead']},{current_condition['control']},{cycle_task_time_sec:.2f},{cycle_risky_time_sec:.2f},"
                f"{cycle_is_risky},{user_response_text},{next_target_floor_z_m:.3f},{is_approved},{latency:.2f},{is_invalid}\n"
            )

        SetReviewPending(False)
        awaiting_worker_answer = False
        completion_sent = False

    # =========================================================
    # 피실험자 정보 입력
    # =========================================================
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
        f"\n [적용 완료] 키: {user_height_cm}cm | 어깨 높이: {user_shoulder_height_cm}cm | 상완: {l1_cm}cm | 하완: {l2_cm}cm"
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

    # =========================================================
    # 인터페이스 및 장치 초기화
    # =========================================================
    # 음성 인식 결과를 rule 기반 의도 parser로 해석하기 위한 인터페이스를 준비한다.
    intent_parser = RuleIntentParser()
    speech_recognizer = ContinuousSpeechRecognizer(
        on_text=lambda text: print(f"🗣️ [음성 인식]: '{text}'")
    )
    speech_recognizer.start()

    # TODO: 구버전 LLM 제어기. LlmIntentParser 기반 의도 분석으로 교체 후 제거한다.
    experiment_controller = PickAndPlaceExperiment(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)

    # 카메라 해상도와 표시 창 크기를 키워 MediaPipe 확인이 쉽도록 한다.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT)
    cv2.namedWindow(DISPLAY_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DISPLAY_WINDOW_NAME, DISPLAY_WINDOW_WIDTH, DISPLAY_WINDOW_HEIGHT)

    mp_pose_instance = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # =========================================================
    # 실험 지표 및 상태 변수 초기화
    # =========================================================
    # 실험 전체에 걸쳐 summary CSV로 저장할 누적 지표를 초기화한다.
    metrics = {
        "completed_transfers": 0, "risky_posture_time_sec": 0,
        "system_intervention_count": 0, "robot_adjustment_count": 0,
        "total_adjustment_magnitude_mm": 0.0, "correction_commands_count": 0, "invalid_cmds": 0,
        "early_stop_flag": 0
    }
    # 8분 제한 실험을 위해 전체 실험 시작 시각을 기록한다.
    MAX_EXPERIMENT_TIME_SEC = 480.0 
    experiment_start_time = time.time()

    cycle_durations = []
    llm_latencies = []

    trial_count = 0
    cycle_start_time = 0.0
    wait_start_time = 0.0
    current_tighten_z_mm = DEFAULT_PASS_FLOOR_Z_CM * 10.0

    cycle_task_time_sec = 0.0
    cycle_risky_time_sec = 0.0
    cycle_is_risky = False
    last_task_sample_time = 0.0
    completion_sent = False
    key = -1

    awaiting_worker_answer = False

    robot_state = "UNKNOWN"
    previous_robot_state = None
    next_robot_state_poll_time = 0.0

    # =========================================================
    # 실시간 HRI 제어 루프
    # =========================================================
    while cap.isOpened():
        # 제한 시간이 지나면 현재까지의 결과를 저장하고 종료한다.
        elapsed_time = time.time() - experiment_start_time
        if elapsed_time >= MAX_EXPERIMENT_TIME_SEC:
            print(f"[⏱️ 시간 종료] 8분({MAX_EXPERIMENT_TIME_SEC}초)이 경과되어 실험을 자동 종료합니다.")
            speak("제한 시간 8분이 경과하여 실험을 종료합니다.")
            break

        current_voice = speech_recognizer.get_and_clear()
        if detect_experiment_stop(key, current_voice, intent_parser):
            speak("실험자의 종료 의사가 확인되어 실험을 종료합니다.")
            metrics["early_stop_flag"] = 1
            break

        ret, frame = cap.read()
        if not ret: break

        # HTTP bridge가 주는 최신 로봇 상태를 주기적으로 읽는다.
        now = time.time()
        if now >= next_robot_state_poll_time:
            fetched_robot_state = GetRobotState()
            if fetched_robot_state:
                robot_state = fetched_robot_state
            next_robot_state_poll_time = now + ROBOT_STATE_POLL_SEC

        # 매 프레임마다 자세를 추정해 현재 어깨/팔꿈치 각도와 RULA를 계산한다.
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose_instance.process(image_rgb)

        shoulder_ang, elbow_ang, current_rula = 0.0, 0.0, 1
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # 💡 2. 오른팔 인식을 위해 11, 13, 15, 23 사용
            sh_x, sh_y, sh_z = lm[11].x, lm[11].y, lm[11].z       
            elb_x, elb_y, elb_z = lm[13].x, lm[13].y, lm[13].z    
            wr_x, wr_y, wr_z = lm[15].x, lm[15].y, lm[15].z       
            hip_x, hip_y, hip_z = lm[23].x, lm[23].y, lm[23].z    

            h, w, _ = frame.shape
            shoulder_pt = [int(sh_x * w), int(sh_y * h)]
            elbow_pt = [int(elb_x * w), int(elb_y * h)]
            wrist_pt = [int(wr_x * w), int(wr_y * h)]
            hip_pt = [int(hip_x * w), int(hip_y * h)]
            
            shoulder_ang = calculate_angle(hip_pt, shoulder_pt, elbow_pt)
            elbow_ang = calculate_angle(shoulder_pt, elbow_pt, wrist_pt)
            current_rula = estimate_rula_score(shoulder_ang, elbow_ang)
            
        entered_at_task = robot_state == "AT_TASK" and previous_robot_state != "AT_TASK"
        entered_returning = robot_state == "RETURNING" and previous_robot_state != "RETURNING"
        entered_picking = robot_state == "PICKING" and previous_robot_state != "PICKING"
        entered_idle = robot_state == "IDLE" and previous_robot_state != "IDLE"

        # Sequence 1) PICKING: 로봇이 다음 pass 위치를 준비하는 구간.
        # HRI는 이전 cycle 흔적만 정리하고 다음 AT_TASK를 기다린다.
        if entered_picking:
            awaiting_worker_answer = False
            completion_sent = False
            SetReviewPending(False)

        # Sequence 2) AT_TASK 진입: 작업자가 실제 체결을 시작하는 시점.
        # 이 순간부터 한 cycle의 자세 측정을 새로 시작한다.
        if entered_at_task:
            speech_recognizer.get_and_clear()
            if previous_robot_state != "AT_TASK":
                speak("블록의 네 개 구멍에 볼트를 체결해 주세요.")
            cycle_start_time = time.time()
            cycle_task_time_sec = 0.0
            cycle_risky_time_sec = 0.0
            cycle_is_risky = False
            last_task_sample_time = cycle_start_time
            awaiting_worker_answer = False
            completion_sent = False

        # Sequence 3) AT_TASK 유지: 작업 중 유효 자세 시간과 위험 자세 시간을 누적한다.
        if robot_state == "AT_TASK":
            sample_time = time.time()
            dt = max(0.0, sample_time - last_task_sample_time) if last_task_sample_time else 0.0
            last_task_sample_time = sample_time
            if shoulder_ang > 0:
                cycle_task_time_sec += dt
                if is_risky_shoulder_angle(shoulder_ang):
                    cycle_risky_time_sec += dt

        # Sequence 4) AT_TASK 완료 처리: 작업 완료 음성/키 입력이 들어오면
        # hold_finished를 보내고, 이번 cycle의 위험 자세 여부를 확정한다.
        if robot_state == "AT_TASK" and not completion_sent:
            is_done = detect_task_completion(key, current_voice, intent_parser)

            if is_done:
                speak("조립 완료블록을 내려놓습니다.")
                SendHoldFinished()
                cycle_is_risky, _ = evaluate_risky_cycle(cycle_task_time_sec, cycle_risky_time_sec)
                completion_sent = True

        # Sequence 5) RETURNING 진입: 작업이 끝나 로봇이 복귀하기 시작한 시점.
        # 여기서 5개 컨디션 중 현재 조건에 맞춰 자동 유지/자동 보정/작업자 질문을 결정한다.
        if entered_returning and completion_sent:
            SetReviewPending(True)
            user_response_text = ""
            policy = decide_returning_policy(current_condition, cycle_is_risky)
            speak(policy["message"])

            if policy["mode"] == "auto":
                if policy["is_risky"] and current_condition["lead"] == "System":
                    metrics["system_intervention_count"] += 1
                apply_next_target(user_response_text, policy["is_approved_rule"])
            else:
                speech_recognizer.get_and_clear()
                wait_start_time = time.time()
                awaiting_worker_answer = True

        # Sequence 6) Worker 주도 조건에서만: RETURNING 중 작업자 답변을 기다렸다가
        # LLM 또는 Rule 계산에 반영해 다음 target_z를 만든다.
        if robot_state == "RETURNING" and awaiting_worker_answer:
            answered, user_response_text, worker_action, elapsed_wait = poll_worker_adjust_answer(
                wait_start_time,
                key,
                current_voice,
                intent_parser,
            )
            cv2.putText(frame, f"Waiting Answer... {elapsed_wait:.1f}s", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            if answered:
                is_approved_rule = worker_action in ("approve", "adjust")
                apply_next_target(user_response_text, is_approved_rule)

        # Sequence 7) IDLE: 한 cycle이 완전히 끝난 뒤의 대기 구간.
        if entered_idle:
            awaiting_worker_answer = False
            completion_sent = False

        # 화면에는 로봇 상태와 HRI가 현재 어느 단계에 있는지 함께 표시한다.
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
        cv2.putText(frame, f"Live RULA: {current_rula}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Shoulder Angle: {shoulder_ang:.1f} deg", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 💡 [타이머 추가 부분] Shoulder Angle 아래(y좌표 190)에 노란색으로 크게 타이머를 띄웁니다.
        current_elapsed_sec = time.time() - experiment_start_time
        elapsed_mins = int(current_elapsed_sec // 60)
        elapsed_secs = int(current_elapsed_sec % 60)
        timer_text = f"Time: {elapsed_mins:02d}:{elapsed_secs:02d} / 08:00"
        cv2.putText(frame, timer_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, "[Manual Override] SPACE: Done | Y: Yes | N: No | Q/ESC: Stop", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)

        cv2.imshow(DISPLAY_WINDOW_NAME, frame)

        key = cv2.waitKey(10) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            metrics["early_stop_flag"] = 1
            break

        previous_robot_state = robot_state

    # =========================================================
    # 종료 처리 및 요약 저장
    # =========================================================
    speech_recognizer.stop()
    cap.release()
    cv2.destroyAllWindows()

    # 💡 횟수가 아닌 '실제 진행된 총 시간(8분 또는 조기종료)'을 바탕으로 평균 작업 시간 산출
    actual_experiment_duration = time.time() - experiment_start_time
    avg_cycle_time = actual_experiment_duration / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0
    
    avg_llm_latency = sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0
    avg_adj_mm = metrics["total_adjustment_magnitude_mm"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"📊 [{current_condition['name']}] 매트릭스 추출 완료 (실제 진행 시간: {actual_experiment_duration:.1f}초)")
    print("=" * 60)

    # 💡 서머리 엑셀 마지막에 Early_Stop_Flag 추가
    file_exists = os.path.isfile(SUMMARY_FILENAME)
    with open(SUMMARY_FILENAME, "a", encoding="utf-8-sig") as f:
        if not file_exists:
            f.write("Condition,Completed_Transfers,Avg_Task_Time_s,Risky_Time_s,System_Interventions,Adjust_Count,Avg_Adj_mm,Correction_Cmds,Invalid_Cmds,Avg_LLM_Latency_s,Early_Stop_Flag\n")
        f.write(f"{current_condition['name']},{metrics['completed_transfers']},{avg_cycle_time:.2f},"
                f"{metrics['risky_posture_time_sec']:.2f},{metrics['system_intervention_count']},"
                f"{metrics['robot_adjustment_count']},{avg_adj_mm:.1f},{metrics['correction_commands_count']},"
                f"{metrics['invalid_cmds']},{avg_llm_latency:.2f},{metrics['early_stop_flag']}\n")

    speak("수고하셨습니다. 실험이 종료되었습니다.")


if __name__ == "__main__":
    main()
