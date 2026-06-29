# main_integrated.py
import cv2
import json
import math
import os
import queue
import threading
import time

import pyttsx3
import speech_recognition as sr

try:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import pose as mp_pose
except (ImportError, AttributeError):
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.pose as mp_pose

from experiment_controller import PickAndPlaceExperiment
from hri_http_sender import GetRobotState, SendHoldFinished, SendPassGoal, SetReviewPending

# =========================================================
# API 및 환경 설정
# =========================================================
OPENAI_API_KEY = "api"  # ⚠️ 여기에 실제 Groq API 키를 입력하세요.
LLAMA_BASE_URL = "https://api.groq.com/openai/v1"

TOTAL_TRIALS_PER_CONDITION = 10
DEFAULT_PASS_FLOOR_Z_CM = 135.5
ROBOT_STATE_POLL_SEC = 0.2
RISK_SHOULDER_DEG = 110.0
LINK0_HEIGHT_MM = 634.0
MIN_LINK0_Z_M = 0.634
MAX_LINK0_Z_M = 1.200

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SAVE_FILENAME = os.path.join(RESULT_DIR, "experiment_log_detailed.json")
SUMMARY_FILENAME = os.path.join(RESULT_DIR, "experiment_summary_matrix.csv")
RAW_CSV_FILENAME = os.path.join(RESULT_DIR, "experiment_raw_data_per_trial.csv")

CONDITIONS = {
    1: {"intervention": "Intervention", "lead": "System", "control": "LLM", "name": "Cond1_Sys_LLM"},
    2: {"intervention": "Intervention", "lead": "System", "control": "Rule", "name": "Cond2_Sys_Rule"},
    3: {"intervention": "Intervention", "lead": "Worker", "control": "LLM", "name": "Cond3_Worker_LLM"},
    4: {"intervention": "Intervention", "lead": "Worker", "control": "Rule", "name": "Cond4_Worker_Rule"},
    5: {"intervention": "Non-Intervention", "lead": "System", "control": "None", "name": "Cond5_Control_NoInterv"},
}

voice_command = None
voice_lock = threading.Lock()
running = True
coordinate_logs = []
tts_queue = queue.Queue()
tts_worker_started = False
tts_worker_lock = threading.Lock()


def tts_worker():
    engine = None
    while True:
        text = tts_queue.get()
        try:
            if engine is None:
                engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS ERROR] {e}")
            try:
                if engine is not None:
                    engine.stop()
            except Exception:
                pass
            engine = None
        finally:
            tts_queue.task_done()


def ensure_tts_worker_started():
    global tts_worker_started
    if tts_worker_started:
        return
    with tts_worker_lock:
        if tts_worker_started:
            return
        threading.Thread(target=tts_worker, daemon=True).start()
        tts_worker_started = True


def speak(text):
    print(f"[TTS] {text}")
    ensure_tts_worker_started()
    tts_queue.put(text)


def speech_recognition_thread():
    global voice_command
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = False
    recognizer.pause_threshold = 0.5

    microphone = sr.Microphone()
    print("[STT] 마이크 상시 대기 모드 켜짐 (노이즈 필터링 없이 즉각 반응)")

    with microphone as source:
        while running:
            try:
                audio = recognizer.listen(source, timeout=1.0, phrase_time_limit=3.0)
                text = recognizer.recognize_google(audio, language="ko-KR")
                print(f"🗣️ [음성 인식]: '{text}'")
                with voice_lock:
                    voice_command = text
            except sr.WaitTimeoutError:
                continue
            except Exception:
                continue


def calculate_angle(a, b, c):
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
    score = 1
    if shoulder_angle > 45:
        score += 2
    elif shoulder_angle > 20:
        score += 1
    if elbow_angle < 60 or elbow_angle > 100:
        score += 1
    return min(score, 7)


def append_coordinate_log(trial, target_z_mm):
    coordinate_logs.append(
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "trial": trial,
            "target_z_mm": target_z_mm,
        }
    )
    with open(SAVE_FILENAME, "w", encoding="utf-8") as f:
        json.dump(coordinate_logs, f, indent=4)


def get_and_clear_voice():
    global voice_command
    with voice_lock:
        local_voice = voice_command
        voice_command = None
    return local_voice


def detect_task_completion(key):
    local_voice = get_and_clear_voice()
    
    # 💡 조기 종료 및 완료 키워드 분리
    early_stop_list = ["그만", "중단", "포기", "아파", "조기", "못하겠", "힘들"]
    kw_list = ["끝", "완료", "다했", "체결", "조립", "다 했", "완료했", "마무리", "오케이", "종료"]
    
    is_done = False
    is_early_stop = False

    if local_voice:
        if any(kw in local_voice for kw in early_stop_list):
            print(f"[🚨 조기 종료 감지]: '{local_voice}'")
            is_early_stop = True
        elif any(kw in local_voice for kw in kw_list):
            print(f"[작업 완료 음성 감지]: '{local_voice}'")
            is_done = True

    if key == ord(" "):
        print("[수동 조작 감지]: 스페이스바(완료) 눌림")
        is_done = True

    return is_done, is_early_stop


def poll_worker_adjust_answer(wait_start_time, key):
    elapsed_wait = time.time() - wait_start_time
    local_voice = get_and_clear_voice()
    user_response_text = local_voice or ""

    manual_yes = key == ord("y") or key == ord("Y")
    manual_no = key == ord("n") or key == ord("N")

    if not (user_response_text or elapsed_wait > 5.0 or manual_yes or manual_no):
        return False, "", False, False, elapsed_wait

    if manual_yes:
        user_response_text = "Yes, please adjust (Manual)"
        print("[수동 조작 감지]: Y 키 (조정 승인)")
    elif manual_no:
        user_response_text = "No, keep it (Manual)"
        print("[수동 조작 감지]: N 키 (조정 거절)")
    elif user_response_text:
        print(f"[작업자 답변]: '{user_response_text}'")
    else:
        print("[대답 없음] 기본값으로 진행합니다.")

    return True, user_response_text, manual_yes, manual_no, elapsed_wait


def resolve_rule_approval(user_response_text, manual_yes, manual_no):
    if manual_yes:
        return True
    if manual_no:
        return False
    pos_kws = ["응", "어", "네", "예", "조정", "해줘", "맞아", "오케이", "ok", "좋아"]
    normalized = user_response_text.replace(" ", "")
    return any(kw in normalized for kw in pos_kws)


def compute_cycle_averages(shoulder_angles, elbow_angles, rula_scores):
    # 💡 5도 단위 그룹화로 가장 오래 유지한 자세(최빈값 구간)의 평균 각도 추출
    if shoulder_angles:
        bins = {}
        for a in shoulder_angles:
            b = int(a // 5) * 5
            if b not in bins: bins[b] = []
            bins[b].append(a)
        max_bin = max(bins.values(), key=len)
        avg_sh = sum(max_bin) / len(max_bin)
    else:
        avg_sh = 30.0
        
    avg_elb = sum(elbow_angles) / len(elbow_angles) if elbow_angles else 80.0
    avg_rula = sum(rula_scores) / len(rula_scores) if rula_scores else 2.0
    
    return avg_sh, avg_elb, avg_rula


def is_risky_posture(avg_shoulder_angle_deg):
    return avg_shoulder_angle_deg >= RISK_SHOULDER_DEG


def compute_recommended_floor_z_mm(
    shoulder_height_mm,
    upper_arm_mm,
    forearm_mm,
    target_shoulder_angle_deg,
):
    shoulder_rad = math.radians(target_shoulder_angle_deg)
    return (
        shoulder_height_mm
        - (
            upper_arm_mm * math.cos(shoulder_rad)
            + forearm_mm * math.cos(math.radians(0.0))
        )
    )


def floor_z_mm_to_link0_m(floor_z_mm):
    return (floor_z_mm - LINK0_HEIGHT_MM) / 1000.0


def link0_m_to_floor_z_mm(link0_z_m):
    return link0_z_m * 1000.0 + LINK0_HEIGHT_MM


def clamp_floor_z_mm_to_robot_limits(floor_z_mm):
    link0_z_m = floor_z_mm_to_link0_m(floor_z_mm)
    clamped_link0_z_m = max(MIN_LINK0_Z_M, min(MAX_LINK0_Z_M, link0_z_m))
    return link0_m_to_floor_z_mm(clamped_link0_z_m)


def decide_returning_policy(condition, cycle_avg_sh):
    lead_type = condition["lead"]
    control_type = condition["control"]
    is_risky = is_risky_posture(cycle_avg_sh)

    if control_type == "None":
        return {
            "mode": "auto",
            "message": "비개입 조건이므로 기존 높이를 그대로 유지합니다.",
            "is_approved_rule": False,
            "is_risky": is_risky,
        }

    if not is_risky:
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
    global running, voice_command

    def run_llm_interface(user_response_text, recommended_floor_z_mm):
        llm_start_time = time.time()
        llm_result = experiment_controller.run_task(
            condition=current_condition,
            sh_angle=shoulder_ang,
            avg_sh_angle=cycle_avg_sh,
            elb_angle=cycle_avg_elb,
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
        # RETURNING 평가가 끝난 뒤 다음 cycle에 쓸 target_z를 계산하고,
        # 필요하면 새 pass goal JSON을 저장/전송한다.
        nonlocal trial_count, current_tighten_z_mm, awaiting_worker_answer, completion_sent

        trial_count += 1
        metrics["completed_transfers"] = trial_count
        metrics["total_rula_score"] += cycle_avg_rula
        cycle_duration = time.time() - cycle_start_time if cycle_start_time else 0.0
        cycle_durations.append(cycle_duration)

        if not user_response_text:
            user_response_text = f"Tightened with avg shoulder {cycle_avg_sh:.1f} deg"

        control_type = current_condition["control"]
        if control_type == "Rule":
            target_shoulder_angle_deg = cycle_avg_sh - 20.0
        else:
            target_shoulder_angle_deg = 20.0

        recommended_floor_z_mm = compute_recommended_floor_z_mm(
            user_shoulder_height_cm * 10,
            l1_cm * 10,
            l2_cm * 10,
            target_shoulder_angle_deg,
        )

        if control_type == "LLM":
            llm_result, latency = run_llm_interface(user_response_text, recommended_floor_z_mm)
            llm_latencies.append(latency)
            final_floor_z_m = llm_result.get("final_z_m", current_tighten_z_mm / 1000.0)
            next_target_z_mm = final_floor_z_m * 1000.0
            is_approved = llm_result.get("is_approved", is_approved_rule)
            is_correction = llm_result.get("is_correction", False)
            is_invalid = llm_result.get("is_invalid", False)
        elif control_type == "Rule" and is_approved_rule:
            latency = 0.0
            next_target_z_mm = recommended_floor_z_mm
            is_approved = True
            is_correction = False
            is_invalid = False
        else:
            latency = 0.0
            next_target_z_mm = current_tighten_z_mm
            is_approved = False
            is_correction = False
            is_invalid = False

        next_target_z_mm = clamp_floor_z_mm_to_robot_limits(next_target_z_mm)
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
            # Receiver key is kept for compatibility, but the value is link0 기준 m.
            target_z_link0_m = floor_z_mm_to_link0_m(next_target_z_mm)
            SendPassGoal({"target_z_mm": target_z_link0_m, "msg": f"Trial {trial_count} Setup"})
        else:
            print("[PASS_GOAL 생략] 이전 목표를 그대로 유지합니다.")

        append_coordinate_log(trial_count, current_tighten_z_mm)

        raw_file_exists = os.path.isfile(RAW_CSV_FILENAME)
        with open(RAW_CSV_FILENAME, "a", encoding="utf-8-sig") as f:
            if not raw_file_exists:
                f.write("Time,Condition,Trial_Num,Lead_Type,Control_Type,Avg_Shoulder,Avg_Elbow,RULA,User_Voice,Final_Z_m,Is_Approved,LLM_Latency_s,Is_Invalid\n")
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')},{current_condition['name']},{trial_count},"
                f"{current_condition['lead']},{current_condition['control']},{cycle_avg_sh:.1f},{cycle_avg_elb:.1f},"
                f"{cycle_avg_rula:.1f},{user_response_text},{next_target_floor_z_m:.3f},{is_approved},{latency:.2f},{is_invalid}\n"
            )

        SetReviewPending(False)
        awaiting_worker_answer = False
        completion_sent = False

        if trial_count < TOTAL_TRIALS_PER_CONDITION:
            speak("다음 사이클을 준비합니다. 로봇 상태가 AT_TASK가 되면 다시 측정을 시작합니다.")
            with voice_lock:
                voice_command = None

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

    stt_thread = threading.Thread(target=speech_recognition_thread, daemon=True)
    stt_thread.start()

    experiment_controller = PickAndPlaceExperiment(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)

    cap = cv2.VideoCapture(0)
    mp_pose_instance = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 💡 1. 평가지표에 조기 종료 카운트 추가
    metrics = {
        "completed_transfers": 0, "total_rula_score": 0, "risky_posture_time_sec": 0,
        "system_intervention_count": 0, "robot_adjustment_count": 0,
        "total_adjustment_magnitude_mm": 0.0, "correction_commands_count": 0, "invalid_cmds": 0,
        "early_stop_count": 0  # 조기 종료 카운트
    }
    
    # 💡 2. 로우데이터 헤더에 Early_Stop_Count 열 추가 (f_raw.write 부분 수정)
    raw_exists = os.path.isfile(SAVE_FILENAME)
    f_raw = open(SAVE_FILENAME, "a", encoding="utf-8-sig")
    if not raw_exists:
        f_raw.write("Timestamp,Condition,Trial,State,Shoulder_Ang,Elbow_Ang,RULA,Tighten_Z_mm,"
                    "Sh_X,Sh_Y,Sh_Z,Elb_X,Elb_Y,Elb_Z,Wr_X,Wr_Y,Wr_Z,Hip_X,Hip_Y,Hip_Z,Early_Stop_Count\n")
        f_raw.flush()

    # 💡 3. 8분(480초) 타이머 시작 지점 설정 (while문 진입 직전에 위치)
    MAX_EXPERIMENT_TIME_SEC = 480.0 
    experiment_start_time = time.time()

    cycle_durations = []
    llm_latencies = []

    trial_count = 0
    cycle_start_time = 0.0
    wait_start_time = 0.0
    current_tighten_z_mm = DEFAULT_PASS_FLOOR_Z_CM * 10.0

    accumulated_shoulder_angles = []
    accumulated_elbow_angles = []
    accumulated_rula_scores = []

    cycle_avg_sh = 0.0
    cycle_avg_elb = 0.0
    cycle_avg_rula = 0.0
    completion_sent = False
    key = -1

    awaiting_worker_answer = False

    robot_state = "UNKNOWN"
    previous_robot_state = None
    next_robot_state_poll_time = 0.0

    while cap.isOpened():
        # 💡 1. 8분(480초) 경과 시 루프 강제 종료 로직
        elapsed_time = time.time() - experiment_start_time
        if elapsed_time >= MAX_EXPERIMENT_TIME_SEC:
            print(f"[⏱️ 시간 종료] 8분({MAX_EXPERIMENT_TIME_SEC}초)이 경과되어 실험을 자동 종료합니다.")
            speak("제한 시간 8분이 경과하여 실험을 종료합니다.")
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
            
            # 💡 3. 어깨 각도가 130도 이상일 때만 위험 시간 누적
            if shoulder_ang >= 130.0: 
                metrics["risky_posture_time_sec"] += 0.1

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
            with voice_lock:
                voice_command = None
            if previous_robot_state != "AT_TASK":
                speak("블록의 네 개 구멍에 볼트를 체결해 주세요.")
            cycle_start_time = time.time()
            accumulated_shoulder_angles.clear()
            accumulated_elbow_angles.clear()
            accumulated_rula_scores.clear()
            awaiting_worker_answer = False
            completion_sent = False

        # Sequence 3) AT_TASK 유지: 작업 중 프레임별 자세값을 계속 누적한다.
        if robot_state == "AT_TASK" and shoulder_ang > 0:
            accumulated_shoulder_angles.append(shoulder_ang)
            accumulated_elbow_angles.append(elbow_ang)
            accumulated_rula_scores.append(current_rula)
            if current_rula >= 4:
                metrics["risky_posture_time_sec"] += 0.1

        # Sequence 4) AT_TASK 완료 처리: 작업 완료 음성/키 입력이 들어오면
        # hold_finished를 보내고, 이번 cycle의 평균 자세값을 확정한다.
        if robot_state == "AT_TASK" and not completion_sent:
            is_done, is_early_stop = detect_task_completion(key)
            
            if is_early_stop:
                speak("실험자의 조기 종료 의사가 확인되어 실험을 즉시 중단합니다.")
                metrics["early_stop_count"] = 1
                break  # 🚨 조기 종료 시 여기서 루프를 탈출하여 엑셀 저장으로 넘어갑니다.
                
            elif is_done:
                speak("조립 완료블록을 내려놓습니다.")
                SendHoldFinished()
                cycle_avg_sh, cycle_avg_elb, cycle_avg_rula = compute_cycle_averages(
                    accumulated_shoulder_angles,
                    accumulated_elbow_angles,
                    accumulated_rula_scores,
                )
                completion_sent = True

        # Sequence 5) RETURNING 진입: 작업이 끝나 로봇이 복귀하기 시작한 시점.
        # 여기서 5개 컨디션 중 현재 조건에 맞춰 자동 유지/자동 보정/작업자 질문을 결정한다.
        if entered_returning and completion_sent:
            SetReviewPending(True)
            user_response_text = ""
            policy = decide_returning_policy(current_condition, cycle_avg_sh)
            speak(policy["message"])

            if policy["mode"] == "auto":
                if policy["is_risky"] and current_condition["lead"] == "System":
                    metrics["system_intervention_count"] += 1
                apply_next_target(user_response_text, policy["is_approved_rule"])
            else:
                with voice_lock:
                    voice_command = None
                wait_start_time = time.time()
                awaiting_worker_answer = True

        # Sequence 6) Worker 주도 조건에서만: RETURNING 중 작업자 답변을 기다렸다가
        # LLM 또는 Rule 계산에 반영해 다음 target_z를 만든다.
        if robot_state == "RETURNING" and awaiting_worker_answer:
            answered, user_response_text, manual_yes, manual_no, elapsed_wait = poll_worker_adjust_answer(
                wait_start_time,
                key,
            )
            cv2.putText(frame, f"Waiting Answer... {5.0 - elapsed_wait:.1f}s", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            if answered:
                is_approved_rule = False
                if current_condition["control"] != "LLM":
                    is_approved_rule = resolve_rule_approval(user_response_text, manual_yes, manual_no)
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
        cv2.putText(frame, f"Trial: {trial_count}/{TOTAL_TRIALS_PER_CONDITION}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Live RULA: {current_rula}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Shoulder Angle: {shoulder_ang:.1f} deg", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 💡 [타이머 추가 부분] Shoulder Angle 아래(y좌표 190)에 노란색으로 크게 타이머를 띄웁니다.
        current_elapsed_sec = time.time() - experiment_start_time
        elapsed_mins = int(current_elapsed_sec // 60)
        elapsed_secs = int(current_elapsed_sec % 60)
        timer_text = f"Time: {elapsed_mins:02d}:{elapsed_secs:02d} / 08:00"
        cv2.putText(frame, timer_text, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(frame, "[Manual Override] SPACE: Done | Y: Yes | N: No", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)

        cv2.imshow("HRI Ergonomic Bolt Fastening Task", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break

        previous_robot_state = robot_state

    running = False
    cap.release()
    cv2.destroyAllWindows()

    avg_cycle_time = sum(cycle_durations) / len(cycle_durations) if cycle_durations else 0.0
    avg_rula = metrics["total_rula_score"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0
    avg_llm_latency = sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0
    avg_adj_mm = metrics["total_adjustment_magnitude_mm"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0


    # 💡 횟수가 아닌 '실제 진행된 총 시간(8분 또는 조기종료)'을 바탕으로 평균 작업 시간 산출
    actual_experiment_duration = time.time() - experiment_start_time
    avg_cycle_time = actual_experiment_duration / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0
    
    avg_rula = metrics["total_rula_score"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0
    avg_llm_latency = sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0
    avg_adj_mm = metrics["total_adjustment_magnitude_mm"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"📊 [{current_condition['name']}] 매트릭스 추출 완료 (실제 진행 시간: {actual_experiment_duration:.1f}초)")
    print("=" * 60)

    # 💡 서머리 엑셀 마지막에 Early_Stop_Count 추가
    file_exists = os.path.isfile(SUMMARY_FILENAME)
    with open(SUMMARY_FILENAME, "a", encoding="utf-8-sig") as f:
        if not file_exists:
            f.write("Condition,Completed_Transfers,Avg_Task_Time_s,Avg_RULA,Risky_Time_s,System_Interventions,Adjust_Count,Avg_Adj_mm,Correction_Cmds,Invalid_Cmds,Avg_LLM_Latency_s,Early_Stop_Count\n")
        f.write(f"{current_condition['name']},{metrics['completed_transfers']},{avg_cycle_time:.2f},{avg_rula:.2f},"
                f"{metrics['risky_posture_time_sec']:.2f},{metrics['system_intervention_count']},"
                f"{metrics['robot_adjustment_count']},{avg_adj_mm:.1f},{metrics['correction_commands_count']},"
                f"{metrics['invalid_cmds']},{avg_llm_latency:.2f},{metrics['early_stop_count']}\n")

    speak("수고하셨습니다. 실험이 종료되었습니다.")


if __name__ == "__main__":
    main()