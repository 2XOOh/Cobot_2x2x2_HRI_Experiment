# main_integrated.py
import cv2
import time
import json
import os 
import pyttsx3                  
import speech_recognition as sr 
import socket                   
import threading                
import math
from datetime import datetime  # 💡 실시간 밀리초 단위 로깅을 위해 추가

try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except (ImportError, AttributeError):
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

from ik_rula_manager import RobotIKManager
from experiment_controller import PickAndPlaceExperiment
from hri_http_sender import SendPassGoal
from real_robot_gripper_source import start_real_robot_gripper_listener

# =========================================================
# API 및 환경 설정
# =========================================================
OPENAI_API_KEY = "apikey" # ⚠️ 여기에 실제 Groq API 키를 다시 입력하세요!
LLAMA_BASE_URL = "https://api.groq.com/openai/v1" 

TOTAL_TRIALS_PER_CONDITION = 10 

# 로봇 하드웨어 안전 한계치 (LINK0 634mm + 로봇 최대 한계 1200mm)
MAX_HARDWARE_Z_MM = 1834.0 

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SAVE_FILENAME = os.path.join(RESULT_DIR, "experiment_log_detailed.json")
SUMMARY_FILENAME = os.path.join(RESULT_DIR, "experiment_summary_matrix.csv")
RAW_CSV_FILENAME = os.path.join(RESULT_DIR, "experiment_raw_data_per_trial.csv")

# 💡 [새로 추가된 진짜 로우데이터 저장 파일]
TRUE_RAW_CSV_FILENAME = os.path.join(RESULT_DIR, "experiment_time_series_raw.csv")

CONDITIONS = {
    1: {"intervention": "Intervention", "lead": "System", "control": "LLM", "name": "Cond1_Sys_LLM"},
    2: {"intervention": "Intervention", "lead": "System", "control": "Rule", "name": "Cond2_Sys_Rule"},
    3: {"intervention": "Intervention", "lead": "Worker", "control": "LLM", "name": "Cond3_Worker_LLM"},
    4: {"intervention": "Intervention", "lead": "Worker", "control": "Rule", "name": "Cond4_Worker_Rule"},
    5: {"intervention": "Non-Intervention", "lead": "System", "control": "None", "name": "Cond5_Control_NoInterv"}
}

gripper_open_event = threading.Event()
voice_command = None
voice_lock = threading.Lock()
running = True
coordinate_logs = []

def speak(text):
    print(f"[TTS] {text}")
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

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
            except sr.WaitTimeoutError: continue
            except Exception: continue

def calculate_angle(a, b, c):
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot_product = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba == 0 or mag_bc == 0: return 0.0
    cosine_angle = max(-1.0, min(1.0, dot_product / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine_angle))

def estimate_rula_score(shoulder_angle, elbow_angle):
    score = 1
    if shoulder_angle > 45: score += 2
    elif shoulder_angle > 20: score += 1
    if elbow_angle < 60 or elbow_angle > 100: score += 1
    return min(score, 7)

def main():
    global voice_command, running

    print("\n" + "="*60)
    print(" 🧑‍🔧 피실험자 신체 정보 입력 (엔터키를 누르면 괄호 안의 기본값 적용)")
    print("="*60)
    
    try:
        in_h = input(" 1. 작업자 키 (cm) [기본: 175.0]: ")
        USER_HEIGHT_CM = float(in_h) if in_h.strip() else 175.0
    except: USER_HEIGHT_CM = 175.0
    
    try:
        default_sh = USER_HEIGHT_CM - 30.0
        in_sh = input(f" 2. 어깨까지의 높이 (cm) [기본: {default_sh}]: ")
        USER_SHOULDER_HEIGHT_CM = float(in_sh) if in_sh.strip() else default_sh
    except: USER_SHOULDER_HEIGHT_CM = USER_HEIGHT_CM - 30.0

    try:
        in_l1 = input(" 3. 상완 길이 (어깨~팔꿈치, cm) [기본: 30.0]: ")
        L1_CM = float(in_l1) if in_l1.strip() else 30.0
    except: L1_CM = 30.0
    
    try:
        in_l2 = input(" 4. 하완 길이 (팔꿈치~손목, cm) [기본: 25.0]: ")
        L2_CM = float(in_l2) if in_l2.strip() else 25.0
    except: L2_CM = 25.0

    print(f"\n [적용 완료] 키: {USER_HEIGHT_CM}cm | 어깨 높이: {USER_SHOULDER_HEIGHT_CM}cm | 상완: {L1_CM}cm | 하완: {L2_CM}cm")

    print("\n" + "="*60)
    for k, v in CONDITIONS.items(): print(f" [{k}] {v['name']}")
    print("="*60)
    try: choice = int(input("수행할 실험 조건 번호를 입력하세요 (1~5): "))
    except: choice = 1
    CURRENT_CONDITION = CONDITIONS.get(choice, CONDITIONS[1])

    stt_thread = threading.Thread(target=speech_recognition_thread, daemon=True)
    stt_thread.start()
    start_real_robot_gripper_listener(gripper_open_event)

    experiment_controller = PickAndPlaceExperiment(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)

    cap = cv2.VideoCapture(0)
    mp_pose_instance = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    metrics = {
        "completed_transfers": 0, "total_rula_score": 0, "risky_posture_time_sec": 0,
        "system_intervention_count": 0, "robot_adjustment_count": 0,
        "total_adjustment_magnitude_mm": 0.0, "correction_commands_count": 0, "invalid_cmds": 0 
    }
    
    cycle_durations = []
    llm_latencies = [] 
    
    current_state = "INIT_START"
    trial_count = 0
    cycle_start_time = time.time()
    wait_start_time = 0.0
    
    ORIGIN_Z_MM = USER_HEIGHT_CM * 1.1 * 10.0 
    
    # 어깨 180도, 팔을 위로 쭉 뻗었을 때의 초기 높이 계산 로직 적용 (안전 한계치 적용)
    initial_calc_z_mm = (USER_SHOULDER_HEIGHT_CM * 10) - (L1_CM * 10 + L2_CM * 10) * math.cos(math.radians(180.0))
    current_tighten_z_mm = min(MAX_HARDWARE_Z_MM, initial_calc_z_mm)
    print(f"\n[초기 세팅] 어깨 180도 기준 목표 높이: {initial_calc_z_mm:.1f} mm")
    print(f"           (로봇 보호를 위해 실제 세팅된 높이: {current_tighten_z_mm:.1f} mm)\n")

    accumulated_shoulder_angles = []
    accumulated_elbow_angles = []
    accumulated_rula_scores = []
    
    cycle_avg_sh = 0.0
    cycle_avg_elb = 0.0
    cycle_avg_rula = 0.0
    user_response_text = ""
    is_approved_rule = False
    key = -1 

    # 💡 [핵심] 초당 30번씩 찍힐 진짜 로우데이터 파일 열기 준비
    true_raw_exists = os.path.isfile(TRUE_RAW_CSV_FILENAME)
    f_raw = open(TRUE_RAW_CSV_FILENAME, "a", encoding="utf-8-sig")
    if not true_raw_exists:
        f_raw.write("Timestamp,Condition,Trial_Num,State,Shoulder_Angle,Elbow_Angle,Current_RULA,Robot_Target_Z_mm\n")

    while cap.isOpened() and trial_count < TOTAL_TRIALS_PER_CONDITION:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose_instance.process(image_rgb)
        
        shoulder_ang, elbow_ang, current_rula = 0.0, 0.0, 1
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark
            shoulder, elbow, wrist, hip = [lm[12].x, lm[12].y], [lm[14].x, lm[14].y], [lm[16].x, lm[16].y], [lm[24].x, lm[24].y]
            shoulder_ang = calculate_angle(hip, shoulder, elbow)
            elbow_ang = calculate_angle(shoulder, elbow, wrist)
            current_rula = estimate_rula_score(shoulder_ang, elbow_ang)
            if current_rula >= 4: metrics["risky_posture_time_sec"] += 0.1

        # 💡 [핵심] 매 프레임마다 시간, 각도, 현재 상태를 CSV에 밀어넣음 (이게 교수님이 원하시는 진짜 데이터!)
        current_time_ms = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        f_raw.write(f"{current_time_ms},{CURRENT_CONDITION['name']},{trial_count},{current_state},{shoulder_ang:.2f},{elbow_ang:.2f},{current_rula},{current_tighten_z_mm:.1f}\n")

        if current_state == "INIT_START":
            speak(f"[{CURRENT_CONDITION['name']}] 어깨 각도 180도 기준 초기 조립 위치로 이동합니다.")
            SendPassGoal({"target_z_mm": current_tighten_z_mm, "msg": "Initial move"})
            
            coordinate_logs.append({
                "time": time.strftime('%Y-%m-%d %H:%M:%S'), "trial": 0, "target_z_mm": current_tighten_z_mm
            })
            with open(SAVE_FILENAME, "w", encoding="utf-8") as f:
                json.dump(coordinate_logs, f, indent=4)
                
            time.sleep(3.0)
            
            with voice_lock: voice_command = None 
            speak("블록의 다섯 개 구멍에 볼트를 체결해 주세요.")
            current_state = "BOLT_TIGHTENING"
            cycle_start_time = time.time()
            accumulated_shoulder_angles.clear(); accumulated_elbow_angles.clear(); accumulated_rula_scores.clear()

        elif current_state == "BOLT_TIGHTENING":
            if shoulder_ang > 0:
                accumulated_shoulder_angles.append(shoulder_ang)
                accumulated_elbow_angles.append(elbow_ang)
                accumulated_rula_scores.append(current_rula)
            
            local_voice = None
            with voice_lock:
                if voice_command:
                    local_voice = voice_command
                    voice_command = None
            
            kw_list = ["끝", "완료", "다했", "체결", "조립", "다 했", "완료했", "마무리", "오케이"]
            voice_detected = local_voice and any(kw in local_voice for kw in kw_list)
            
            if voice_detected or key == ord(' '):
                if key == ord(' '): print("[수동 조작 감지]: 스페이스바(완료) 눌림")
                else: print(f"[작업 완료 음성 감지]: '{local_voice}'")
                current_state = "RELEASE_BLOCK"

        elif current_state == "RELEASE_BLOCK":
            speak("조립 완료. 그리퍼를 해제합니다. 블록을 내려놓으세요.")
            SendPassGoal({"command": "GRIPPER_OPEN"})
            
            cycle_avg_sh = sum(accumulated_shoulder_angles) / len(accumulated_shoulder_angles) if accumulated_shoulder_angles else 30.0
            cycle_avg_elb = sum(accumulated_elbow_angles) / len(accumulated_elbow_angles) if accumulated_elbow_angles else 80.0
            cycle_avg_rula = sum(accumulated_rula_scores) / len(accumulated_rula_scores) if accumulated_rula_scores else 2.0
            
            time.sleep(2.5)
            current_state = "MOVE_TO_ORIGIN" 

        elif current_state == "MOVE_TO_ORIGIN":
            speak("로봇이 원점으로 복귀하여 대기합니다.")
            SendPassGoal({"target_z_mm": ORIGIN_Z_MM, "msg": "Return to origin"})
            time.sleep(3.0) 
            current_state = "EVALUATE_POSTURE"

        elif current_state == "EVALUATE_POSTURE":
            lead_type = CURRENT_CONDITION["lead"]
            control_type = CURRENT_CONDITION["control"]
            is_risky = (cycle_avg_sh >= 130.0)
            
            user_response_text = ""
            is_approved_rule = False

            if control_type == "None":
                speak("비개입 조건이므로 기존 높호를 그대로 유지합니다.")
                is_approved_rule = False
                current_state = "APPLY_NEXT_TARGET"
            else:
                if is_risky:
                    if lead_type == "System":
                        speak("방금 전 130도 이상의 위험 자세가 감지되어 시스템이 다음 높이를 보정합니다.")
                        metrics["system_intervention_count"] += 1
                        is_approved_rule = True
                        current_state = "APPLY_NEXT_TARGET"
                    elif lead_type == "Worker":
                        speak("방금 전 자세 불편이 감지되었습니다. 높이 조정을 진행할까요?")
                        with voice_lock: voice_command = None 
                        wait_start_time = time.time()
                        current_state = "WAIT_ADJUST_ANSWER"
                else:
                    speak("안전한 자세입니다. 동일한 높이로 다음 사이클을 진행합니다.")
                    is_approved_rule = False
                    current_state = "APPLY_NEXT_TARGET"

        elif current_state == "WAIT_ADJUST_ANSWER":
            elapsed_wait = time.time() - wait_start_time
            cv2.putText(frame, f"Waiting Answer... {5.0 - elapsed_wait:.1f}s", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            with voice_lock:
                if voice_command:
                    user_response_text = voice_command
                    voice_command = None
            
            manual_yes = (key == ord('y') or key == ord('Y'))
            manual_no = (key == ord('n') or key == ord('N'))
            
            if user_response_text or elapsed_wait > 5.0 or manual_yes or manual_no:
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
                
                if CURRENT_CONDITION["control"] != "LLM":
                    if manual_yes: is_approved_rule = True
                    elif manual_no: is_approved_rule = False
                    else:
                        pos_kws = ["응", "어", "네", "예", "조정", "해줘", "맞아", "오케이", "ok", "좋아"]
                        is_approved_rule = any(kw in user_response_text.replace(" ", "") for kw in pos_kws)
                current_state = "APPLY_NEXT_TARGET"

        elif current_state == "APPLY_NEXT_TARGET":
            trial_count += 1
            metrics["completed_transfers"] = trial_count
            metrics["total_rula_score"] += cycle_avg_rula
            cycle_duration = time.time() - cycle_start_time
            cycle_durations.append(cycle_duration)

            if not user_response_text:
                user_response_text = f"Tightened with avg shoulder {cycle_avg_sh:.1f} deg"

            target_sh_angle_for_rule = cycle_avg_sh - 20.0
            recommended_z_mm = (USER_SHOULDER_HEIGHT_CM * 10) - (L1_CM * 10 + L2_CM * 10) * math.cos(math.radians(target_sh_angle_for_rule))

            llm_start_time = time.time()
            llm_result = experiment_controller.run_task(
                condition=CURRENT_CONDITION, sh_angle=shoulder_ang, avg_sh_angle=cycle_avg_sh,
                elb_angle=cycle_avg_elb, target_pass_floor_z_mm=recommended_z_mm, adj_mm=0.0,
                current_pass_floor_z_mm=current_tighten_z_mm, h_sh=USER_SHOULDER_HEIGHT_CM * 10, l1=L1_CM * 10, l2=L2_CM * 10,
                user_voice_text=user_response_text, is_approved_rule=is_approved_rule
            )
            latency = time.time() - llm_start_time
            if CURRENT_CONDITION["control"] == "LLM": llm_latencies.append(latency)
            
            if llm_result.get("is_emergency_stop"):
                speak("위험 상황이 감지되었습니다. 로봇 시스템을 비상 정지합니다.")
                print("🚨 [LLM 결정] 비상 정지(Emergency Stop) 작동!")
                break
                
            if llm_result.get("clarification_question"):
                speak(llm_result["clarification_question"])
                print(f"❓ [LLM 질문]: {llm_result['clarification_question']}")
            
            next_target_z_m = llm_result.get("final_z_m", current_tighten_z_mm / 1000.0)
            next_target_z_mm = next_target_z_m * 1000.0
            
            adj_mag = abs(next_target_z_mm - current_tighten_z_mm)
            metrics["total_adjustment_magnitude_mm"] += adj_mag
            if adj_mag > 10.0: metrics["robot_adjustment_count"] += 1
            if llm_result.get("is_correction"): metrics["correction_commands_count"] += 1
            if llm_result.get("is_invalid"): metrics["invalid_cmds"] += 1
                
            current_tighten_z_mm = next_target_z_mm
            
            SendPassGoal({"target_z_mm": current_tighten_z_mm, "msg": f"Trial {trial_count} Setup"})
            
            coordinate_logs.append({
                "time": time.strftime('%Y-%m-%d %H:%M:%S'), "trial": trial_count, "target_z_mm": current_tighten_z_mm
            })
            with open(SAVE_FILENAME, "w", encoding="utf-8") as f:
                json.dump(coordinate_logs, f, indent=4)
            
            raw_file_exists = os.path.isfile(RAW_CSV_FILENAME)
            with open(RAW_CSV_FILENAME, "a", encoding="utf-8-sig") as f:
                if not raw_file_exists: f.write("Time,Condition,Trial_Num,Lead_Type,Control_Type,Avg_Shoulder,Avg_Elbow,RULA,User_Voice,Final_Z_m,Is_Approved,LLM_Latency_s,Is_Invalid\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{CURRENT_CONDITION['name']},{trial_count},"
                        f"{CURRENT_CONDITION['lead']},{CURRENT_CONDITION['control']},{cycle_avg_sh:.1f},{cycle_avg_elb:.1f},"
                        f"{cycle_avg_rula:.1f},{user_response_text},{next_target_z_m:.3f},{llm_result.get('is_approved', is_approved_rule)},{latency:.2f},{llm_result.get('is_invalid', False)}\n")

            if trial_count < TOTAL_TRIALS_PER_CONDITION:
                speak(f"{trial_count + 1}번째 작업을 위해 로봇이 이동합니다.")
                time.sleep(4.0) 
                
                with voice_lock: voice_command = None
                current_state = "BOLT_TIGHTENING"
                cycle_start_time = time.time()
                accumulated_shoulder_angles.clear(); accumulated_elbow_angles.clear(); accumulated_rula_scores.clear()
            else:
                current_state = "END_EXPERIMENT"
        
        cv2.putText(frame, f"State: {current_state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Trial: {trial_count}/{TOTAL_TRIALS_PER_CONDITION}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Live RULA: {current_rula}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Shoulder Angle: {shoulder_ang:.1f} deg", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "[Manual Override] SPACE: Done | Y: Yes | N: No", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 200), 2)
        
        cv2.imshow("HRI Ergonomic Bolt Fastening Task", frame)
        
        key = cv2.waitKey(10) & 0xFF
        if key == 27: break

    # 💡 루프가 끝나면 찐 로우데이터 파일도 안전하게 닫아줍니다.
    f_raw.close()
    
    running = False
    cap.release()
    cv2.destroyAllWindows()

    avg_cycle_time = sum(cycle_durations) / len(cycle_durations) if cycle_durations else 0.0
    avg_rula = metrics["total_rula_score"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0
    avg_llm_latency = sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0.0
    avg_adj_mm = metrics["total_adjustment_magnitude_mm"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0
    
    print("\n" + "="*60)
    print(f"📊 [{CURRENT_CONDITION['name']}] 매트릭스 추출 완료")
    print("="*60)
    
    file_exists = os.path.isfile(SUMMARY_FILENAME)
    with open(SUMMARY_FILENAME, "a", encoding="utf-8-sig") as f:
        if not file_exists:
            f.write("Condition,Completed_Transfers,Avg_Task_Time_s,Avg_RULA,Risky_Time_s,System_Interventions,Adjust_Count,Avg_Adj_mm,Correction_Cmds,Invalid_Cmds,Avg_LLM_Latency_s\n")
        f.write(f"{CURRENT_CONDITION['name']},{metrics['completed_transfers']},{avg_cycle_time:.2f},{avg_rula:.2f},"
                f"{metrics['risky_posture_time_sec']:.2f},{metrics['system_intervention_count']},"
                f"{metrics['robot_adjustment_count']},{avg_adj_mm:.1f},{metrics['correction_commands_count']},"
                f"{metrics['invalid_cmds']},{avg_llm_latency:.2f}\n")
    
    speak("수고하셨습니다. 실험이 완료되었습니다.")

if __name__ == "__main__":
    main()