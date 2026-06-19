# main_integrated.py
import cv2
import time
import json
import os 
import pyttsx3                  # TTS (음성 출력)
import speech_recognition as sr # STT (음성 인식)
import socket                   # 로봇과의 TCP/IP 통신용
import threading                # 비전 카메라와 음성 인식을 동시에 처리하기 위한 스레드
import math

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
OPENAI_API_KEY = "apikey"
LLAMA_BASE_URL = "https://api.groq.com/openai/v1" 

# 작업자 정보 및 실험 필수 상수 설정
USER_HEIGHT_CM = 175.0          
USER_SHOULDER_HEIGHT_CM = 145.0 
L1_CM = 30.0                    
L2_CM = 25.0                    

TOTAL_TRIALS_PER_CONDITION = 10 # 한 컨디션당 10회

DEFAULT_PASS_FLOOR_Z_CM = 135.5
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

SAVE_FILENAME = os.path.join(RESULT_DIR, "experiment_log_detailed.json")
SUMMARY_FILENAME = os.path.join(RESULT_DIR, "experiment_summary.csv")
RAW_CSV_FILENAME = os.path.join(RESULT_DIR, "experiment_raw_data_per_trial.csv")

# 5가지 실험 조건 매트릭스
CONDITIONS = {
    1: {"intervention": "개입", "lead": "시스템", "control": "llm", "name": "Cond1_Sys_LLM"},
    2: {"intervention": "개입", "lead": "시스템", "control": "rule", "name": "Cond2_Sys_Rule"},
    3: {"intervention": "개입", "lead": "작업자", "control": "llm", "name": "Cond3_Worker_LLM"},
    4: {"intervention": "개입", "lead": "작업자", "control": "rule", "name": "Cond4_Worker_Rule"},
    5: {"intervention": "비개입", "lead": "시스템", "control": "none", "name": "Cond5_Control_NoInterv"}
}

# 공유 변수 및 이벤트
gripper_open_event = threading.Event()
voice_command = None
voice_lock = threading.Lock()
running = True

def speak(text):
    print(f"[TTS] {text}")
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# STT 스레드
def speech_recognition_thread():
    global voice_command
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    print("[STT] 음성 인식 대기 중...")
    while running:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            try:
                audio = recognizer.listen(source, timeout=1.0, phrase_time_limit=4.0)
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
    
    print("="*60)
    for k, v in CONDITIONS.items(): print(f" [{k}] {v['name']}")
    print("="*60)
    try:
        choice = int(input("수행할 실험 조건 번호를 입력하세요 (1~5): "))
        CURRENT_CONDITION = CONDITIONS[choice]
    except:
        CURRENT_CONDITION = CONDITIONS[1]

    stt_thread = threading.Thread(target=speech_recognition_thread, daemon=True)
    stt_thread.start()
    start_real_robot_gripper_listener(gripper_open_event)

    ik_manager = RobotIKManager(h_shoulder_cm=USER_SHOULDER_HEIGHT_CM, l1_cm=L1_CM, l2_cm=L2_CM, current_pass_floor_z_cm=DEFAULT_PASS_FLOOR_Z_CM)
    experiment_controller = PickAndPlaceExperiment(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)

    cap = cv2.VideoCapture(0)
    mp_pose_instance = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    metrics = {
        "completed_transfers": 0,
        "risky_posture_time_sec": 0,
        "total_rula_score": 0,
        "robot_adjustment_count": 0,
        "system_intervention_count": 0,
        "correction_commands_count": 0,
        "total_adjustment_magnitude_mm": 0.0
    }
    
    cycle_durations = []
    
    # 상태 제어 관련 변수들
    current_state = "INIT_START"
    trial_count = 0
    cycle_start_time = time.time()
    wait_start_time = 0.0
    
    # 위치 변수: 로봇 원점 높이 & 현재 작업 높이 분리
    ORIGIN_Z_MM = USER_HEIGHT_CM * 1.1 * 10.0 
    current_tighten_z_mm = DEFAULT_PASS_FLOOR_Z_CM * 10.0

    accumulated_shoulder_angles = []
    accumulated_elbow_angles = []
    accumulated_rula_scores = []
    
    cycle_avg_sh = 0.0
    cycle_avg_elb = 0.0
    cycle_avg_rula = 0.0
    user_response_text = ""
    is_approved_rule = False

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

        # =========================================================
        # 🤖 1. 상태: 실험 시작 및 최초 위치 하강
        # =========================================================
        if current_state == "INIT_START":
            speak(f"[{CURRENT_CONDITION['name']}] 첫 번째 조립 위치로 이동합니다.")
            SendPassGoal({"target_z_mm": current_tighten_z_mm, "msg": "Initial move to tighten pos"})
            time.sleep(3.0)
            
            with voice_lock: voice_command = None 
            speak("블록의 다섯 개 구멍에 볼트를 체결해 주세요.")
            current_state = "BOLT_TIGHTENING"
            cycle_start_time = time.time()
            accumulated_shoulder_angles.clear(); accumulated_elbow_angles.clear(); accumulated_rula_scores.clear()

        # =========================================================
        # 🤖 2. 상태: 조립 작업 및 각도 모니터링
        # =========================================================
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
            
            if local_voice and any(kw in local_voice for kw in ["끝", "완료", "다했", "체결", "조립"]):
                print(f"[작업 완료 음성 감지]: '{local_voice}'")
                current_state = "RELEASE_BLOCK"

        # =========================================================
        # 🤖 3. 상태: 블록 해제
        # =========================================================
        elif current_state == "RELEASE_BLOCK":
            speak("조립 완료. 그리퍼를 해제합니다. 블록을 내려놓으세요.")
            SendPassGoal({"command": "GRIPPER_OPEN"})
            
            cycle_avg_sh = sum(accumulated_shoulder_angles) / len(accumulated_shoulder_angles) if accumulated_shoulder_angles else 30.0
            cycle_avg_elb = sum(accumulated_elbow_angles) / len(accumulated_elbow_angles) if accumulated_elbow_angles else 80.0
            cycle_avg_rula = sum(accumulated_rula_scores) / len(accumulated_rula_scores) if accumulated_rula_scores else 2.0
            
            time.sleep(2.5)
            current_state = "MOVE_TO_ORIGIN" 

        # =========================================================
        # 🤖 4. 상태: 원점 복귀 (다음 블록 대기)
        # =========================================================
        elif current_state == "MOVE_TO_ORIGIN":
            speak("로봇이 원점으로 복귀하여 대기합니다.")
            SendPassGoal({"target_z_mm": ORIGIN_Z_MM, "msg": "Return to origin standby"})
            time.sleep(3.0) 
            current_state = "EVALUATE_POSTURE"

        # =========================================================
        # 🤖 5. 상태: 원점에서 자세 평가 및 조절 여부 질의
        # =========================================================
        elif current_state == "EVALUATE_POSTURE":
            lead_type = CURRENT_CONDITION["lead"]
            control_type = CURRENT_CONDITION["control"]
            is_risky = (cycle_avg_sh >= 90.0) 
            
            user_response_text = ""
            is_approved_rule = False

            # [조건 분기] 비개입(none) 조건이면 자세와 상관없이 무시
            if control_type == "none":
                speak("비개입 조건이므로 기존 높이를 그대로 유지합니다.")
                is_approved_rule = False
                current_state = "APPLY_NEXT_TARGET"
            else:
                if is_risky:
                    if lead_type == "시스템":
                        speak("방금 전 위험 자세가 감지되어 시스템이 다음 높이를 보정합니다.")
                        metrics["system_intervention_count"] += 1
                        is_approved_rule = True
                        current_state = "APPLY_NEXT_TARGET"
                    elif lead_type == "작업자":
                        speak("방금 전 자세 불편이 감지되었습니다. 높이 조정을 진행할까요?")
                        with voice_lock: voice_command = None 
                        wait_start_time = time.time()
                        current_state = "WAIT_ADJUST_ANSWER"
                else:
                    speak("안전한 자세입니다. 동일한 높이로 다음 사이클을 진행합니다.")
                    is_approved_rule = False
                    current_state = "APPLY_NEXT_TARGET"

        # =========================================================
        # 🤖 6. 상태: 작업자 질의 응답 대기 (화면 유지)
        # =========================================================
        elif current_state == "WAIT_ADJUST_ANSWER":
            elapsed_wait = time.time() - wait_start_time
            cv2.putText(frame, f"Waiting Answer... {5.0 - elapsed_wait:.1f}s", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            with voice_lock:
                if voice_command:
                    user_response_text = voice_command
                    voice_command = None
            
            if user_response_text or elapsed_wait > 5.0:
                if user_response_text: print(f"[작업자 답변]: '{user_response_text}'")
                else: print("[대답 없음] 기본값(미승인)으로 진행합니다.")
                
                if CURRENT_CONDITION["control"] != "llm":
                    pos_kws = ["응", "어", "네", "예", "조정", "해줘", "맞아", "오케이", "ok", "좋아"]
                    is_approved_rule = any(kw in user_response_text.replace(" ", "") for kw in pos_kws)
                current_state = "APPLY_NEXT_TARGET"

        # =========================================================
        # 🤖 7. 상태: 높이 계산 후 하강
        # =========================================================
        elif current_state == "APPLY_NEXT_TARGET":
            trial_count += 1
            metrics["completed_transfers"] = trial_count
            metrics["total_rula_score"] += cycle_avg_rula
            cycle_duration = time.time() - cycle_start_time
            cycle_durations.append(cycle_duration)

            if not user_response_text:
                user_response_text = f"평균 어깨 각도 {cycle_avg_sh:.1f}도로 체결함"

            llm_result = experiment_controller.run_task(
                condition=CURRENT_CONDITION, sh_angle=shoulder_ang, avg_sh_angle=cycle_avg_sh,
                elb_angle=cycle_avg_elb, target_pass_floor_z_mm=current_tighten_z_mm, adj_mm=0.0,
                current_pass_floor_z_mm=current_tighten_z_mm, h_sh=USER_SHOULDER_HEIGHT_CM * 10, l1=L1_CM * 10, l2=L2_CM * 10,
                user_voice_text=user_response_text, is_approved_rule=is_approved_rule
            )
            
            next_target_z_m = llm_result.get("final_z_m", current_tighten_z_mm / 1000.0)
            next_target_z_mm = next_target_z_m * 1000.0
            
            # 메트릭스 계산
            adj_mag = abs(next_target_z_mm - current_tighten_z_mm)
            metrics["total_adjustment_magnitude_mm"] += adj_mag
            if adj_mag > 10.0: metrics["robot_adjustment_count"] += 1
            if llm_result.get("is_correction"): metrics["correction_commands_count"] += 1
                
            current_tighten_z_mm = next_target_z_mm
            ik_manager.current_pass_floor_z_mm = next_target_z_mm
            
            SendPassGoal({"target_z_mm": current_tighten_z_mm, "msg": f"Trial {trial_count} Setup"})
            
            # 로우데이터 CSV 저장
            raw_file_exists = os.path.isfile(RAW_CSV_FILENAME)
            with open(RAW_CSV_FILENAME, "a", encoding="utf-8") as f:
                if not raw_file_exists: f.write("Time,Condition,Trial_Num,Lead_Type,Control_Type,Avg_Shoulder,Avg_Elbow,RULA,User_Voice,Final_Z_m,Is_Approved\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{CURRENT_CONDITION['name']},{trial_count},"
                        f"{CURRENT_CONDITION['lead']},{CURRENT_CONDITION['control']},{cycle_avg_sh:.1f},{cycle_avg_elb:.1f},"
                        f"{cycle_avg_rula:.1f},{user_response_text},{next_target_z_m:.3f},{llm_result.get('is_approved', is_approved_rule)}\n")

            if trial_count < TOTAL_TRIALS_PER_CONDITION:
                speak(f"{trial_count + 1}번째 작업을 위해 로봇이 목표 위치로 하강합니다.")
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
        cv2.imshow("HRI Ergonomic Bolt Fastening Task", frame)
        
        if cv2.waitKey(10) & 0xFF == 27: break

    running = False
    cap.release()
    cv2.destroyAllWindows()

    avg_cycle_time = sum(cycle_durations) / len(cycle_durations) if cycle_durations else 0.0
    avg_rula = metrics["total_rula_score"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0
    
    print("\n" + "="*60)
    print(f"📊 [{CURRENT_CONDITION['name']}] 총 {metrics['completed_transfers']}회 조립 실험 종료")
    print(f" - 평균 소요 시간: {avg_cycle_time:.2f} 초 | 총평균 RULA: {avg_rula:.2f} 점")
    print("="*60)
    
    # 종합 요약 CSV 저장 (엑셀 매트릭스 양식 완벽 대응)
    file_exists = os.path.isfile(SUMMARY_FILENAME)
    with open(SUMMARY_FILENAME, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Time,Condition,Total_Trials,Avg_Task_Time(s),Avg_RULA,Adjust_Count,System_Interventions,Correction_Cmds,Total_Adj_mm,Risky_Duration(s)\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{CURRENT_CONDITION['name']},{metrics['completed_transfers']},"
                f"{avg_cycle_time:.2f},{avg_rula:.2f},{metrics['robot_adjustment_count']},"
                f"{metrics['system_intervention_count']},{metrics['correction_commands_count']},"
                f"{metrics['total_adjustment_magnitude_mm']:.1f},{metrics['risky_posture_time_sec']:.2f}\n")
    
    speak("수고하셨습니다. 실험 세션이 완료되었습니다.")

if __name__ == "__main__":
    main()