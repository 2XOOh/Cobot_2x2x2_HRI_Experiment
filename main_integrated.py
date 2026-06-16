# main_integrated.py
import cv2
import time
import json
import os 
import pyttsx3                  # TTS (음성 출력)
import speech_recognition as sr # STT (음성 인식)
import socket                   # 로봇과의 TCP/IP 통신용
import threading                # 비전 카메라와 통신을 동시에 돌리기 위한 스레드

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
OPENAI_API_KEY = "api-key"
LLAMA_BASE_URL = "https://api.groq.com/openai/v1" 

DEFAULT_PASS_FLOOR_Z_CM = 135.5
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
SAVE_FILENAME = os.path.join(RESULT_DIR, "experiment_log_detailed.json")
SUMMARY_FILENAME = os.path.join(RESULT_DIR, "experiment_summary_metrics.csv")

# 5가지 실험 조건 매트릭스
CONDITIONS = {
    1: {"intervention": "개입", "lead": "시스템", "control": "llm", "name": "Cond1_Sys_LLM"},
    2: {"intervention": "개입", "lead": "시스템", "control": "rule", "name": "Cond2_Sys_Rule"},
    3: {"intervention": "개입", "lead": "작업자", "control": "llm", "name": "Cond3_Worker_LLM"},
    4: {"intervention": "개입", "lead": "작업자", "control": "rule", "name": "Cond4_Worker_Rule"},
    5: {"intervention": "비개입", "lead": "시스템", "control": "none", "name": "Cond5_Control_NoInterv"}
}

gripper_open_event = threading.Event()

# =========================================================
# 통신 및 음성 인터페이스
# =========================================================
def indy7_socket_listener():
    HOST = '0.0.0.0' 
    PORT = 9999      
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"\n[통신 대기중] 뉴로메카 Indy7 로봇 접속 대기 (Port:{PORT})")
        while True:
            client_socket, addr = server_socket.accept()
            print(f"[연결 성공] 로봇 접속됨: {addr}")
            while True:
                try:
                    data = client_socket.recv(1024)
                    if not data: break
                    msg = data.decode('utf-8').strip()
                    if msg == "GRIPPER_OPEN":
                        gripper_open_event.set()
                        print("\n[로봇 신호] 그리퍼 열림 -> 15초 데이터 수집 및 태스크 개시!")
                except ConnectionResetError: break
            client_socket.close()
            print("[연결 종료] 로봇 연결 끊어짐.")
    except Exception as e: print(f"[소켓 에러] {e}")
    finally: server_socket.close()

def speak_voice(text):
    """비동기 TTS 음성 출력"""
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

def listen_for_command():
    """작업자 음성 명령 원문 인식 텍스트 반환"""
    r = sr.Recognizer()
    r.energy_threshold = 300 
    r.dynamic_energy_threshold = True 
    with sr.Microphone() as source:
        print("\n🎤 [마이크 대기] 의도를 말씀해 주세요...")
        try:
            audio = r.listen(source, timeout=4.0, phrase_time_limit=5.0) 
            text = r.recognize_google(audio, language='ko-KR')
            print(f"🗣️ [인식 완료]: '{text}'")
            return text
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            print("⚠️ [인식 실패] 음성이 감지되지 않았거나 불명확합니다.")
            return ""
        except Exception as e:
            print(f"🚨 [마이크 에러] {e}")
            return ""

def check_positive_keywords(text):
    """Rule-based 구체화된 긍정 판단 리스트"""
    clean_text = text.replace(" ", "").strip()
    positive_keywords = ["응", "어", "네", "예", "조정", "그래", "해줘", "맞아", "오케이", "ok", "좋아", "이동"]
    return any(keyword in clean_text for keyword in positive_keywords)

# =========================================================
# 메인 실험 루프 및 15초/10분 상태 머신
# =========================================================
def main():
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    
    print("="*60)
    for k, v in CONDITIONS.items(): print(f" [{k}] {v['name']}")
    print("="*60)
    try:
        choice = int(input("수행할 실험 조건 번호를 입력하세요 (1~5): "))
        CURRENT_CONDITION = CONDITIONS[choice]
    except:
        CURRENT_CONDITION = CONDITIONS[5]
    
    try:
        h_sh = float(input("1. 바닥~어깨 높이 (cm): "))
        l1 = float(input("2. 상완 길이 (cm): "))
        l2 = float(input("3. 하완 길이 (cm): "))
        initial_pass_floor_z_cm = float(input("4. 초기 pass 높이 (cm): "))
    except ValueError:
        h_sh, l1, l2, initial_pass_floor_z_cm = 130.0, 28.0, 24.0, DEFAULT_PASS_FLOOR_Z_CM

    ik_engine = RobotIKManager(h_shoulder_cm=h_sh, l1_cm=l1, l2_cm=l2, current_pass_floor_z_cm=initial_pass_floor_z_cm)
    controller = PickAndPlaceExperiment(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)
   
    socket_thread = threading.Thread(target=indy7_socket_listener, daemon=True)
    socket_thread.start()
    
    # 엑셀 매트릭스 평가지표 
    metrics = {
        "completed_transfers": 0,
        "robot_adjustment_count": 0,
        "system_intervention_count": 0,
        "correction_commands_count": 0,
        "invalid_failed_command_count": 0,
        "total_adjustment_magnitude_mm": 0.0,
        "risky_posture_total_time_sec": 0.0,
        "total_rula_score": 0.0 # RULA 누적 점수 추가
    }
    cycle_durations = [] # 사이클당 소요시간 저장 배열

    # 15초 타임라인 상태 머신
    STATE_IDLE = "IDLE"
    STATE_TRACKING = "TRACKING"
    current_state = STATE_IDLE
    cycle_start_time = 0.0
    cycle_angles = []

    # 10분 실험 제한 타이머 (600초)
    CONDITION_DURATION = 600.0  
    condition_start_time = time.time()

    cap = cv2.VideoCapture(0)
    speak_voice(f"실험 세션을 시작합니다. 현재 조건은 {CURRENT_CONDITION['name']} 입니다.")
   
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        lead_type = CURRENT_CONDITION["lead"]
        control_type = CURRENT_CONDITION["control"]
       
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 10분 만료 체크
            elapsed_total = time.time() - condition_start_time
            time_left = max(0.0, CONDITION_DURATION - elapsed_total)
            if time_left <= 0:
                print(f"\n⏱ [시간 종료] 10분이 경과하여 해당 실험 조건이 종료되었습니다.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            current_frame_sh_deg = 0.0
            elb_deg = 0.0
            target_pass_floor_z_mm = ik_engine.current_pass_floor_z_mm
            adj_mm = 0.0
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                
                if lm[12].visibility > 0.5 and lm[14].visibility > 0.5 and lm[16].visibility > 0.5:
                    sh_deg, avg_sh_deg, tmp_elb_deg, tmp_adj_mm, tmp_target_z = ik_engine.calculate_ik(lm[12], lm[14], lm[16], control_type)
                    current_frame_sh_deg = avg_sh_deg
                    elb_deg = tmp_elb_deg
                    adj_mm = tmp_adj_mm
                    target_pass_floor_z_mm = tmp_target_z
                    
            # ---------------------------------------------------------
            # 🔄 1사이클 (15초 구간) 비동기 타임라인 흐름
            # ---------------------------------------------------------
            if current_state == STATE_IDLE:
                if gripper_open_event.is_set():
                    gripper_open_event.clear()
                    current_state = STATE_TRACKING
                    cycle_start_time = time.time() 
                    cycle_angles = []
                    speak_voice("블록을 안정적으로 인계받고 칠판의 점 개수를 세어 주세요.")
                    
            elif current_state == STATE_TRACKING:
                elapsed_task = time.time() - cycle_start_time
                
                if elapsed_task < 15.0:
                    if current_frame_sh_deg > 0:
                        cycle_angles.append(current_frame_sh_deg)
                    
                    if elapsed_task < 5.0:
                        phase_msg = "Phase 1: Stabilization Wait (5s)"
                        color = (0, 255, 255)
                    else:
                        phase_msg = "Phase 2: Counting Red Dots (10s)"
                        color = (0, 165, 255)
                        
                    cv2.putText(frame, f"Task Progress: {elapsed_task:.1f}s / 15.0s", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, phase_msg, (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                else:
                    # 15초 태스크 종료 -> 평균 계산 및 룰라 판단
                    final_avg_sh_angle = sum(cycle_angles) / len(cycle_angles) if cycle_angles else 0.0
                    
                    # [추가] 이번 사이클의 RULA 어깨 상지 위험 점수 추출
                    cycle_rula_score = ik_engine.calculate_rula_score(final_avg_sh_angle)
                    metrics["total_rula_score"] += cycle_rula_score
                    
                    print(f"\n⏹ [15초 태스크 완료] 구간 평균 각도: {final_avg_sh_angle:.1f}도 | RULA 위험 점수: {cycle_rula_score}점")
                    
                    is_risky = (final_avg_sh_angle >= 60.0)
                    is_finally_approved = False
                    
                    if is_risky:
                        metrics["risky_posture_total_time_sec"] += 15.0 
                        user_voice_text = ""
                        is_approved_rule = False
                        
                        if lead_type == "시스템":
                            speak_voice("위험 각도가 확인되어 시스템이 개입하여 높이를 보정합니다.")
                            metrics["system_intervention_count"] += 1
                            is_approved_rule = True
                        elif lead_type == "작업자":
                            speak_voice("자세 오차가 발견되었습니다. 위치 조정을 진행할까요?")
                            user_voice_text = listen_for_command()
                            if control_type != "llm":
                                is_approved_rule = check_positive_keywords(user_voice_text)

                        # LLM 의도 해석 및 제어 타겟 캡슐화 모듈 호출
                        final_json_str, llm_metrics, final_z_m = controller.run_task(
                            condition=CURRENT_CONDITION, sh_angle=current_frame_sh_deg, avg_sh_angle=final_avg_sh_angle,
                            elb_angle=elb_deg, target_pass_floor_z_mm=target_pass_floor_z_mm, adj_mm=adj_mm, 
                            current_pass_floor_z_mm=ik_engine.current_pass_floor_z_mm, h_sh=h_sh, l1=l1, l2=l2,
                            user_voice_text=user_voice_text, is_approved_rule=is_approved_rule
                        )
                        
                        is_finally_approved = llm_metrics.get("is_approved", is_approved_rule)
                        if is_finally_approved:
                            speak_voice("네, 안내된 좌표로 로봇을 조정합니다.")
                            metrics["robot_adjustment_count"] += 1
                            metrics["total_adjustment_magnitude_mm"] += abs(adj_mm)
                            
                            print("\n[Indy7 네트워크 패킷 JSON 전송]")
                            print(final_json_str)
                            SendPassGoal(final_json_str) 
                            
                            ik_engine.current_pass_floor_z_mm = (final_z_m * 1000) + ik_engine.LINK0_HEIGHT_MM
                        else:
                            speak_voice("기존 높이를 유지합니다.")
                        
                        if llm_metrics.get("is_correction"): metrics["correction_commands_count"] += 1
                        if llm_metrics.get("is_invalid"): metrics["invalid_failed_command_count"] += 1
                    else:
                        print(f"[안전 확인] 15초 유지 구간 평균 각도({final_avg_sh_angle:.1f}도)가 허용 한계(60도) 미만입니다.")
                    
                    # 1사이클 전체 태스크 소요 시간 산출 및 지표 기록 완료
                    metrics["completed_transfers"] += 1
                    task_completion_time = time.time() - cycle_start_time
                    cycle_durations.append(task_completion_time)
                    
                    # [추가] 상세 로그 JSON에 rula_score 필드 저장
                    log_data = {
                        "time": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "condition": CURRENT_CONDITION["name"],
                        "cycle_index": metrics["completed_transfers"],
                        "task_completion_time_sec": round(task_completion_time, 2),
                        "avg_shoulder_angle": round(final_avg_sh_angle, 1),
                        "rula_score": cycle_rula_score,
                        "is_risky": is_risky,
                        "is_approved": is_finally_approved
                    }
                    with open(SAVE_FILENAME, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                    
                    current_state = STATE_IDLE

            min_left, sec_left = int(time_left // 60), int(time_left % 60)
            cv2.putText(frame, f"Cond: {CURRENT_CONDITION['name']} | Time Left: {min_left:02d}:{sec_left:02d}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Transfers: {metrics['completed_transfers']} | Cur Frame Angle: {current_frame_sh_deg:.1f}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('HRI Experiment System', frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

    # 10분 통계 매트릭스 CSV 누적 작성 (RULA 평균 점수 추가)
    avg_cycle_time = sum(cycle_durations) / len(cycle_durations) if cycle_durations else 0.0
    avg_rula = metrics["total_rula_score"] / metrics["completed_transfers"] if metrics["completed_transfers"] > 0 else 0.0
    
    print("\n" + "="*50)
    print(f"📊 [{CURRENT_CONDITION['name']}] 10분 실험 세션 종료")
    print(f" - 총 완료 블록 전달 횟수 (Transfers): {metrics['completed_transfers']} 회")
    print(f" - 평균 1사이클 소요 시간 (Task Completion Time): {avg_cycle_time:.2f} 초")
    print(f" - 10분 구간 평균 RULA 점수: {avg_rula:.2f} 점")
    print("="*50)
    
    file_exists = os.path.isfile(SUMMARY_FILENAME)
    with open(SUMMARY_FILENAME, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Time,Condition,Duration(s),Transfers,Avg_Task_Completion_Time(s),Avg_RULA_Score,Adjustments,Interventions,Correction_Cmds,Invalid_Cmds,Total_Adj_mm,Risky_Time(s)\n")
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{CURRENT_CONDITION['name']},{CONDITION_DURATION},"
                f"{metrics['completed_transfers']},{avg_cycle_time:.2f},{avg_rula:.2f},{metrics['robot_adjustment_count']},"
                f"{metrics['system_intervention_count']},{metrics['correction_commands_count']},"
                f"{metrics['invalid_failed_command_count']},{metrics['total_adjustment_magnitude_mm']:.1f},"
                f"{metrics['risky_posture_total_time_sec']:.1f}\n")

if __name__ == "__main__":
    main()