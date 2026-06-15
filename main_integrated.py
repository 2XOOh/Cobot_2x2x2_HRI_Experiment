#main_integrated.py (메인 비전 및 음성 소통 엔진 - Indy7 통신 버전) 구글 STT가 인식한 발화 원문을 가져와 LLM에 그대로 넘기도록 수정했으며, 교수님이 주신 엑셀 매트릭스 지표(사이클 수, 소요시간, 위험지속시간, 개입/번복 횟수 등)를 완벽하게 카운팅하여 로그에 남김
import cv2
import time
import json
import os 
import pyttsx3                  
import speech_recognition as sr 
import socket                   
import threading                

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

CURRENT_CONDITION = {
    "intervention": "개입",   
    "lead": "작업자",         
    "control": "llm"          
}

DEFAULT_PASS_FLOOR_Z_CM = 135.5
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
SAVE_FILENAME = os.path.join(RESULT_DIR, "experiment_log.json")

gripper_open_event = threading.Event()

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
                        print("\n[로봇 신호 수신] 그리퍼 열림 데이터 파싱 완료!")
                except ConnectionResetError:
                    break
            client_socket.close()
            print("[연결 종료] 로봇 연결 끊어짐.")
    except Exception as e:
        print(f"[소켓 에러] {e}")
    finally:
        server_socket.close()

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160) 
    engine.say(text)
    engine.runAndWait()

def listen_for_command():
    """작업자의 음성을 텍스트 원문으로 반환 (LLM 뉘앙스 분석용)"""
    r = sr.Recognizer()
    r.energy_threshold = 300 
    r.dynamic_energy_threshold = True 
    
    with sr.Microphone() as source:
        print("\n🎤 [마이크 열림] 대답해 주세요...")
        try:
            # 4초 대기 타임아웃, 발화 제한 5초 (무한 대기 방지)
            audio = r.listen(source, timeout=4.0, phrase_time_limit=5.0) 
            print("[음성 수신 완료] 텍스트 변환 중...")
            text = r.recognize_google(audio, language='ko-KR')
            print(f"🗣️ [인식 성공 결과]: '{text}'")
            return text
        except sr.WaitTimeoutError:
            print("⚠️ [음성 대기 시간 초과]")
            return ""
        except sr.UnknownValueError:
            print("⚠️ [음성 파싱 불가] 소리가 너무 작거나 인지 실패")
            return ""
        except Exception as e:
            print(f"🚨 [마이크 시스템 에러] {e}")
            return ""

def check_positive_keywords(text):
    """(요구사항 3 적용) Rule-based 구체화된 긍정 판단 리스트"""
    clean_text = text.replace(" ", "").strip()
    positive_keywords = ["응", "어", "네", "예", "조정", "그래", "해줘", "맞아", "오케이", "ok", "좋아"]
    return any(keyword in clean_text for keyword in positive_keywords)

def main():
    if not os.path.exists(RESULT_DIR): os.makedirs(RESULT_DIR)
    print("\n" + "="*40)
    try:
        h_sh = float(input("1. 바닥~어깨 높이 (cm): "))
        l1 = float(input("2. 상완 길이 (cm): "))
        l2 = float(input("3. 하완 길이 (cm): "))
        initial_pass_floor_z_cm = float(input("4. 초기 pass 높이 (cm): "))
    except ValueError:
        h_sh, l1, l2 = 130.0, 28.0, 24.0
        initial_pass_floor_z_cm = DEFAULT_PASS_FLOOR_Z_CM

    ik_engine = RobotIKManager(
        h_shoulder_cm=h_sh, l1_cm=l1, l2_cm=l2, current_pass_floor_z_cm=initial_pass_floor_z_cm
    )
    controller = PickAndPlaceExperiment(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)
   
    socket_thread = threading.Thread(target=indy7_socket_listener, daemon=True)
    socket_thread.start()

    # --- [요구사항 5] 프로젝트 매트릭스 지표 초기화 ---
    metrics = {
        "completed_transfers": 0,
        "robot_adjustment_count": 0,
        "system_intervention_count": 0,
        "correction_commands_count": 0,
        "invalid_failed_command_count": 0,
        "total_adjustment_magnitude_mm": 0.0,
        "risky_posture_total_time_sec": 0.0,
    }
    cycle_start_time = time.time()
    risky_start_time = None

    cap = cv2.VideoCapture(0)
    speak("실험 시스템이 시작되었습니다. 로봇의 그리퍼 작동을 대기합니다.")
   
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        lead_type = CURRENT_CONDITION["lead"]
        control_type = CURRENT_CONDITION["control"]
       
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = results.pose_landmarks.landmark
                
                if lm[12].visibility > 0.5 and lm[14].visibility > 0.5 and lm[16].visibility > 0.5:
                    
                    (
                        sh_deg, avg_sh_deg, elb_deg, adj_mm, target_pass_floor_z_mm
                    ) = ik_engine.calculate_ik(lm[12], lm[14], lm[16], control_type=control_type)
                   
                    is_risky = (avg_sh_deg > 60.0)
                    
                    # [매트릭스 지표] 위험 자세 지속 시간 누적
                    if is_risky and risky_start_time is None:
                        risky_start_time = time.time()
                    elif not is_risky and risky_start_time is not None:
                        metrics["risky_posture_total_time_sec"] += (time.time() - risky_start_time)
                        risky_start_time = None

                    cv2.putText(frame, f"Avg Armpit Angle: {avg_sh_deg:.1f}", (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_risky else (0, 255, 0), 2)
                    cv2.putText(frame, f"Waiting Indy7 Robot Signal...", (30, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if gripper_open_event.is_set():
                        gripper_open_event.clear()
                        
                        # [매트릭스 지표] 완료된 Task(Transfer) 누적 및 사이클 타임 계산
                        metrics["completed_transfers"] += 1
                        task_completion_time = time.time() - cycle_start_time
                        cycle_start_time = time.time()
                        
                        if is_risky:
                            user_voice_text = ""
                            is_approved_rule = False
                            cmd_latency_start = time.time()
                            
                            if lead_type == "시스템":
                                speak("조정하겠습니다.")
                                metrics["system_intervention_count"] += 1
                                is_approved_rule = True
                                
                            elif lead_type == "작업자":
                                speak("조정할까요?")
                                user_voice_text = listen_for_command()
                                
                                # 룰베이스일 경우 긍정 단어 리스트로 필터링
                                if control_type != "llm":
                                    is_approved_rule = check_positive_keywords(user_voice_text)

                            # [요구사항 4] LLM에게 음성 텍스트를 전달하여 뉘앙스/번복 명령 최종 해석
                            final_json_str, llm_metrics, final_z_m = controller.run_task(
                                condition=CURRENT_CONDITION, sh_angle=sh_deg, avg_sh_angle=avg_sh_deg, elb_angle=elb_deg,
                                target_pass_floor_z_mm=target_pass_floor_z_mm, adj_mm=adj_mm, 
                                current_pass_floor_z_mm=ik_engine.current_pass_floor_z_mm, h_sh=h_sh, l1=l1, l2=l2,
                                user_voice_text=user_voice_text, is_approved_rule=is_approved_rule
                            )
                            
                            # LLM이 해석을 끝낸 뒤 TTS 출력
                            if lead_type == "작업자":
                                final_approved = llm_metrics.get("is_approved", False) if control_type == "llm" else is_approved_rule
                                if final_approved:
                                    speak("네, 위치를 조정하겠습니다.")
                                else:
                                    speak("현재 위치를 유지합니다.")
                            
                            cmd_latency = time.time() - cmd_latency_start

                            # [매트릭스 지표] 각종 데이터 갱신
                            is_finally_approved = llm_metrics.get("is_approved", is_approved_rule)
                            if is_finally_approved:
                                metrics["robot_adjustment_count"] += 1
                                metrics["total_adjustment_magnitude_mm"] += abs(adj_mm)
                            
                            if llm_metrics.get("is_correction"): metrics["correction_commands_count"] += 1
                            if llm_metrics.get("is_invalid"): metrics["invalid_failed_command_count"] += 1
                            
                            print("\n[Indy7 로봇 전송용 데이터 출력 (JSON)]")
                            print(final_json_str)
                            SendPassGoal(final_json_str)
                            
                            # [요구사항 5] 엑셀 매트릭스와 연동되는 종합 로그 기록
                            log_data = {
                                "time": time.strftime('%Y-%m-%d %H:%M:%S'),
                                "condition": CURRENT_CONDITION,
                                "is_approved": is_finally_approved,
                                "robot_payload": json.loads(final_json_str),
                                "metrics": {
                                    "task_completion_time_sec": round(task_completion_time, 2),
                                    "completed_transfers": metrics["completed_transfers"],
                                    "robot_adjustment_count": metrics["robot_adjustment_count"],
                                    "system_intervention_count": metrics["system_intervention_count"],
                                    "command_to_action_latency_sec": round(cmd_latency, 2),
                                    "correction_commands_count": metrics["correction_commands_count"],
                                    "invalid_failed_command_count": metrics["invalid_failed_command_count"],
                                    "risky_posture_total_time_sec": round(metrics["risky_posture_total_time_sec"], 2)
                                }
                            }
                            
                            with open(SAVE_FILENAME, "a", encoding="utf-8") as f:
                                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                                print("✅ [시스템] 데이터 전송 및 지표 로그 저장 완료!")
                             
                            ik_engine.current_pass_floor_z_mm = (final_z_m * 1000) + LINK0_HEIGHT_MM
                        else:
                            print("[안내] 안전 각도(60도 이하)이므로 조정 없음")

            cv2.imshow('HRI Experiment System', frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()