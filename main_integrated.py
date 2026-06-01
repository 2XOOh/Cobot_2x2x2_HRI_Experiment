#main_integrated.py (메인 비전 및 음성 소통 엔진 - Indy7 통신 버전)
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

CURRENT_CONDITION = {
    "intervention": "개입",   
    "lead": "작업자",         # 시스템 / 작업자
    "control": "llm"          # llm 
}

DEFAULT_PASS_FLOOR_Z_CM = 135.5
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
SAVE_FILENAME = os.path.join(RESULT_DIR, "experiment_log.json")

# =========================================================
# 로봇 통신 (TCP/IP Server) 변수 및 스레드 함수
# =========================================================
gripper_open_event = threading.Event()

def indy7_socket_listener():
    HOST = '0.0.0.0' 
    PORT = 9999      
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"\n[통신 대기중] 뉴로메카 Indy7 로봇의 접속을 기다립니다. (Port:{PORT})")
        
        while True:
            client_socket, addr = server_socket.accept()
            print(f"[연결 성공] 로봇 접속됨: {addr}")
            
            while True:
                try:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    
                    msg = data.decode('utf-8').strip()
                    if msg == "GRIPPER_OPEN":
                        gripper_open_event.set()
                        print("\n[로봇 신호 수신] 그리퍼 열림 데이터 파싱 완료!")
                        
                except ConnectionResetError:
                    break
            client_socket.close()
            print("[연결 종료] 로봇과의 연결이 끊어졌습니다. 다시 대기합니다.")
            
    except Exception as e:
        print(f"[소켓 에러] {e}")
    finally:
        server_socket.close()

# =========================================================
# 음성(Voice) 관련 함수
# =========================================================
def speak(text):
    """텍스트를 음성으로 출력합니다."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 160) 
    engine.say(text)
    engine.runAndWait()

def listen_for_yes_no():
    """
    [초반 순정 버전 복구] 목소리를 노이즈로 오인하게 만들던 소음 측정 기능을 제거하고,
    시간 제한을 없애 초반처럼 사용자가 말할 때까지 안정적으로 기다리는 순정 로직입니다.
    """
    r = sr.Recognizer()
    
    # 구글 기본 라이브러리 권장 순정 세팅
    r.energy_threshold = 300 
    r.dynamic_energy_threshold = True 
    
    with sr.Microphone() as source:
        print("\n🎤 [마이크 열림] 대답해 주세요...")
        try:
            # 타임아웃과 글자수 자르기를 모두 제거하여 사용자가 편하게 말할 때까지 기다림
            audio = r.listen(source) 
            print("[음성 수신 완료] 분석 중...")
            
            text = r.recognize_google(audio, language='ko-KR')
            print(f"🗣️ [인식 성공 결과]: '{text}'")
            
            # 긍정 판단 단어 리스트
            clean_text = text.replace(" ", "").strip()
            positive_keywords = ["응", "어", "네", "예", "조정", "그래", "해줘", "맞아", "오케이", "ok", "좋아"]
            
            if any(keyword in clean_text for keyword in positive_keywords):
                return True
                
        except sr.UnknownValueError:
            print("⚠️ [음성 파싱 불가] 소리가 너무 작거나 인지할 수 없는 발음입니다.")
        except Exception as e:
            print(f"🚨 [마이크 시스템 에러] {e}")
            
    return False

# =========================================================
# 메인 비전 루프
# =========================================================
def main():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("\n" + "="*40)
    try:
        h_sh = float(input("1. 바닥~어깨 높이 (cm): "))
        l1 = float(input("2. 상완 길이 (cm): "))
        l2 = float(input("3. 하완 길이 (cm): "))
        initial_pass_floor_z_cm = float(
            input("4. 초기 pass 높이 (바닥 기준, cm): ")
        )
    except ValueError:
        h_sh, l1, l2 = 130.0, 28.0, 24.0
        initial_pass_floor_z_cm = DEFAULT_PASS_FLOOR_Z_CM

    ik_engine = RobotIKManager(
        h_shoulder_cm=h_sh,
        l1_cm=l1,
        l2_cm=l2,
        current_pass_floor_z_cm=initial_pass_floor_z_cm,
    )
    controller = PickAndPlaceExperiment(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)
   
    socket_thread = threading.Thread(target=indy7_socket_listener, daemon=True)
    socket_thread.start()

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
                        sh_deg,
                        avg_sh_deg,
                        adj_mm,
                        target_pass_floor_z_mm,
                    ) = ik_engine.calculate_ik(
                        lm[12],
                        lm[14],
                        lm[16],
                        control_type=control_type,
                    )
                   
                    is_risky = (avg_sh_deg > 60.0)
                    
                    cv2.putText(frame, f"Avg Armpit Angle: {avg_sh_deg:.1f}", (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if is_risky else (0, 255, 0), 2)
                    
                    cv2.putText(frame, f"Waiting Indy7 Robot Socket Signal...", (30, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    if gripper_open_event.is_set():
                        gripper_open_event.clear()
                        
                        if is_risky:
                            is_approved = True
                            
                            if lead_type == "시스템":
                                speak("조정하겠습니다.")
                                is_approved = True
                                
                            elif lead_type == "작업자":
                                speak("조정할까요?")
                                # 순정 초고감도 마이크 리스너 호출
                                if listen_for_yes_no():
                                    speak("네, 위치를 조정하겠습니다.")
                                    is_approved = True
                                else:
                                    speak("현재 위치를 유지합니다.")
                                    is_approved = False
                            
                            # 마이크 인식 실패 또는 거절 시, 이동량을 0으로 설정하여 무조건 정상 로그가 남도록 수정
                            if not is_approved:
                                adj_mm = 0.0
                                target_pass_floor_z_mm = ik_engine.current_pass_floor_z_mm
                            
                            # 동의 여부나 마이크 에러 상태와 상관없이 JSON 생성 및 데이터 전송을 진행
                            final_json_str = controller.run_task(
                                condition=CURRENT_CONDITION, 
                                sh_angle=sh_deg, 
                                avg_sh_angle=avg_sh_deg, 
                                target_pass_floor_z_mm=target_pass_floor_z_mm, 
                                adj_mm=adj_mm, 
                                current_pass_floor_z_mm=ik_engine.current_pass_floor_z_mm,
                                h_sh=h_sh, l1=l1, l2=l2
                            )
                            
                            print("\n[Indy7 로봇 전송용 데이터 출력 (JSON)]")
                            print(final_json_str)
                            SendPassGoal(final_json_str)
                            
                            log_data = {
                                "time": time.strftime('%Y-%m-%d %H:%M:%S'),
                                "condition": CURRENT_CONDITION,
                                "is_approved": is_approved,
                                "robot_payload": json.loads(final_json_str)
                            }
                            
                            with open(SAVE_FILENAME, "a", encoding="utf-8") as f:
                                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
                                print("✅ [시스템] 데이터 전송 및 로그 저장 완료!")
                             
                            # 내부 상태 반영
                            ik_engine.current_pass_floor_z_mm = target_pass_floor_z_mm
                        else:
                            print("[안내] 안전 각도(60도 이하)이므로 조정 없음")

            cv2.imshow('HRI Experiment System', frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()