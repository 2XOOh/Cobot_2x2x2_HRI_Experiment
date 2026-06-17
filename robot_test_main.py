# robot_test_main.py
import cv2
import time
import json
import socket
import threading
import pyttsx3
import speech_recognition as sr

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
# 기본 세팅 및 API (API 키를 반드시 입력하세요!)
# =========================================================
OPENAI_API_KEY = "api-key"
LLAMA_BASE_URL = "https://api.groq.com/openai/v1" 

CONDITIONS = {
    1: {"intervention": "개입", "lead": "시스템", "control": "llm", "name": "Cond1_Sys_LLM"},
    2: {"intervention": "개입", "lead": "시스템", "control": "rule", "name": "Cond2_Sys_Rule"},
    3: {"intervention": "개입", "lead": "작업자", "control": "llm", "name": "Cond3_Worker_LLM"},
    4: {"intervention": "개입", "lead": "작업자", "control": "rule", "name": "Cond4_Worker_Rule"},
    5: {"intervention": "비개입", "lead": "시스템", "control": "none", "name": "Cond5_Control_NoInterv"}
}

gripper_open_event = threading.Event()
tts_lock = threading.Lock()

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
        print(f"📡 [통신 대기중] 더미/실제 로봇 접속 대기 (Port:{PORT})")
        while True:
            client_socket, addr = server_socket.accept()
            while True:
                try:
                    data = client_socket.recv(1024)
                    if not data: break
                    msg = data.decode('utf-8').strip()
                    if msg == "GRIPPER_OPEN":
                        gripper_open_event.set()
                        print("\n🔔 [로봇 신호 감지!] 현재 카메라의 자세 각도를 바탕으로 즉시 연산을 시작합니다.")
                except ConnectionResetError: break
            client_socket.close()
    except Exception as e: print(f"[소켓 에러] {e}")
    finally: server_socket.close()

def speak_voice(text):
    def _speak():
        with tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
                engine.say(text)
                engine.runAndWait()
            except RuntimeError: pass
    threading.Thread(target=_speak, daemon=True).start()

def listen_for_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n🎤 [마이크 대기] 의도를 말씀해 주세요 (예: '응 해줘', '아니 싫어')...")
        try:
            audio = r.listen(source, timeout=4.0, phrase_time_limit=5.0) 
            text = r.recognize_google(audio, language='ko-KR')
            print(f"🗣️ [음성 인식 완료]: '{text}'")
            return text
        except:
            print("⚠️ [인식 실패] 음성이 감지되지 않았습니다.")
            return ""

def check_positive_keywords(text):
    clean_text = text.replace(" ", "").strip()
    return any(keyword in clean_text for keyword in ["응", "어", "네", "예", "조정", "그래", "해줘", "맞아", "오케이", "ok", "좋아", "이동"])

# =========================================================
# 메인 테스트 루프 (카메라 기반)
# =========================================================
def main():
    print("🚀 [카메라 연동] 초경량 로봇 구동 테스트 모드 시작")
    
    # 1. 통신 리스너 시작 (더미 TCP + 실제 리눅스 신호 모두 대기)
    threading.Thread(target=indy7_socket_listener, daemon=True).start()
    start_real_robot_gripper_listener(gripper_open_event)

    # 2. 제어 모듈 초기화
    controller = PickAndPlaceExperiment(api_key=OPENAI_API_KEY, base_url=LLAMA_BASE_URL)
    # 실험 편의를 위해 신체 사이즈는 기본값으로 고정 (130cm, 28cm, 24cm, 135.5cm)
    h_sh, l1, l2, initial_pass_floor_z_cm = 130.0, 28.0, 24.0, 135.5
    ik_engine = RobotIKManager(h_shoulder_cm=h_sh, l1_cm=l1, l2_cm=l2, current_pass_floor_z_cm=initial_pass_floor_z_cm)

    # 3. 컨디션 선택 루프
    while True:
        print("\n" + "="*50)
        print("🤖 카메라를 켤 실험 조건을 선택하세요")
        for k, v in CONDITIONS.items(): print(f"  [{k}] {v['name']}")
        print("  [0] 테스트 종료")
        print("="*50)
        
        choice = input("👉 번호 입력 (0~5): ")
        if choice == '0': break
        if not choice.isdigit() or int(choice) not in CONDITIONS: continue
        
        CURRENT_CONDITION = CONDITIONS[int(choice)]
        lead_type = CURRENT_CONDITION["lead"]
        control_type = CURRENT_CONDITION["control"]
        
        print(f"\n▶ [{CURRENT_CONDITION['name']}] 카메라를 켭니다. (종료하려면 화면 클릭 후 ESC)")
        cap = cv2.VideoCapture(0)
        
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb_frame)

                current_frame_sh_deg, elb_deg, target_pass_floor_z_mm, adj_mm = 0.0, 0.0, ik_engine.current_pass_floor_z_mm, 0.0
                
                # 뼈대 인식 및 실시간 각도 연산
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    lm = results.pose_landmarks.landmark
                    sh_deg, current_frame_sh_deg, elb_deg, adj_mm, target_pass_floor_z_mm = ik_engine.calculate_ik(lm[12], lm[14], lm[16], control_type)

                # ==================================================
                # 핵심! 신호가 오면 이 순간의 각도로 판단을 진행합니다
                # ==================================================
                if gripper_open_event.is_set():
                    gripper_open_event.clear()
                    
                    print(f"\n⏹ [순간 캡처 완료] 현재 오른쪽 어깨 각도: {current_frame_sh_deg:.1f}도")
                    is_risky = (current_frame_sh_deg >= 60.0)
                    final_json_str = '{"description": "안전 각도(60도 미만) 유지로 인한 로봇 이동 없음"}'
                    
                    if is_risky:
                        user_voice_text = ""
                        is_approved_rule = False
                        
                        if lead_type == "시스템":
                            speak_voice("위험 각도가 확인되어 시스템이 개입하여 높이를 보정합니다.")
                            is_approved_rule = True
                        elif lead_type == "작업자":
                            speak_voice("자세 오차가 발견되었습니다. 위치 조정을 진행할까요?")
                            user_voice_text = listen_for_command()  # 마이크로 답변 대기
                            if control_type != "llm":
                                is_approved_rule = check_positive_keywords(user_voice_text)

                        # LLM 의도 해석 및 최종 Z좌표 산출
                        final_json_str, llm_metrics, final_z_m = controller.run_task(
                            condition=CURRENT_CONDITION, sh_angle=current_frame_sh_deg, avg_sh_angle=current_frame_sh_deg,
                            elb_angle=elb_deg, target_pass_floor_z_mm=target_pass_floor_z_mm, adj_mm=adj_mm, 
                            current_pass_floor_z_mm=ik_engine.current_pass_floor_z_mm, h_sh=h_sh, l1=l1, l2=l2,
                            user_voice_text=user_voice_text, is_approved_rule=is_approved_rule
                        )
                        
                        if llm_metrics.get("is_approved", is_approved_rule):
                            speak_voice("네, 안내된 좌표로 로봇을 조정합니다.")
                            ik_engine.current_pass_floor_z_mm = (final_z_m * 1000) + ik_engine.LINK0_HEIGHT_MM
                        else:
                            speak_voice("기존 높이를 유지합니다.")
                            
                    else:
                        print(f"[안전 확인] 각도({current_frame_sh_deg:.1f}도)가 허용 한계(60도) 미만이므로 로봇은 움직이지 않습니다.")

                    print("\n📦 [생성된 로봇 제어 JSON (실시간 전송됨)]")
                    print(final_json_str)
                    
                    # 로그 저장
                    with open("test_robot_log.json", "a", encoding="utf-8") as f:
                        log_entry = {"time": time.strftime('%Y-%m-%d %H:%M:%S'), "condition": CURRENT_CONDITION["name"], "angle": round(current_frame_sh_deg, 1), "robot_payload": json.loads(final_json_str)}
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                        
                    # 로봇으로 전송!
                    SendPassGoal(final_json_str)
                    print("\n👀 카메라 루프로 복귀합니다. 다음 신호를 기다립니다...")

                # 카메라 텍스트 오버레이
                cv2.putText(frame, f"Cond: {CURRENT_CONDITION['name']}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Cur Angle: {current_frame_sh_deg:.1f}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Robot Light Test', frame)
                
                if cv2.waitKey(1) & 0xFF == 27: 
                    break # ESC 누르면 현재 컨디션 종료
                    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()