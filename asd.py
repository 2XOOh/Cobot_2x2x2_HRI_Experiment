import cv2
import math
from collections import deque
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing

class RobotIKManager:
    def __init__(self, h_shoulder_cm, l1_cm, l2_cm, current_robot_cm, window_size=5):
        self.H_shoulder = h_shoulder_cm * 10
        self.L1 = l1_cm * 10
        self.L2 = l2_cm * 10
        self.current_robot_mm = current_robot_cm * 10
        
        self.prev_target_z = None
        self.alpha_smoothing = 0.2
        self.angle_window = deque(maxlen=window_size)
        
        self.Z_ideal_arm = (self.L1 * math.cos(math.radians(20))) + \
                           (self.L2 * math.cos(math.radians(45)))
                           
        self.Z_base_target = self.H_shoulder - self.Z_ideal_arm
        self.Delta_Z_buffer = 5.0 

    def calculate_target_z(self, sh, el, wr):
        # 내부 수학 계산 (유지)
        dy1 = el.y - sh.y  
        dx1 = abs(el.x - sh.x)
        sh_vertical_rad = math.atan2(dx1, dy1) 
        sh_deg = math.degrees(sh_vertical_rad) 
        
        dy2 = wr.y - el.y
        dx2 = abs(wr.x - el.x)
        el_vertical_rad = math.atan2(dx2, dy2)
        
        self.angle_window.append(sh_deg)
        avg_sh_deg = sum(self.angle_window) / len(self.angle_window)
        
        if avg_sh_deg > 30.0:
            raw_target_robot_mm = self.Z_base_target + self.Delta_Z_buffer
        else:
            raw_target_robot_mm = self.current_robot_mm

        if self.prev_target_z is None:
            target_robot_mm = raw_target_robot_mm
        else:
            target_robot_mm = (raw_target_robot_mm * self.alpha_smoothing) + \
                              (self.prev_target_z * (1.0 - self.alpha_smoothing))
        self.prev_target_z = target_robot_mm

        z_curr_arm = (self.L1 * math.cos(sh_vertical_rad)) + (self.L2 * math.cos(el_vertical_rad))
        wrist_height_ground = self.H_shoulder - z_curr_arm
        
        v_es = (sh.x - el.x, sh.y - el.y)
        v_ew = (wr.x - el.x, wr.y - el.y)
        dot_prod = v_es[0] * v_ew[0] + v_es[1] * v_ew[1]
        mag_es = math.sqrt(v_es[0]**2 + v_es[1]**2)
        mag_ew = math.sqrt(v_ew[0]**2 + v_ew[1]**2)
        cosine_angle = max(-1.0, min(1.0, dot_prod / (mag_es * mag_ew + 1e-6)))
        inner_elbow_deg = math.degrees(math.acos(cosine_angle))
        
        el_beta_deg = math.degrees(el_vertical_rad)
        
        return sh_deg, avg_sh_deg, inner_elbow_deg, el_beta_deg, target_robot_mm, z_curr_arm, wrist_height_ground

# --- 시연 세션 시작 ---
print("-" * 50)
print("   [수현님 - 범위 디텍션 + 직접 측정 IK 로봇 제어 시스템]   ")
print("-" * 50)
try:
    h_sh_input = float(input("1. 바닥에서 어깨까지의 실측 높이 (cm): "))
    l1_input = float(input("2. 상완 실측 길이 (cm): "))
    l2_input = float(input("3. 하완 실측 길이 (cm): "))
    r_input = float(input("4. 현재 로봇 초기 작업대 높이 (cm): "))
except ValueError:
    print("입력 오류! 기본 임의값으로 시작합니다.")
    h_sh_input, l1_input, l2_input, r_input = 130, 28, 24, 100

manager = RobotIKManager(h_sh_input, l1_input, l2_input, r_input)
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            
            # 연산은 그대로 진행
            sh_angle, avg_sh_angle, el_inner_angle, el_beta_deg, target_z_mm, z_curr_arm, wrist_height_ground = manager.calculate_target_z(lm[12], lm[14], lm[16])
            
            mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # --- UI 텍스트 시각화 (요청하신 딱 4개 핵심 정보만 출력) ---
            
            # 1. 알파 각도 (겨드랑이)
            cv2.putText(frame, f"1. Alpha Angle (Armpit): {sh_angle:.1f} deg", (30, 40), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 255), 2)
            
            # 2. 베타 각도 (팔꿈치)
            cv2.putText(frame, f"2. Beta Angle (Elbow): {el_beta_deg:.1f} deg", (30, 80), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 0), 2)
            
            # 3. 어깨부터 손목까지 팔이 수직으로 내려간 길이
            cv2.putText(frame, f"3. Arm Vertical Drop Length: {z_curr_arm:.1f} mm", (30, 120), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 130, 255), 2)
            
            # 4. 지면부터 손목 포인트까지의 수직 높이 (전달 높이)
            cv2.putText(frame, f"4. Delivery Height from Ground: {wrist_height_ground:.1f} mm", (30, 160), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 100), 2)

        cv2.imshow('Robot IK Absolute Optimization', frame)
        if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()