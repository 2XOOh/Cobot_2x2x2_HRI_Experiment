#ik_rula_manager.py (물리 연산 엔진) 평가 지표에 포함된 'Elbow angle(팔꿈치 각도)' 값을 산출하여 리턴하도록 함수에 연산 로직을 추가
import math
from collections import deque

class RobotIKManager:
    def __init__(
        self,
        h_shoulder_cm,
        l1_cm,
        l2_cm,
        current_pass_floor_z_cm,
        window_size=5,
    ):
        self.raw_h_sh = h_shoulder_cm
        self.raw_l1 = l1_cm
        self.raw_l2 = l2_cm
        
        self.current_pass_floor_z_mm = current_pass_floor_z_cm * 10
        self.H_shoulder_mm = h_shoulder_cm * 10
        self.L1 = l1_cm * 10
        self.L2 = l2_cm * 10
        
        self.prev_target_pass_floor_z_mm = None
        self.alpha_smoothing = 0.2  
        self.angle_window = deque(maxlen=window_size)
        
        self.Delta_Z_buffer = 5.0
        self.FIXED_ADJUST_MM = -50.0

        # 인체공학적 이상적인 목표 깊이 (어깨 20도, 팔꿈치 45도 중립 자세)
        self.Z_ideal = self.H_shoulder_mm - (
            self.L1 * math.cos(math.radians(20))
            + self.L2 * math.cos(math.radians(45))
        )

    def calculate_ik(self, lm_shoulder, lm_elbow, lm_wrist, control_type="llm"):
        # 1. 어깨(겨드랑이) 각도 산출 로직
        dy = lm_elbow.y - lm_shoulder.y
        dx = lm_elbow.x - lm_shoulder.x
        
        sh_angle_rad = math.atan2(dy, dx)
        sh_deg = math.degrees(sh_angle_rad)
        if sh_deg < 0:
            sh_deg += 360
        
        armpit_deg = abs(sh_deg - 90.0)
        self.angle_window.append(armpit_deg)
        avg_sh_deg = sum(self.angle_window) / len(self.angle_window)

        # 2. [추가] 매트릭스 지표 측정용 팔꿈치 각도 산출 (어깨-팔꿈치 벡터와 팔꿈치-손목 벡터 내적)
        v1 = [lm_shoulder.x - lm_elbow.x, lm_shoulder.y - lm_elbow.y, lm_shoulder.z - lm_elbow.z]
        v2 = [lm_wrist.x - lm_elbow.x, lm_wrist.y - lm_elbow.y, lm_wrist.z - lm_elbow.z]
        dot = sum(a*b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a*a for a in v1))
        mag2 = math.sqrt(sum(a*a for a in v2))
        try:
            elb_deg = math.degrees(math.acos(dot / (mag1 * mag2)))
        except ValueError:
            elb_deg = 0.0

        # 3. Z축(높이) 조정 로직
        z_curr_arm = self.H_shoulder_mm - (
            self.L1 * math.cos(math.radians(avg_sh_deg))
            + self.L2 * math.cos(math.radians(45))
        )
        
        if control_type == "llm":
            raw_adjustment_mm = self.Z_ideal - z_curr_arm
            raw_target_pass_floor_z_mm = (
                self.current_pass_floor_z_mm
                - raw_adjustment_mm
                + self.Delta_Z_buffer
            )
        else:
            raw_target_pass_floor_z_mm = (
                self.current_pass_floor_z_mm + self.FIXED_ADJUST_MM
            )
        
        # EMA 필터 적용
        if self.prev_target_pass_floor_z_mm is None:
            target_pass_floor_z_mm = raw_target_pass_floor_z_mm
        else:
            target_pass_floor_z_mm = (
                raw_target_pass_floor_z_mm * self.alpha_smoothing
            ) + (
                self.prev_target_pass_floor_z_mm * (1.0 - self.alpha_smoothing)
            )
            
        self.prev_target_pass_floor_z_mm = target_pass_floor_z_mm
        
        adj_mm = target_pass_floor_z_mm - self.current_pass_floor_z_mm
        
        # 반환값에 elb_deg 추가
        return armpit_deg, avg_sh_deg, elb_deg, adj_mm, target_pass_floor_z_mm