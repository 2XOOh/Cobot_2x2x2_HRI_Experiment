# ik_rula_manager.py
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
        self.LINK0_HEIGHT_MM = 634.0
        
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

    def calculate_rula_score(self, shoulder_angle_deg):
        """
        어깨(Upper Arm) 굴곡 각도 기반 단순화된 RULA 점수 산출
        (기준: 0~20도 = 1점, 20~45도 = 2점, 45~90도 = 3점, 90도 이상 = 4점)
        """
        if shoulder_angle_deg <= 20.0:
            return 1
        elif 20.0 < shoulder_angle_deg <= 45.0:
            return 2
        elif 45.0 < shoulder_angle_deg <= 90.0:
            return 3
        else:
            return 4

    def calculate_ik(self, lm_shoulder, lm_elbow, lm_wrist, control_type="llm"):
        # 1. 어깨(겨드랑이) 각도 산출
        dy = lm_elbow.y - lm_shoulder.y
        dx = lm_elbow.x - lm_shoulder.x
        
        sh_angle_rad = math.atan2(dy, dx)
        sh_deg = math.degrees(sh_angle_rad)
        if sh_deg < 0:
            sh_deg += 360
        
        armpit_deg = abs(sh_deg - 90.0)
        self.angle_window.append(armpit_deg)
        avg_sh_deg = sum(self.angle_window) / len(self.angle_window)

        # 2. 매트릭스 지표용 팔꿈치 각도 산출 (3차원 벡터 내적)
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
        
        # 4. EMA 필터 적용
        if self.prev_target_pass_floor_z_mm is None:
            target_pass_floor_z_mm = raw_target_pass_floor_z_mm
        else:
            target_pass_floor_z_mm = (
                raw_target_pass_floor_z_mm * self.alpha_smoothing
            ) + (
                self.prev_target_pass_floor_z_mm * (1.0 - self.alpha_smoothing)
            )
            
        # 5. 로봇팔 하한선 강제 방어벽
        if target_pass_floor_z_mm < self.LINK0_HEIGHT_MM:
            target_pass_floor_z_mm = self.LINK0_HEIGHT_MM
            
        self.prev_target_pass_floor_z_mm = target_pass_floor_z_mm
        adj_mm = target_pass_floor_z_mm - self.current_pass_floor_z_mm
        
        return armpit_deg, avg_sh_deg, elb_deg, adj_mm, target_pass_floor_z_mm