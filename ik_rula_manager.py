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
        
        # mm 단위 변환
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

        # 인체공학적 이상적인 목표 깊이 (어깨 20도, 팔꿈치 0도 안정적인 어깨각도로 팔을 쭉 뻗었을때 자세 기준)
        self.Z_ideal = self.H_shoulder_mm - (
            self.L1 * math.cos(math.radians(20))
            + self.L2 * math.cos(math.radians(0))
        )

    def calculate_angles_and_target(self, sh_deg, elb_deg, control_type="llm"):
        self.angle_window.append((sh_deg, elb_deg))
        avg_sh_deg = sum(w[0] for w in self.angle_window) / len(self.angle_window)
        
        # 기하학 역기하 계산을 위한 암 보정
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
        
        # 부드러운 하드웨어 조작을 위한 EMA 지수 평활 필터링
        if self.prev_target_pass_floor_z_mm is None:
            target_pass_floor_z_mm = raw_target_pass_floor_z_mm
        else:
            target_pass_floor_z_mm = (
                raw_target_pass_floor_z_mm * self.alpha_smoothing
            ) + (
                self.prev_target_pass_floor_z_mm * (1.0 - self.alpha_smoothing)
            )
        
        self.prev_target_pass_floor_z_mm = target_pass_floor_z_mm
        return target_pass_floor_z_mm