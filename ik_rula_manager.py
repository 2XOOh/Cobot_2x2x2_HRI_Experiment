#ik_rula_manager.py (물리 연산 엔진)
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
        # 컨트롤러로 다시 넘겨주기 위해 원본 cm 데이터 저장 (LLM 프롬프트용)
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
        self.Z_ideal = (self.L1 * math.cos(math.radians(20))) + \
                       (self.L2 * math.cos(math.radians(45)))

    def calculate_ik(self, sh, el, wr, control_type="llm"):
        dy1, dx1 = el.y - sh.y, abs(el.x - sh.x)
        sh_rad = math.atan2(dx1, dy1)
        sh_deg = math.degrees(sh_rad) # 겨드랑이 각도 산출
        
        dy2, dx2 = wr.y - el.y, abs(wr.x - el.x)
        el_rad = math.atan2(dx2, dy2)
        
        self.angle_window.append(sh_deg)
        avg_sh_deg = sum(self.angle_window) / len(self.angle_window)
        
        z_curr_arm = (self.L1 * math.cos(sh_rad)) + (self.L2 * math.cos(el_rad))
        
        # [수정됨] 60도 초과 등의 위험 판단은 메인 루프에 위임하고, 
        # 이 계산 엔진은 "현재 각도 기준으로 목표까지 몇 mm 이동해야 하는지"를 무조건 산출하여 대기합니다.
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
        
        # 1. 노이즈를 막기 위한 지수 이동 평균(EMA) 필터 적용
        if self.prev_target_pass_floor_z_mm is None:
            target_pass_floor_z_mm = raw_target_pass_floor_z_mm
        else:
            target_pass_floor_z_mm = (
                raw_target_pass_floor_z_mm * self.alpha_smoothing
            ) + (
                self.prev_target_pass_floor_z_mm * (1.0 - self.alpha_smoothing)
            )
            
        # 2. [수정됨] 로봇팔이 내려갈 수 있는 최저 높이(63.4cm) 하한선 강제 적용
        MIN_Z_LIMIT_MM = 634.0
        if target_pass_floor_z_mm < MIN_Z_LIMIT_MM:
            target_pass_floor_z_mm = MIN_Z_LIMIT_MM
            
        # 3. [수정됨] 하한선(63.4cm)에 걸렸을 경우를 대비해 최종 이동량(adjustment) 재계산
        # 이래야 LLM 프롬프트에 들어가는 "얼마나 움직였는지" 값이 거짓말을 안 함
        final_adjustment_mm = self.current_pass_floor_z_mm - target_pass_floor_z_mm
        
        self.prev_target_pass_floor_z_mm = target_pass_floor_z_mm
        
        # 메인 루프가 필요할 때 언제든 값을 가져다 쓸 수 있도록 4개의 주요 데이터를 리턴
        return sh_deg, avg_sh_deg, final_adjustment_mm, target_pass_floor_z_mm