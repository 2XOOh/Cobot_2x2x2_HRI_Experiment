#experiment_controller.py (실험 조건 제어 뇌)
import json
from datetime import datetime
from openai import OpenAI

LINK0_HEIGHT_MM = 634.0

class PickAndPlaceExperiment:
    def __init__(self, api_key, base_url=None):
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def run_task(
        self,
        condition,
        sh_angle,
        avg_sh_angle,
        target_pass_floor_z_mm,
        adj_mm,
        current_pass_floor_z_mm,
        h_sh,
        l1,
        l2,
    ):
        timestamp_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")

        interv = condition.get("intervention", "개입")  
        lead = condition.get("lead", "시스템")          
        control = condition.get("control", "llm")       

        # 최종 도달할 실제 로봇 좌표 Z값 계산 (link0 기준, meter 단위)
        final_z_m = round((target_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000, 3)
        no_interv_z_m = round((current_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000, 3)

        if interv == "비개입":
            final_target_pass_floor_z_mm = current_pass_floor_z_mm
            msg = f"[No-Intervention] 위험 각도({avg_sh_angle:.1f}도) 감지되었으나 비개입 조건이므로 최종 좌표 {no_interv_z_m:.3f}m를 유지합니다."
        else:
            final_target_pass_floor_z_mm = target_pass_floor_z_mm
            if control == "llm":
                # [수정] 프롬프트에 상대 이동량(adj_mm) 대신 최종 이동할 좌표 정보(final_z_m)를 주입합니다.
                msg = self._generate_gpt_msg(
                    lead,
                    avg_sh_angle,
                    final_z_m,
                    h_sh,
                    l1,
                    l2,
                    current_pass_floor_z_mm,
                )
            else:
                msg = self._generate_rule_msg(lead, final_z_m)

        # JSON 패키지 빌드
        result_json = {
            "frame_id": "link0",
            "timestamp": timestamp_iso,
            "armpit_angle_deg": round(avg_sh_angle, 1), 
            "position": {
                "x": 0.45,
                "y": 0.00,
                "z": round(
                    (final_target_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000,
                    3,
                )
            },
            "orientation": {
                "x": 0.0,
                "y": 0.9239,
                "z": 0.0,
                "w": 0.3827
            },
            "description": msg  
        }
        
        return json.dumps(result_json, indent=2, ensure_ascii=False)

    def _generate_gpt_msg(self, lead, avg_sh_angle, final_z_m, h_sh, l1, l2, curr_z):
        # [수정] 오직 '최종 이동할 좌표 사실'만 명확하게 한 문장으로 답변하도록 유도
        prompt = f"""
        작업자의 겨드랑이 각도가 {avg_sh_angle:.1f}도로 위험 임계치(60도)를 초과했습니다.
        안전 자세 복귀를 위한 최종 목표 Z 좌표는 {final_z_m:.3f}m 입니다.
        
        [지시사항]
        인체공학 보고서 양식이나 서론/결론, 설명조의 문장을 모두 제외하고 오직 다음 형태로만 딱 한 문장으로 답변하세요:
        "위험 각도({avg_sh_angle:.1f}도)가 감지되어 작업 공간 최종 Z 좌표를 {final_z_m:.3f}m로 조정합니다."
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[
                    {"role": "system", "content": "너는 오직 지정된 최종 목표 좌표 사실만 한 문장으로 명확하게 답변하는 로봇 관제 엔진이다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"위험 각도 감지로 인해 작업 공간 최종 Z 좌표를 {final_z_m:.3f}m로 정량 조정합니다."

    def _generate_rule_msg(self, lead, final_z_m):
        if lead == "시스템":
            return f"[Rule-Base] 60도 초과 감지. 최종 목표 Z 좌표({final_z_m:.3f}m)로 이동합니다."
        else: 
            return f"[Rule-Base] 작업자 동의 수신. 최종 목표 Z 좌표({final_z_m:.3f}m)로 이동합니다."