# experiment_controller.py
import json
import re
from datetime import datetime
from openai import OpenAI

LINK0_HEIGHT_MM = 634.0
MIN_Z_LIMIT_M = 0.634  
MAX_Z_LIMIT_M = 1.200  

class PickAndPlaceExperiment:
    def __init__(self, api_key, base_url=None):
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def run_task(
        self, condition, sh_angle, avg_sh_angle, elb_angle, target_pass_floor_z_mm, 
        adj_mm, current_pass_floor_z_mm, h_sh, l1, l2, user_voice_text="", is_approved_rule=False
    ):
        timestamp_iso = datetime.now().astimezone().isoformat(timespec="milliseconds")

        interv = condition.get("intervention", "개입")  
        lead = condition.get("lead", "시스템")          
        control = condition.get("control", "llm")       

        current_z_m = round((current_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000, 3)
        calculated_target_z_m = round((target_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000, 3)

        llm_metrics = {"is_approved": False, "is_correction": False, "is_invalid": False}

        if interv == "비개입":
            final_z_m = current_z_m
            msg = f"[No-Intervention] 위험 각도({avg_sh_angle:.1f}도) 감지되었으나 비개입 조건이므로 최종 좌표 {final_z_m:.3f}m를 유지합니다."
        else:
            if control == "llm":
                response_json = self._analyze_intent_gpt(lead, avg_sh_angle, calculated_target_z_m, current_z_m, user_voice_text)
                
                if response_json:
                    final_z_m = response_json.get("final_z_m", current_z_m)
                    msg = response_json.get("description", "위험 각도에 따라 조정합니다.")
                    llm_metrics["is_approved"] = response_json.get("is_approved", False)
                    llm_metrics["is_correction"] = response_json.get("is_correction", False)
                else:
                    final_z_m = current_z_m
                    msg = "[LLM Error] 음성 분석 실패로 현재 위치를 유지합니다."
                    llm_metrics["is_invalid"] = True
            else:
                llm_metrics["is_approved"] = is_approved_rule
                if is_approved_rule:
                    final_z_m = calculated_target_z_m
                    msg = f"[Rule-Base] 동의 수신. 최종 목표 Z 좌표({final_z_m:.3f}m)로 이동합니다."
                else:
                    final_z_m = current_z_m
                    msg = f"[Rule-Base] 거절 수신. 현재 Z 좌표({final_z_m:.3f}m)를 유지합니다."

        result_json = {
            "frame_id": "link0",
            "timestamp": timestamp_iso,
            "armpit_angle_deg": round(avg_sh_angle, 1), 
            "elbow_angle_deg": round(elb_angle, 1),
            "position": {
                "x": 0.45,
                "y": 0.00,
                "z": final_z_m
            },
            "orientation": {
                "x": 0.0,
                "y": 0.9239,
                "z": 0.0,
                "w": 0.3827
            },
            "description": msg  
        }
        
        return json.dumps(result_json, indent=2, ensure_ascii=False), llm_metrics, final_z_m

    def _analyze_intent_gpt(self, lead, avg_sh_angle, calculated_target_z_m, current_z_m, user_text):
        prompt = f"""
        [시스템 역할 및 페르소나 (Persona)]
        당신은 작업자의 근골격계 질환을 예방하기 위해 인체공학적 자세를 분석하고 로봇의 작업 공간(Z좌표)을 제어하는 '안전 제어 AI 관제 엔진'입니다.
        당신의 목표는 작업자의 음성 명령 의도를 정확히 파악하고(번복 명령 포함), 관절 한계값을 준수하여 최종 로봇 이동 좌표를 결정하는 것입니다.

        [로봇 및 환경 현재 상태 (State)]
        - 15초 구간 평균 겨드랑이 각도: {avg_sh_angle:.1f}도 (인간공학적 위험 임계치: 60도 초과 시 위험)
        - 현재 로봇의 Z 좌표: {current_z_m:.3f}m
        - 시스템이 계산한 안전 복귀용 추천 Z 좌표: {calculated_target_z_m:.3f}m
        - 로봇 Z축 이동 한계 (Min/Max): 최소 {MIN_Z_LIMIT_M}m ~ 최대 {MAX_Z_LIMIT_M}m (범위 밖 이동 시 실패 조건)
        - 작업자 음성 명령: "{user_text if user_text else '음성 발화 없음(또는 시스템 주도)'}"

        [안전 기반 판단 및 의도 분석 지침 (Safety & Intent Rules)]
        1. 긍정/동의 뉘앙스 파악: 작업자의 명령에 "응, 어, 네, 예, 조정, 그래, 해줘, 맞아, 오케이, ok, 좋아" 등의 키워드나 이와 비슷한 뉘앙스가 포함되어 있다면 긍정(진행)으로 판단하세요.
        2. 구체적 수치 금지 제약 극복: 작업자가 구체적으로 "몇 cm 올려/내려"라고 말하지 않아도 됩니다. 뉘앙스만 동의하면 시스템 추천 좌표({calculated_target_z_m:.3f}m)로 이동을 진행합니다.
        3. 번복 명령 (최우선 순위): 발화 중 명령을 번복하는 경우(예: "해줘... 아 아니야 하지마"), 반드시 가장 마지막에 발화된 의도를 추출하여 최종 제어로 이어지게 하세요. 
        4. 범위 실패 조건: 결정된 Z 좌표가 Min/Max 범위를 벗어나면, 이동을 거부하고(현재 좌표 유지) 이유를 명시하세요.

        [생각 과정 (Auto Chain-of-Thought)]
        반드시 JSON 응답 내 "thought" 필드에 단계별 판단 논리(번복 여부 판단 -> 한계값 범위 체크 -> 최종 승인 결정)를 가장 먼저 작성하세요.
        
        [출력 포맷 (오직 JSON만 출력)]
        {{
            "thought": "판단 근거 작성",
            "is_approved": true 또는 false,
            "is_correction": true 또는 false, 
            "final_z_m": 0.000,
            "description": "상황에 맞는 최종 결과 1문장"
        }}

        [Few-Shot Examples]
        User State: Z추천 0.850m. 음성: "어 올려줘 아니 잠깐만 그냥 둬"
        Assistant: {{ "thought": "명령 초반에 '올려줘'라고 했으나, 마지막에 '그냥 둬'라고 번복하였으므로 최종 의도는 부정/거절이다. 안전 한계 범위는 통과했으나 작업자 거절로 취소.", "is_approved": false, "is_correction": false, "final_z_m": {current_z_m:.3f}, "description": "작업자 거절(번복)로 인해 현재 위치를 유지합니다." }}

        User State: Z추천 1.300m. 음성: "응 오케이 해줘"
        Assistant: {{ "thought": "긍정적인 뉘앙스 동의 확인. 하지만 추천 좌표(1.300m)가 최대 한계값(1.200m)을 초과하므로 실패 조건에 해당하여 이동을 취소한다.", "is_approved": false, "is_correction": false, "final_z_m": {current_z_m:.3f}, "description": "작업자가 동의하였으나 로봇 이동 한계(1.200m)를 초과하여 이동을 취소합니다." }}
        """

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant", 
                messages=[
                    {"role": "system", "content": "너는 오직 지정된 JSON 포맷으로만 답변하는 로봇 관제 엔진이다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            raw_content = re.sub(r'^```json\s*', '', raw_content)
            raw_content = re.sub(r'\s*```$', '', raw_content)
            
            parsed_json = json.loads(raw_content)
            return parsed_json
        except Exception as e:
            print(f"🚨 [LLM JSON 파싱 에러]: {e}")
            return None