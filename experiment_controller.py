# experiment_controller.py
import json
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

        interv = condition.get("intervention", "Intervention")  
        lead = condition.get("lead", "System")          
        control = condition.get("control", "LLM")       

        current_z_m = round((current_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000.0, 3)
        recommended_z_m = round((target_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000.0, 3)

        # 💡 비개입(None)일 경우 조기 반환
        if control == "None" or interv == "Non-Intervention":
            return {
                "decision_action": "유지",
                "normalized_intent": "비개입 실험 조건",
                "thought": "비개입(통제) 조건이므로 작업자의 자세나 음성에 상관없이 현재 높이를 유지합니다.",
                "is_approved": False,
                "is_correction": False,
                "is_invalid": False,
                "is_emergency_stop": False,
                "final_z_m": current_z_m,
                "clarification_question": "",
                "description": "비개입 조건 유지"
            }

        # 💡 Rule 기반 처리 (계산된 20도 하향 목표인 recommended_z_m 적용)
        if is_approved_rule and control == "Rule":
            final_z_m = max(MIN_Z_LIMIT_M, min(MAX_Z_LIMIT_M, recommended_z_m))
            return {
                "decision_action": "하향",
                "normalized_intent": "어깨 각도 20도 하향 (Rule-based)",
                "thought": "Rule-based 조건 제어. 팔을 뻗은 상태를 가정하고 현재 어깨 각도에서 20도 낮춘 목표를 계산하여 적용합니다.",
                "is_approved": True,
                "is_correction": False,
                "is_invalid": False,
                "is_emergency_stop": False,
                "final_z_m": final_z_m,
                "clarification_question": "",
                "description": f"Rule-based 높이 변경: {final_z_m:.3f}m"
            }
        elif not is_approved_rule and control == "Rule":
            return {
                "decision_action": "유지",
                "normalized_intent": "조정 거부 (Rule-based)",
                "thought": "Rule-based 유지 (작업자 거절 또는 정상 범위).",
                "is_approved": False,
                "is_correction": False,
                "is_invalid": False,
                "is_emergency_stop": False,
                "final_z_m": current_z_m,
                "clarification_question": "",
                "description": "Rule-based 유지"
            }

        # 💡 LLM 프롬프트 (130도 기준 상향, JSON 스키마 구조 최신화)
        system_prompt = f"""
        당신은 인간-로봇 상호작용(HRI) 실험을 통제하는 인공지능 제어 뇌입니다.
        현재 작업자는 볼트 조립을 마치고 로봇이 대기 위치로 돌아간 상태에서 다음 조립 위치 조절 여부를 결정해야 합니다.

        [시스템 제어 및 안전 사양 가이드라인]
        1. 로봇 팔 Z 한계치: 최소 {MIN_Z_LIMIT_M}m ~ 최대 {MAX_Z_LIMIT_M}m.
        2. 평균 어깨 각도가 130도 이상이면 작업 영역이 과도하게 높아서 팔이 무리하게 들린 상태이므로 낮춰주어야 인체공학적으로 안전합니다.
        3. 작업자가 '조금' 올려/낮춰 달라고 요구할 경우 3~5cm(0.03~0.05m), '많이'라고 하면 10~15cm(0.10~0.15m) 내외로 계산하여 final_z_m을 결정하세요.
        4. 작업자의 음성이 맥락에 안 맞거나(예: "아 배고파", "어어") 알아들을 수 없으면 is_invalid를 true로 반환하고 높이를 유지하세요.
        5. 작업자가 "아파", "위험해", "멈춰", "그만" 등 비상 정지나 위험을 나타내는 말을 하면 is_emergency_stop을 true로 반환하세요.
        6. 작업자의 요구가 너무 모호하여 정확한 판단이 불가능하면 clarification_question에 작업자에게 되물을 질문을 한글로 작성하세요.

        반드시 아래의 JSON 스키마 규격을 충족하는 순수한 JSON 오브젝트 1개만 출력하세요. 
        {{
           "decision_action": "유지, 상향, 하향, 비상정지, 재질문 중 택1",
           "normalized_intent": "작업자의 명확한 의도 요약",
           "thought": "관절 각도(130도 기준) 분석 및 음성 의도 파악에 대한 판단 근거 요약",
           "is_approved": true 또는 false,
           "is_correction": true 또는 false,
           "is_invalid": true 또는 false,
           "is_emergency_stop": true 또는 false,
           "final_z_m": 최종 결정된 로봇 팔 목표 Z 좌표 값 (단위: m),
           "clarification_question": "되물을 질문 (없으면 빈 문자열 표기)",
           "description": "결과 요약"
        }}
        """

        user_content = f"""
        - 조건: {interv} / 주도: {lead} / 방식: {control}
        - 현재 높이: {current_z_m:.3f}m | 인체공학 추천 높이: {recommended_z_m:.3f}m
        - 이전 사이클 어깨/팔꿈치 평균 각도: {avg_sh_angle:.1f}도 / {elb_angle:.1f}도
        - 작업자 음성: "{user_voice_text}"
        """

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            parsed_json = json.loads(response.choices[0].message.content.strip())
            
            # API 통신 성공 시
            if "final_z_m" in parsed_json:
                z = float(parsed_json["final_z_m"])
                parsed_json["final_z_m"] = max(MIN_Z_LIMIT_M, min(MAX_Z_LIMIT_M, z))
                
            # 기본값 방어 로직
            if "is_invalid" not in parsed_json: parsed_json["is_invalid"] = False
            if "is_emergency_stop" not in parsed_json: parsed_json["is_emergency_stop"] = False
            if "clarification_question" not in parsed_json: parsed_json["clarification_question"] = ""
            
            return parsed_json
            
        except Exception as e:
            return {
                "decision_action": "유지",
                "normalized_intent": "API 에러",
                "thought": f"API 연산 예외(오류): {e}",
                "is_approved": False,
                "is_correction": False,
                "is_invalid": True, 
                "is_emergency_stop": False,
                "final_z_m": current_z_m,
                "clarification_question": "",
                "description": "API 통신 오류로 현재 위치 유지"
            }