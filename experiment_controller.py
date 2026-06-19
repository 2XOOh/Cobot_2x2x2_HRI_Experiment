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

        interv = condition.get("intervention", "개입")  
        lead = condition.get("lead", "시스템")          
        control = condition.get("control", "llm")       

        current_z_m = round((current_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000.0, 3)
        recommended_z_m = round((target_pass_floor_z_mm - LINK0_HEIGHT_MM) / 1000.0, 3)

        # 💡 비개입(none)일 경우 조기 반환
        if control == "none" or interv == "비개입":
            return {
                "thought": "비개입(통제) 조건 유지.",
                "is_approved": False,
                "is_correction": False,
                "is_invalid": False,
                "final_z_m": current_z_m,
                "description": "비개입 조건 유지"
            }

        # 💡 Rule 기반 처리
        if is_approved_rule and control == "rule":
            final_z_m = max(MIN_Z_LIMIT_M, min(MAX_Z_LIMIT_M, recommended_z_m))
            return {
                "thought": "Rule-based 조건 제어.",
                "is_approved": True,
                "is_correction": False,
                "is_invalid": False,
                "final_z_m": final_z_m,
                "description": f"Rule-based 높이 변경: {final_z_m:.3f}m"
            }
        elif not is_approved_rule and control == "rule":
            return {
                "thought": "Rule-based 유지.",
                "is_approved": False,
                "is_correction": False,
                "is_invalid": False,
                "final_z_m": current_z_m,
                "description": "Rule-based 유지"
            }

        # LLM 프롬프트 (오류 판독 요소 추가)
        system_prompt = f"""
        당신은 인간-로봇 상호작용(HRI) 실험을 통제하는 인공지능 제어 뇌입니다.
        현재 작업자는 볼트 조립을 마치고 로봇이 대기 위치로 돌아간 상태에서 다음 조립 위치 조절 여부를 결정해야 합니다.

        [시스템 제어 및 안전 사양 가이드라인]
        1. 로봇 팔 Z 한계치: 최소 {MIN_Z_LIMIT_M}m ~ 최대 {MAX_Z_LIMIT_M}m.
        2. 평균 어깨 각도가 90도 이상이면 작업 영역이 높아서 팔이 들린 상태이므로 낮춰주어야 인체공학적으로 안전합니다.
        3. 작업자의 음성이 맥락에 안 맞거나(예: "아 배고파", "어어"), 알아들을 수 없는 말이면 is_invalid를 true로 반환하고 높이를 유지({current_z_m}m)하세요.

        반드시 아래의 JSON 스키마 규격을 충족하는 순수한 JSON 오브젝트 1개만 출력하세요. 
        {{
           "thought": "관절 각도 분석 및 음성 의도 파악 요약",
           "is_approved": true 또는 false,
           "is_correction": true 또는 false (음성을 통한 임의 보정 발생 여부),
           "is_invalid": true 또는 false (명령 인식 실패, 엉뚱한 대답 여부),
           "final_z_m": 최종 결정된 로봇 팔 목표 Z 좌표 값,
           "description": "결과 요약"
        }}

        [Few-Shot Examples]
        User State: Z추천 0.850m. 음성: "어 올려줘 아니 잠깐만 그냥 둬"
        Assistant: {{ "thought": "번복하였으므로 최종 의도는 거절이다.", "is_approved": false, "is_correction": false, "is_invalid": false, "final_z_m": {current_z_m:.3f}, "description": "작업자 거절(번복) 유지." }}

        User State: Z추천 1.300m. 음성: "오늘 점심 뭐 먹지"
        Assistant: {{ "thought": "작업과 무관한 발화이므로 명령 인식 실패로 간주한다.", "is_approved": false, "is_correction": false, "is_invalid": true, "final_z_m": {current_z_m:.3f}, "description": "잘못된 명령어 인식으로 인한 유지." }}
        """

        user_content = f"""
        - 조건: {interv} / 주도: {lead} / 방식: {control}
        - 현재 높이: {current_z_m:.3f}m | 추천 높이: {recommended_z_m:.3f}m
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
            if "is_invalid" not in parsed_json: parsed_json["is_invalid"] = False
            return parsed_json
            
        except Exception as e:
            # 💡 API 에러 발생 시 is_invalid 카운트 증가
            return {
                "thought": f"API 연산 예외(오류): {e}",
                "is_approved": False,
                "is_correction": False,
                "is_invalid": True, 
                "final_z_m": current_z_m,
                "description": "API 통신 오류로 현재 위치 유지"
            }