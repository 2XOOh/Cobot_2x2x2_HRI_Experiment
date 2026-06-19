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

        # 💡 [필터 추가] 조건 5: 비개입(none)일 경우 무조건 현재 높이 유지 및 조기 반환
        if control == "none" or interv == "비개입":
            return {
                "thought": "비개입(통제) 조건이므로 로봇의 높이를 변경하지 않고 기존 높이를 유지합니다.",
                "is_approved": False,
                "is_correction": False,
                "final_z_m": current_z_m,
                "description": "비개입 조건 유지"
            }

        # Rule 기반 즉시 반환 처리 분기
        if is_approved_rule and control == "rule":
            final_z_m = max(MIN_Z_LIMIT_M, min(MAX_Z_LIMIT_M, recommended_z_m))
            return {
                "thought": "Rule-based 조건 제어 규칙에 의거하여 추천 계산값으로 일치 이동합니다.",
                "is_approved": True,
                "is_correction": False,
                "final_z_m": final_z_m,
                "description": f"Rule-based 높이 변경: {final_z_m:.3f}m"
            }
        elif not is_approved_rule and control == "rule":
            return {
                "thought": "Rule-based 조건에서 작업자가 거절하였거나 위험하지 않으므로 위치를 유지합니다.",
                "is_approved": False,
                "is_correction": False,
                "final_z_m": current_z_m,
                "description": "Rule-based 유지"
            }

        # 💡 LLM 연산 프롬프트 빌딩 (Few-Shot 예시 복구)
        system_prompt = f"""
        당신은 인간-로봇 상호작용(HRI) 실험을 통제하는 인공지능 제어 뇌입니다.
        현재 작업자는 볼트 조립을 마치고 로봇이 대기 위치로 돌아간 상태에서 다음 조립 위치 조절 여부를 결정해야 합니다.

        [시스템 제어 및 안전 사양 가이드라인]
        1. 로봇 팔 Z 한계치: 최소 {MIN_Z_LIMIT_M}m ~ 최대 {MAX_Z_LIMIT_M}m.
        2. 평균 어깨 각도(avg_sh_angle)가 90도 이상이면 작업 영역이 높아서 팔이 들린 상태이므로 낮춰주어야 인체공학적으로 안전합니다.
        3. 작업자 음성 내용에 거절, 번복(예: "아니 그냥 둬")이 있으면 is_approved를 false로 하고 높이를 유지({current_z_m}m)하세요.

        반드시 아래의 JSON 스키마 규격을 충족하는 순수한 JSON 오브젝트 1개만 출력하세요. 
        {{
           "thought": "관절 각도 분석 및 음성 의도 파악 요약",
           "is_approved": true 또는 false,
           "is_correction": true 또는 false (음성을 통한 임의 보정 발생 여부),
           "final_z_m": 최종 결정된 로봇 팔 목표 Z 좌표 값,
           "description": "결과 요약"
        }}

        [Few-Shot Examples]
        User State: Z추천 0.850m. 음성: "어 올려줘 아니 잠깐만 그냥 둬"
        Assistant: {{ "thought": "명령 초반에 '올려줘'라고 했으나, 마지막에 '그냥 둬'라고 번복하였으므로 최종 의도는 부정/거절이다. 안전 한계 범위는 통과했으나 작업자 거절로 취소.", "is_approved": false, "is_correction": false, "final_z_m": {current_z_m:.3f}, "description": "작업자 거절(번복)로 인해 현재 위치를 유지합니다." }}

        User State: Z추천 1.300m. 음성: "응 오케이 해줘"
        Assistant: {{ "thought": "긍정적인 뉘앙스 동의 확인. 하지만 추천 좌표(1.300m)가 최대 한계값({MAX_Z_LIMIT_M}m)을 초과하므로 최대 한계값으로 보정하여 이동한다.", "is_approved": true, "is_correction": false, "final_z_m": {MAX_Z_LIMIT_M}, "description": "작업자가 동의하였으나 로봇 이동 한계를 초과하여 한계값으로 보정합니다." }}
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
            
            if "final_z_m" in parsed_json:
                z = float(parsed_json["final_z_m"])
                parsed_json["final_z_m"] = max(MIN_Z_LIMIT_M, min(MAX_Z_LIMIT_M, z))
            return parsed_json
        except Exception as e:
            return {
                "thought": f"API 연산 예외: {e}",
                "is_approved": True,
                "is_correction": False,
                "final_z_m": recommended_z_m,
                "description": "오류 발생으로 기본 안전 가이드 적용"
            }