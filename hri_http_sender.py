# hri_http_sender.py
import json
import urllib.error
import urllib.request

LINUX_RECEIVER_URL = "http://192.168.0.2:8000/pass_goal"

def SendPassGoal(payload, url=LINUX_RECEIVER_URL, timeout_sec=2.0):
    """
    HRI 코드가 생성한 의사결정 제어 JSON payload를 네트워크 수신기로 전송합니다.
    """
    if isinstance(payload, str):
        payload_dict = json.loads(payload)
    elif isinstance(payload, dict):
        payload_dict = payload
    else:
        raise TypeError("payload 형식은 무조건 dict 또는 JSON string 구조여야 합니다.")

    body = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")

    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            response_body = response.read().decode("utf-8", errors="replace")
            print(f"📡 [HTTP 네트워크 전송 성공] 상태코드={response.status}, 응답본문={response_body}")
    except Exception as e:
        print(f"📡 [네트워크 전송 실패 알림] 목적지 수신기가 꺼져있거나 연결 지연 중입니다: {e}")