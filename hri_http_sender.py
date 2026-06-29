import json
import urllib.error
import urllib.parse
import urllib.request

BASE_RECEIVER_URL = "http://192.168.0.2:8000"
PASS_GOAL_ENDPOINT = "/pass_goal"
HOLD_FINISHED_ENDPOINT = "/hold_finished"
REVIEW_PENDING_ENDPOINT = "/review_pending"
ROBOT_STATE_ENDPOINT = "/robot_state"


def _make_url(path, base_url=BASE_RECEIVER_URL):
    return urllib.parse.urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))


def _request_json(method, path, payload=None, timeout_sec=2.0, base_url=BASE_RECEIVER_URL):
    body = None
    headers = {}
    if payload is not None:
        if isinstance(payload, str):
            payload_dict = json.loads(payload)
        elif isinstance(payload, dict):
            payload_dict = payload
        else:
            raise TypeError("payload 형식은 dict 또는 JSON string 이어야 합니다.")
        body = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = urllib.request.Request(
        url=_make_url(path, base_url=base_url),
        data=body,
        headers=headers,
        method=method,
    )

    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        response_body = response.read().decode("utf-8", errors="replace")
        if not response_body:
            return {"status": response.status}
        try:
            return json.loads(response_body)
        except json.JSONDecodeError:
            return {"status": response.status, "raw": response_body}


def SendPassGoal(payload, timeout_sec=2.0, base_url=BASE_RECEIVER_URL):
    """
    HRI 코드가 생성한 목표 JSON payload를 로봇 인터페이스로 전송합니다.
    """
    try:
        result = _request_json(
            "POST",
            PASS_GOAL_ENDPOINT,
            payload=payload,
            timeout_sec=timeout_sec,
            base_url=base_url,
        )
        print(f"📡 [PASS_GOAL 전송 성공] {result}")
        return True
    except Exception as e:
        print(f"📡 [PASS_GOAL 전송 실패] {e}")
        return False


def SendHoldFinished(timeout_sec=2.0, base_url=BASE_RECEIVER_URL):
    """
    작업자가 AT_TASK 구간 작업을 끝냈음을 로봇 인터페이스에 알립니다.
    """
    try:
        result = _request_json(
            "POST",
            HOLD_FINISHED_ENDPOINT,
            payload={"hold_finished": True},
            timeout_sec=timeout_sec,
            base_url=base_url,
        )
        print(f"📡 [HOLD_FINISHED 전송 성공] {result}")
        return True
    except Exception as e:
        print(f"📡 [HOLD_FINISHED 전송 실패] {e}")
        return False


def SetReviewPending(pending, timeout_sec=2.0, base_url=BASE_RECEIVER_URL):
    """
    HRI 평가 및 다음 goal 생성 진행 상태를 로봇 인터페이스에 전달합니다.
    """
    try:
        result = _request_json(
            "POST",
            REVIEW_PENDING_ENDPOINT,
            payload={"pending": bool(pending)},
            timeout_sec=timeout_sec,
            base_url=base_url,
        )
        print(f"📡 [REVIEW_PENDING 전송 성공] pending={bool(pending)} | {result}")
        return True
    except Exception as e:
        print(f"📡 [REVIEW_PENDING 전송 실패] {e}")
        return False


def GetRobotState(timeout_sec=1.0, base_url=BASE_RECEIVER_URL):
    """
    현재 로봇 상태 문자열을 읽어옵니다.
    """
    try:
        result = _request_json(
            "GET",
            ROBOT_STATE_ENDPOINT,
            timeout_sec=timeout_sec,
            base_url=base_url,
        )
        robot_state = result.get("robot_state")
        if robot_state:
            return str(robot_state)
    except Exception as e:
        print(f"📡 [ROBOT_STATE 조회 실패] {e}")
    return None