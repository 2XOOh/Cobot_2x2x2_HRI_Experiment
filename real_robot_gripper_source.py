"""
real_robot_gripper_source.py

실제 IndyDCP3 그리퍼 상태를 폴링해서
닫힘 -> 열림 순간에만 공유 open_event를 set하는 최소 구현.
(neuromeka 모듈이 없는 환경에서도 에러 없이 작동하도록 방어 코드 추가)
"""

from __future__ import annotations

import time
from threading import Event, Thread

# =============================================================
# [수정] neuromeka 라이브러리가 없어도 프로그램이 죽지 않도록 방어 코드 추가
# =============================================================
try:
    from neuromeka import IndyDCP3
    from neuromeka.enums import EndtoolState
    ROBOT_LIB_AVAILABLE = True
except ImportError:
    print("\n[경고] 'neuromeka' 모듈을 찾을 수 없습니다. 더미 로봇 모드로 시뮬레이션을 진행합니다.")
    IndyDCP3 = None
    EndtoolState = None
    ROBOT_LIB_AVAILABLE = False

# 요청대로 IP 고정
ROBOT_IP = "166.104.214.96"
PORT_NAME = "C"

# EndtoolState가 상단 try-except에서 None이 되었을 때를 대비한 안전장치
OPEN_STATE = EndtoolState.HIGH_PNP if EndtoolState else None
POLL_SEC = 0.05

# 열림 엣지(트리거) 횟수 저장
open_edge_count = 0


def record_open_edge_count() -> int:
    """열림 엣지 횟수를 +1 하고 현재 누적값을 반환한다."""
    global open_edge_count
    open_edge_count += 1
    return open_edge_count


def fetch_gripper_is_open(indy: IndyDCP3) -> bool:
    """
    현재 그리퍼 open 여부를 읽는다.
    gripper_node.py와 동일하게 get_endtool_do() 기반.
    """
    # 라이브러리가 없는 환경이면 무조건 False 반환 (폴링 에러 방지)
    if not ROBOT_LIB_AVAILABLE or indy is None:
        return False

    do_state = indy.get_endtool_do()

    # {"signals": [{"port": "C", "states": [-2]}]} 형태
    if isinstance(do_state, dict) and "signals" in do_state:
        for signal in do_state["signals"]:
            if signal.get("port") == PORT_NAME:
                states = signal.get("states", [])
                if len(states) > 0:
                    # -2 가 HIGH_PNP(열림 상태)를 의미한다고 가정
                    return states[0] == -2
    return False


def poll_real_robot_gripper_edge(open_event: Event) -> None:
    """
    무한 루프 폴링:
    닫힘 -> 열림 전이에서만 전달받은 open_event를 set.
    """
    # 라이브러리가 없으면 폴링 루프를 돌지 않고 즉시 종료합니다.
    if not ROBOT_LIB_AVAILABLE:
        return

    print(
        "[REAL ROBOT] IndyDCP3 연결 시도: "
        f"ip={ROBOT_IP}, port={PORT_NAME}, open_state={OPEN_STATE}"
    )
    try:
        indy = IndyDCP3(ROBOT_IP)
        print("[REAL ROBOT] IndyDCP3 연결 완료")
    except Exception as e:
        print(f"[REAL ROBOT] 로봇 연결 실패 (IP를 확인하세요): {e}")
        return

    prev_open = False
    first_state_logged = False

    while True:
        try:
            curr_open = fetch_gripper_is_open(indy)
            if not first_state_logged:
                print(
                    "[REAL ROBOT] 최초 상태 읽기 성공: "
                    f"is_open={curr_open}"
                )
                first_state_logged = True
            if (not prev_open) and curr_open:
                count = record_open_edge_count()
                if count == 1:
                    print("[REAL ROBOT] 첫 GRIPPER_OPEN edge는 무시합니다")
                else:
                    open_event.set()
                    print(f"[REAL ROBOT] GRIPPER_OPEN edge #{count} -> event=set")
            prev_open = curr_open
        except Exception as exc:  # noqa: BLE001
            print(f"[REAL ROBOT] gripper poll error: {exc}")

        time.sleep(POLL_SEC)


def start_real_robot_gripper_listener(open_event: Event) -> Thread | None:
    """
    메인 프로그램에서 그리퍼 폴링 스레드를 시작할 때 호출하는 함수
    """
    # [수정] 라이브러리가 없으면 백그라운드 스레드를 시작하지 않고 None을 반환합니다.
    if not ROBOT_LIB_AVAILABLE:
        print("[안내] 실제 로봇 라이브러리가 없어 그리퍼 리스너를 시작하지 않습니다. (더미 테스트 전용)")
        return None

    thread = Thread(
        target=poll_real_robot_gripper_edge,
        args=(open_event,),
        daemon=True,
    )
    thread.start()
    return thread