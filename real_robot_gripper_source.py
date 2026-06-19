"""
real_robot_gripper_source.py
실제 IndyDCP3 그리퍼 상태를 폴링해서 열림 이벤트(edge)를 감지합니다.
"""

from __future__ import annotations
import time
from threading import Event, Thread

# neuromeka 모듈이 없는 환경에서도 에러 없이 작동하도록 방어 코드 추가
try:
    from neuromeka import IndyDCP3
    from neuromeka.enums import EndtoolState
    ROBOT_LIB_AVAILABLE = True
except ImportError:
    print("\n[경고] 'neuromeka' 모듈을 찾을 수 없습니다. 더미 로봇 모드로 시뮬레이션을 진행합니다.")
    IndyDCP3 = None
    EndtoolState = None
    ROBOT_LIB_AVAILABLE = False

ROBOT_IP = "166.104.214.96"
PORT_NAME = "C"

OPEN_STATE = EndtoolState.HIGH_PNP if EndtoolState else None
POLL_SEC = 0.05
open_edge_count = 0

def record_open_edge_count() -> int:
    global open_edge_count
    open_edge_count += 1
    return open_edge_count

def fetch_gripper_is_open(indy: IndyDCP3 | None) -> bool:
    if not ROBOT_LIB_AVAILABLE or indy is None:
        return False
    try:
        status = indy.get_di_status()
        return status.get(PORT_NAME) == OPEN_STATE
    except Exception:
        return False

def gripper_polling_loop(open_event: Event):
    print("[그리퍼 모니터링] 실제 로봇 기기 신호 대기열을 구동합니다.")
    if not ROBOT_LIB_AVAILABLE:
        print("[더미 상태] 가상 모드로 유지되며 메인 프로세스의 제어 흐름을 방해하지 않습니다.")
        return

    try:
        indy = IndyDCP3(ROBOT_IP)
        print("[REAL ROBOT] IndyDCP3 하드웨어 연결에 성공했습니다.")
    except Exception as e:
        print(f"[REAL ROBOT] 장치 연결에 실패했습니다 (IP 주소를 점검하세요): {e}")
        return

    prev_open = False
    first_state_logged = False

    while True:
        try:
            curr_open = fetch_gripper_is_open(indy)
            if not first_state_logged:
                print(f"[REAL ROBOT] 그리퍼 초기화 모니터링 확인 완료 -> 최초 오픈 상태: {curr_open}")
                first_state_logged = True
            if (not prev_open) and curr_open:
                count = record_open_edge_count()
                if count == 1:
                    print("[REAL ROBOT] 첫 번째 그리퍼 오픈 이벤트 마진은 스킵 처리됩니다.")
                else:
                    open_event.set()
                    print(f"[REAL ROBOT] 그리퍼 오픈 감지 시그널 활성화 #{count}")
            prev_open = curr_open
        except Exception:
            pass
        time.sleep(POLL_SEC)

def start_real_robot_gripper_listener(open_event: Event) -> Thread | None:
    t = Thread(target=gripper_polling_loop, args=(open_event,), daemon=True)
    t.start()
    return t