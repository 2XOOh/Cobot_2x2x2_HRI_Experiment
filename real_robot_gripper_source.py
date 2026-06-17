# real_robot_gripper_source.py
from __future__ import annotations
import time
from threading import Event, Thread

try:
    from neuromeka import IndyDCP3
    from neuromeka.enums import EndtoolState
    ROBOT_LIB_AVAILABLE = True
except ImportError:
    print("\n[경고] 'neuromeka' 모듈을 찾을 수 없습니다. (테스트 진행 불가)")
    IndyDCP3 = None
    EndtoolState = None
    ROBOT_LIB_AVAILABLE = False

ROBOT_IP = "166.104.214.96"
PORT_NAME = "C"

OPEN_STATE = EndtoolState.HIGH_PNP if EndtoolState else None
POLL_SEC = 0.05
open_edge_count = 0

def fetch_gripper_is_open(indy: IndyDCP3 | None) -> bool:
    if not ROBOT_LIB_AVAILABLE or indy is None:
        return False
    try:
        status = indy.get_robot_status()
        return status.get("endtool_state") == OPEN_STATE
    except:
        return False

def record_open_edge_count() -> int:
    global open_edge_count
    open_edge_count += 1
    return open_edge_count

def gripper_poll_loop(open_event: Event):
    if not ROBOT_LIB_AVAILABLE:
        return
        
    print(f"[REAL ROBOT] IndyDCP3 연결 시도 ({ROBOT_IP})")
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
                print(f"[REAL ROBOT] 최초 상태 읽기 성공: is_open={curr_open}")
                first_state_logged = True
            if (not prev_open) and curr_open:
                count = record_open_edge_count()
                if count == 1:
                    print("[REAL ROBOT] 첫 GRIPPER_OPEN edge는 무시합니다")
                else:
                    open_event.set()
                    print(f"[REAL ROBOT] GRIPPER_OPEN edge #{count} -> 신호 전달 완료!")
            prev_open = curr_open
        except Exception as exc: 
            print(f"[REAL ROBOT] gripper poll error: {exc}")
        time.sleep(POLL_SEC)

def start_real_robot_gripper_listener(open_event: Event) -> Thread | None:
    t = Thread(target=gripper_poll_loop, args=(open_event,), daemon=True)
    t.start()
    return t