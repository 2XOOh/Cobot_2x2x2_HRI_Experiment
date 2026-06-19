# dummy_robot.py
import socket

def send_gripper_signal():
    HOST = '127.0.0.1'  
    PORT = 9999         

    print("=== [HRI 실험 보조 도구] 가짜 가상 로봇 신호기 가동 ===")
    while True:
        input("엔터(Enter) 키를 누르면 그리퍼 오픈 수동 작동 트리거를 메인 프로세스에 전달합니다. (종료: Ctrl+C)")
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))
            
            client_socket.sendall("GRIPPER_OPEN".encode('utf-8'))
            print(" 로봇 신호기: 'GRIPPER_OPEN' 이벤트를 메인 제어 프로그램으로 전송하였습니다.\n")
            
            client_socket.close()
        except ConnectionRefusedError:
            print("에러 알림: 메인 통합 제어 프로그램(main_integrated.py)이 현재 켜져 있지 않습니다.\n")

if __name__ == "__main__":
    send_gripper_signal()