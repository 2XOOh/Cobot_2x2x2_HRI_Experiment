# dummy_robot.py
import socket

def send_gripper_signal():
    HOST = '127.0.0.1'  
    PORT = 9999         

    print("="*50)
    print("🤖 [가짜 로봇 신호기 실행됨]")
    print(" - 이 창을 띄워두고 엔터를 치면 로봇이 열린 것처럼 신호가 갑니다.")
    print("="*50)
    
    while True:
        input("👉 엔터(Enter) 키를 누르면 그리퍼 열림 신호를 보냅니다! (종료: Ctrl+C)")
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))
            
            client_socket.sendall("GRIPPER_OPEN".encode('utf-8'))
            print(" ✅ 로봇: 'GRIPPER_OPEN' 신호를 메인 PC로 전송했습니다!\n")
            
            client_socket.close()
        except ConnectionRefusedError:
            print(" 🚨 에러: 메인 프로그램이 켜져 있지 않거나 연결을 거부했습니다.\n")

if __name__ == "__main__":
    send_gripper_signal()