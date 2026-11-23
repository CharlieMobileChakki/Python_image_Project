"""SSH Port Forwarder Tunnel: Bridges local 127.0.0.1:8001 to AIC Cloud RTX 3090 GPU Worker."""
import socket
import select
import sys
import threading
import time
import paramiko

HOSTNAME = "ssh2.aiccloud.one"
PORT = 38738
USERNAME = "root"
PASSWORD = "EFt6xkCntCv56cQ2"
REMOTE_PORT = 8001
LOCAL_PORT = 8001

def forward(source, destination):
    string = ' '
    while string:
        try:
            string = source.recv(1024)
            if string:
                destination.sendall(string)
            else:
                source.shutdown(socket.SHUT_RD)
                destination.shutdown(socket.SHUT_WR)
        except Exception:
            break

def handle_client(client_socket, transport):
    try:
        remote_channel = transport.open_channel(
            "direct-tcpip",
            ("127.0.0.1", REMOTE_PORT),
            client_socket.getpeername(),
        )
    except Exception as exc:
        print(f"Failed to open remote channel: {exc}")
        client_socket.close()
        return

    if remote_channel is None:
        client_socket.close()
        return

    t1 = threading.Thread(target=forward, args=(client_socket, remote_channel), daemon=True)
    t2 = threading.Thread(target=forward, args=(remote_channel, client_socket), daemon=True)
    t1.start()
    t2.start()

def main():
    print(f"Connecting to GPU host {HOSTNAME}:{PORT}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=HOSTNAME,
        port=PORT,
        username=USERNAME,
        password=PASSWORD,
        timeout=25,
        banner_timeout=35,
    )
    transport = ssh.get_transport()
    print("SSH Transport established!")

    local_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    local_server.bind(("127.0.0.1", LOCAL_PORT))
    local_server.listen(10)
    print(f"Tunnel Active: 127.0.0.1:{LOCAL_PORT} -> {HOSTNAME}:{PORT} (RTX 3090 : {REMOTE_PORT})")

    while True:
        try:
            client_sock, addr = local_server.accept()
            th = threading.Thread(target=handle_client, args=(client_sock, transport), daemon=True)
            th.start()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Accept error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
