"""Arangtik GPU Tunnel Daemon
Bridges local 127.0.0.1:8001 -> AIC Cloud RTX A6000 (48GB VRAM) GPU Worker with auto-reconnect.
"""
import socket
import sys
import threading
import time
import logging
import paramiko

logging.basicConfig(level=logging.INFO, format="%(asctime)s [GPU-TUNNEL] %(message)s")
logger = logging.getLogger("gpu-tunnel")

HOSTNAME = "gpu-b9900a.aiccloud.one"
PORT = 36939
USERNAME = "root"
PASSWORD = "9Vp3qUP9DicZHLDf"
REMOTE_PORT = 8001
LOCAL_PORT = 8001

def forward_stream(source, dest):
    try:
        while True:
            data = source.recv(8192)
            if not data:
                break
            dest.sendall(data)
    except Exception:
        pass
    finally:
        try:
            source.close()
        except Exception:
            pass
        try:
            dest.close()
        except Exception:
            pass

class GpuTunnel:
    def __init__(self):
        self.ssh = None
        self.transport = None
        self.server_sock = None
        self.running = True

    def connect_ssh(self):
        while self.running:
            try:
                logger.info("Connecting SSH to GPU Worker %s:%s...", HOSTNAME, PORT)
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.connect(
                    hostname=HOSTNAME,
                    port=PORT,
                    username=USERNAME,
                    password=PASSWORD,
                    timeout=20,
                    banner_timeout=30,
                )
                self.ssh = ssh
                self.transport = ssh.get_transport()
                logger.info("SSH connection to RTX A6000 (48GB VRAM) established successfully!")
                return True
            except Exception as e:
                logger.warning("SSH connection failed: %s. Retrying in 5s...", e)
                time.sleep(5)
        return False

    def handle_client(self, client_sock):
        if not self.transport or not self.transport.is_active():
            logger.warning("Transport inactive; dropping connection and reconnecting...")
            client_sock.close()
            return

        try:
            remote_chan = self.transport.open_channel(
                "direct-tcpip",
                ("127.0.0.1", REMOTE_PORT),
                client_sock.getpeername(),
            )
        except Exception as exc:
            logger.error("Failed to open remote channel on GPU: %s", exc)
            client_sock.close()
            return

        if remote_chan is None:
            client_sock.close()
            return

        t1 = threading.Thread(target=forward_stream, args=(client_sock, remote_chan), daemon=True)
        t2 = threading.Thread(target=forward_stream, args=(remote_chan, client_sock), daemon=True)
        t1.start()
        t2.start()

    def start(self):
        self.connect_ssh()

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(("127.0.0.1", LOCAL_PORT))
        self.server_sock.listen(25)
        logger.info("Listening on 127.0.0.1:%d -> RTX 3090 :%d", LOCAL_PORT, REMOTE_PORT)

        while self.running:
            try:
                client_sock, _ = self.server_sock.accept()
                if not self.transport or not self.transport.is_active():
                    self.connect_ssh()
                th = threading.Thread(target=self.handle_client, args=(client_sock,), daemon=True)
                th.start()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Accept error: %s", e)
                time.sleep(1)

        if self.server_sock:
            self.server_sock.close()
        if self.ssh:
            self.ssh.close()

if __name__ == "__main__":
    tunnel = GpuTunnel()
    tunnel.start()
