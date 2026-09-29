import socket
import sys
from pathlib import Path

GUI_SCRIPT = Path(__file__).resolve().with_name("gui.py")


def find_available_port(start_port: int = 8501, max_tries: int = 20) -> int:
    for port in range(start_port, start_port + max_tries):
        if _is_port_available(port):
            return port
    raise RuntimeError(f"Could not find an available port in {start_port}-{start_port + max_tries - 1}")


def _is_port_available(port: int) -> bool:
    checks = [
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET, ("0.0.0.0", port)),
        (socket.AF_INET6, ("::1", port)),
        (socket.AF_INET6, ("::", port)),
    ]
    for family, address in checks:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(address)
        except OSError:
            if family == socket.AF_INET6:
                continue
            return False
    return True


def launch_gui(port: int | None = None, headless: bool = False) -> int:
    import streamlit.web.cli as stcli

    if port is None:
        port = find_available_port()
        if port != 8501:
            print(f"Port 8501 is busy. Launching Streamlit on port {port}.", flush=True)
    sys.argv = [
        "streamlit",
        "run",
        str(GUI_SCRIPT),
        "--server.port",
        str(port),
        "--server.maxUploadSize",
        "4096",
        "--global.developmentMode=false",
    ]
    if headless:
        sys.argv.append("--server.headless=true")
    return stcli.main()
