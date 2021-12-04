import subprocess
import sys
import time
import getopt
from pyngrok import ngrok
from train import LOG_DIR


def parse_cli_args():
    usage_str = f"Usage: {sys.argv[0]} [-h] -p <LOCAL_PORT>"

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hp:")
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)

    is_port_present = False

    for opt, arg in opts:
        if opt == "-h":
            print(usage_str)
            sys.exit(0)
        elif opt == "-p":
            is_port_present = True
            port = int(arg)

    if not is_port_present:
        print("Port not specified.")
        print(usage_str)
        sys.exit(2)

    return port


if __name__ == "__main__":
    port = parse_cli_args()

    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", LOG_DIR, "--port", str(port)])
    time.sleep(3)

    try:
        http_tunnel = ngrok.connect(port)
    except:
        print("Couldn't open ngrok tunnel")
        tensorboard_process.terminate()

    print(http_tunnel)

    ngrok_process = ngrok.get_ngrok_process()
    try:
        ngrok_process.proc.wait()
    except KeyboardInterrupt:
        print("Shutting down server.")

    ngrok.kill()
    tensorboard_process.terminate()
