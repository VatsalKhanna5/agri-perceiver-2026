import time
import subprocess
from datetime import datetime

LOG_FILE = "gpu_usage.log"

def get_gpu_stats():
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]
    result = subprocess.check_output(cmd).decode("utf-8").strip()
    return result

while True:
    try:
        stats = get_gpu_stats()
        with open(LOG_FILE, "a") as f:
            f.write(stats + "\n")
    except Exception as e:
        print("GPU logging error:", e)

    time.sleep(60)  # log every 60 seconds


# To run this logger in the background, use: nohup python monitoring/gpu_logger.py &