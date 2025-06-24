import argparse
import json
import os
import time


def monitor_gpus(interval: float = 10, duration: float = 24 * 3600, log_fpath="gpu_usage.log"):
    from nvitop import Device

    start_time = time.time()
    os.makedirs(os.path.dirname(log_fpath), exist_ok=True)
    with open(log_fpath, "w") as f:
        while (time.time() - start_time) < duration:
            # timestamp = datetime.datetime.now().isoformat()
            timestamp = time.time()
            devices = Device.all()
            for device in devices:
                stats = {
                    "timestamp": timestamp - start_time,
                    "gpu_id": device.index,
                    "temperature": device.temperature(),
                    "gpu_utilization": device.gpu_utilization(),
                    "memory_total": device.memory_total(),
                    "memory_used": device.memory_used(),
                }
                f.write(json.dumps(stats) + "\n")
            f.flush()
            time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor NVIDIA GPU status")
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=10.0,
        help="Recording time interval in seconds",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=24 * 3600.0,
        help="Maximum monitoring duration in seconds",
    )
    parser.add_argument("--log_fpath", "-l", type=str, default="gpu_usage.log", help="Logging file path")
    args = parser.parse_args()

    monitor_gpus(**vars(args))
