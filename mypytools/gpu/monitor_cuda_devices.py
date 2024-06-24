import argparse
import datetime
import json
import time


def monitor_gpus(interval: int = 10, duration: int = 24 * 3600, log_file="gpu_usage.log"):
    from nvitop import Device

    start_time = time.time()
    with open(log_file, "w") as f:
        while (time.time() - start_time) < duration:
            timestamp = datetime.datetime.now().isoformat()
            devices = Device.all()
            stats = {
                "timestamp": timestamp,
                "devices": dict(),
            }
            for device in devices:
                stats["devices"][device.index] = {
                    "temperature": device.temperature(),
                    "gpu_utilization": device.gpu_utilization(),
                    "memory_total": device.memory_total(),
                    "memory_used": device.memory_used(),
                }
            f.write(json.dumps(stats) + "\n")
            f.flush()
            time.sleep(interval)


# NOTE: not yet finished!!!
def visualize_gpus_usage(log_fpath: str, ax=None):
    import matplotlib.pyplot as plt
    import pandas

    # Prepare the DataFrame from log data
    records_loaded = []

    with open(log_fpath) as file:
        for line in file:
            record_this_line = json.loads(line)
            for gpu_id, data in record_this_line["devices"].items():
                records_loaded.append(
                    {
                        "timestamp": record_this_line["timestamp"],
                        "gpu_id": gpu_id,
                        "temperature": data["temperature"],
                        "gpu_utilization": data["gpu_utilization"],
                        "memory_total": data["memory_total"],
                        "memory_used": data["memory_used"],
                    }
                )

    df = pandas.DataFrame(records_loaded)
    df["timestamp"] = pandas.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    # If no specific Axes object is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        own_figure = True
    else:
        own_figure = False

    # Plot GPU Utilization and Temperature
    for key, grp in df.groupby("gpu_id"):
        ax[0].plot(grp.index, grp["gpu_utilization"], label=f"GPU {key} Utilization")
        ax[0].plot(grp.index, grp["temperature"], label=f"GPU {key} Temperature", linestyle="--")

    ax[0].set_title("GPU Utilization (%) and Temperature (C)")
    ax[0].set_ylabel("Percentage / Celsius")
    ax[0].legend()

    # Plot Memory Usage
    for key, grp in df.groupby("gpu_id"):
        ax[1].plot(grp.index, grp["memory_used"], label=f"GPU {key} Memory Used")

    ax[1].set_title("Memory Used (MiB)")
    ax[1].set_ylabel("Memory (MiB)")
    ax[1].legend()

    # Adjust layout
    if own_figure:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor NVIDIA GPU status")
    parser.add_argument("--interval", "-i", type=int, default=10, help="Recording time interval in seconds")
    parser.add_argument("--duration", "-d", type=int, default=24 * 3600, help="Maximum monitoring duration in seconds")
    parser.add_argument("--log_file", "-l", type=str, default="gpu_usage.log", help="Logging file path")
    args = parser.parse_args()

    monitor_gpus(**vars(args))
