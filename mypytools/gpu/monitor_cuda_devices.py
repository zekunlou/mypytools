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


def visualize_gpus_usage(log_fpath: str, ax=None):
    import matplotlib.pyplot as plt
    import pandas

    # Prepare the DataFrame from log data
    records_loaded = []

    with open(log_fpath) as file:
        # recorded_loaded.append(json.loads(line))
        for line in file:
            records_loaded.append(json.loads(line))

    df = pandas.DataFrame(records_loaded)
    df.set_index("timestamp", inplace=True)

    # If no specific Axes object is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)

    # Plot GPU Utilization and Temperature
    for key, grp in df.groupby("gpu_id"):
        ax[0].plot(grp.index, grp["gpu_utilization"], label=f"GPU {key} Utilization")
        ax[0].plot(grp.index, grp["temperature"], label=f"GPU {key} Temperature", linestyle="--")

    ax[0].set_title("GPU Utilization (%) and Temperature (C)")
    ax[0].set_ylabel("Percentage / Degree")
    ax[0].set_xlim(min(grp.index), max(grp.index))
    ax[0].grid()
    ax[0].legend()

    # Plot Memory Usage
    for color_idx, (key, grp) in enumerate(df.groupby("gpu_id")):
        ax[1].plot(grp.index, grp["memory_used"] / 1024**3, color=f"C{color_idx}", label=f"GPU {key}")
        ax[1].axhline(grp["memory_total"].iloc[0] / 1024**3, min(grp.index), max(grp.index), linestyle="dotted", color=f"C{color_idx}")

    ax[1].set_title("Memory Used (GB)")
    ax[1].set_ylabel("Memory (GB)")
    ax[1].set_xlim(min(grp.index), max(grp.index))
    ax[1].grid()
    ax[1].legend()

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor NVIDIA GPU status")
    parser.add_argument("--interval", "-i", type=float, default=10.0, help="Recording time interval in seconds")
    parser.add_argument("--duration", "-d", type=float, default=24 * 3600.0, help="Maximum monitoring duration in seconds")
    parser.add_argument("--log_fpath", "-l", type=str, default="gpu_usage.log", help="Logging file path")
    args = parser.parse_args()

    monitor_gpus(**vars(args))
