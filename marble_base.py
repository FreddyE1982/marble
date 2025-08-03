from __future__ import annotations

from marble_imports import *
from system_metrics import (
    get_system_memory_usage,
    get_gpu_memory_usage,
    get_cpu_usage,
)
from torch.utils.tensorboard import SummaryWriter
import json
from experiment_tracker import ExperimentTracker
from backup_utils import BackupScheduler
from event_bus import global_event_bus


def clear_output(wait: bool = True) -> None:
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def log_metrics(epoch, tar_idx, loss, vram_usage):
    with open("training_log.txt", "a") as f:
        f.write(
            f"Epoch {epoch}, Tar {tar_idx}: Loss={loss:.4f}, VRAM={vram_usage:.2f}MB\n"
        )


def download_and_process_tar(url, temp_dir):
    print(f"Starting download from {url}")
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
    response.raise_for_status()
    content = response.content

    if len(content) < 1000 and b"oid sha256:" in content:
        print("Detected LFS pointer file. Downloading actual tar file...")
        lfs_info = content.decode("utf-8").strip().splitlines()
        oid_line = next(line for line in lfs_info if line.startswith("oid sha256:"))
        oid = oid_line.split(":")[1].strip()
        lfs_url = f"https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{url.split('data_')[-1]}"
        response = requests.get(
            lfs_url, stream=True, headers=headers, allow_redirects=True
        )
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        tar_path = Path(temp_dir) / "current.tar"
        with open(tar_path, "wb") as f:
            with tqdm(
                total=total_size,
                desc="Downloading tar (LFS)",
                unit="iB",
                unit_scale=True,
                position=1,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        total_size = int(response.headers.get("content-length", 0))
        tar_path = Path(temp_dir) / "current.tar"
        with open(tar_path, "wb") as f:
            with tqdm(
                total=total_size,
                desc="Downloading tar",
                unit="iB",
                unit_scale=True,
                position=1,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    extract_dir = Path(temp_dir) / "extracted"
    extract_dir.mkdir(exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_dir)
    except tarfile.ReadError as e:
        raise ValueError(f"Failed to extract tar file: {e}")

    samples = []
    for root, dirs, files in os.walk(extract_dir):
        images = {}
        jsons = {}
        for file in files:
            base, ext = os.path.splitext(file)
            full_path = Path(root) / file
            if ext.lower() in [".png", ".jpg", ".jpeg"]:
                images[base] = full_path
            elif ext.lower() == ".json":
                jsons[base] = full_path
        for base in images:
            if base in jsons:
                samples.append((str(images[base]), str(jsons[base])))
    samples.sort()
    return samples


class MetricsVisualizer:
    def __init__(
        self,
        fig_width=10,
        fig_height=6,
        refresh_rate=1,
        color_scheme="default",
        show_neuron_ids=False,
        dpi=100,
        track_memory_usage=False,
        track_cpu_usage=False,
        log_dir: str | None = None,
        csv_log_path: str | None = None,
        json_log_path: str | None = None,
        tracker: "ExperimentTracker" | None = None,
        backup_dir: str | None = None,
        backup_interval: float = 3600.0,
        anomaly_std_threshold: float = 3.0,
    ):
        self.metrics = {
            "loss": [],
            "vram_usage": [],
            "batch_processing_time": [],
            "learning_efficiency": [],
            "memory_efficiency": [],
            "arousal": [],
            "stress": [],
            "reward": [],
            "plasticity_threshold": [],
            "message_passing_change": [],
            "compression_ratio": [],
            "meta_loss_avg": [],
            "representation_variance": [],
            "ram_usage": [],
            "gpu_usage": [],
            "cpu_usage": [],
        }
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.refresh_rate = refresh_rate
        self.color_scheme = color_scheme
        self.show_neuron_ids = show_neuron_ids
        self.dpi = dpi
        self.track_memory_usage = track_memory_usage
        self.track_cpu_usage = track_cpu_usage
        self.anomaly_std_threshold = anomaly_std_threshold
        self.writer = SummaryWriter(log_dir) if log_dir else None
        self.tracker = tracker
        self.csv_log_path = csv_log_path
        self._csv_writer = None
        if csv_log_path:
            dir_name = os.path.dirname(csv_log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            if not os.path.exists(csv_log_path):
                with open(csv_log_path, "w", encoding="utf-8") as f:
                    f.write(",".join(["step"] + list(self.metrics.keys())) + "\n")
            self._csv_writer = open(csv_log_path, "a", encoding="utf-8")
        self.json_log_path = json_log_path
        self._json_writer = None
        if json_log_path:
            dir_name = os.path.dirname(json_log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._json_writer = open(json_log_path, "a", encoding="utf-8")
        self.backup_scheduler: BackupScheduler | None = None
        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)
            src = log_dir if log_dir else os.getcwd()
            self.backup_scheduler = BackupScheduler(src, backup_dir, backup_interval)
            self.backup_scheduler.start()
        self._step = 0
        self.events: list[tuple[str, dict]] = []
        self.setup_plot()

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        self.ax.set_title("MARBLE Training Metrics Live View")
        self.ax.set_xlabel("Batches")
        self.ax.set_ylabel("Loss / VRAM Usage")
        self.ax.grid(True)

    def update(self, new_metrics):
        for key, value in new_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            if self.writer:
                self.writer.add_scalar(key, value, self._step)
            if self.anomaly_std_threshold and len(self.metrics[key]) > 5:
                arr = np.asarray(self.metrics[key], dtype=float)
                mean = arr.mean()
                std = arr.std()
                if std > 0 and abs(value - mean) > self.anomaly_std_threshold * std:
                    msg = (
                        f"Anomaly detected in {key}: {value} (mean {mean:.3f}, std {std:.3f})"
                    )
                    if self.writer:
                        self.writer.add_text("anomaly", msg, self._step)
                    print(msg)
        if self.track_memory_usage:
            self.metrics["ram_usage"].append(get_system_memory_usage())
            self.metrics["gpu_usage"].append(get_gpu_memory_usage())
        if self.track_cpu_usage:
            self.metrics["cpu_usage"].append(get_cpu_usage())
        if self.csv_log_path and self._csv_writer:
            row = [str(self._step)]
            for k in self.metrics:
                row.append(str(self.metrics[k][-1]) if self.metrics[k] else "")
            self._csv_writer.write(",".join(row) + "\n")
            self._csv_writer.flush()
        if self._json_writer:
            record = {"step": self._step}
            for k in self.metrics:
                record[k] = self.metrics[k][-1] if self.metrics[k] else None
            self._json_writer.write(json.dumps(record) + "\n")
            self._json_writer.flush()
        if self.tracker:
            self.tracker.log_metrics(new_metrics, self._step)
        self._step += 1
        clear_output(wait=True)
        self.plot_metrics()

    def log_event(self, name: str, data: dict | None = None) -> None:
        """Record a training event and broadcast it on the global event bus."""
        payload = data or {}
        self.events.append((name, payload))
        global_event_bus.publish(name, payload)

    def plot_metrics(self):
        self.ax.clear()
        self.ax.plot(self.metrics["loss"], "b-", label="Loss")
        self.ax_twin = self.ax.twinx()
        self.ax_twin.plot(self.metrics["vram_usage"], "r-", label="VRAM (MB)")
        if self.track_memory_usage and self.metrics["ram_usage"]:
            self.ax_twin.plot(
                self.metrics["ram_usage"], "g-", label="RAM (MB)"
            )
        if self.track_memory_usage and self.metrics["gpu_usage"]:
            self.ax_twin.plot(
                self.metrics["gpu_usage"], "c-", label="GPU (MB)"
            )
        if self.track_cpu_usage and self.metrics["cpu_usage"]:
            self.ax_twin.plot(
                self.metrics["cpu_usage"], "y-", label="CPU (%)"
            )
        if self.metrics["arousal"]:
            self.ax_twin.plot(self.metrics["arousal"], "g--", label="Arousal")
        if self.metrics["stress"]:
            self.ax_twin.plot(self.metrics["stress"], "m--", label="Stress")
        if self.metrics["reward"]:
            self.ax_twin.plot(self.metrics["reward"], "c--", label="Reward")
        if self.metrics["meta_loss_avg"]:
            self.ax.plot(self.metrics["meta_loss_avg"], "y-", label="MetaLossAvg")
        self.ax.set_xlabel("Batches")
        self.ax.set_title("Training Metrics")
        self.ax.grid(True)
        self.ax.legend(loc="upper left")
        self.ax_twin.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

    def close(self) -> None:
        if self.writer:
            self.writer.close()
        if self._csv_writer:
            self._csv_writer.close()
        if self._json_writer:
            self._json_writer.close()
        if self.tracker:
            self.tracker.finish()
        if self.backup_scheduler:
            self.backup_scheduler.stop()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["writer"] = None
        state["_csv_writer"] = None
        state["_json_writer"] = None
        state["backup_scheduler"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


    def __del__(self) -> None:
        self.close()


class MetricsAggregator:
    """Aggregate metrics from multiple visualizers."""

    def __init__(self) -> None:
        self.sources: list[MetricsVisualizer] = []

    def add_source(self, source: MetricsVisualizer) -> None:
        self.sources.append(source)

    def aggregate(self) -> dict[str, float]:
        combined: dict[str, list[float]] = {}
        for src in self.sources:
            for key, values in src.metrics.items():
                combined.setdefault(key, []).extend(values)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in combined.items()}
