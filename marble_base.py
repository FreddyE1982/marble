from marble_imports import *


def clear_output(wait: bool = True) -> None:
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def log_metrics(epoch, tar_idx, loss, vram_usage):
    with open('training_log.txt', 'a') as f:
        f.write(f"Epoch {epoch}, Tar {tar_idx}: Loss={loss:.4f}, VRAM={vram_usage:.2f}MB\n")

def download_and_process_tar(url, temp_dir):
    print(f"Starting download from {url}")
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/91.0.4472.124 Safari/537.36')
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
        response = requests.get(lfs_url, stream=True, headers=headers, allow_redirects=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        tar_path = Path(temp_dir) / "current.tar"
        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, desc="Downloading tar (LFS)", unit='iB', unit_scale=True, position=1) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    else:
        total_size = int(response.headers.get('content-length', 0))
        tar_path = Path(temp_dir) / "current.tar"
        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, desc="Downloading tar", unit='iB', unit_scale=True, position=1) as pbar:
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
            if ext.lower() in ['.png', '.jpg', '.jpeg']:
                images[base] = full_path
            elif ext.lower() == '.json':
                jsons[base] = full_path
        for base in images:
            if base in jsons:
                samples.append((str(images[base]), str(jsons[base])))
    samples.sort()
    return samples

class MetricsVisualizer:
    def __init__(self, fig_width=10, fig_height=6):
        self.metrics = {
            'loss': [],
            'vram_usage': [],
            'batch_processing_time': [],
            'learning_efficiency': [],
            'memory_efficiency': []
        }
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.setup_plot()
    
    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(self.fig_width, self.fig_height))
        self.ax.set_title('MARBLE Training Metrics Live View')
        self.ax.set_xlabel('Batches')
        self.ax.set_ylabel('Loss / VRAM Usage')
        self.ax.grid(True)
    
    def update(self, new_metrics):
        for key, value in new_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
        clear_output(wait=True)
        self.plot_metrics()
    
    def plot_metrics(self):
        self.ax.clear()
        self.ax.plot(self.metrics['loss'], 'b-', label='Loss')
        self.ax_twin = self.ax.twinx()
        self.ax_twin.plot(self.metrics['vram_usage'], 'r-', label='VRAM (MB)')
        self.ax.set_xlabel('Batches')
        self.ax.set_title('Training Metrics')
        self.ax.grid(True)
        self.ax.legend(loc='upper left')
        self.ax_twin.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
