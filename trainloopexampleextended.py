import os
import sys
import torch
import numpy as np
import json
import tarfile
import tempfile
import requests
import time
from pathlib import Path
from tqdm.notebook import tqdm  # Verwende die Jupyter-optimierte Version
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Beispiel: Metriken loggen
def log_metrics(epoch, tar_idx, loss, vram_usage):
    with open('training_log.txt', 'a') as f:
        f.write(f"Epoch {epoch}, Tar {tar_idx}: Loss={loss:.4f}, VRAM={vram_usage:.2f}MB\n")

# ---------------------------------------------------------------------
# Überarbeitete download_and_process_tar-Funktion
# ---------------------------------------------------------------------
def download_and_process_tar(url, temp_dir):
    import tarfile
    from pathlib import Path
    from tqdm.notebook import tqdm  # Jupyter-optimierte Version verwenden
    import os

    print(f"Starting download from {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    # Zuerst laden wir die Datei (möglicherweise ein LFS-Pointer) herunter:
    response = requests.get(url, stream=True, headers=headers, allow_redirects=True)
    response.raise_for_status()
    content = response.content

    # Prüfe, ob es sich um einen LFS-Pointer handelt (klein und enthält "oid sha256:")
    if len(content) < 1000 and b"oid sha256:" in content:
        print("Detected LFS pointer file. Downloading actual tar file...")
        # Lese den Pointer als Text und parse ihn (hier nur exemplarisch – der oid-Wert wird zwar ausgelesen, 
        # aber wir nutzen ihn nicht weiter; stattdessen konstruieren wir den LFS-Download-URL)
        lfs_info = content.decode("utf-8").strip().splitlines()
        oid_line = next(line for line in lfs_info if line.startswith("oid sha256:"))
        oid = oid_line.split(":")[1].strip()
        # Beispielhafter Aufbau des LFS-Download-URLs (angepasst an den verwendeten Dataset-Pfad)
        lfs_url = f"https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{url.split('data_')[-1]}"
        # Lade die tatsächliche Tar-Datei von der LFS-URL herunter:
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
        # Kein LFS-Pointer – direkte Speicherung des heruntergeladenen Inhalts
        total_size = int(response.headers.get('content-length', 0))
        tar_path = Path(temp_dir) / "current.tar"
        with open(tar_path, 'wb') as f:
            with tqdm(total=total_size, desc="Downloading tar", unit='iB', unit_scale=True, position=1) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        
    # Nun versuchen wir, das Tar-Archiv zu öffnen und zu extrahieren
    extract_dir = Path(temp_dir) / "extracted"
    extract_dir.mkdir(exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_dir)
    except tarfile.ReadError as e:
        raise ValueError(f"Failed to extract tar file: {e}")
    
    # Durchsuche den extrahierten Ordner nach passenden Bild- und JSON-Dateien.
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
        # Falls sowohl Bild als auch JSON mit demselben Basisnamen vorhanden sind, füge sie als Paar hinzu
        for base in images:
            if base in jsons:
                samples.append((str(images[base]), str(jsons[base])))
    samples.sort()
    return samples

# ---------------------------------------------------------------------
# Beispiel einer MetricsVisualizer-Klasse (leicht angepasst)
# ---------------------------------------------------------------------
class MetricsVisualizer:
    def __init__(self):
        self.metrics = {
            'loss': [],
            'vram_usage': [],
            'batch_processing_time': [],
            'learning_efficiency': [],
            'memory_efficiency': [],
            'neuron_count': [],
            'synapse_count': [],
            'brain_growth_rate': [],
            'mean_processing_speed': [],
            'synapse_density': [],
            'blitz_count': [],
            'shortest_route': [],
            'longest_route': [],
            'mean_route_length': [],
            'route_length_variance': [],
            'route_efficiency': [],
            'activation_density': []
        }
        self.setup_plot()
    
    def setup_plot(self):
        self.fig, self.axes = plt.subplots(4, 2, figsize=(15, 20))
        self.fig.suptitle('MARBLE Training Metrics Live View', fontsize=16)
    
    def update(self, new_metrics):
        # Aktualisiere gespeicherte Metriken
        for key, value in new_metrics.items():
            self.metrics[key].append(value)
        
        # Hier wird der gesamte Notebook-Output (einschließlich aller tqdm-Balken) gelöscht.
        # Falls möglich, sollte der Plot in einem separaten Output-Bereich erfolgen.
        clear_output(wait=True)
        self.plot_metrics()
        plt.tight_layout()
        plt.show()
    
    def plot_metrics(self):
        # Beispielhafte Plotimplementierung (wie im Originalcode)
        # Performance Metrics
        self.axes[0,0].clear()
        self.axes[0,0].plot(self.metrics['loss'], 'b-', label='Loss')
        self.axes[0,0].set_title('Training Loss')
        self.axes[0,0].set_xlabel('Batch')
        self.axes[0,0].grid(True)
        ax2 = self.axes[0,0].twinx()
        ax2.plot(self.metrics['vram_usage'], 'r-', label='VRAM')
        ax2.set_ylabel('VRAM (MB)')

        self.axes[0,1].clear()
        self.axes[0,1].plot(self.metrics['learning_efficiency'], 'g-', label='Learning')
        self.axes[0,1].plot(self.metrics['route_efficiency'], 'b-', label='Route')
        self.axes[0,1].set_title('System Efficiency')
        self.axes[0,1].legend()
        self.axes[0,1].grid(True)

        self.axes[1,0].clear()
        ln1 = self.axes[1,0].plot(self.metrics['neuron_count'], 'b-', label='Neurons')
        ax3 = self.axes[1,0].twinx()
        ln2 = ax3.plot(self.metrics['synapse_count'], 'r-', label='Synapses')
        self.axes[1,0].set_title('Network Growth')
        self.axes[1,0].grid(True)
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        self.axes[1,0].legend(lns, labs)

        self.axes[1,1].clear()
        self.axes[1,1].plot(self.metrics['brain_growth_rate'], 'g-', label='Growth')
        self.axes[1,1].plot(self.metrics['synapse_density'], 'm-', label='Density')
        self.axes[1,1].plot(self.metrics['activation_density'], 'y-', label='Activity')
        self.axes[1,1].set_title('Network Dynamics')
        self.axes[1,1].legend()
        self.axes[1,1].grid(True)

        self.axes[2,0].clear()
        self.axes[2,0].plot(self.metrics['mean_processing_speed'], 'c-')
        self.axes[2,0].set_title('Processing Speed (ms)')
        self.axes[2,0].grid(True)

        self.axes[2,1].clear()
        self.axes[2,1].plot(self.metrics['memory_efficiency'], 'm-')
        self.axes[2,1].set_title('Memory Efficiency')
        self.axes[2,1].grid(True)

        self.axes[3,0].clear()
        self.axes[3,0].plot(self.metrics['blitz_count'], 'c-', label='Count')
        self.axes[3,0].set_title('Neuronenblitz Activity')
        self.axes[3,0].grid(True)

        self.axes[3,1].clear()
        self.axes[3,1].plot(self.metrics['shortest_route'], 'g-', label='Shortest')
        self.axes[3,1].plot(self.metrics['longest_route'], 'r-', label='Longest')
        self.axes[3,1].plot(self.metrics['mean_route_length'], 'b-', label='Mean')
        if len(self.metrics['route_length_variance']) > 0:
            self.axes[3,1].fill_between(
                range(len(self.metrics['mean_route_length'])),
                np.array(self.metrics['mean_route_length']) - np.sqrt(self.metrics['route_length_variance']),
                np.array(self.metrics['mean_route_length']) + np.sqrt(self.metrics['route_length_variance']),
                alpha=0.2
            )
        self.axes[3,1].set_title('Route Lengths')
        self.axes[3,1].legend()
        self.axes[3,1].grid(True)

# ---------------------------------------------------------------------
# Beispiel: Funktion, um die Tar-URLs zu erhalten
# ---------------------------------------------------------------------
def get_tar_urls():
    base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/raw/main/data_512_2M/data_{:06d}.tar"
    return [base_url.format(i) for i in range(48)]

# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
if __name__ == '__main__':
    print("Starting MARBLE initialization...")
    # Hier würden normalerweise Core, Neuronenblitz, DataLoader, Brain etc. initialisiert werden.
    # Für dieses Beispiel setzen wir nur den MetricsVisualizer ein.
    visualizer = MetricsVisualizer()

    tar_urls = get_tar_urls()
    print(f"Found {len(tar_urls)} tar archives to process")
    
    num_epochs = 10
    batch_size = 4
    best_loss = float('inf')
    
    # Haupt-Trainingsschleife (vereinfacht)
    for epoch in range(num_epochs):
        print(f"\nStarting Epoch {epoch+1}/{num_epochs}")
        for tar_idx, tar_url in enumerate(tar_urls):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Lade das Tar-Archiv herunter und erhalte die Samples (Paar(e) von Bild- und JSON-Pfaden)
                samples = download_and_process_tar(tar_url, temp_dir)
                
                batch_losses = []
                previous_neuron_count = 0  # Dummy-Wert; in der echten Implementierung z. B. len(core.neurons)
                
                for img_path, json_path in tqdm(samples, desc=f"Processing samples (Tar {tar_idx})", position=2, leave=False):
                    try:
                        start_process_time = time.perf_counter_ns()
                        
                        with open(json_path, 'r') as f:
                            prompt_data = json.load(f)
                        with open(img_path, 'rb') as f:
                            img_data = f.read()
                        
                        # Dummy-Verarbeitung: Simuliere Prompt-Encoding und Bildgenerierung
                        prompt_str = prompt_data.get("prompt", "")
                        if prompt_str:
                            input_scalar = float(np.mean([ord(c) for c in prompt_str]))
                        else:
                            input_scalar = 0.0
                        
                        # Simuliere eine kurze Verarbeitungszeit
                        time.sleep(0.1)
                        process_time = (time.perf_counter_ns() - start_process_time) / 1e6  # in ms
                        
                        # Dummy-Berechnung des Loss (Mean Squared Error zwischen zwei Arrays)
                        target_img = Image.open(BytesIO(img_data)).convert("RGB")
                        target_arr = np.array(target_img).astype(np.float32)
                        generated_arr = target_arr * 0.95  # Dummy-Generierung
                        loss = np.mean((target_arr - generated_arr) ** 2)
                        batch_losses.append(loss)
                        
                        # Dummy-Metriken (ersetze diese durch echte Berechnungen)
                        route_lengths = [5, 10, 7]
                        active_neurons = 1000
                        blitz_delta = 1
                        
                        current_metrics = {
                            'loss': np.mean(batch_losses),
                            'vram_usage': 100.0,  # Dummy-Wert
                            'batch_processing_time': process_time,
                            'neuron_count': previous_neuron_count + 10,
                            'synapse_count': previous_neuron_count + 8,
                            'brain_growth_rate': 10 / process_time if process_time > 0 else 0,
                            'mean_processing_speed': process_time,
                            'synapse_density': (previous_neuron_count + 8) / (previous_neuron_count + 10),
                            'blitz_count': blitz_delta,
                            'shortest_route': min(route_lengths),
                            'longest_route': max(route_lengths),
                            'mean_route_length': np.mean(route_lengths),
                            'route_length_variance': np.var(route_lengths),
                            'route_efficiency': blitz_delta / len(route_lengths) if route_lengths else 0,
                            'activation_density': active_neurons / (previous_neuron_count + 10)
                        }
                        
                        visualizer.update(current_metrics)
                        
                        if np.mean(batch_losses) < best_loss:
                            best_loss = np.mean(batch_losses)
                            print(f"New best loss: {best_loss:.4f}, model saved.")
                        
                        batch_losses = []
                        previous_neuron_count += 10
                    except Exception as e:
                        print(f"Error processing sample: {str(e)}")
                        continue
                
                # Erzeuge alle 5 Tar-Archive Testbilder
                if tar_idx % 5 == 0:
                    test_prompts = [
                        "A stunning mountain landscape at sunset with dramatic clouds",
                        "A detailed close-up of a butterfly on a flower",
                        "An abstract representation of urban life in neon colors"
                    ]
                    os.makedirs(f"epoch_{epoch}_tar_{tar_idx}_samples", exist_ok=True)
                    for idx, prompt in enumerate(test_prompts):
                        # Erzeuge ein Dummy-Bild als Testausgabe
                        dummy_img = Image.new("RGB", (1024, 1024), color=(int(255 * idx / len(test_prompts)), 100, 150))
                        dummy_img.save(f"epoch_{epoch}_tar_{tar_idx}_samples/sample_{idx}.png")
    
    print("Training completed. Final model saved.")
