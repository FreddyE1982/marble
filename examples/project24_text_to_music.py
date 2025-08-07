import os, sys, io, requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torchaudio
from datasets import load_dataset

from highlevel_pipeline import HighLevelPipeline
from bit_tensor_streaming_dataset import BitTensorStreamingDataset

def _decode_audio(url: str) -> list[float]:
    """Download ``url`` and decode into a mono float waveform."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    audio_buf = io.BytesIO(resp.content)
    waveform, _ = torchaudio.load(audio_buf)
    return waveform.mean(dim=0).tolist()

def main() -> None:
    hf_stream = load_dataset(
        "sleeping-ai/Udio-24MX1", split="train", streaming=True
    )

    def format_record(record: dict) -> dict:
        audio = _decode_audio(record["song_path"])
        return {
            "input": {
                "lyrics": record.get("lyrics", ""),
                "prompt": record.get("prompt", ""),
                "tags": record.get("tags", []),
                "duration": record.get("duration", 0.0),
            },
            "target": audio,
        }

    formatted_stream = hf_stream.map(format_record, remove_columns=hf_stream.column_names)
    dataset = BitTensorStreamingDataset(formatted_stream, virtual_batch_size=2)

    pipeline = (
        HighLevelPipeline()
        .marble_interface.new_marble_system()
        .marble_interface.train_marble_system(train_examples=dataset, epochs=1)
    )
    pipeline.execute()

if __name__ == "__main__":
    main()
