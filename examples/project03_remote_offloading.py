import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from datasets import load_dataset
from remote_offload import RemoteBrainServer, RemoteBrainClient
from marble_main import MARBLE
from config_loader import load_config


def main() -> None:
    ds = load_dataset("mnist", split="train[:100]")
    pairs = [
        (
            float(record["image"].resize((8, 8)).convert("L").getpixel((0, 0)) / 255.0),
            float(record["label"]),
        )
        for record in ds
    ]
    train = pairs[:80]
    val = pairs[80:]

    server = RemoteBrainServer(port=8005)
    server.start()
    client = RemoteBrainClient("http://localhost:8005")
    cfg = load_config()
    marble = MARBLE(cfg["core"], remote_client=client)
    marble.neuronenblitz.remote_timeout = 10.0
    brain = marble.brain
    brain.offload_enabled = True
    brain.lobe_manager.genesis(range(len(marble.core.neurons)))
    brain.offload_high_attention(threshold=-1.0)
    brain.train(train, epochs=1, validation_examples=val)
    print("Validation loss:", brain.validate(val))
    server.stop()


if __name__ == "__main__":
    main()
