"""Example entry point for MARBLE projects."""

from dataset_loader import load_dataset
from marble_main import train
from config_loader import load_config


def main() -> None:
    config = load_config("config.yaml")
    data = load_dataset("data.csv", offline=config["dataset"].get("offline", False))
    train(data, config)


if __name__ == "__main__":
    main()
