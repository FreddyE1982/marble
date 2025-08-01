import argparse
from pipeline import Pipeline
from config_loader import create_marble_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MARBLE pipeline")
    parser.add_argument("pipeline", help="Path to pipeline JSON file")
    parser.add_argument("--config", required=True, help="YAML config for MARBLE")
    args = parser.parse_args()

    marble = create_marble_from_config(args.config)
    with open(args.pipeline, "r", encoding="utf-8") as f:
        pipe = Pipeline.load_json(f)
    pipe.execute(marble)


if __name__ == "__main__":
    main()
