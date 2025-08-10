import argparse
from pipeline import Pipeline
from config_loader import create_marble_from_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MARBLE pipeline")
    parser.add_argument("pipeline", help="Path to pipeline JSON file")
    parser.add_argument("--config", required=True, help="YAML config for MARBLE")
    args = parser.parse_args()

    marble = create_marble_from_config(args.config)
    cfg = load_config(args.config)
    pipeline_cfg = cfg.get("pipeline", {})
    cache_dir = pipeline_cfg.get("cache_dir")
    default_limit = pipeline_cfg.get("default_step_memory_limit_mb")
    with open(args.pipeline, "r", encoding="utf-8") as f:
        pipe = Pipeline.load_json(f)
    pipe.execute(
        marble,
        cache_dir=cache_dir,
        default_memory_limit_mb=default_limit,
    )


if __name__ == "__main__":
    main()
