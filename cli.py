import argparse

from config_loader import create_marble_from_config
from dataset_loader import load_dataset
from marble_core import benchmark_message_passing
from marble_interface import (
    evaluate_marble_system,
    save_core_json_file,
    save_marble_system,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MARBLE command line interface")
    parser.add_argument("--config", "-c", help="Path to config YAML", default=None)
    parser.add_argument("--train", help="Path or URL to training dataset")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument("--validate", help="Optional validation dataset path")
    parser.add_argument("--evaluate", help="Evaluation dataset for measuring MSE")
    parser.add_argument("--save", help="Path to save trained model")
    parser.add_argument(
        "--export-core",
        help="Path to export the core JSON after training",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "exponential", "cyclic"],
        help="Learning rate scheduler to use",
    )
    parser.add_argument("--scheduler-steps", type=int, help="Scheduler cycle length")
    parser.add_argument(
        "--scheduler-gamma", type=float, help="Gamma for exponential scheduler"
    )
    parser.add_argument(
        "--scheduler-plugin",
        choices=["thread", "asyncio"],
        help="Plugin used to schedule asynchronous tasks",
    )
    parser.add_argument("--min-lr", type=float, help="Minimum learning rate")
    parser.add_argument("--max-lr", type=float, help="Maximum learning rate")
    parser.add_argument(
        "--parallel-wanderers",
        type=int,
        help="Number of parallel Neuronenblitz worker threads",
    )
    parser.add_argument(
        "--sync-interval-ms",
        type=int,
        help="Milliseconds between cross-device tensor synchronizations (100-10000 recommended)",
    )
    parser.add_argument(
        "--unified-learning",
        action="store_true",
        help="Enable UnifiedLearner meta-controller",
    )
    parser.add_argument(
        "--unified-learning-gating-hidden",
        type=int,
        help="Hidden units for UnifiedLearner gating network",
    )
    parser.add_argument(
        "--unified-learning-log-path",
        help="Path to UnifiedLearner decision log",
    )
    parser.add_argument(
        "--pipeline",
        help="Path to a pipeline JSON file to execute after initialization",
    )
    parser.add_argument(
        "--causal-attention",
        action="store_true",
        help="Enable causal masking in attention modules",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--early-stopping-delta", type=float, help="Delta for early stopping"
    )
    parser.add_argument(
        "--benchmark-msgpass",
        type=int,
        metavar="N",
        help="Run message passing benchmark for N iterations",
    )
    parser.add_argument(
        "--grid-search",
        help="YAML file describing parameter grid for hyperparameter search",
    )
    parser.add_argument(
        "--no-early-stop", action="store_true", help="Disable early stopping"
    )
    parser.add_argument(
        "--precompile-graphs",
        action="store_true",
        help="Precompile compute graphs before training",
    )
    parser.add_argument(
        "--sync-config",
        nargs="+",
        metavar="DEST",
        help="Synchronise config to destination paths and exit",
    )
    parser.add_argument(
        "--sync-src",
        help="Source config to synchronise (defaults to --config)",
    )
    parser.add_argument(
        "--remote-retries",
        type=int,
        help="Maximum number of retries for remote calls",
    )
    parser.add_argument(
        "--remote-backoff",
        type=float,
        help="Backoff factor for remote call retries",
    )
    parser.add_argument(
        "--quantize",
        type=int,
        help="Quantize tensors to the specified bit width (1-8)",
    )
    parser.add_argument("--cv-folds", type=int, help="Number of folds for k-fold cross-validation")
    parser.add_argument("--cv-seed", type=int, help="Random seed for cross-validation splits")
    args = parser.parse_args()

    if args.sync_config:
        src = args.sync_src or args.config
        if src is None:
            raise ValueError("Provide --sync-src or --config")
        from config_sync_service import sync_config

        sync_config(src, args.sync_config)
        return

    overrides: dict[str, dict] = {
        "neuronenblitz": {},
        "brain": {},
        "sync": {},
        "unified_learning": {},
        "network": {"remote_client": {}},
        "core": {},
        "cross_validation": {},
    }
    if args.lr_scheduler:
        overrides["neuronenblitz"]["lr_scheduler"] = args.lr_scheduler
    if args.scheduler_steps is not None:
        overrides["neuronenblitz"]["scheduler_steps"] = args.scheduler_steps
    if args.scheduler_gamma is not None:
        overrides["neuronenblitz"]["scheduler_gamma"] = args.scheduler_gamma
    if args.scheduler_plugin:
        overrides.setdefault("scheduler", {})["plugin"] = args.scheduler_plugin
    if args.min_lr is not None:
        overrides["neuronenblitz"]["min_learning_rate"] = args.min_lr
    if args.max_lr is not None:
        overrides["neuronenblitz"]["max_learning_rate"] = args.max_lr
    if args.parallel_wanderers is not None:
        overrides["neuronenblitz"]["parallel_wanderers"] = args.parallel_wanderers
    if args.sync_interval_ms is not None:
        overrides["sync"]["interval_ms"] = args.sync_interval_ms
    if args.early_stopping_patience is not None:
        overrides["brain"]["early_stopping_patience"] = args.early_stopping_patience
    if args.early_stopping_delta is not None:
        overrides["brain"]["early_stopping_delta"] = args.early_stopping_delta
    if args.unified_learning:
        overrides["unified_learning"]["enabled"] = True
    if args.unified_learning_gating_hidden is not None:
        overrides["unified_learning"]["gating_hidden"] = args.unified_learning_gating_hidden
    if args.unified_learning_log_path is not None:
        overrides["unified_learning"]["log_path"] = args.unified_learning_log_path
    if args.remote_retries is not None:
        overrides["network"]["remote_client"]["max_retries"] = args.remote_retries
    if args.remote_backoff is not None:
        overrides["network"]["remote_client"]["backoff_factor"] = args.remote_backoff
    if args.quantize is not None:
        overrides["core"]["quantization_bits"] = args.quantize
    if args.cv_folds is not None:
        overrides["cross_validation"]["folds"] = args.cv_folds
    if args.cv_seed is not None:
        overrides["cross_validation"]["seed"] = args.cv_seed
    if args.causal_attention:
        overrides["core"]["attention_causal"] = True
    if args.precompile_graphs:
        overrides["brain"]["precompile_graphs"] = True
    marble = create_marble_from_config(args.config, overrides=overrides)
    if args.grid_search:
        import yaml

        with open(args.grid_search, "r", encoding="utf-8") as f:
            param_grid = yaml.safe_load(f)
        if args.pipeline:
            from pipeline import Pipeline

            with open(args.pipeline, "r", encoding="utf-8") as f:
                pipe = Pipeline.load_json(f)

            def score_fn(outputs):
                last = outputs[-1]
                return float(last) if isinstance(last, (int, float)) else 0.0

            results = pipe.hyperparameter_search(param_grid, score_fn, marble=marble)
        else:
            from hyperparameter_search import grid_search

            results = grid_search(param_grid, lambda p: 0.0)
        print(results)
        return
    if args.no_early_stop:
        marble.get_brain().early_stop_enabled = False
    if args.train:
        train_data = load_dataset(args.train)
        val_data = load_dataset(args.validate) if args.validate else None
        marble.get_brain().train(
            train_data, epochs=args.epochs, validation_examples=val_data
        )
    if args.evaluate:
        eval_data = load_dataset(args.evaluate)
        mse = evaluate_marble_system(marble, eval_data)
        print(f"Evaluation MSE: {mse:.6f}")
    if args.pipeline:
        from pipeline import Pipeline

        with open(args.pipeline, "r", encoding="utf-8") as f:
            pipe = Pipeline.load_json(f)
        results = pipe.execute(marble)
        print(results)
    if args.save:
        save_marble_system(marble, args.save)
    if args.export_core:
        save_core_json_file(marble, args.export_core)
    if args.benchmark_msgpass is not None:
        iters, sec = benchmark_message_passing(
            marble.get_core(), iterations=args.benchmark_msgpass
        )
        print(
            f"Message passing benchmark: {iters} iterations in {sec:.6f}s per iteration"
        )


if __name__ == "__main__":
    main()
