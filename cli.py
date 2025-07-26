import argparse
from config_loader import create_marble_from_config
from dataset_loader import load_dataset
from marble_interface import save_marble_system, evaluate_marble_system


def main() -> None:
    parser = argparse.ArgumentParser(description="MARBLE command line interface")
    parser.add_argument("--config", "-c", help="Path to config YAML", default=None)
    parser.add_argument("--train", help="Path or URL to training dataset")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--validate", help="Optional validation dataset path")
    parser.add_argument("--evaluate", help="Evaluation dataset for measuring MSE")
    parser.add_argument("--save", help="Path to save trained model")
    args = parser.parse_args()

    marble = create_marble_from_config(args.config)
    if args.train:
        train_data = load_dataset(args.train)
        val_data = load_dataset(args.validate) if args.validate else None
        marble.get_brain().train(train_data, epochs=args.epochs, validation_examples=val_data)
    if args.evaluate:
        eval_data = load_dataset(args.evaluate)
        mse = evaluate_marble_system(marble, eval_data)
        print(f"Evaluation MSE: {mse:.6f}")
    if args.save:
        save_marble_system(marble, args.save)


if __name__ == "__main__":
    main()
