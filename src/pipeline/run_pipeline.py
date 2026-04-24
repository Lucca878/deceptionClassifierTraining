import argparse

try:
    from src.pipeline.evaluate import evaluate_model_on_datasets
    from src.pipeline.train import make_config, run_cv_only, run_full_only, run_training
except ModuleNotFoundError:
    from evaluate import evaluate_model_on_datasets
    from train import make_config, run_cv_only, run_full_only, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end deception pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "train", "eval", "cv", "full"],
        default="all",
        help=(
            "all: train (cv+full) + eval, train: cv+full only, "
            "cv: only cross-validation, full: only full-data training, "
            "eval: only evaluate existing model"
        ),
    )
    parser.add_argument(
        "--model",
        choices=["distilbert", "bert", "sbert", "modernbert"],
        default="distilbert",
        help="Model preset used for training",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional Hugging Face model name override",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Existing local model dir used in eval mode",
    )
    parser.add_argument("--output_root", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trained_model_dir = None

    if args.mode in {"all", "train", "cv", "full"}:
        cfg = make_config(
            model_key=args.model,
            model_name=args.model_name,
            output_root=args.output_root,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
        )
        print(f"Training model preset: {cfg.model_key}")
        print(f"Backbone: {cfg.model_name}")

        if args.mode == "cv":
            cv_path = run_cv_only(cfg)
            print(f"Cross-validation results saved at: {cv_path}")
        elif args.mode == "full":
            trained_model_dir = run_full_only(cfg)
            print(f"Trained model saved at: {trained_model_dir}")
        else:
            trained_model_dir = run_training(cfg)
            print(f"Trained model saved at: {trained_model_dir}")

    if args.mode in {"all", "eval"}:
        eval_model_dir = args.model_dir if args.model_dir else trained_model_dir
        if not eval_model_dir:
            raise ValueError("Provide --model_dir when using --mode eval")

        summary_path = evaluate_model_on_datasets(
            model_dir=eval_model_dir,
            output_dir=args.results_dir,
        )
        print(f"Evaluation summary saved at: {summary_path}")


if __name__ == "__main__":
    main()
