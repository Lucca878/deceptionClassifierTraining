import argparse

try:
    from src.pipeline.evaluate import evaluate_model_on_datasets, filter_existing_per_model_csvs
    from src.pipeline.train import make_config, run_cv_only, run_cv_only_with_best_model, run_full_only, run_training
except ModuleNotFoundError:
    from evaluate import evaluate_model_on_datasets, filter_existing_per_model_csvs
    from train import make_config, run_cv_only, run_cv_only_with_best_model, run_full_only, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end deception pipeline")
    parser.add_argument(
        "--mode",
        choices=["all", "train", "eval", "cv", "full", "filter"],
        default="all",
        help=(
            "all: train (cv+full) + eval, train: cv+full only, "
            "cv: only cross-validation, full: only full-data training, "
            "eval: only evaluate existing model, filter: only filter existing per-model CSV outputs"
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
    parser.add_argument(
        "--labeled_output",
        choices=["combined", "per-model", "both"],
        default="combined",
        help="How evaluation writes labeled CSVs: combined per dataset, per-model files, or both",
    )
    parser.add_argument(
        "--filter_correct_only",
        action="store_true",
        help=(
            "When evaluating with per-model outputs, additionally write reduced CSVs "
            "containing only correctly predicted rows"
        ),
    )
    parser.add_argument(
        "--filter_prob_min",
        type=float,
        default=None,
        help="Minimum probability threshold for reduced per-model CSVs",
    )
    parser.add_argument(
        "--filter_prob_max",
        type=float,
        default=None,
        help="Maximum probability threshold for reduced per-model CSVs",
    )
    parser.add_argument(
        "--filter_prob_preset",
        choices=["70-100", "80-100", "70-90", "80-90"],
        default=None,
        help=(
            "Shortcut threshold presets for reduced CSVs; explicit min/max values override preset sides"
        ),
    )
    parser.add_argument(
        "--filter_all_ranges",
        action="store_true",
        help=(
            "Run all built-in probability ranges in one pass: "
            "70-100, 80-100, 70-90, 80-90"
        ),
    )
    parser.add_argument(
        "--filter_print_stats",
        action="store_true",
        help="Print row and class-distribution stats for each filtered output",
    )
    parser.add_argument(
        "--filter_datasets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated dataset names to filter (e.g., hippocorpus_test,decop); "
            "by default all evaluated datasets are filtered"
        ),
    )
    parser.add_argument(
        "--filter_model_tag",
        type=str,
        default=None,
        help="Optional model tag filter for --mode filter (e.g., distilBERT_finetuned)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_best_cv_model",
        action="store_true",
        help=(
            "When running cross-validation, also save the best fold model selected by the Trainer "
            "to a cv_best_model directory inside the run output"
        ),
    )
    parser.add_argument(
        "--cv_selection_metric",
        choices=["accuracy", "eval_accuracy", "eval_loss", "validation_loss", "loss"],
        default="accuracy",
        help=(
            "Metric used by the custom cross-fold selection layer when saving a single CV model. "
            "Use accuracy/eval_accuracy to maximize performance or eval_loss/validation_loss/loss to minimize loss."
        ),
    )
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
            cv_selection_metric=args.cv_selection_metric,
        )
        print(f"Training model preset: {cfg.model_key}")
        print(f"Backbone: {cfg.model_name}")

        if args.mode == "cv":
            if args.save_best_cv_model:
                cv_path, best_model_path = run_cv_only_with_best_model(cfg)
                print(f"Best CV-selected model saved at: {best_model_path}")
            else:
                cv_path = run_cv_only(cfg)
            print(f"Cross-validation results saved at: {cv_path}")
        elif args.mode == "full":
            trained_model_dir = run_full_only(cfg)
            print(f"Trained model saved at: {trained_model_dir}")
        else:
            trained_model_dir = run_training(cfg, save_best_cv_model=args.save_best_cv_model)
            print(f"Trained model saved at: {trained_model_dir}")

    if args.mode in {"all", "eval"}:
        eval_model_dir = args.model_dir if args.model_dir else trained_model_dir
        if not eval_model_dir:
            raise ValueError("Provide --model_dir when using --mode eval")

        filter_datasets = None
        if args.filter_datasets:
            filter_datasets = {
                ds.strip() for ds in args.filter_datasets.split(",") if ds.strip()
            }

        summary_path = evaluate_model_on_datasets(
            model_dir=eval_model_dir,
            output_dir=args.results_dir,
            labeled_output=args.labeled_output,
            filter_correct_only=args.filter_correct_only,
            filter_prob_min=args.filter_prob_min,
            filter_prob_max=args.filter_prob_max,
            filter_prob_preset=args.filter_prob_preset,
            filter_all_ranges=args.filter_all_ranges,
            filter_print_stats=args.filter_print_stats,
            filter_datasets=filter_datasets,
        )
        print(f"Evaluation summary saved at: {summary_path}")

    if args.mode == "filter":
        filter_datasets = None
        if args.filter_datasets:
            filter_datasets = {
                ds.strip() for ds in args.filter_datasets.split(",") if ds.strip()
            }

        written_outputs = filter_existing_per_model_csvs(
            output_dir=args.results_dir,
            filter_correct_only=args.filter_correct_only,
            filter_prob_min=args.filter_prob_min,
            filter_prob_max=args.filter_prob_max,
            filter_prob_preset=args.filter_prob_preset,
            filter_all_ranges=args.filter_all_ranges,
            filter_datasets=filter_datasets,
            filter_model_tag=args.filter_model_tag,
        )
        print(f"Filtered CSV files written: {len(written_outputs)}")
        for item in written_outputs:
            print(f" - {item['path']}")
            if args.filter_print_stats:
                print(
                    f"   dataset={item['dataset']} model={item['model']} "
                    f"criteria={item['filter']} | {item['stats']}"
                )


if __name__ == "__main__":
    main()
