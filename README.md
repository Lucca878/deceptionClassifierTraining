# Deception Classifier Pipeline

Clean end-to-end training and evaluation pipeline for deception detection.

The workflow is script-first and reproducible:
1. Train one model architecture.
2. Save the final model locally.
3. Evaluate the saved model across all datasets in `data/`.

By default, evaluation writes one combined labeled CSV per dataset and appends the current model's columns into that file.

## Fresh Clone Flow

If you are new to this repo:
1. Clone/download the repository.
2. Create and activate the environment.
3. Train one model (`distilbert`, `bert`, `sbert`, or `modernbert`).
4. Evaluate that trained model on all datasets.
5. Use the generated combined per-dataset labeled CSV outputs and summary CSV.

## Setup

```bash
conda env create -f environment.yml
conda activate deception
```

If conda shell hooks are not active yet:

```bash
source ~/.zshrc
```

## Single End-to-End Command

```bash
python src/pipeline/run_pipeline.py --mode all --model distilbert
```

Supported model presets:
- `distilbert`
- `bert`
- `sbert`
- `modernbert`

Example runs:

```bash
# DistilBERT
python src/pipeline/run_pipeline.py --mode all --model distilbert

# BERT
python src/pipeline/run_pipeline.py --mode all --model bert

# SBERT
python src/pipeline/run_pipeline.py --mode all --model sbert

# ModernBERT
python src/pipeline/run_pipeline.py --mode all --model modernbert
```

## Modes

```bash
# Train only (CV + full-data training)
python src/pipeline/run_pipeline.py --mode train --model distilbert

# CV only (tuning stage)
python src/pipeline/run_pipeline.py --mode cv --model distilbert

# CV only + save a single CV-selected model
python src/pipeline/run_pipeline.py --mode cv --model distilbert --save_best_cv_model

# CV only + save a single CV-selected model using loss as cross-fold selector
python src/pipeline/run_pipeline.py --mode cv --model distilbert --save_best_cv_model --cv_selection_metric eval_loss

# Full-data training only (after choosing settings from CV)
python src/pipeline/run_pipeline.py --mode full --model distilbert

 # Evaluate only (existing trained model)
 # For models trained by this pipeline:
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/<model>_<timestamp>/model
 
 # For pre-trained models (e.g., distilBERT_finetuned):
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/distilBERT_finetuned

 # Control labeled CSV output style:
 # combined (default): one labeled_<dataset>.csv with appended columns for each evaluated model
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/distilBERT_finetuned --labeled_output combined

 # per-model: one labeled_<dataset>_<model>.csv per evaluation run
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/distilBERT_finetuned --labeled_output per-model

 # both: write both combined and per-model labeled CSV outputs
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/distilBERT_finetuned --labeled_output both

 # Optional: write reduced per-model CSVs (filter by confidence and correctness)
 # This creates extra files like labeled_<dataset>_<model>_filtered_*.csv
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/distilBERT_finetuned --labeled_output per-model --filter_correct_only --filter_prob_min 0.70 --filter_prob_max 1.0

 # Optional: apply filtering only to specific datasets
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/distilBERT_finetuned --labeled_output per-model --filter_correct_only --filter_prob_min 0.80 --filter_prob_max 1.0 --filter_datasets hippocorpus_test

 # Filter only (no model inference): reduce already-generated per-model CSV files in results/
 python src/pipeline/run_pipeline.py --mode filter --results_dir results --filter_correct_only --filter_prob_preset 80-100

 # Filter only for a specific model tag and dataset
 python src/pipeline/run_pipeline.py --mode filter --results_dir results --filter_correct_only --filter_prob_preset 70-90 --filter_model_tag distilBERT_finetuned --filter_datasets hippocorpus_test

 # Filter only shortcut: run all probability ranges in one command and print stats
 python src/pipeline/run_pipeline.py --mode filter --results_dir results --filter_correct_only --filter_all_ranges --filter_print_stats

 # Same all-ranges shortcut during evaluation (writes per-model + filtered outputs + stats)
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/distilBERT_finetuned --labeled_output per-model --filter_correct_only --filter_all_ranges --filter_print_stats
```

## Colab GPU Training

You can run training in Colab with GPU by using the same command interface.

Typical Colab flow (same as before):
1. Open `train_colab.ipynb`.
2. Runtime -> Change runtime type -> GPU.
3. Run the setup/GPU cells.
4. Set hyperparameters and model.
5. Run CV.
6. Run full training and package the model.

New Colab options:
1. Save one CV-selected model during CV (`SAVE_BEST_CV_MODEL = True`).
2. Choose how that single CV model is selected across folds (`CV_SELECTION_METRIC`).
3. Choose artifact source in packaging stage (`EXPORT_MODEL_SOURCE = "full_train"` or `"cv_best"`).

`CV_SELECTION_METRIC` choices:
- `accuracy`
- `eval_accuracy`
- `eval_loss`
- `validation_loss`
- `loss`

Selection direction:
- Accuracy metrics are maximized.
- Loss metrics are minimized.

Important behavior note:
- The per-fold Trainer still chooses the best checkpoint inside each fold.
- The custom layer only chooses which fold-model to keep as `cv_best_model`.

```bash
# CV (inspect cv_results.csv)
python src/pipeline/run_pipeline.py --mode cv --model distilbert

# CV + save one model selected across folds (default selector: accuracy)
python src/pipeline/run_pipeline.py --mode cv --model distilbert --save_best_cv_model

# CV + save one model selected across folds by validation loss
python src/pipeline/run_pipeline.py --mode cv --model distilbert --save_best_cv_model --cv_selection_metric eval_loss

# Full training after you are satisfied with CV
python src/pipeline/run_pipeline.py --mode full --model distilbert
```

If `--save_best_cv_model` is enabled, an additional artifact is written at:
- `models/<model>_<timestamp>/cv_best_model/`

The full-data trained model is saved under:
- `models/<model>_<timestamp>/model/`

In the Colab notebook (`train_colab.ipynb`):
- Cell 8 configures `SAVE_BEST_CV_MODEL`, `CV_SELECTION_METRIC`, and `EXPORT_MODEL_SOURCE`.
- Cell 12 runs CV with the selected cross-fold metric.
- Cell 18 packages either `full_train` or `cv_best` based on `EXPORT_MODEL_SOURCE`.

## Training Settings (DistilBERT Notebook Equivalent)

The training path is aligned with the DistilBERT notebook logic:
- training data: `data/hippocorpus/hippocorpus_training_truncated.csv`
- text column: `text_truncated`
- label column: `condition`
- label mapping: `truthful -> 0`, `deceptive -> 1`
- CV: 5-fold, group-aware using `truth-dec_pairId` when available
- epochs: 2
- learning rate: `5e-5`
- train/eval batch size: `32`
- weight decay: `0.01`
- eval/save strategy: per epoch
- fp16: disabled (as in notebook)

## Recommended Settings by Architecture

Use this section as the current reference for model-specific defaults from CV runs.
Update values as new tuning results become available.

- `distilbert` (confirmed baseline):
	- epochs: `2`
	- learning rate: `5e-5`
	- batch size: `32`
	- weight decay: `0.01`
- `bert`:
	- status: `confirmed`
	- epochs: `2`
	- learning rate: `2.5e-5`
	- batch size: `32`
	- weight decay: `0.01`
	- latest CV reference: accuracy `0.7609`, recall truthful `0.7468`, recall deceptive `0.7739`, overfit folds `0/5`
- `sbert`:
	- status: `confirmed`
	- epochs: `2`
	- learning rate: `5e-5`
	- batch size: `32`
	- weight decay: `0.01`
	- latest CV reference: accuracy `0.7922`, recall truthful `0.7881`, recall deceptive `0.7959`, overfit folds `0/5`
- `modernbert`:
	- status: `confirmed`
	- epochs: `2`
	- learning rate: `4e-5`
	- batch size: `32`
	- weight decay: `0.01`
	- latest CV reference: accuracy `0.7686`, recall truthful `0.7404`, recall deceptive `0.7947`, mean eval accuracy `0.7509`, mean eval loss `0.5158`, mean train loss `0.4786`, overfit folds `0/5`

Important label semantics:
- Training follows notebook semantics internally: `truthful -> 0`, `deceptive -> 1`.
- Evaluation outputs follow project semantics: `deceptive -> 0`, `truthful -> 1`.
- This conversion is handled explicitly during evaluation output generation.

## Outputs

Training artifacts:
- `models/<model>_<timestamp>/cv_results.csv`
- `models/<model>_<timestamp>/config.json`
- `models/<model>_<timestamp>/model/`

Evaluation artifacts:
- `results/summary_all_datasets.csv`
- `results/labeled_<dataset>.csv` (default, combined output)
- `results/labeled_<dataset>_<model>.csv` (optional, when `--labeled_output per-model` or `--labeled_output both` is used)
- `results/labeled_<dataset>_<model>_filtered_*.csv` (optional, when filter flags are used with per-model output)

`summary_all_datasets.csv` is append-only: each new evaluation run appends rows instead of overwriting prior results.

Combined labeled dataset CSVs contain the original dataset columns plus one set of prediction columns per evaluated model:
- `<model>_label_numeric`: numeric prediction (`deceptive=0`, `truthful=1`)
- `<model>_label`: string prediction (`deceptive` or `truthful`)
- `<model>_probability`: class probability of the predicted label

`--labeled_output combined` is the default and appends new model columns into the existing `labeled_<dataset>.csv` files.

`--labeled_output per-model` writes standalone labeled files for the current evaluated model only.

`--labeled_output both` writes both formats in the same evaluation run.

Reduced per-model CSV options:
- `--filter_correct_only`: keep only rows where prediction matches mapped ground truth label
- `--filter_prob_min`: keep rows with `<model>_probability >= min`
- `--filter_prob_max`: keep rows with `<model>_probability <= max`
- `--filter_prob_preset`: threshold shortcut (`70-100`, `80-100`, `70-90`, `80-90`)
- `--filter_all_ranges`: run all built-in threshold ranges in one command (`70-100`, `80-100`, `70-90`, `80-90`)
- `--filter_print_stats`: print row count and class proportions in terminal for each filtered output
- `--filter_datasets`: comma-separated dataset names to filter (e.g., `hippocorpus_test,decop`)
- `--filter_model_tag`: when using `--mode filter`, only process per-model CSVs for that model tag

Note: filtering reduced CSVs requires per-model predictions, so use `--labeled_output per-model` or `--labeled_output both`.

Filter-only mode:
- `--mode filter` scans `results/` (or `--results_dir`) for existing per-model labeled CSVs and writes reduced `..._filtered_*.csv` files without running evaluation again.

## Reproducibility Note

This pipeline is designed to reproduce the same training logic as the DistilBERT notebook.

To get as close as possible to identical results across runs, keep these fixed:
- same dataset files
- same model backbone
- same seed
- same library versions (use `environment.yml`)
- same hardware/runtime type when possible

Small numeric differences can still happen across different machines/runtimes, but the algorithmic training and evaluation logic is aligned.

## Git / Large Files

The entire `models/` directory is ignored in `.gitignore`.
This keeps large checkpoints out of Git history by default.
  