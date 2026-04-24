# Deception Classifier Pipeline

Clean end-to-end training and evaluation pipeline for deception detection.

The workflow is script-first and reproducible:
1. Train one model architecture.
2. Save the final model locally.
3. Evaluate the saved model across all datasets in `data/`.

## Fresh Clone Flow

If you are new to this repo:
1. Clone/download the repository.
2. Create and activate the environment.
3. Train one model (`distilbert`, `bert`, `sbert`, or `modernbert`).
4. Evaluate that trained model on all datasets.
5. Use the generated per-dataset labeled CSV outputs and summary CSV.

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

# Full-data training only (after choosing settings from CV)
python src/pipeline/run_pipeline.py --mode full --model distilbert

 # Evaluate only (existing trained model)
 # For models trained by this pipeline:
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/<model>_<timestamp>/model
 
 # For pre-trained models (e.g., distilBERT_finetuned):
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/distilBERT_finetuned
```

## Colab GPU Training

You can run training in Colab with GPU by using the same command interface.

Typical Colab flow:
1. Upload or clone this repo in Colab.
2. Runtime -> Change runtime type -> GPU.
3. Install environment dependencies (pip in Colab).
4. Run CV first, then full training.

```bash
# CV (inspect cv_results.csv)
python src/pipeline/run_pipeline.py --mode cv --model distilbert

# Full training after you are satisfied with CV
python src/pipeline/run_pipeline.py --mode full --model distilbert
```

The trained model is saved under `models/<model>_<timestamp>/model/`.

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
	- status: `TBD`

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
- `results/labeled_<dataset>_<model>.csv`

`summary_all_datasets.csv` is append-only: each new evaluation run appends rows instead of overwriting prior results.

Each labeled dataset CSV contains the original dataset columns plus:
- `<model>_label_numeric`: numeric prediction (`deceptive=0`, `truthful=1`)
- `<model>_label`: string prediction (`deceptive` or `truthful`)
- `<model>_probability`: class probability of the predicted label

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
  