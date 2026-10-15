# Deception Classifier Training & Evaluation Pipeline

[![Paper](https://img.shields.io/badge/arXiv-2610.13363-b31b1b.svg)](https://doi.org/10.48550/arXiv.2610.13363)
[![Model DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23196791.svg)](https://doi.org/10.5281/zenodo.23196791)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the training and evaluation pipeline for the deception classifiers used in:

> Pfründer, L., Loconte, R., Verschuere, B. & Kleinberg, B. Adversarial Attacks Reveal Divergent Deception Competence in Humans and Generative Language Models. Preprint at https://doi.org/10.48550/ARXIV.2610.13363 (2026).


The pipeline fine-tunes transformer encoders (DistilBERT, BERT, SBERT, ModernBERT) on the [hippocorpus](data/hippocorpus/) training split and evaluates the resulting classifiers across five deception datasets in `data/`. The workflow is seed-controlled, and reproducible end to end:
1. Train one model architecture.
2. Save the final model locally.
3. Evaluate the saved model on all datasets.

## Resources

- **Paper**: https://doi.org/10.48550/arXiv.2610.13363
- **Trained ModernBERT classifier** (automated deception classifier used in the paper): **https://doi.org/10.5281/zenodo.23196791**

## Pretrained Model

The trained ModernBERT classifier (`models/modernbert_trained`) is archived on Zenodo: [https://doi.org/10.5281/zenodo.23196791](https://doi.org/10.5281/zenodo.23196791).

Download and extract the archive, then load it with Hugging Face Transformers:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_dir = "modernbert_trained"  # path to the extracted Zenodo archive
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
```

Label semantics of the released model (training mapping): `0 = truthful`, `1 = deceptive`.

Or evaluate it on all datasets with this pipeline:

```bash
python src/pipeline/run_pipeline.py --mode eval --model_dir modernbert_trained
```

## Quickstart

1. Clone/download the repository.
2. Create and activate the environment (see Setup below).
3. Either download the trained ModernBERT model from Zenodo (above), or train a model (`distilbert`, `bert`, `sbert`, or `modernbert`).
4. Evaluate the model on all datasets.
5. A per-dataset labeled CSVs and a summary CSV is stored in `results/`.

## Setup

```bash
conda env create -f environment.yml
conda activate deception
```

If conda shell hooks are not active yet:

```bash
source ~/.zshrc
```

**Note:** All dependencies are in both `environment.yml` (conda) and `requirements.txt` (pip). Use `environment.yml` for the recommended setup with conda, or `requirements.txt` if installing with pip.

## Single End-to-End Command

```bash
python src/pipeline/run_pipeline.py --mode all --model modernbert
```

Supported model presets:
- `distilbert`
- `bert`
- `sbert`
- `modernbert`

**Default behavior:**
- Training artifacts are saved to `models/` (use `--output_root` to override)
- Evaluation results are saved to `results/` (use `--results_dir` to override)
- The pipeline is reproducible with seed `42` (use `--seed` to change)

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
python src/pipeline/run_pipeline.py --mode train --model modernbert

# CV only (tuning stage)
python src/pipeline/run_pipeline.py --mode cv --model modernbert

# CV only + save a single CV-selected model
python src/pipeline/run_pipeline.py --mode cv --model modernbert --save_best_cv_model

# CV only + save a single CV-selected model using loss as cross-fold selector
python src/pipeline/run_pipeline.py --mode cv --model modernbert --save_best_cv_model --cv_selection_metric eval_loss

# Full-data training only (after choosing settings from CV)
python src/pipeline/run_pipeline.py --mode full --model modernbert

 # Evaluate only (existing trained model)
 # For models trained by this pipeline:
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/<model>_<timestamp>/model
 
 # For a downloaded/pre-trained model (e.g., the released modernbert_trained):
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/modernbert_trained

 # Control labeled CSV output style:
 # combined (default): one labeled_<dataset>.csv with appended columns for each evaluated model
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/modernbert_trained --labeled_output combined

 # per-model: one labeled_<dataset>_<model>.csv per evaluation run
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/modernbert_trained --labeled_output per-model

 # both: write both combined and per-model labeled CSV outputs
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/modernbert_trained --labeled_output both

 # Optional: write reduced per-model CSVs (filter by confidence and correctness)
 # This creates extra files like labeled_<dataset>_<model>_filtered_*.csv
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/modernbert_trained --labeled_output per-model --filter_correct_only --filter_prob_min 0.70 --filter_prob_max 1.0

 # Optional: apply filtering only to specific datasets
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/modernbert_trained --labeled_output per-model --filter_correct_only --filter_prob_min 0.80 --filter_prob_max 1.0 --filter_datasets hippocorpus_test

 # Filter only (no model inference): reduce already-generated per-model CSV files in results/
 python src/pipeline/run_pipeline.py --mode filter --results_dir results --filter_correct_only --filter_prob_preset 80-100

 # Filter only for a specific model tag and dataset
 python src/pipeline/run_pipeline.py --mode filter --results_dir results --filter_correct_only --filter_prob_preset 70-90 --filter_model_tag modernbert_trained --filter_datasets hippocorpus_test

 # Filter only shortcut: run all probability ranges in one command and print stats
 python src/pipeline/run_pipeline.py --mode filter --results_dir results --filter_correct_only --filter_all_ranges --filter_print_stats

 # Same all-ranges shortcut during evaluation (writes per-model + filtered outputs + stats)
 python src/pipeline/run_pipeline.py --mode eval --model_dir models/modernbert_trained --labeled_output per-model --filter_correct_only --filter_all_ranges --filter_print_stats
```

## Colab GPU Training

You can run training in Colab with GPU by using the same command interface and defaults as the local setup.

**Where outputs land — Colab vs. local:**

- **Local (VS Code or terminal):** everything is written straight into your local project — trained models to `models/`, evaluation results to `results/`. Nothing else to do.
- **Colab:** training and packaging first write into the *ephemeral Colab runtime* (`models/` and a zipped model, section 5). Because the runtime is wiped when the session ends, section 6 optionally copies that model zip to Google Drive so it persists. Drive is an export step only — not the primary save location — and is skipped entirely for local runs.

Typical Colab flow:
1. Open `train_colab.ipynb`.
2. Runtime → Change runtime type → GPU (e.g., A100).
3. Run the setup/GPU cells (clone, install dependencies, verify GPU).
4. Configure hyperparameters in section 2.5 (optional; leave `None` for defaults).
5. Select model in section 2.6.
6. Run cross-validation in section 3.
7. Inspect CV results in section 4.
8. Run full training and package the model in section 5 (writes the zip inside the runtime).
9. (Optional) Copy the model zip to Google Drive in section 6 so it survives the session.
10. (Optional) Run evaluation in section 7 and copy the results zip to Drive — or skip and evaluate locally instead.

Note on evaluation: for most workflows, evaluate **locally** so results are written directly into your local `results/` project folder. Colab evaluation (section 7) is only needed if you want to keep results produced inside the Colab runtime.

Key Colab options:
1. **Save best CV model:** Set `SAVE_BEST_CV_MODEL = True` in section 2.5 to save the best model selected across CV folds.
2. **CV selection metric:** Configure `CV_SELECTION_METRIC` to choose how fold models are compared (default: `accuracy`).
3. **Export source:** Set `EXPORT_MODEL_SOURCE` in section 5 to package either the full-trained model or the best CV model.

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
python src/pipeline/run_pipeline.py --mode cv --model modernbert

# CV + save one model selected across folds (default selector: accuracy)
python src/pipeline/run_pipeline.py --mode cv --model modernbert --save_best_cv_model

# CV + save one model selected across folds by validation loss
python src/pipeline/run_pipeline.py --mode cv --model modernbert --save_best_cv_model --cv_selection_metric eval_loss

# Full training after you are satisfied with CV
python src/pipeline/run_pipeline.py --mode full --model modernbert
```

If `--save_best_cv_model` is enabled, an additional artifact is written at:
- `models/<model>_<timestamp>/cv_best_model/`

The full-data trained model is saved under:
- `models/<model>_<timestamp>/model/`

In the Colab notebook (`train_colab.ipynb`):
- Cell 8 configures `SAVE_BEST_CV_MODEL`, `CV_SELECTION_METRIC`, and `EXPORT_MODEL_SOURCE`.
- Cell 12 runs CV with the selected cross-fold metric.
- Cell 18 packages either `full_train` or `cv_best` based on `EXPORT_MODEL_SOURCE`.

## Training Settings (Default)

The default training settings are the following:
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
- fp16: disabled

## Recommended Settings by Architecture

Use this section as the current reference for model-specific defaults from CV runs. These are the training settings for all metrics reported in the paper.

- `distilbert` :
	- status: `confirmed`
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
- `sbert`:
	- status: `confirmed`
	- epochs: `2`
	- learning rate: `5e-5`
	- batch size: `32`
	- weight decay: `0.01`
- `modernbert`:
	- status: `confirmed`
	- epochs: `2`
	- learning rate: `4e-5`
	- batch size: `32`
	- weight decay: `0.01`

Important label semantics:
- Training labeling: `truthful -> 0`, `deceptive -> 1`.
- Evaluation outputs follow project labeling: `deceptive -> 0`, `truthful -> 1`.
- This conversion is handled explicitly during evaluation output generation.

## Outputs

Training artifacts:
- `models/<model>_<timestamp>/cv_results.csv` — Cross-validation results
- `models/<model>_<timestamp>/config.json` — Training configuration
- `models/<model>_<timestamp>/model/` — Final trained model

Evaluation artifacts:
- `results/summary_all_datasets.csv` — Summary metrics across all datasets
- `results/labeled_<dataset>.csv` — Default combined labeled output
- `results/labeled_<dataset>_<model>.csv` — Per-model labeled output (optional)
- `results/labeled_<dataset>_<model>_filtered_*.csv` — Filtered per-model output (optional)

`summary_all_datasets.csv` is append-only: each new evaluation run appends rows instead of overwriting prior results.

AUC behavior:
- During evaluation (`--mode eval` and `--mode all`), ROC AUC is computed directly from model probabilities and written to `results/summary_all_datasets.csv` as `auc`.
- After evaluation, missing `auc` values in existing summary rows can be backfilled from combined labeled files (`results/labeled_<dataset>.csv`) when `<model>_label_numeric` and `<model>_probability` columns are present.
- This means you do not need to rerun model training to recover AUC when those prediction columns already exist.

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

## Repository Structure

```
data/                  Input datasets (CSV)
models/                Locally saved model checkpoints (git-ignored)
results/               Evaluation outputs: labeled CSVs + summary metrics
src/pipeline/
  run_pipeline.py      Single CLI: train / cv / full / eval / filter modes
  train.py             Group-aware cross-validation + full-data training
  evaluate.py          Multi-dataset evaluation and labeled-CSV filtering
train_colab.ipynb      Colab notebook wrapper around the same pipeline
environment.yml        Conda environment (pinned)
requirements.txt       Pip requirements (pinned)
```

## Datasets

Training uses the hippocorpus training split only (`data/hippocorpus/hippocorpus_training_truncated.csv`). Evaluation covers five datasets:

| Key (in `results/`) | Source | n | Description |
|---|---|---|---|
| `hippocorpus_test` | `data/hippocorpus/hippocorpus_test_truncated.csv` | 505 | hippocorpus test split (truncated stories) |
| `deceptive_intentions` | `data/deceptiveIntention/deceptiveIntentions.csv` | 1,640 | Deceptive Intentions dataset |
| `opinion_spam` | `data/opinionSpam/deceptive-opinion.csv` | 1,600 | Deceptive Opinion Spam corpus |
| `decop` | `data/opinionSpam2/DeCop.csv` | 2,500 | DeCop dataset |
| `real_life_trial` | `data/realLifeTrial/realLifeTrial.csv` | 121 | Real-Life Trial transcripts |

### Dataset References

All five datasets are openly available from the original authors. Please cite the original publications when using them:

- **hippocorpus**: Sap, M., Horvitz, E., Choi, Y., Smith, N. A. & Pennebaker, J. *Recollection versus Imagination: Exploring Human Memory and Cognition via Neural Language Models.* In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020), 1970–1978. https://doi.org/10.18653/v1/2020.acl-main.178
- **Deceptive Intentions**: Kleinberg, B., Van Der Toolen, Y., Vrij, A., Arntz, A. & Verschuere, B. *Automated verbal credibility assessment of intentions: The model statement technique and predictive modeling.* Applied Cognitive Psychology 32, 354–366 (2018). https://doi.org/10.1002/acp.3407
- **Deceptive Opinion Spam**: Ott, M., Choi, Y., Cardie, C. & Hancock, J. T. *Finding Deceptive Opinion Spam by Any Stretch of the Imagination.* In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011), 309–319.
- **DeCop**: Capuozzo, P., Lauriola, I., Strapparava, C., Aiolli, F. & Sartori, G. *DecOp: A Multilingual and Multi-domain Corpus For Detecting Deception In Typed Text.* In Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020), 1423–1430.
- **Real-Life Trial**: Pérez-Rosas, V., Abouelenien, M., Mihalcea, R. & Burzo, M. *Deception Detection using Real-life Trial Data.* In Proceedings of the 2015 ACM on International Conference on Multimodal Interaction (ICMI 2015), 59–66. https://doi.org/10.1145/2818346.2820758

## Reproducibility Note

To reproduce the paper's results, keep these fixed:
- same dataset files
- same models
- same seed
- same training settings
- same library versions (use `environment.yml`)
- same hardware/runtime type when possible

## Git / Large Files

The entire `models/` directory is ignored in `.gitignore`.
This keeps large checkpoints out of Git history by default.
The released ModernBERT model is distributed via Zenodo instead (see [Pretrained Model](#pretrained-model)).

## Citation

If you use this code or the trained model, please cite the paper:

```bibtex
@article{pfrunder2026adversarial,
  title   = {Adversarial Attacks Reveal Divergent Deception Competence in Humans and Generative Language Models},
  author  = {Pfründer, Lucca and Loconte, Riccardo and Verschuere, Bruno and Kleinberg, Bennett},
  journal = {arXiv preprint arXiv:2610.13363},
  year    = {2026},
  doi     = {10.48550/arXiv.2610.13363},
  url     = {https://doi.org/10.48550/arXiv.2610.13363}
}
```

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — see [LICENSE](LICENSE).
  