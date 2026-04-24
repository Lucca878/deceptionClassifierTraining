import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict

import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_PRESETS: Dict[str, str] = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "sbert": "sentence-transformers/all-mpnet-base-v2",
    "modernbert": "answerdotai/ModernBERT-base",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Data
    data_path: str = "data/hippocorpus/hippocorpus_training_truncated.csv"
    text_col: str = "text_truncated"
    label_col: str = "condition"          # raw column: "truthful" / "deceptive"
    group_col: str = "truth-dec_pairId"   # keeps paired stories together in splits

    # Model
    model_key: str = "distilbert"
    model_name: str = "distilbert-base-uncased"
    output_dir: str = "models"

    # Training — same defaults as the original DistilBERT notebook
    seed: int = 42
    num_folds: int = 5
    epochs: int = 2
    lr: float = 5e-5
    batch_size: int = 32
    weight_decay: float = 0.01


def make_config(
    model_key: str,
    model_name: str | None = None,
    output_root: str = "models",
    seed: int = 42,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    weight_decay: float | None = None,
) -> TrainConfig:
    if model_key not in MODEL_PRESETS:
        raise ValueError(f"Unsupported model '{model_key}'. Choices: {sorted(MODEL_PRESETS)}")

    chosen_model_name = model_name if model_name else MODEL_PRESETS[model_key]
    cfg = TrainConfig(
        model_key=model_key,
        model_name=chosen_model_name,
        output_dir=output_root,
        seed=seed,
    )
    
    # Override hyperparameters if provided
    if epochs is not None:
        cfg.epochs = epochs
    if lr is not None:
        cfg.lr = lr
    if batch_size is not None:
        cfg.batch_size = batch_size
    if weight_decay is not None:
        cfg.weight_decay = weight_decay
    
    return cfg


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(cfg: TrainConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    # Map text labels to binary: truthful=0, deceptive=1
    df["labels_binary"] = df[cfg.label_col].map({"truthful": 0, "deceptive": 1})
    df = df.dropna(subset=["labels_binary"]).copy()
    df["labels_binary"] = df["labels_binary"].astype(int)
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str)
    return df


# ---------------------------------------------------------------------------
# Splits — GroupKFold keeps truthful/deceptive pairs together
# ---------------------------------------------------------------------------

def create_splits(df: pd.DataFrame, cfg: TrainConfig):
    """
    Returns two dicts: train_splits and test_splits.
    Each key is a fold name (split_1 ... split_n).

    GroupKFold.split() returns row indices into df.
    We use df.iloc[train_idx] directly — this correctly keeps
    all rows sharing the same truth-dec_pairId on the same side
    of the split, with no data dropped.
    """
    group_kfold = GroupKFold(n_splits=cfg.num_folds)
    train_splits, test_splits = {}, {}

    for i, (train_idx, test_idx) in enumerate(
        group_kfold.split(df, groups=df[cfg.group_col]), start=1
    ):
        train_splits[f"split_{i}"] = df.iloc[train_idx].copy()
        test_splits[f"split_{i}"] = df.iloc[test_idx].copy()

    return train_splits, test_splits


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def tokenize(dataset: Dataset, tokenizer, cfg: TrainConfig) -> Dataset:
    def preprocess(examples):
        tokenized = tokenizer(examples[cfg.text_col], truncation=True)
        tokenized["labels"] = [int(v) for v in examples["labels_binary"]]
        return tokenized
    return dataset.map(preprocess, batched=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": float(accuracy_score(labels, predictions))}


# ---------------------------------------------------------------------------
# Training arguments — same as original notebook
# ---------------------------------------------------------------------------

def make_training_args(output_dir: str, cfg: TrainConfig, use_eval: bool = True) -> TrainingArguments:
    common_args = dict(
        output_dir=output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.epochs,
        fp16=False,
        push_to_hub=False,
        logging_strategy="steps",
        logging_steps=10,
        report_to=[],
        seed=cfg.seed,
    )

    if use_eval:
        return TrainingArguments(
            **common_args,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
        )

    return TrainingArguments(
        **common_args,
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
    )


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(df: pd.DataFrame, cfg: TrainConfig, run_dir: str) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_splits, test_splits = create_splits(df, cfg)
    all_results = []

    for split_name in train_splits:
        print(f"\n--- {split_name} ---")

        # Build HuggingFace datasets
        tok_train = tokenize(Dataset.from_pandas(train_splits[split_name], preserve_index=False), tokenizer, cfg)
        tok_test  = tokenize(Dataset.from_pandas(test_splits[split_name],  preserve_index=False), tokenizer, cfg)

        # Fresh model for each fold
        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

        trainer = Trainer(
            model=model,
            args=make_training_args(os.path.join(run_dir, "cv", split_name), cfg),
            train_dataset=tok_train,
            eval_dataset=tok_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        # Collect predictions
        preds = np.argmax(trainer.predict(tok_test).predictions, axis=1)
        labels = np.array(test_splits[split_name]["labels_binary"])

        fold_df = pd.DataFrame({
            "Split": split_name,
            "Prediction": preds,
            "Label": labels,
        })
        fold_df["Correct"] = fold_df["Prediction"] == fold_df["Label"]
        all_results.append(fold_df)

        del model, trainer
        torch.cuda.empty_cache()

    return pd.concat(all_results, ignore_index=True)


# ---------------------------------------------------------------------------
# Full training + save
# ---------------------------------------------------------------------------

def train_and_save(df: pd.DataFrame, cfg: TrainConfig, run_dir: str) -> str:
    """Train on the full dataset and save the model."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tok_full = tokenize(Dataset.from_pandas(df, preserve_index=False), tokenizer, cfg)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

    trainer = Trainer(
        model=model,
        args=make_training_args(os.path.join(run_dir, "full_train"), cfg, use_eval=False),
        train_dataset=tok_full,
        data_collator=data_collator,
    )
    trainer.train()

    save_dir = os.path.join(run_dir, "model")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to: {save_dir}")
    return save_dir


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_training(cfg: TrainConfig) -> str:
    set_seed(cfg.seed)

    run_dir = os.path.join(cfg.output_dir, f"{cfg.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    df = load_data(cfg)
    print(f"Loaded {len(df)} rows from {cfg.data_path}")

    # Cross-validation
    cv_results = run_cv(df, cfg, run_dir)
    cv_results.to_csv(os.path.join(run_dir, "cv_results.csv"), sep=";", index=False)

    acc = cv_results["Correct"].mean()
    print(f"\nCV accuracy: {acc:.4f}")

    # Full training
    final_dir = train_and_save(df, cfg, run_dir)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return final_dir


def run_cv_only(cfg: TrainConfig) -> str:
    """Run cross-validation only and save CV artifacts."""
    set_seed(cfg.seed)

    run_dir = os.path.join(cfg.output_dir, f"{cfg.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    df = load_data(cfg)
    print(f"Loaded {len(df)} rows from {cfg.data_path}")

    cv_results = run_cv(df, cfg, run_dir)
    cv_path = os.path.join(run_dir, "cv_results.csv")
    cv_results.to_csv(cv_path, sep=";", index=False)

    acc = cv_results["Correct"].mean()
    print(f"\nCV accuracy: {acc:.4f}")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"CV results saved at: {cv_path}")
    return cv_path


def run_full_only(cfg: TrainConfig) -> str:
    """Run full-data training only and save final model."""
    set_seed(cfg.seed)

    run_dir = os.path.join(cfg.output_dir, f"{cfg.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    df = load_data(cfg)
    print(f"Loaded {len(df)} rows from {cfg.data_path}")

    final_dir = train_and_save(df, cfg, run_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return final_dir


if __name__ == "__main__":
    cfg = TrainConfig()
    run_training(cfg)
