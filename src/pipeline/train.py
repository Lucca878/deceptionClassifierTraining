import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, KFold
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


@dataclass
class TrainConfig:
    data_path: str = "data/hippocorpus/hippocorpus_training_truncated.csv"
    text_col: str = "text_truncated"
    label_col: str = "condition"
    group_col: str = "truth-dec_pairId"
    model_key: str = "distilbert"
    model_name: str = "distilbert-base-uncased"
    output_root: str = "models"
    seed: int = 42
    num_folds: int = 5
    epochs: int = 2
    lr: float = 5e-5
    train_batch_size: int = 32
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    save_total_limit: int = 3
    logging_steps: int = 10
    max_length: int = 0
    report_to: str = "tensorboard"


def make_config(
    model_key: str,
    model_name: str | None = None,
    output_root: str = "models",
    seed: int = 42,
) -> TrainConfig:
    if model_key not in MODEL_PRESETS:
        raise ValueError(f"Unsupported model '{model_key}'. Choices: {sorted(MODEL_PRESETS)}")

    chosen_model_name = model_name if model_name else MODEL_PRESETS[model_key]
    return TrainConfig(
        model_key=model_key,
        model_name=chosen_model_name,
        output_root=output_root,
        seed=seed,
    )


def load_and_prepare_data(cfg: TrainConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)

    if cfg.text_col not in df.columns:
        raise ValueError(f"Missing text column '{cfg.text_col}' in {cfg.data_path}")
    if cfg.label_col not in df.columns:
        raise ValueError(f"Missing label column '{cfg.label_col}' in {cfg.data_path}")

    df = df.copy()
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str)
    df["labels"] = df[cfg.label_col]

    # Keep label mapping identical to the DistilBERT notebook.
    df["labels_binary"] = df["labels"].map({"truthful": 0, "deceptive": 1})
    df = df.dropna(subset=["labels_binary"]).copy()
    df["labels_binary"] = df["labels_binary"].astype(int)

    return df


def create_splits(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    train_splits: Dict[str, pd.DataFrame] = {}
    test_splits: Dict[str, pd.DataFrame] = {}

    if cfg.group_col in df.columns:
        splitter = GroupKFold(n_splits=cfg.num_folds)
        split_iter = splitter.split(df, groups=df[cfg.group_col])
    else:
        splitter = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)
        split_iter = splitter.split(df)

    for idx, (train_idx, test_idx) in enumerate(split_iter, start=1):
        split_name = f"split_{idx}"
        train_splits[split_name] = df.iloc[train_idx].copy()
        test_splits[split_name] = df.iloc[test_idx].copy()

    return train_splits, test_splits


def build_preprocess_fn(tokenizer, cfg: TrainConfig):
    def preprocess_function(examples):
        kwargs = {"truncation": True}
        if cfg.max_length > 0:
            kwargs["max_length"] = cfg.max_length

        tokenized_examples = tokenizer(examples[cfg.text_col], **kwargs)
        tokenized_examples["labels"] = [int(v) for v in examples["labels_binary"]]
        return tokenized_examples

    return preprocess_function


def build_training_args(output_dir: str, cfg: TrainConfig) -> TrainingArguments:
    report_to = [] if cfg.report_to == "none" else [cfg.report_to]
    # Keep this fixed to match the DistilBERT notebook training arguments.
    use_fp16 = False

    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        num_train_epochs=cfg.epochs,
        load_best_model_at_end=True,
        fp16=use_fp16,
        push_to_hub=False,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        report_to=report_to,
        seed=cfg.seed,
    )


def run_cv_training(df: pd.DataFrame, cfg: TrainConfig, run_dir: str) -> pd.DataFrame:
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": float(accuracy_score(labels, predictions))}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    preprocess_function = build_preprocess_fn(tokenizer, cfg)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_splits, test_splits = create_splits(df, cfg)
    all_results = []

    for split_name in train_splits:
        data_train = Dataset.from_pandas(train_splits[split_name], preserve_index=False)
        data_test = Dataset.from_pandas(test_splits[split_name], preserve_index=False)

        tok_data_train = data_train.map(preprocess_function, batched=True)
        tok_data_test = data_test.map(preprocess_function, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

        split_out_dir = os.path.join(run_dir, "cv_checkpoints", split_name)
        training_args = build_training_args(split_out_dir, cfg)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tok_data_train,
            eval_dataset=tok_data_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.evaluate()

        predictions = trainer.predict(tok_data_test)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = np.array(test_splits[split_name]["labels_binary"]).astype(int)

        participant_col = "index" if "index" in test_splits[split_name].columns else None
        participant_values = (
            test_splits[split_name][participant_col].to_numpy()
            if participant_col
            else test_splits[split_name].index.to_numpy()
        )

        split_df = pd.DataFrame(
            {
                "Participant_id": participant_values,
                "Split": split_name,
                "Prediction": pred_labels,
                "Label": true_labels,
            }
        )
        split_df["Correct_predictions"] = split_df["Prediction"] == split_df["Label"]
        all_results.append(split_df)

    return pd.concat(all_results, ignore_index=True)


def train_full_and_save(df: pd.DataFrame, cfg: TrainConfig, run_dir: str) -> str:
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": float(accuracy_score(labels, predictions))}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    preprocess_function = build_preprocess_fn(tokenizer, cfg)

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)
    full_ds = Dataset.from_pandas(df, preserve_index=False)
    tok_full_ds = full_ds.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    full_train_dir = os.path.join(run_dir, "full_train_checkpoints")
    training_args = build_training_args(full_train_dir, cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_full_ds,
        eval_dataset=tok_full_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_dir = os.path.join(run_dir, "model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    return final_dir


def save_run_metadata(cfg: TrainConfig, run_dir: str, cv_results: pd.DataFrame, final_dir: str) -> None:
    results_path = os.path.join(run_dir, "cv_results.csv")
    cv_results.to_csv(results_path, sep=";", index=False)

    deceptive = cv_results[cv_results["Label"] == 1]
    truthful = cv_results[cv_results["Label"] == 0]

    summary = {
        "overall_accuracy": float(cv_results["Correct_predictions"].mean()),
        "deceptive_recall": float(deceptive["Correct_predictions"].mean()) if len(deceptive) else None,
        "truthful_recall": float(truthful["Correct_predictions"].mean()) if len(truthful) else None,
        "n_rows": int(len(cv_results)),
    }

    payload = {
        "config": asdict(cfg),
        "summary": summary,
        "artifacts": {
            "cv_results_csv": results_path,
            "final_model_dir": final_dir,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_training(cfg: TrainConfig) -> str:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    set_seed(cfg.seed)

    run_name = f"{cfg.model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(cfg.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    df = load_and_prepare_data(cfg)
    cv_results = run_cv_training(df, cfg, run_dir)
    final_dir = train_full_and_save(df, cfg, run_dir)
    save_run_metadata(cfg, run_dir, cv_results, final_dir)

    return final_dir
