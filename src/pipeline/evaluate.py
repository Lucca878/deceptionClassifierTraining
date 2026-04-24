import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class DatasetSpec:
    name: str
    path: str
    text_col: str
    label_col: str
    sep: str = ","


DATASETS: List[DatasetSpec] = [
    DatasetSpec(
        name="deceptive_intentions",
        path="data/deceptiveIntention/deceptiveIntentions.csv",
        text_col="q1",
        label_col="outcome_class",
    ),
    DatasetSpec(
        name="hippocorpus_test",
        path="data/hippocorpus/hippocorpus_test_truncated.csv",
        text_col="text_truncated",
        label_col="condition",
    ),
    DatasetSpec(
        name="opinion_spam",
        path="data/opinionSpam/deceptive-opinion.csv",
        text_col="text",
        label_col="condition",
        sep=";",
    ),
    DatasetSpec(
        name="decop",
        path="data/opinionSpam2/DeCop.csv",
        text_col="sent",
        label_col="labels",
    ),
    DatasetSpec(
        name="real_life_trial",
        path="data/realLifeTrial/realLifeTrial.csv",
        text_col="text",
        label_col="condition",
    ),
]


def _normalize_label(value):
    if pd.isna(value):
        return None

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"deceptive", "deception", "lie", "false", "f", "1"}:
            return 1
        if v in {"truthful", "truth", "true", "t", "0"}:
            return 0
        return None

    if isinstance(value, (int, np.integer, float, np.floating)):
        if int(value) == 1:
            return 1
        if int(value) == 0:
            return 0

    return None


def _predict_batch(texts, model, tokenizer, device: str, batch_size: int = 32):
    preds = []
    probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
            batch_preds = np.argmax(batch_probs, axis=1).tolist()

        preds.extend(batch_preds)
        probs.extend(batch_probs.tolist())

    return preds, probs


def _raw_to_project_label(raw_label: int) -> int:
    """
    Notebook training uses truthful=0 and deceptive=1.
    Project output convention requires deceptive=0 and truthful=1.
    """
    return 1 - int(raw_label)


def _project_label_to_str(label_num: int) -> str:
    return "truthful" if int(label_num) == 1 else "deceptive"


def _raw_probs_to_project_conf(raw_probs: list[float], project_label: int) -> float:
    # project 0 (deceptive) corresponds to raw class 1; project 1 (truthful) to raw class 0.
    raw_idx = 1 - int(project_label)
    return float(raw_probs[raw_idx])


def _shared_labeled_path(output_dir: str, dataset_name: str) -> str:
    return os.path.join(output_dir, f"labeled_{dataset_name}.csv")


def _per_model_labeled_path(output_dir: str, dataset_name: str, model_tag: str) -> str:
    return os.path.join(output_dir, f"labeled_{dataset_name}_{model_tag}.csv")


def _can_reuse_labeled_df(candidate_df: pd.DataFrame, source_df: pd.DataFrame) -> bool:
    return len(candidate_df) == len(source_df) and all(col in candidate_df.columns for col in source_df.columns)


def _load_or_init_labeled_df(source_df: pd.DataFrame, output_dir: str, dataset_name: str) -> pd.DataFrame:
    shared_path = _shared_labeled_path(output_dir, dataset_name)
    if os.path.exists(shared_path):
        labeled_df = pd.read_csv(shared_path)
        if not _can_reuse_labeled_df(labeled_df, source_df):
            raise ValueError(
                f"Existing labeled file has incompatible shape or columns: {shared_path}"
            )
        return labeled_df

    labeled_df = source_df.copy()
    legacy_pattern = os.path.join(output_dir, f"labeled_{dataset_name}_*.csv")
    for legacy_path in sorted(glob.glob(legacy_pattern)):
        legacy_df = pd.read_csv(legacy_path)
        if not _can_reuse_labeled_df(legacy_df, source_df):
            continue

        for column in legacy_df.columns:
            if column not in labeled_df.columns:
                labeled_df[column] = legacy_df[column]

    return labeled_df


def _ensure_prediction_columns(labeled_df: pd.DataFrame, model_tag: str) -> tuple[str, str, str]:
    numeric_col = f"{model_tag}_label_numeric"
    label_col = f"{model_tag}_label"
    probability_col = f"{model_tag}_probability"

    if numeric_col not in labeled_df.columns:
        labeled_df[numeric_col] = pd.Series(
            pd.array([pd.NA] * len(labeled_df), dtype="Int64"),
            index=labeled_df.index,
        )
    else:
        labeled_df[numeric_col] = pd.array(labeled_df[numeric_col], dtype="Int64")

    if label_col not in labeled_df.columns:
        labeled_df[label_col] = ""
    else:
        labeled_df[label_col] = labeled_df[label_col].fillna("").astype(str)

    if probability_col not in labeled_df.columns:
        labeled_df[probability_col] = pd.Series(
            pd.array([pd.NA] * len(labeled_df), dtype="Float64"),
            index=labeled_df.index,
        )
    else:
        labeled_df[probability_col] = pd.array(labeled_df[probability_col], dtype="Float64")

    return numeric_col, label_col, probability_col


def _write_predictions(
    labeled_df: pd.DataFrame,
    valid_idx: list[int],
    pred_label_num: list[int],
    pred_label_str: list[str],
    pred_conf: list[float],
    model_tag: str,
) -> pd.DataFrame:
    numeric_col, label_col, probability_col = _ensure_prediction_columns(labeled_df, model_tag)

    for idx, num, label, conf in zip(valid_idx, pred_label_num, pred_label_str, pred_conf):
        labeled_df.at[idx, numeric_col] = int(num)
        labeled_df.at[idx, label_col] = label
        labeled_df.at[idx, probability_col] = round(conf, 6)

    return labeled_df


def evaluate_model_on_datasets(
    model_dir: str,
    output_dir: str = "results",
    labeled_output: str = "combined",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if labeled_output not in {"combined", "per-model", "both"}:
        raise ValueError("labeled_output must be one of: combined, per-model, both")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device).eval()

    summary_rows = []
    model_tag = os.path.basename(os.path.normpath(model_dir)) or "model"
    evaluated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for ds in DATASETS:
        if not os.path.exists(ds.path):
            continue

        df = pd.read_csv(ds.path, sep=ds.sep)
        if ds.text_col not in df.columns or ds.label_col not in df.columns:
            continue

        data = df[[ds.text_col, ds.label_col]].copy()
        data[ds.text_col] = data[ds.text_col].fillna("").astype(str)
        data["label"] = data[ds.label_col].apply(_normalize_label)
        data = data.dropna(subset=["label"]).copy()
        if len(data) == 0:
            continue

        # Ground truth in project convention (deceptive=0, truthful=1).
        y_true = [1 - int(v) for v in data["label"].astype(int).tolist()]

        raw_pred, raw_probs = _predict_batch(
            data[ds.text_col].tolist(), model, tokenizer, device=device
        )
        y_pred = [_raw_to_project_label(p) for p in raw_pred]

        pred_label_num = y_pred
        pred_label_str = [_project_label_to_str(v) for v in pred_label_num]
        pred_conf = [
            _raw_probs_to_project_conf(prob_vec, proj_lbl)
            for prob_vec, proj_lbl in zip(raw_probs, pred_label_num)
        ]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        summary_rows.append(
            {
                "evaluated_at": evaluated_at,
                "model": model_tag,
                "dataset": ds.name,
                "n": len(data),
                "accuracy": round(acc, 4),
                "f1_macro": round(f1, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

        valid_idx = data.index.tolist()

        if labeled_output in {"combined", "both"}:
            labeled_df = _load_or_init_labeled_df(df, output_dir, ds.name)
            labeled_df = _write_predictions(
                labeled_df,
                valid_idx,
                pred_label_num,
                pred_label_str,
                pred_conf,
                model_tag,
            )
            labeled_out = _shared_labeled_path(output_dir, ds.name)
            labeled_df.to_csv(labeled_out, index=False)

        if labeled_output in {"per-model", "both"}:
            per_model_df = _write_predictions(
                df.copy(),
                valid_idx,
                pred_label_num,
                pred_label_str,
                pred_conf,
                model_tag,
            )
            labeled_out = _per_model_labeled_path(output_dir, ds.name, model_tag)
            per_model_df.to_csv(labeled_out, index=False)

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(output_dir, "summary_all_datasets.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

    summary_df.to_csv(out_path, index=False)
    return out_path
