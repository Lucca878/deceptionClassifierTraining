import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Set

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


FILTER_PRESETS: dict[str, tuple[float, float]] = {
    "70-100": (0.70, 1.0),
    "80-100": (0.80, 1.0),
    "70-90": (0.70, 0.90),
    "80-90": (0.80, 0.90),
}
FILTER_PRESET_ORDER: list[str] = ["70-100", "80-100", "70-90", "80-90"]


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


def _format_threshold_token(value: Optional[float]) -> str:
    if value is None:
        return "na"
    return str(value).replace(".", "p")


def _filtered_per_model_labeled_path(
    output_dir: str,
    dataset_name: str,
    model_tag: str,
    correct_only: bool,
    prob_min: Optional[float],
    prob_max: Optional[float],
) -> str:
    suffix = _build_filter_suffix(correct_only, prob_min, prob_max)
    return os.path.join(output_dir, f"labeled_{dataset_name}_{model_tag}_{suffix}.csv")


def _build_filter_suffix(
    correct_only: bool,
    prob_min: Optional[float],
    prob_max: Optional[float],
) -> str:
    parts = ["filtered"]
    if correct_only:
        parts.append("correct")
    if prob_min is not None or prob_max is not None:
        parts.append(f"prob_{_format_threshold_token(prob_min)}_{_format_threshold_token(prob_max)}")
    return "_".join(parts)


def _resolve_filter_thresholds(
    prob_min: Optional[float],
    prob_max: Optional[float],
    prob_preset: Optional[str],
) -> tuple[Optional[float], Optional[float]]:
    if prob_preset is not None:
        if prob_preset not in FILTER_PRESETS:
            valid = ", ".join(sorted(FILTER_PRESETS))
            raise ValueError(f"Unknown filter preset '{prob_preset}'. Valid presets: {valid}")

        preset_min, preset_max = FILTER_PRESETS[prob_preset]
        if prob_min is None:
            prob_min = preset_min
        if prob_max is None:
            prob_max = preset_max

    return prob_min, prob_max


def _resolve_filter_ranges(
    prob_min: Optional[float],
    prob_max: Optional[float],
    prob_preset: Optional[str],
    all_ranges: bool,
) -> list[tuple[Optional[float], Optional[float]]]:
    if all_ranges:
        if prob_preset is not None or prob_min is not None or prob_max is not None:
            raise ValueError(
                "--filter_all_ranges cannot be combined with --filter_prob_preset, --filter_prob_min, or --filter_prob_max"
            )
        return [FILTER_PRESETS[p] for p in FILTER_PRESET_ORDER]

    resolved_min, resolved_max = _resolve_filter_thresholds(prob_min, prob_max, prob_preset)
    return [(resolved_min, resolved_max)]


def _extract_model_tag_from_columns(df: pd.DataFrame) -> Optional[str]:
    numeric_cols = [c for c in df.columns if c.endswith("_label_numeric")]
    if len(numeric_cols) != 1:
        return None
    return numeric_cols[0][: -len("_label_numeric")]


def _extract_dataset_name_from_per_model_filename(path: str, model_tag: str) -> Optional[str]:
    base = os.path.basename(path)
    prefix = "labeled_"
    suffix = f"_{model_tag}.csv"
    if not base.startswith(prefix) or not base.endswith(suffix):
        return None
    return base[len(prefix) : -len(suffix)]


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


def _filter_per_model_df(
    per_model_df: pd.DataFrame,
    ds: Optional[DatasetSpec],
    model_tag: str,
    correct_only: bool,
    prob_min: Optional[float],
    prob_max: Optional[float],
) -> pd.DataFrame:
    numeric_col = f"{model_tag}_label_numeric"
    probability_col = f"{model_tag}_probability"

    if numeric_col not in per_model_df.columns:
        raise ValueError(f"Missing expected prediction column: {numeric_col}")

    mask = pd.Series(True, index=per_model_df.index, dtype=bool)

    if prob_min is not None or prob_max is not None:
        if probability_col not in per_model_df.columns:
            raise ValueError(f"Missing expected probability column: {probability_col}")

        probs = pd.to_numeric(per_model_df[probability_col], errors="coerce")
        prob_mask = probs.notna()
        if prob_min is not None:
            prob_mask &= probs >= prob_min
        if prob_max is not None:
            prob_mask &= probs <= prob_max
        mask &= prob_mask

    if correct_only:
        if ds is None:
            raise ValueError("correct-only filter requires a known dataset spec")

        if ds.label_col not in per_model_df.columns:
            raise ValueError(
                f"Cannot apply correct-only filter for dataset '{ds.name}': missing label column '{ds.label_col}'"
            )

        true_raw = per_model_df[ds.label_col].apply(_normalize_label)
        true_project = true_raw.map(lambda v: pd.NA if v is None else 1 - int(v))
        pred_project = pd.to_numeric(per_model_df[numeric_col], errors="coerce")
        correct_mask = (
            true_project.notna()
            & pred_project.notna()
            & (pred_project.astype("Int64") == true_project.astype("Int64"))
        )
        mask &= correct_mask

    return per_model_df.loc[mask].copy()


def _filter_stats_text(filtered_df: pd.DataFrame, model_tag: str) -> str:
    label_col = f"{model_tag}_label"
    total = len(filtered_df)

    truthful = 0
    deceptive = 0
    if label_col in filtered_df.columns:
        normalized = filtered_df[label_col].fillna("").astype(str).str.lower().str.strip()
        truthful = int((normalized == "truthful").sum())
        deceptive = int((normalized == "deceptive").sum())

    truthful_pct = (100.0 * truthful / total) if total > 0 else 0.0
    deceptive_pct = (100.0 * deceptive / total) if total > 0 else 0.0

    return (
        f"rows={total} | truthful={truthful} ({truthful_pct:.1f}%) | "
        f"deceptive={deceptive} ({deceptive_pct:.1f}%)"
    )


def _compute_auc_from_saved_predictions(
    output_dir: str,
    dataset_name: str,
    model_tag: str,
) -> Optional[float]:
    ds = next((spec for spec in DATASETS if spec.name == dataset_name), None)
    if ds is None:
        return None

    shared_path = _shared_labeled_path(output_dir, dataset_name)
    if not os.path.exists(shared_path):
        return None

    df = pd.read_csv(shared_path)
    numeric_col = f"{model_tag}_label_numeric"
    probability_col = f"{model_tag}_probability"

    if ds.label_col not in df.columns or numeric_col not in df.columns or probability_col not in df.columns:
        return None

    true_raw = df[ds.label_col].apply(_normalize_label)
    true_project = true_raw.map(lambda v: pd.NA if v is None else 1 - int(v))
    pred_project = pd.to_numeric(df[numeric_col], errors="coerce")
    pred_conf = pd.to_numeric(df[probability_col], errors="coerce")

    valid = (
        true_project.notna()
        & pred_project.notna()
        & pred_conf.notna()
        & pred_project.isin([0, 1])
    )
    if int(valid.sum()) == 0:
        return None

    y_true = true_project.loc[valid].astype(int)
    if y_true.nunique() < 2:
        return None

    pred_vals = pred_project.loc[valid].astype(int)
    conf_vals = pred_conf.loc[valid].astype(float)
    y_score = np.where(pred_vals.to_numpy() == 1, conf_vals.to_numpy(), 1.0 - conf_vals.to_numpy())

    return float(roc_auc_score(y_true.to_numpy(), y_score))


def _backfill_auc_in_summary(summary_df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    if summary_df.empty:
        if "auc" not in summary_df.columns:
            summary_df["auc"] = pd.Series(dtype="Float64")
        return summary_df

    if "auc" not in summary_df.columns:
        summary_df["auc"] = pd.NA

    cache: dict[tuple[str, str], Optional[float]] = {}

    for idx, row in summary_df.iterrows():
        existing_auc = row.get("auc")
        if pd.notna(existing_auc):
            continue

        dataset_name = row.get("dataset")
        model_tag = row.get("model")
        if not isinstance(dataset_name, str) or not isinstance(model_tag, str):
            continue

        key = (dataset_name, model_tag)
        if key not in cache:
            cache[key] = _compute_auc_from_saved_predictions(
                output_dir=output_dir,
                dataset_name=dataset_name,
                model_tag=model_tag,
            )

        auc_value = cache[key]
        if auc_value is not None:
            summary_df.at[idx, "auc"] = round(auc_value, 4)

    return summary_df


def filter_existing_per_model_csvs(
    output_dir: str = "results",
    filter_correct_only: bool = False,
    filter_prob_min: Optional[float] = None,
    filter_prob_max: Optional[float] = None,
    filter_prob_preset: Optional[str] = None,
    filter_all_ranges: bool = False,
    filter_datasets: Optional[Set[str]] = None,
    filter_model_tag: Optional[str] = None,
) -> list[dict[str, str]]:
    ranges = _resolve_filter_ranges(
        filter_prob_min,
        filter_prob_max,
        filter_prob_preset,
        filter_all_ranges,
    )

    for range_min, range_max in ranges:
        if range_min is not None and range_max is not None and range_min > range_max:
            raise ValueError("filter_prob_min cannot be greater than filter_prob_max")

    filter_enabled = (
        filter_correct_only
        or any((range_min is not None or range_max is not None) for range_min, range_max in ranges)
    )
    if not filter_enabled:
        raise ValueError(
            "No filter criteria provided. Use --filter_correct_only, --filter_prob_min/--filter_prob_max, --filter_prob_preset, or --filter_all_ranges."
        )

    dataset_by_name = {ds.name: ds for ds in DATASETS}
    written_outputs: list[dict[str, str]] = []

    pattern = os.path.join(output_dir, "labeled_*.csv")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        if "_filtered_" in base:
            continue

        df = pd.read_csv(path)
        model_tag = _extract_model_tag_from_columns(df)
        if not model_tag:
            continue
        if filter_model_tag and model_tag != filter_model_tag:
            continue

        dataset_name = _extract_dataset_name_from_per_model_filename(path, model_tag)
        if not dataset_name:
            continue
        if filter_datasets is not None and dataset_name not in filter_datasets:
            continue

        ds_spec = dataset_by_name.get(dataset_name)
        for range_min, range_max in ranges:
            filtered_df = _filter_per_model_df(
                per_model_df=df,
                ds=ds_spec,
                model_tag=model_tag,
                correct_only=filter_correct_only,
                prob_min=range_min,
                prob_max=range_max,
            )
            out_path = _filtered_per_model_labeled_path(
                output_dir=output_dir,
                dataset_name=dataset_name,
                model_tag=model_tag,
                correct_only=filter_correct_only,
                prob_min=range_min,
                prob_max=range_max,
            )
            filtered_df.to_csv(out_path, index=False)
            written_outputs.append(
                {
                    "path": out_path,
                    "dataset": dataset_name,
                    "model": model_tag,
                    "filter": _build_filter_suffix(filter_correct_only, range_min, range_max),
                    "stats": _filter_stats_text(filtered_df, model_tag),
                }
            )

    return written_outputs


def evaluate_model_on_datasets(
    model_dir: str,
    output_dir: str = "results",
    labeled_output: str = "combined",
    filter_correct_only: bool = False,
    filter_prob_min: Optional[float] = None,
    filter_prob_max: Optional[float] = None,
    filter_prob_preset: Optional[str] = None,
    filter_all_ranges: bool = False,
    filter_print_stats: bool = False,
    filter_datasets: Optional[Set[str]] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    if labeled_output not in {"combined", "per-model", "both"}:
        raise ValueError("labeled_output must be one of: combined, per-model, both")

    ranges = _resolve_filter_ranges(
        filter_prob_min,
        filter_prob_max,
        filter_prob_preset,
        filter_all_ranges,
    )

    for range_min, range_max in ranges:
        if range_min is not None and range_max is not None and range_min > range_max:
            raise ValueError("filter_prob_min cannot be greater than filter_prob_max")

    filter_enabled = (
        filter_correct_only
        or any((range_min is not None or range_max is not None) for range_min, range_max in ranges)
    )

    if filter_enabled and labeled_output == "combined":
        raise ValueError(
            "Filtering reduced CSVs requires per-model predictions. Use --labeled_output per-model or --labeled_output both."
        )

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

        y_score_positive = [float(prob_vec[0]) for prob_vec in raw_probs]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_true, y_score_positive)

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
                "auc": round(float(auc), 4),
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

            should_filter_dataset = (
                filter_enabled
                and (filter_datasets is None or ds.name in filter_datasets)
            )

            if should_filter_dataset:
                for range_min, range_max in ranges:
                    filtered_df = _filter_per_model_df(
                        per_model_df=per_model_df,
                        ds=ds,
                        model_tag=model_tag,
                        correct_only=filter_correct_only,
                        prob_min=range_min,
                        prob_max=range_max,
                    )
                    filtered_out = _filtered_per_model_labeled_path(
                        output_dir=output_dir,
                        dataset_name=ds.name,
                        model_tag=model_tag,
                        correct_only=filter_correct_only,
                        prob_min=range_min,
                        prob_max=range_max,
                    )
                    filtered_df.to_csv(filtered_out, index=False)
                    if filter_print_stats:
                        stats = _filter_stats_text(filtered_df, model_tag)
                        print(
                            f"[filter] dataset={ds.name} model={model_tag} "
                            f"criteria={_build_filter_suffix(filter_correct_only, range_min, range_max)} | {stats}"
                        )

    summary_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(output_dir, "summary_all_datasets.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        summary_df = pd.concat([existing, summary_df], ignore_index=True)

    summary_df = _backfill_auc_in_summary(summary_df, output_dir)

    summary_df.to_csv(out_path, index=False)
    return out_path
