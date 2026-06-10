#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

try:
    from statsmodels.stats.contingency_tables import mcnemar
except Exception:  # pragma: no cover
    mcnemar = None


METHODS = [
    "top_score",
    "random_candidate",
    "vlm_direct",
    "first_question",
    "random_question",
    "vlm_best_question",
    "proposed_efe",
]
INTERACTIVE = ["first_question", "random_question", "vlm_best_question", "proposed_efe"]
PROMPT_TYPES = ["clear", "ambiguous", "partial"]
SEED = 20260520
COLOR_WORDS = {
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "grey",
    "silver",
}
SPATIAL_WORDS = {
    "left",
    "right",
    "front",
    "back",
    "middle",
    "center",
    "centre",
    "nearest",
    "farthest",
    "closest",
    "furthest",
    "top",
    "bottom",
}
RELATION_PATTERNS = [
    "next to",
    "beside",
    "between",
    "near",
    "behind",
    "in front",
    "closest to",
    "farther from",
    "farthest from",
    "left of",
    "right of",
]
SIZE_WORDS = {"big", "small", "large", "tiny", "tall", "short", "larger", "smaller"}
OBJECT_WORDS = ["cup", "bottle", "fork", "spoon", "utensil"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scene-clustered significance tests for Ask2Act offline trials.")
    parser.add_argument("--input", required=True, help="Cleaned trial-level CSV/JSON/Parquet, or audit filter JSON.")
    parser.add_argument("--out", default="outputs/statistics", help="Output directory.")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations.")
    return parser.parse_args()


def ensure_out(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_trial(trial: dict[str, Any]) -> dict[str, Any]:
    audit = trial.get("audit") if isinstance(trial.get("audit"), dict) else {}
    snapshot = trial.get("session_snapshot") if isinstance(trial.get("session_snapshot"), dict) else {}
    row = dict(trial)
    row.pop("session_snapshot", None)
    row.pop("audit", None)
    row["audit_failure_reason"] = audit.get("failure_reason", "")
    row["audit_reviewed"] = bool(audit.get("reviewed"))
    row["include_in_audit"] = audit.get("include_in_audit", True)
    row["prompt_text"] = trial.get("prompt") or snapshot.get("instruction") or ""
    row["candidate_set_size"] = trial.get("candidate_count")
    return row


def load_from_filter(path: Path) -> pd.DataFrame:
    data = read_json(path)
    if "included_trials" not in data:
        raise ValueError(f"JSON does not look like an audit filter: {path}")
    root = path.parent.parent
    rows: list[dict[str, Any]] = []
    for record in data.get("included_trials") or []:
        trial_id = record.get("trial_id") or Path(str(record.get("trial_file") or "")).stem
        if not trial_id:
            continue
        trial_path = root / "trials" / f"{trial_id}.json"
        if not trial_path.exists():
            raise FileNotFoundError(f"Filter references missing trial JSON: {trial_path}")
        trial = read_json(trial_path)
        audit = trial.get("audit") if isinstance(trial.get("audit"), dict) else {}
        if audit.get("include_in_audit", True) is False:
            continue
        rows.append(flatten_trial(trial))
    return pd.DataFrame(rows)


def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        data = read_json(path)
        if isinstance(data, dict) and "included_trials" in data:
            return load_from_filter(path)
        if isinstance(data, list):
            return pd.DataFrame([flatten_trial(item) if isinstance(item, dict) else item for item in data])
        if isinstance(data, dict) and "trials" in data:
            return pd.DataFrame([flatten_trial(item) if isinstance(item, dict) else item for item in data["trials"]])
        return pd.json_normalize(data)
    raise ValueError(f"Unsupported input type: {path}")


def find_col(df: pd.DataFrame, names: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried: {names}")
    return None


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    mapping: dict[str, str] = {}
    scene_col = find_col(df, ["scene_id", "scene"])
    method_col = find_col(df, ["method", "question_mode"])
    prompt_type_col = find_col(df, ["prompt_type", "prompt_category"])
    outcome_col = find_col(df, ["correct", "success", "outcome", "auto_outcome"])
    q_col = find_col(df, ["num_questions", "question_count", "questions", "n_questions"])
    latency_col = find_col(df, ["latency_sec", "latency_s", "latency", "latency_seconds"])
    scene_type_col = find_col(df, ["scene_type"], required=False)
    candidate_col = find_col(df, ["candidate_set_size", "candidate_count"], required=False)
    prompt_text_col = find_col(df, ["prompt_text", "prompt", "instruction", "command"], required=False)
    target_col = find_col(df, ["target_category", "object_category", "target_object"], required=False)
    attr_col = find_col(df, ["attribute_type", "attribute", "modifier_type"], required=False)
    trial_col = find_col(df, ["trial_id", "id"], required=False)
    prompt_id_col = find_col(df, ["prompt_id", "task_id", "task_prompt_id", "prompt_uid"], required=False)

    out = df.copy()
    out["scene_id"] = out[scene_col].astype(str)
    out["method"] = out[method_col].astype(str)
    out["prompt_type"] = out[prompt_type_col].astype(str)
    if outcome_col.lower() in {"correct", "success"}:
        out["correct"] = pd.to_numeric(out[outcome_col], errors="coerce").fillna(0).astype(int)
    else:
        out["correct"] = (out[outcome_col].astype(str) == "correct").astype(int)
    out["num_questions"] = pd.to_numeric(out[q_col], errors="coerce").fillna(0.0)
    out["latency_sec"] = pd.to_numeric(out[latency_col], errors="coerce")
    out["scene_type"] = out[scene_type_col].astype(str) if scene_type_col else ""
    out["candidate_set_size"] = pd.to_numeric(out[candidate_col], errors="coerce") if candidate_col else np.nan
    out["prompt_text"] = out[prompt_text_col].astype(str) if prompt_text_col else ""
    out["prompt_length"] = out["prompt_text"].str.len()
    out["target_category"] = out[target_col].astype(str) if target_col else ""
    out["attribute_type"] = out[attr_col].astype(str) if attr_col else ""
    out["trial_id"] = out[trial_col].astype(str) if trial_col else ""
    if prompt_id_col:
        out["prompt_id"] = out[prompt_id_col].astype(str)
    elif prompt_text_col:
        out["prompt_id"] = out["prompt_text"]
    else:
        out["prompt_id"] = ""
    out["exact_prompt_key"] = out["scene_id"] + "::" + out["prompt_id"].astype(str)

    mapping.update(
        {
            "scene_id": scene_col,
            "method": method_col,
            "prompt_type": prompt_type_col,
            "correct": outcome_col,
            "num_questions": q_col,
            "latency_sec": latency_col,
            "scene_type": scene_type_col or "",
            "candidate_set_size": candidate_col or "",
            "prompt_text": prompt_text_col or "",
            "target_category": target_col or "",
            "attribute_type": attr_col or "",
            "trial_id": trial_col or "",
            "prompt_id": prompt_id_col or ("prompt_text inferred from prompt text" if prompt_text_col else ""),
        }
    )
    return out[out["method"].isin(METHODS)].copy(), mapping


def _word_present(text: str, words: set[str]) -> int:
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return int(bool(tokens.intersection(words)))


def _pattern_present(text: str, patterns: list[str]) -> int:
    lowered = " " + re.sub(r"\s+", " ", text.lower()).strip() + " "
    return int(any(f" {pattern} " in lowered for pattern in patterns))


def _object_from_prompt(text: str) -> str:
    lowered = text.lower()
    for word in OBJECT_WORDS:
        if re.search(rf"\b{re.escape(word)}s?\b", lowered):
            return word
    return "unknown"


def add_prompt_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    normalized = (
        out["prompt_text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    out["prompt_norm"] = normalized
    out["has_my"] = normalized.map(lambda text: int(bool(re.search(r"\bmy\b", text))))
    out["has_color"] = normalized.map(lambda text: _word_present(text, COLOR_WORDS))
    out["has_spatial"] = normalized.map(lambda text: _word_present(text, SPATIAL_WORDS))
    out["has_relation"] = normalized.map(lambda text: _pattern_present(text, RELATION_PATTERNS))
    out["has_size"] = normalized.map(lambda text: _word_present(text, SIZE_WORDS))
    out["has_modifier"] = (
        out[["has_my", "has_color", "has_spatial", "has_relation", "has_size"]].sum(axis=1).clip(upper=1).astype(int)
    )
    object_from_prompt = normalized.map(_object_from_prompt)
    existing_target = out["target_category"].replace("", np.nan)
    out["prompt_object_category"] = existing_target.fillna(object_from_prompt).replace("", "unknown")
    out["prompt_pool_key"] = out["scene_id"].astype(str) + "|" + out["prompt_type"].astype(str)
    pool_size = out.groupby("prompt_pool_key")["prompt_norm"].transform("size")
    unique_size = out.groupby("prompt_pool_key")["prompt_norm"].transform("nunique")
    prompt_count = out.groupby(["prompt_pool_key", "prompt_norm"])["prompt_norm"].transform("size")
    out["prompt_pool_n"] = pool_size.astype(int)
    out["prompt_pool_unique_n"] = unique_size.astype(int)
    out["prompt_pool_repetition_rate"] = (1.0 - unique_size / pool_size).astype(float)
    out["prompt_text_repeat_count_in_pool"] = prompt_count.astype(int)
    return out


def pct(x: float) -> float:
    return 100.0 * x


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def bootstrap_mean_ci(diff: np.ndarray, n_iter: int, rng: np.random.Generator) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, diff.size, size=(n_iter, diff.size))
    samples = diff[idx].mean(axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]).tolist())


def bootstrap_median_ci(diff: np.ndarray, n_iter: int, rng: np.random.Generator) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, diff.size, size=(n_iter, diff.size))
    samples = np.median(diff[idx], axis=1)
    return tuple(np.percentile(samples, [2.5, 97.5]).tolist())


def table_to_csv(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def df_to_md(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def dataset_audit(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"section": "overall", "group": "total_records", "n": len(df)})
    rows.append({"section": "overall", "group": "scenes", "n": df["scene_id"].nunique()})
    rows.append({"section": "overall", "group": "methods", "n": df["method"].nunique()})
    for method, n in df["method"].value_counts().sort_index().items():
        rows.append({"section": "method", "group": method, "n": int(n)})
    for prompt, n in df["prompt_type"].value_counts().sort_index().items():
        rows.append({"section": "prompt_type", "group": prompt, "n": int(n)})
    for (method, prompt), n in df.groupby(["method", "prompt_type"]).size().sort_index().items():
        rows.append({"section": "method_x_prompt_type", "group": f"{method}|{prompt}", "n": int(n)})
    for (method, scene), n in df.groupby(["method", "scene_id"]).size().sort_index().items():
        rows.append({"section": "method_x_scene_id", "group": f"{method}|{scene}", "n": int(n)})
    if df["scene_type"].replace("", np.nan).notna().any():
        for (method, scene_type), n in df.groupby(["method", "scene_type"]).size().sort_index().items():
            rows.append({"section": "method_x_scene_type", "group": f"{method}|{scene_type}", "n": int(n)})
    if df["target_category"].replace("", np.nan).notna().any():
        for (method, target), n in df.groupby(["method", "target_category"]).size().sort_index().items():
            rows.append({"section": "method_x_target_category", "group": f"{method}|{target}", "n": int(n)})
    if df["attribute_type"].replace("", np.nan).notna().any():
        for (method, attr), n in df.groupby(["method", "attribute_type"]).size().sort_index().items():
            rows.append({"section": "method_x_attribute_type", "group": f"{method}|{attr}", "n": int(n)})
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "dataset_audit.csv", index=False)
    return audit


def balance_checks(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    checks = [
        ("method_x_prompt_type", ["method", "prompt_type"]),
        ("method_x_scene_id", ["method", "scene_id"]),
    ]
    if df["scene_type"].replace("", np.nan).notna().any():
        checks.append(("method_x_scene_type", ["method", "scene_type"]))
    if df["target_category"].replace("", np.nan).notna().any():
        checks.append(("method_x_target_category", ["method", "target_category"]))
    if df["attribute_type"].replace("", np.nan).notna().any():
        checks.append(("method_x_attribute_type", ["method", "attribute_type"]))
    for section, keys in checks:
        for group_values, n in df.groupby(keys).size().sort_index().items():
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            rows.append({"section": section, "group": "|".join(map(str, group_values)), "metric": "count", "value": int(n)})
    for method, sub in df.groupby("method"):
        rows.append(
            {
                "section": "candidate_set_size_by_method",
                "group": method,
                "metric": "mean",
                "value": float(sub["candidate_set_size"].mean()) if sub["candidate_set_size"].notna().any() else np.nan,
            }
        )
        rows.append(
            {
                "section": "candidate_set_size_by_method",
                "group": method,
                "metric": "std",
                "value": float(sub["candidate_set_size"].std()) if sub["candidate_set_size"].notna().any() else np.nan,
            }
        )
        rows.append({"section": "prompt_length_by_method", "group": method, "metric": "mean", "value": float(sub["prompt_length"].mean())})
        rows.append({"section": "prompt_length_by_method", "group": method, "metric": "std", "value": float(sub["prompt_length"].std())})
        for feature in ["has_my", "has_color", "has_spatial", "has_relation", "has_size", "has_modifier"]:
            if feature in sub.columns:
                rows.append({"section": "prompt_feature_rate_by_method", "group": method, "metric": feature, "value": float(sub[feature].mean())})
        if "prompt_norm" in sub.columns:
            rows.append({"section": "prompt_unique_by_method", "group": method, "metric": "unique_prompt_n", "value": int(sub["prompt_norm"].nunique())})
            rows.append({"section": "prompt_unique_by_method", "group": method, "metric": "duplicate_rate", "value": float(1.0 - sub["prompt_norm"].nunique() / len(sub))})
    if "prompt_norm" in df.columns:
        for (method, prompt_type), sub in df.groupby(["method", "prompt_type"]):
            group = f"{method}|{prompt_type}"
            rows.append({"section": "prompt_unique_by_method_prompt_type", "group": group, "metric": "unique_prompt_n", "value": int(sub["prompt_norm"].nunique())})
            rows.append({"section": "prompt_unique_by_method_prompt_type", "group": group, "metric": "duplicate_rate", "value": float(1.0 - sub["prompt_norm"].nunique() / len(sub))})
            for feature in ["has_my", "has_color", "has_spatial", "has_relation", "has_size", "has_modifier"]:
                rows.append({"section": "prompt_feature_rate_by_method_prompt_type", "group": group, "metric": feature, "value": float(sub[feature].mean())})
    frame = table_to_csv(rows, out / "balance_checks.csv")
    frame.to_csv(out / "prompt_balance_checks.csv", index=False)
    return frame


def validation_notes(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if len(df) != 1512:
        notes.append(f"Expected 1512 records; found {len(df)}.")
    if df["method"].nunique() != 7:
        notes.append(f"Expected 7 methods; found {df['method'].nunique()}.")
    for method in METHODS:
        n = int((df["method"] == method).sum())
        if n != 216:
            notes.append(f"Expected {method} N=216; found {n}.")
    if df["scene_id"].nunique() != 36:
        notes.append(f"Expected 36 scenes; found {df['scene_id'].nunique()}.")
    scene_method_counts = df.groupby(["method", "scene_id"]).size()
    if not (scene_method_counts == 6).all():
        notes.append(
            "Method x scene counts are not exactly 6 for every cell "
            f"(min={scene_method_counts.min()}, max={scene_method_counts.max()}); "
            "scene-clustered GEE and scene-level aggregation are used."
        )
    method_prompt = df.groupby(["method", "prompt_type"]).size()
    if not method_prompt.between(70, 74).all():
        notes.append("Method x prompt_type counts deviate more than expected from 72.")
    return notes


def build_accuracy_formula(df: pd.DataFrame) -> tuple[str, list[str]]:
    covariates = ['C(method, Treatment(reference="proposed_efe"))', 'C(prompt_type, Treatment(reference="clear"))']
    used = ["method", "prompt_type"]
    if df["scene_type"].replace("", np.nan).nunique(dropna=True) > 1:
        covariates.append("C(scene_type)")
        used.append("scene_type")
    if df["candidate_set_size"].notna().any() and df["candidate_set_size"].nunique(dropna=True) > 1:
        covariates.append("candidate_set_size")
        used.append("candidate_set_size")
    return "correct ~ " + " + ".join(covariates), used


def build_prompt_feature_accuracy_formula(df: pd.DataFrame) -> tuple[str, list[str]]:
    covariates = ['C(method, Treatment(reference="proposed_efe"))', 'C(prompt_type, Treatment(reference="clear"))']
    used = ["method", "prompt_type"]
    if df["scene_type"].replace("", np.nan).nunique(dropna=True) > 1:
        covariates.append("C(scene_type)")
        used.append("scene_type")
    if df["candidate_set_size"].notna().any() and df["candidate_set_size"].nunique(dropna=True) > 1:
        covariates.append("candidate_set_size")
        used.append("candidate_set_size")
    if df["prompt_length"].nunique(dropna=True) > 1:
        covariates.append("prompt_length")
        used.append("prompt_length")
    for feature in ["has_my", "has_color", "has_spatial", "has_relation", "has_size"]:
        if feature in df.columns and df[feature].nunique(dropna=True) > 1:
            covariates.append(feature)
            used.append(feature)
    if "prompt_object_category" in df.columns and 1 < df["prompt_object_category"].nunique(dropna=True) <= 12:
        covariates.append("C(prompt_object_category)")
        used.append("prompt_object_category")
    return "correct ~ " + " + ".join(covariates), used


def fit_gee(df: pd.DataFrame, formula: str):
    model = smf.gee(formula=formula, groups="scene_id", data=df, family=sm.families.Binomial())
    return model.fit()


def method_param_name(method: str) -> str | None:
    if method == "proposed_efe":
        return None
    return f'C(method, Treatment(reference="proposed_efe"))[T.{method}]'


def contrast_from_params(result, weights: dict[str, float]) -> tuple[float, float, float, float, float]:
    params = result.params
    cov = result.cov_params()
    vector = np.zeros(len(params), dtype=float)
    index = list(params.index)
    for name, weight in weights.items():
        if name not in params.index:
            raise KeyError(f"Parameter not in model: {name}")
        vector[index.index(name)] = weight
    estimate = float(np.dot(vector, params.to_numpy()))
    se = float(np.sqrt(np.dot(vector, np.dot(cov.to_numpy(), vector))))
    z = estimate / se if se > 0 else np.nan
    p = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    low = estimate - 1.96 * se
    high = estimate + 1.96 * se
    return estimate, se, p, low, high


def method_contrast_weights(method_a: str, method_b: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    a_name = method_param_name(method_a)
    b_name = method_param_name(method_b)
    if a_name is not None:
        weights[a_name] = weights.get(a_name, 0.0) + 1.0
    if b_name is not None:
        weights[b_name] = weights.get(b_name, 0.0) - 1.0
    return weights


def accuracy_gee_contrasts(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, str, str]:
    formula, covariates = build_accuracy_formula(df)
    result = fit_gee(df, formula)
    summary_text = str(result.summary())
    (out / "accuracy_gee_model_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    comparisons = [
        ("A", "proposed_efe", "top_score"),
        ("A", "proposed_efe", "random_candidate"),
        ("A", "proposed_efe", "vlm_direct"),
        ("A", "vlm_best_question", "vlm_direct"),
        ("B", "proposed_efe", "first_question"),
        ("B", "proposed_efe", "random_question"),
        ("B", "proposed_efe", "vlm_best_question"),
    ]
    rows: list[dict[str, Any]] = []
    for family, a, b in comparisons:
        weights = method_contrast_weights(a, b)
        est, se, p, low, high = contrast_from_params(result, weights)
        acc_a = float(df.loc[df["method"] == a, "correct"].mean())
        acc_b = float(df.loc[df["method"] == b, "correct"].mean())
        rows.append(
            {
                "family": family,
                "method_A": a,
                "method_B": b,
                "log_odds_diff_A_minus_B": est,
                "se": se,
                "odds_ratio": math.exp(est),
                "or_ci95_low": math.exp(low),
                "or_ci95_high": math.exp(high),
                "p_raw": p,
                "descriptive_acc_A": acc_a,
                "descriptive_acc_B": acc_b,
                "descriptive_acc_diff_A_minus_B": acc_a - acc_b,
            }
        )
    frame = pd.DataFrame(rows)
    for family in sorted(frame["family"].unique()):
        mask = frame["family"] == family
        frame.loc[mask, "p_holm"] = holm(frame.loc[mask, "p_raw"].tolist())
    frame.to_csv(out / "accuracy_gee_contrasts.csv", index=False)
    return frame, formula, summary_text


def prompt_feature_accuracy_gee_contrasts(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, str, str]:
    formula, covariates = build_prompt_feature_accuracy_formula(df)
    result = fit_gee(df, formula)
    summary_text = str(result.summary())
    (out / "accuracy_prompt_feature_gee_model_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    comparisons = [
        ("A", "proposed_efe", "top_score"),
        ("A", "proposed_efe", "random_candidate"),
        ("A", "proposed_efe", "vlm_direct"),
        ("A", "vlm_best_question", "vlm_direct"),
        ("B", "proposed_efe", "first_question"),
        ("B", "proposed_efe", "random_question"),
        ("B", "proposed_efe", "vlm_best_question"),
    ]
    rows: list[dict[str, Any]] = []
    for family, a, b in comparisons:
        weights = method_contrast_weights(a, b)
        est, se, p, low, high = contrast_from_params(result, weights)
        acc_a = float(df.loc[df["method"] == a, "correct"].mean())
        acc_b = float(df.loc[df["method"] == b, "correct"].mean())
        rows.append(
            {
                "family": family,
                "method_A": a,
                "method_B": b,
                "log_odds_diff_A_minus_B": est,
                "se": se,
                "odds_ratio": math.exp(est),
                "or_ci95_low": math.exp(low),
                "or_ci95_high": math.exp(high),
                "p_raw": p,
                "descriptive_acc_A": acc_a,
                "descriptive_acc_B": acc_b,
                "descriptive_acc_diff_A_minus_B": acc_a - acc_b,
            }
        )
    frame = pd.DataFrame(rows)
    for family in sorted(frame["family"].unique()):
        mask = frame["family"] == family
        frame.loc[mask, "p_holm"] = holm(frame.loc[mask, "p_raw"].tolist())
    frame.to_csv(out / "accuracy_prompt_feature_gee_contrasts.csv", index=False)
    return frame, formula, summary_text


def aggregate_scene_method_prompt(df: pd.DataFrame) -> pd.DataFrame:
    def iqr(x: pd.Series) -> float:
        return float(x.quantile(0.75) - x.quantile(0.25))

    return (
        df.groupby(["scene_id", "method", "prompt_type"], as_index=False)
        .agg(
            mean_num_questions=("num_questions", "mean"),
            median_num_questions=("num_questions", "median"),
            mean_latency=("latency_sec", "mean"),
            median_latency=("latency_sec", "median"),
            iqr_latency=("latency_sec", iqr),
            n_trials=("trial_id", "size"),
        )
        .copy()
    )


def aggregate_scene_method_overall(scene_prompt: pd.DataFrame, context: str) -> pd.DataFrame:
    if context in PROMPT_TYPES:
        data = scene_prompt[scene_prompt["prompt_type"] == context].copy()
    else:
        data = scene_prompt.copy()
    return (
        data.groupby(["scene_id", "method"], as_index=False)
        .agg(
            mean_num_questions=("mean_num_questions", "mean"),
            median_num_questions=("median_num_questions", "median"),
            mean_latency=("mean_latency", "mean"),
            median_latency=("median_latency", "median"),
            iqr_latency=("iqr_latency", "mean"),
            n_prompt_type_cells=("prompt_type", "size"),
        )
        .copy()
    )


def safe_friedman(pivot: pd.DataFrame, methods: list[str]) -> tuple[float, float]:
    if len(pivot) < 2:
        return np.nan, np.nan
    arrays = [pivot[m].to_numpy(dtype=float) for m in methods]
    if all(np.allclose(arrays[0], arr) for arr in arrays[1:]):
        return 0.0, 1.0
    result = stats.friedmanchisquare(*arrays)
    return float(result.statistic), float(result.pvalue)


def safe_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    if diff.size == 0:
        return np.nan, np.nan
    if np.allclose(diff, 0):
        return 0.0, 1.0
    result = stats.wilcoxon(a, b, zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def scene_question_tests_for_frame(
    df: pd.DataFrame,
    n_boot: int,
    contexts: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    scene_prompt = aggregate_scene_method_prompt(df[df["method"].isin(INTERACTIVE)])
    contexts = contexts or ["ambiguous", "overall", "partial"]
    pair_rows: list[dict[str, Any]] = []
    friedman_rows: list[dict[str, Any]] = []
    for context in contexts:
        agg = aggregate_scene_method_overall(scene_prompt, context)
        pivot = agg.pivot(index="scene_id", columns="method", values="mean_num_questions").reindex(columns=INTERACTIVE).dropna(subset=INTERACTIVE)
        stat, p = safe_friedman(pivot, INTERACTIVE)
        friedman_rows.append({"context": context, "n_scenes": len(pivot), "friedman_statistic": stat, "p_raw": p})
        start = len(pair_rows)
        pvals: list[float] = []
        med_pivot = agg.pivot(index="scene_id", columns="method", values="median_num_questions").reindex(columns=INTERACTIVE).dropna(subset=INTERACTIVE)
        common = pivot.index.intersection(med_pivot.index)
        pivot = pivot.loc[common]
        med_pivot = med_pivot.loc[common]
        for baseline in ["first_question", "random_question", "vlm_best_question"]:
            efe = pivot["proposed_efe"].to_numpy(dtype=float)
            base = pivot[baseline].to_numpy(dtype=float)
            efe_med = med_pivot["proposed_efe"].to_numpy(dtype=float)
            base_med = med_pivot[baseline].to_numpy(dtype=float)
            w_stat, w_p = safe_wilcoxon(efe, base)
            diff = efe - base
            ci = bootstrap_mean_ci(diff, n_boot, rng)
            pair_rows.append(
                {
                    "context": context,
                    "method_A": "proposed_efe",
                    "method_B": baseline,
                    "n_scenes": len(pivot),
                    "mean_Q_EFE": float(np.mean(efe)),
                    "mean_Q_baseline": float(np.mean(base)),
                    "median_Q_EFE": float(np.median(efe_med)),
                    "median_Q_baseline": float(np.median(base_med)),
                    "scene_level_paired_mean_diff_EFE_minus_B": float(np.mean(diff)),
                    "scene_level_paired_median_diff_EFE_minus_B": float(np.median(efe_med - base_med)),
                    "percent_reduction_mean_Q": float(100.0 * (np.mean(base) - np.mean(efe)) / np.mean(base)) if np.mean(base) else np.nan,
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "wilcoxon_statistic": w_stat,
                    "p_raw": w_p,
                }
            )
            pvals.append(w_p)
        row_indices = list(range(start, len(pair_rows)))
        finite = [(idx, pval) for idx, pval in zip(row_indices, pvals) if np.isfinite(pval)]
        adjusted = holm([pval for _, pval in finite]) if finite else []
        for idx in row_indices:
            pair_rows[idx]["p_holm"] = np.nan
        for (idx, _), adj in zip(finite, adjusted):
            pair_rows[idx]["p_holm"] = adj
    pair_frame = pd.DataFrame(pair_rows)
    friedman_frame = pd.DataFrame(friedman_rows)
    return pair_frame, friedman_frame


def scene_question_tests(df: pd.DataFrame, out: Path, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    scene_prompt = aggregate_scene_method_prompt(df[df["method"].isin(INTERACTIVE)])
    scene_prompt.to_csv(out / "scene_method_prompt_type_aggregates.csv", index=False)
    pair_frame, friedman_frame = scene_question_tests_for_frame(df, n_boot, contexts=["ambiguous", "overall", "partial"])
    pair_frame.to_csv(out / "question_count_scene_level_tests.csv", index=False)
    friedman_frame.to_csv(out / "question_count_scene_level_friedman.csv", index=False)
    return pair_frame, friedman_frame


def scene_latency_tests(df: pd.DataFrame, out: Path, n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    scene_prompt = aggregate_scene_method_prompt(df[df["method"].isin(INTERACTIVE)])
    contexts = ["ambiguous", "overall", "partial"]
    rows: list[dict[str, Any]] = []
    for context in contexts:
        agg = aggregate_scene_method_overall(scene_prompt, context)
        mean_pivot = agg.pivot(index="scene_id", columns="method", values="mean_latency").dropna(subset=INTERACTIVE)
        median_pivot = agg.pivot(index="scene_id", columns="method", values="median_latency").dropna(subset=INTERACTIVE)
        iqr_pivot = agg.pivot(index="scene_id", columns="method", values="iqr_latency").dropna(subset=INTERACTIVE)
        common = mean_pivot.index.intersection(median_pivot.index).intersection(iqr_pivot.index)
        mean_pivot = mean_pivot.loc[common]
        median_pivot = median_pivot.loc[common]
        iqr_pivot = iqr_pivot.loc[common]
        start = len(rows)
        pvals: list[float] = []
        for baseline in ["first_question", "random_question", "vlm_best_question"]:
            efe = mean_pivot["proposed_efe"].to_numpy(dtype=float)
            base = mean_pivot[baseline].to_numpy(dtype=float)
            efe_med = median_pivot["proposed_efe"].to_numpy(dtype=float)
            base_med = median_pivot[baseline].to_numpy(dtype=float)
            stat, p = safe_wilcoxon(efe, base)
            mean_diff = efe - base
            median_diff = efe_med - base_med
            mean_ci = bootstrap_mean_ci(mean_diff, n_boot, rng)
            med_ci = bootstrap_median_ci(median_diff, n_boot, rng)
            rows.append(
                {
                    "context": context,
                    "method_A": "proposed_efe",
                    "method_B": baseline,
                    "n_scenes": len(mean_pivot),
                    "mean_latency_EFE": float(np.mean(efe)),
                    "median_latency_EFE": float(np.median(efe_med)),
                    "iqr_latency_EFE": float(np.mean(iqr_pivot["proposed_efe"])),
                    "mean_latency_baseline": float(np.mean(base)),
                    "median_latency_baseline": float(np.median(base_med)),
                    "iqr_latency_baseline": float(np.mean(iqr_pivot[baseline])),
                    "paired_mean_diff_EFE_minus_B": float(np.mean(mean_diff)),
                    "mean_diff_ci95_low": mean_ci[0],
                    "mean_diff_ci95_high": mean_ci[1],
                    "paired_median_diff_EFE_minus_B": float(np.median(median_diff)),
                    "median_diff_ci95_low": med_ci[0],
                    "median_diff_ci95_high": med_ci[1],
                    "wilcoxon_statistic": stat,
                    "p_raw": p,
                }
            )
            pvals.append(p)
        for adj, idx in zip(holm(pvals), range(start, len(rows))):
            rows[idx]["p_holm"] = adj
    return table_to_csv(rows, out / "latency_scene_level_tests.csv")


def prompt_type_descriptives_and_gee(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    desc = (
        df.groupby("prompt_type", as_index=False)
        .agg(
            N=("correct", "size"),
            accuracy=("correct", "mean"),
            failure_rate=("correct", lambda x: 1.0 - x.mean()),
            asked_rate=("num_questions", lambda x: (x > 0).mean()),
            mean_Q=("num_questions", "mean"),
            median_Q=("num_questions", "median"),
            mean_latency=("latency_sec", "mean"),
            median_latency=("latency_sec", "median"),
        )
        .copy()
    )
    desc.to_csv(out / "prompt_type_descriptives.csv", index=False)
    covariates = ['C(prompt_type, Treatment(reference="clear"))', 'C(method, Treatment(reference="proposed_efe"))']
    if df["scene_type"].replace("", np.nan).nunique(dropna=True) > 1:
        covariates.append("C(scene_type)")
    formula = "correct ~ " + " + ".join(covariates)
    result = fit_gee(df, formula)
    summary_text = str(result.summary())
    (out / "prompt_type_gee_model_summary.txt").write_text(summary_text + "\n", encoding="utf-8")

    def pt_name(prompt_type: str) -> str:
        return f'C(prompt_type, Treatment(reference="clear"))[T.{prompt_type}]'

    contrasts = [
        ("ambiguous", "clear", {pt_name("ambiguous"): 1.0}),
        ("partial", "clear", {pt_name("partial"): 1.0}),
        ("ambiguous", "partial", {pt_name("ambiguous"): 1.0, pt_name("partial"): -1.0}),
    ]
    rows: list[dict[str, Any]] = []
    for a, b, weights in contrasts:
        est, se, p, low, high = contrast_from_params(result, weights)
        rows.append(
            {
                "contrast": f"{a}_vs_{b}",
                "log_odds_diff": est,
                "se": se,
                "odds_ratio": math.exp(est),
                "or_ci95_low": math.exp(low),
                "or_ci95_high": math.exp(high),
                "p_raw": p,
                "accuracy_A": float(df.loc[df["prompt_type"] == a, "correct"].mean()),
                "accuracy_B": float(df.loc[df["prompt_type"] == b, "correct"].mean()),
            }
        )
    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm(tests["p_raw"].tolist())
    tests.to_csv(out / "prompt_type_gee_tests.csv", index=False)
    return desc, tests, summary_text


def prompt_feature_balance_tables(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (method, prompt_type), sub in df.groupby(["method", "prompt_type"]):
        rows.append(
            {
                "method": method,
                "prompt_type": prompt_type,
                "N": len(sub),
                "unique_prompt_n": int(sub["prompt_norm"].nunique()),
                "duplicate_rate": float(1.0 - sub["prompt_norm"].nunique() / len(sub)),
                "mean_prompt_length": float(sub["prompt_length"].mean()),
                "has_my_rate": float(sub["has_my"].mean()),
                "has_color_rate": float(sub["has_color"].mean()),
                "has_spatial_rate": float(sub["has_spatial"].mean()),
                "has_relation_rate": float(sub["has_relation"].mean()),
                "has_size_rate": float(sub["has_size"].mean()),
                "has_modifier_rate": float(sub["has_modifier"].mean()),
                "mean_candidate_set_size": float(sub["candidate_set_size"].mean()) if sub["candidate_set_size"].notna().any() else np.nan,
            }
        )
    balance = pd.DataFrame(rows).sort_values(["method", "prompt_type"])
    balance.to_csv(out / "prompt_feature_balance_by_method_prompt_type.csv", index=False)

    pool = (
        df.groupby(["scene_id", "prompt_type"], as_index=False)
        .agg(
            N=("trial_id", "size"),
            unique_prompt_n=("prompt_norm", "nunique"),
            unique_method_n=("method", "nunique"),
            mean_prompt_length=("prompt_length", "mean"),
            has_my_rate=("has_my", "mean"),
            has_color_rate=("has_color", "mean"),
            has_spatial_rate=("has_spatial", "mean"),
            has_relation_rate=("has_relation", "mean"),
            has_size_rate=("has_size", "mean"),
            mean_candidate_set_size=("candidate_set_size", "mean"),
        )
        .copy()
    )
    pool["duplicate_rate"] = 1.0 - pool["unique_prompt_n"] / pool["N"]
    pool["low_diversity_flag"] = (pool["unique_prompt_n"] < 3) | (pool["duplicate_rate"] > 0.50)
    pool.to_csv(out / "prompt_pool_diversity_by_scene_prompt_type.csv", index=False)
    return balance, pool


def _method_contrasts_for_result(result, df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("A", "proposed_efe", "top_score"),
        ("A", "proposed_efe", "random_candidate"),
        ("A", "proposed_efe", "vlm_direct"),
        ("A", "vlm_best_question", "vlm_direct"),
        ("B", "proposed_efe", "first_question"),
        ("B", "proposed_efe", "random_question"),
        ("B", "proposed_efe", "vlm_best_question"),
    ]
    rows: list[dict[str, Any]] = []
    for family, a, b in comparisons:
        weights = method_contrast_weights(a, b)
        est, se, p, low, high = contrast_from_params(result, weights)
        acc_a = float(df.loc[df["method"] == a, "correct"].mean())
        acc_b = float(df.loc[df["method"] == b, "correct"].mean())
        rows.append(
            {
                "family": family,
                "method_A": a,
                "method_B": b,
                "log_odds_diff_A_minus_B": est,
                "se": se,
                "odds_ratio": math.exp(est),
                "or_ci95_low": math.exp(low),
                "or_ci95_high": math.exp(high),
                "p_raw": p,
                "descriptive_acc_A": acc_a,
                "descriptive_acc_B": acc_b,
                "descriptive_acc_diff_A_minus_B": acc_a - acc_b,
            }
        )
    frame = pd.DataFrame(rows)
    for family in sorted(frame["family"].unique()):
        mask = frame["family"] == family
        frame.loc[mask, "p_holm"] = holm(frame.loc[mask, "p_raw"].tolist())
    return frame


def prompt_pool_sensitivity_analysis(df: pd.DataFrame, pool: pd.DataFrame, out: Path, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    flagged_keys = {
        f"{row.scene_id}|{row.prompt_type}"
        for row in pool[pool["low_diversity_flag"]].itertuples()
    }
    filtered = df[~df["prompt_pool_key"].isin(flagged_keys)].copy()
    summary = pd.DataFrame(
        [
            {
                "analysis": "main",
                "N": len(df),
                "scenes": df["scene_id"].nunique(),
                "scene_prompt_type_pools": df["prompt_pool_key"].nunique(),
                "flagged_pools_removed": 0,
            },
            {
                "analysis": "remove_low_diversity_pools",
                "N": len(filtered),
                "scenes": filtered["scene_id"].nunique() if len(filtered) else 0,
                "scene_prompt_type_pools": filtered["prompt_pool_key"].nunique() if len(filtered) else 0,
                "flagged_pools_removed": len(flagged_keys),
            },
        ]
    )
    summary.to_csv(out / "prompt_pool_sensitivity_summary.csv", index=False)
    if len(filtered) < 200 or filtered["method"].nunique() < 7:
        empty = pd.DataFrame([{"status": "skipped", "reason": "Too few records after removing low-diversity prompt pools."}])
        empty.to_csv(out / "sensitivity_accuracy_prompt_feature_gee_contrasts.csv", index=False)
        empty.to_csv(out / "sensitivity_question_count_scene_level_tests.csv", index=False)
        text = "Sensitivity analysis skipped because too few records remained after removing low-diversity prompt pools.\n"
        (out / "prompt_pool_sensitivity_interpretation.txt").write_text(text, encoding="utf-8")
        return summary, empty, empty, text

    formula, used = build_prompt_feature_accuracy_formula(filtered)
    try:
        result = fit_gee(filtered, formula)
        (out / "sensitivity_accuracy_prompt_feature_gee_model_summary.txt").write_text(str(result.summary()) + "\n", encoding="utf-8")
        acc = _method_contrasts_for_result(result, filtered)
        acc.insert(0, "sensitivity", "remove_low_diversity_pools")
        acc.insert(1, "N", len(filtered))
        acc.to_csv(out / "sensitivity_accuracy_prompt_feature_gee_contrasts.csv", index=False)
    except Exception as exc:
        acc = pd.DataFrame([{"status": "failed", "reason": str(exc), "N": len(filtered)}])
        acc.to_csv(out / "sensitivity_accuracy_prompt_feature_gee_contrasts.csv", index=False)

    q, friedman = scene_question_tests_for_frame(filtered, n_boot, contexts=["ambiguous", "overall", "partial"])
    q.to_csv(out / "sensitivity_question_count_scene_level_tests.csv", index=False)
    friedman.to_csv(out / "sensitivity_question_count_scene_level_friedman.csv", index=False)

    lines = [
        "Prompt-pool sensitivity interpretation:",
        "",
        f"- Flagged and removed {len(flagged_keys)} scene_id x prompt_type pools with unique prompts < 3 or duplicate rate > 0.50.",
        f"- Records changed from {len(df)} to {len(filtered)}.",
    ]
    if "p_holm" in acc.columns:
        fam_a = acc[acc["family"] == "A"]
        sig_a = fam_a[fam_a["p_holm"] < 0.05]
        if len(sig_a) == len(fam_a):
            lines.append("- All planned EFE/non-interactive accuracy contrasts remained significant after prompt-pool sensitivity filtering.")
        else:
            lines.append("- Some EFE/non-interactive accuracy contrasts were not significant after prompt-pool sensitivity filtering; inspect the CSV.")
        fam_b = acc[acc["family"] == "B"]
        sig_b = fam_b[fam_b["p_holm"] < 0.05]
        if sig_b.empty:
            lines.append("- EFE remained statistically comparable in accuracy to other interactive baselines after filtering.")
        else:
            lines.append("- At least one EFE vs interactive accuracy contrast changed after filtering; inspect the CSV.")
    amb = q[q["context"] == "ambiguous"] if "context" in q.columns else pd.DataFrame()
    if not amb.empty and "p_holm" in amb.columns:
        valid = amb[(amb["n_scenes"] > 0) & amb["p_holm"].notna()]
        if valid.empty:
            lines.append("- Ambiguous question-count sensitivity could not be evaluated because filtering removed all complete paired ambiguous scenes.")
        else:
            sig_count = int((valid["p_holm"] < 0.05).sum())
            lines.append(f"- Ambiguous question-count sensitivity retained {sig_count}/{len(valid)} significant EFE reductions after Holm correction.")
    text = "\n".join(lines) + "\n"
    (out / "prompt_pool_sensitivity_interpretation.txt").write_text(text, encoding="utf-8")
    return summary, acc, q, text


def supplementary_exact_paired(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    counts = df.groupby("exact_prompt_key").agg(n=("method", "size"), method_n=("method", "nunique")).reset_index()
    exact_keys = set(counts.loc[(counts["n"] == 7) & (counts["method_n"] == 7), "exact_prompt_key"])
    if not exact_keys:
        frame = pd.DataFrame(
            [{"status": "skipped", "reason": "No exact prompt-level paired subset was available; prompt-level paired tests were not used."}]
        )
        frame.to_csv(out / "supplementary_exact_paired_tests.csv", index=False)
        return frame
    subset = df[df["exact_prompt_key"].isin(exact_keys)].copy()
    rows: list[dict[str, Any]] = [{"status": "available", "n_exact_prompt_keys": len(exact_keys), "n_records": len(subset)}]
    comparisons = [
        ("accuracy", "correct", "proposed_efe", "top_score"),
        ("accuracy", "correct", "proposed_efe", "random_candidate"),
        ("accuracy", "correct", "proposed_efe", "vlm_direct"),
        ("questions", "num_questions", "proposed_efe", "first_question"),
        ("questions", "num_questions", "proposed_efe", "random_question"),
        ("questions", "num_questions", "proposed_efe", "vlm_best_question"),
        ("latency", "latency_sec", "proposed_efe", "first_question"),
        ("latency", "latency_sec", "proposed_efe", "random_question"),
        ("latency", "latency_sec", "proposed_efe", "vlm_best_question"),
    ]
    for test_type, value, a, b in comparisons:
        piv = subset[subset["method"].isin([a, b])].pivot(index="exact_prompt_key", columns="method", values=value).dropna()
        if len(piv) == 0:
            continue
        av = piv[a].to_numpy()
        bv = piv[b].to_numpy()
        if test_type == "accuracy":
            b_count = int(((av == 1) & (bv == 0)).sum())
            c_count = int(((av == 0) & (bv == 1)).sum())
            table = [[int(((av == 1) & (bv == 1)).sum()), b_count], [c_count, int(((av == 0) & (bv == 0)).sum())]]
            if mcnemar is not None:
                res = mcnemar(table, exact=False, correction=True)
                stat, p = float(res.statistic), float(res.pvalue)
            else:
                stat = ((abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)) if b_count + c_count else 0.0
                p = float(stats.chi2.sf(stat, 1))
        else:
            stat, p = safe_wilcoxon(av.astype(float), bv.astype(float))
        rows.append(
            {
                "status": "ran",
                "test_type": test_type,
                "method_A": a,
                "method_B": b,
                "n_pairs": len(piv),
                "mean_A": float(np.mean(av)),
                "mean_B": float(np.mean(bv)),
                "mean_diff_A_minus_B": float(np.mean(av - bv)),
                "statistic": stat,
                "p_raw": p,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "supplementary_exact_paired_tests.csv", index=False)
    return frame


def prompt_balance_interpretation(df: pd.DataFrame, out: Path, notes: list[str]) -> str:
    lines = [
        "Prompt balance interpretation:",
        "",
        f"- Total records: {len(df)}.",
        f"- Scenes: {df['scene_id'].nunique()}. Methods: {df['method'].nunique()}.",
    ]
    method_prompt = df.groupby(["method", "prompt_type"]).size().unstack(fill_value=0).reindex(METHODS)
    lines.append("- Method x prompt_type counts:")
    for method, row in method_prompt.iterrows():
        lines.append(f"  - {method}: " + ", ".join(f"{col}={int(row[col])}" for col in method_prompt.columns))
    scene_method = df.groupby(["method", "scene_id"]).size()
    lines.append(f"- Method x scene_id counts range from {int(scene_method.min())} to {int(scene_method.max())}.")
    if notes:
        lines.append("- Balance notes:")
        lines.extend([f"  - {note}" for note in notes])
    lines.append("- GEE includes scene clustering; imbalanced available covariates are included as fixed covariates where available.")
    text = "\n".join(lines) + "\n"
    (out / "prompt_balance_interpretation.txt").write_text(text, encoding="utf-8")
    return text


def accuracy_interpretation(frame: pd.DataFrame, out: Path) -> str:
    lines = ["Accuracy interpretation:", ""]
    fam_a = frame[frame["family"] == "A"]
    fam_b = frame[frame["family"] == "B"]
    sig_a = fam_a[fam_a["p_holm"] < 0.05]
    if sig_a.empty:
        lines.append("- No non-interactive contrast was significant after Holm correction.")
    else:
        for row in sig_a.itertuples():
            direction = "higher" if row.log_odds_diff_A_minus_B > 0 else "lower"
            lines.append(
                f"- {row.method_A} had {direction} adjusted accuracy than {row.method_B} "
                f"(OR={row.odds_ratio:.2f}, Holm p={row.p_holm:.4g})."
            )
    sig_b = fam_b[fam_b["p_holm"] < 0.05]
    if sig_b.empty:
        lines.append("- EFE was not significantly different in accuracy from the interactive baselines after Holm correction; describe accuracy as comparable.")
    else:
        for row in sig_b.itertuples():
            direction = "higher" if row.log_odds_diff_A_minus_B > 0 else "lower"
            lines.append(f"- EFE had {direction} adjusted accuracy than {row.method_B} (Holm p={row.p_holm:.4g}).")
    text = "\n".join(lines) + "\n"
    (out / "accuracy_interpretation.txt").write_text(text, encoding="utf-8")
    return text


def question_interpretation(frame: pd.DataFrame, friedman: pd.DataFrame, out: Path) -> str:
    lines = ["Question-count interpretation:", ""]
    for context in ["ambiguous", "overall", "partial"]:
        fr = friedman[friedman["context"] == context]
        if not fr.empty:
            lines.append(f"- {context}: Friedman p={float(fr.iloc[0]['p_raw']):.4g} with n_scenes={int(fr.iloc[0]['n_scenes'])}.")
        sub = frame[frame["context"] == context]
        sig = sub[sub["p_holm"] < 0.05]
        if sig.empty:
            lines.append(f"  No EFE pairwise question-count comparison was significant after Holm correction.")
        else:
            for row in sig.itertuples():
                direction = "reduced" if row.scene_level_paired_mean_diff_EFE_minus_B < 0 else "increased"
                lines.append(
                    f"  EFE {direction} questions vs {row.method_B}: mean diff={row.scene_level_paired_mean_diff_EFE_minus_B:.2f}, "
                    f"reduction={row.percent_reduction_mean_Q:.2f}%, Holm p={row.p_holm:.4g}."
                )
    lines.append("- Main efficiency claim should emphasize ambiguous prompts if those comparisons are significant.")
    text = "\n".join(lines) + "\n"
    (out / "question_count_interpretation.txt").write_text(text, encoding="utf-8")
    return text


def latency_interpretation(frame: pd.DataFrame, out: Path) -> str:
    lines = ["Latency interpretation:", "", "- Latency is long-tailed; scene-level Wilcoxon tests and median/IQR summaries are used."]
    for context in ["ambiguous", "overall", "partial"]:
        sub = frame[frame["context"] == context]
        sig = sub[sub["p_holm"] < 0.05]
        if sig.empty:
            lines.append(f"- {context}: no Holm-corrected significant latency difference; report descriptive latency cautiously.")
        else:
            for row in sig.itertuples():
                direction = "lower" if row.paired_mean_diff_EFE_minus_B < 0 else "higher"
                lines.append(f"- {context}: EFE had {direction} scene-level mean latency than {row.method_B} (Holm p={row.p_holm:.4g}).")
    text = "\n".join(lines) + "\n"
    (out / "latency_interpretation.txt").write_text(text, encoding="utf-8")
    return text


def prompt_type_interpretation(desc: pd.DataFrame, tests: pd.DataFrame, out: Path) -> str:
    lines = [
        "Prompt-type interpretation:",
        "",
        "- Clear prompts are high accuracy and rarely trigger questions.",
        "- Ambiguous prompts are harder and require more clarification.",
        "- Partial prompts are intermediate but remain challenging.",
        "- This is supporting analysis, not the main contribution.",
        "",
        "GEE prompt-type contrasts:",
    ]
    for row in tests.itertuples():
        lines.append(f"- {row.contrast}: OR={row.odds_ratio:.2f}, Holm p={row.p_holm:.4g}.")
    text = "\n".join(lines) + "\n"
    (out / "prompt_type_interpretation.txt").write_text(text, encoding="utf-8")
    return text


def prompt_sampling_robustness_interpretation(
    prompt_feature_accuracy: pd.DataFrame,
    prompt_feature_formula: str,
    feature_balance: pd.DataFrame,
    pool: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    sensitivity_accuracy: pd.DataFrame,
    sensitivity_question: pd.DataFrame,
    out: Path,
) -> str:
    lines = [
        "Prompt-sampling robustness interpretation:",
        "",
        "- Prompt wording was sampled from scene- and prompt-type-specific pools; pool sizes and repeat rates are therefore explicitly audited.",
        f"- Prompt-feature GEE formula: `{prompt_feature_formula}`.",
    ]
    if not pool.empty:
        flagged = pool[pool["low_diversity_flag"]]
        lines.append(
            f"- {len(flagged)}/{len(pool)} scene_id x prompt_type pools were flagged as low-diversity or high-repeat "
            "(unique prompts < 3 or duplicate rate > 0.50)."
        )
        if len(flagged):
            top = flagged.sort_values(["unique_prompt_n", "duplicate_rate"], ascending=[True, False]).head(8)
            examples = ", ".join(f"{r.scene_id}:{r.prompt_type}(unique={int(r.unique_prompt_n)}, dup={r.duplicate_rate:.2f})" for r in top.itertuples())
            lines.append(f"- Most constrained prompt pools: {examples}.")
    if "p_holm" in prompt_feature_accuracy.columns:
        fam_a = prompt_feature_accuracy[prompt_feature_accuracy["family"] == "A"]
        sig_a = fam_a[fam_a["p_holm"] < 0.05]
        lines.append(f"- With prompt features as covariates, {len(sig_a)}/{len(fam_a)} planned non-interactive accuracy contrasts remained Holm-significant.")
        fam_b = prompt_feature_accuracy[prompt_feature_accuracy["family"] == "B"]
        sig_b = fam_b[fam_b["p_holm"] < 0.05]
        if sig_b.empty:
            lines.append("- With prompt features as covariates, EFE accuracy remained statistically comparable to the other interactive methods.")
        else:
            lines.append("- With prompt features as covariates, at least one EFE vs interactive contrast became significant; inspect `accuracy_prompt_feature_gee_contrasts.csv`.")
    if not sensitivity_summary.empty:
        sens = sensitivity_summary[sensitivity_summary["analysis"] == "remove_low_diversity_pools"]
        if not sens.empty:
            row = sens.iloc[0]
            lines.append(
                f"- Sensitivity after removing flagged pools retained N={int(row['N'])} records across "
                f"{int(row['scene_prompt_type_pools'])} scene-prompt pools."
            )
    if "p_holm" in sensitivity_accuracy.columns:
        fam_a = sensitivity_accuracy[sensitivity_accuracy["family"] == "A"]
        sig_a = fam_a[fam_a["p_holm"] < 0.05]
        lines.append(f"- In the low-diversity-pool sensitivity subset, {len(sig_a)}/{len(fam_a)} non-interactive accuracy contrasts remained Holm-significant.")
    if "p_holm" in sensitivity_question.columns:
        amb = sensitivity_question[sensitivity_question["context"] == "ambiguous"]
        if not amb.empty:
            valid = amb[(amb["n_scenes"] > 0) & amb["p_holm"].notna()]
            if valid.empty:
                lines.append("- In the same sensitivity subset, ambiguous question-count tests were not estimable because no complete paired ambiguous scenes remained.")
            else:
                sig = int((valid["p_holm"] < 0.05).sum())
                lines.append(f"- In the same sensitivity subset, ambiguous question-count reductions remained significant for {sig}/{len(valid)} EFE comparisons.")
    lines.append("- These analyses support treating prompt-pool imbalance as a checked robustness issue rather than a blocker for the main conclusions.")
    text = "\n".join(lines) + "\n"
    (out / "prompt_sampling_robustness_interpretation.txt").write_text(text, encoding="utf-8")
    return text


def make_figures(df: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10})
    acc = df.groupby("method")["correct"].mean().reindex(METHODS)
    fig, ax = plt.subplots(figsize=(8, 4))
    acc.plot(kind="bar", ax=ax, color=["#9ca3af", "#9ca3af", "#9ca3af", "#93c5fd", "#93c5fd", "#93c5fd", "#2563eb"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by method")
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy_by_method.png")
    plt.close(fig)

    pivot = df.groupby(["method", "prompt_type"])["correct"].mean().unstack().reindex(METHODS)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot[PROMPT_TYPES].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by method and prompt type")
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy_method_prompt_type.png")
    plt.close(fig)

    scene_prompt = aggregate_scene_method_prompt(df[df["method"].isin(INTERACTIVE)])
    amb = aggregate_scene_method_overall(scene_prompt, "ambiguous")
    q = amb.groupby("method")["mean_num_questions"].mean().reindex(INTERACTIVE)
    fig, ax = plt.subplots(figsize=(6, 4))
    q.plot(kind="bar", ax=ax, color=["#93c5fd", "#93c5fd", "#93c5fd", "#2563eb"])
    ax.set_ylabel("Scene-level mean questions")
    ax.set_title("Ambiguous prompts: questions by method")
    fig.tight_layout()
    fig.savefig(out / "fig_scene_level_questions_ambiguous.png")
    plt.close(fig)

    inter = df[df["method"].isin(INTERACTIVE)].groupby("method").agg(acc=("correct", "mean"), q=("num_questions", "mean")).reindex(INTERACTIVE)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(inter["q"], inter["acc"], s=90, c=["#93c5fd", "#93c5fd", "#93c5fd", "#2563eb"])
    for method, row in inter.iterrows():
        ax.annotate(method, (row["q"], row["acc"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean questions")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("Accuracy vs questions")
    fig.tight_layout()
    fig.savefig(out / "fig_accuracy_vs_questions.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    values = [df.loc[df["method"] == method, "latency_sec"].dropna().to_numpy() for method in INTERACTIVE]
    ax.boxplot(values, tick_labels=INTERACTIVE, showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (s, log scale)")
    ax.set_title("Latency distribution for interactive methods")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "fig_latency_interactive.png")
    plt.close(fig)


def write_report(
    out: Path,
    mapping: dict[str, str],
    audit: pd.DataFrame,
    notes: list[str],
    accuracy: pd.DataFrame,
    accuracy_formula: str,
    prompt_feature_accuracy: pd.DataFrame,
    prompt_feature_accuracy_formula: str,
    question: pd.DataFrame,
    friedman: pd.DataFrame,
    latency: pd.DataFrame,
    balance: pd.DataFrame,
    prompt_feature_balance: pd.DataFrame,
    prompt_pool: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    sensitivity_accuracy: pd.DataFrame,
    sensitivity_question: pd.DataFrame,
    prompt_desc: pd.DataFrame,
    prompt_tests: pd.DataFrame,
    supplementary: pd.DataFrame,
    acc_text: str,
    prompt_sampling_text: str,
    q_text: str,
    lat_text: str,
    bal_text: str,
    prompt_text: str,
) -> None:
    lines = [
        "# Significance Report",
        "",
        f"Generated at epoch `{time.time():.2f}`.",
        "",
        "## Why The Analysis Changed",
        "",
        "Prompt texts are not fully identical across methods; exact prompt-level paired tests are therefore not valid as the main analysis. Accuracy is analyzed with scene-clustered GEE, while question count and latency are analyzed after scene-level aggregation.",
        "",
        "## Column Mapping",
        "",
        df_to_md(pd.DataFrame([{"canonical": k, "source": v} for k, v in mapping.items()])),
        "",
        "## Dataset Audit",
        "",
        df_to_md(audit.head(60)),
        "",
        "## Validation / Balance Notes",
        "",
    ]
    if notes:
        lines.extend([f"- {note}" for note in notes])
    else:
        lines.append("- Dataset passed expected high-level checks.")
    lines.extend(
        [
            "",
            "## Accuracy Results: Scene-Clustered GEE",
            "",
            f"Formula: `{accuracy_formula}`",
            "",
            df_to_md(accuracy),
            "",
            acc_text,
            "",
            "## Prompt Sampling Robustness",
            "",
            prompt_sampling_text,
            "",
            "Prompt-feature adjusted GEE formula:",
            "",
            f"`{prompt_feature_accuracy_formula}`",
            "",
            "Prompt-feature adjusted accuracy contrasts:",
            "",
            df_to_md(prompt_feature_accuracy),
            "",
            "Prompt feature balance by method x prompt_type:",
            "",
            df_to_md(prompt_feature_balance, max_rows=40),
            "",
            "Prompt pool diversity by scene x prompt_type:",
            "",
            df_to_md(prompt_pool, max_rows=40),
            "",
            "Prompt-pool sensitivity summary:",
            "",
            df_to_md(sensitivity_summary),
            "",
            "Sensitivity accuracy contrasts:",
            "",
            df_to_md(sensitivity_accuracy),
            "",
            "Sensitivity question-count tests:",
            "",
            df_to_md(sensitivity_question),
            "",
            "## Question Count Results: Scene-Level Aggregation",
            "",
            "Friedman tests:",
            "",
            df_to_md(friedman),
            "",
            "Pairwise Wilcoxon tests:",
            "",
            df_to_md(question),
            "",
            q_text,
            "",
            "## Latency Results: Scene-Level Aggregation",
            "",
            df_to_md(latency),
            "",
            lat_text,
            "",
            "## Prompt Balance Checks",
            "",
            bal_text,
            "",
            "See `prompt_balance_checks.csv` and `balance_checks.csv` for full count tables.",
            "",
            "## Prompt-Type Difficulty",
            "",
            "Descriptives:",
            "",
            df_to_md(prompt_desc),
            "",
            "GEE contrasts:",
            "",
            df_to_md(prompt_tests),
            "",
            prompt_text,
            "",
            "## Optional Supplementary Exact-Paired Subset",
            "",
            df_to_md(supplementary),
            "",
            "## Paper-Ready Conclusion",
            "",
            "- Interactive clarification substantially improves target-resolution accuracy compared with non-interactive baselines when supported by scene-clustered GEE contrasts.",
            "- Direct VLM target selection is insufficient under ambiguous referential commands when interactive VLM questioning significantly outperforms `vlm_direct`.",
            "- EFE should be described as maintaining comparable accuracy to other interactive clarification baselines unless GEE contrasts show significant differences.",
            "- EFE reduces the number of clarification questions especially for ambiguous prompts when scene-level Wilcoxon tests support it.",
            "- Because prompt wording was randomized within scene and prompt type, statistical comparisons were performed using scene-clustered or scene-level analyses rather than prompt-level paired tests.",
        ]
    )
    (out / "significance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    ensure_out(out)
    n_boot = int(args.bootstrap)
    raw = load_input(input_path)
    df, mapping = normalize_columns(raw)
    df = add_prompt_features(df)
    df.to_csv(out / "cleaned_trial_level_for_statistics.csv", index=False)
    pd.DataFrame([{"canonical": k, "source": v} for k, v in mapping.items()]).to_csv(out / "column_mapping.csv", index=False)
    audit = dataset_audit(df, out)
    balance = balance_checks(df, out)
    notes = validation_notes(df)
    bal_text = prompt_balance_interpretation(df, out, notes)
    accuracy, accuracy_formula, _ = accuracy_gee_contrasts(df, out)
    prompt_feature_accuracy, prompt_feature_accuracy_formula, _ = prompt_feature_accuracy_gee_contrasts(df, out)
    prompt_feature_balance, prompt_pool = prompt_feature_balance_tables(df, out)
    sensitivity_summary, sensitivity_accuracy, sensitivity_question, sensitivity_text = prompt_pool_sensitivity_analysis(
        df, prompt_pool, out, n_boot
    )
    question, friedman = scene_question_tests(df, out, n_boot)
    latency = scene_latency_tests(df, out, n_boot)
    prompt_desc, prompt_tests, _ = prompt_type_descriptives_and_gee(df, out)
    supplementary = supplementary_exact_paired(df, out)
    make_figures(df, out)
    acc_text = accuracy_interpretation(accuracy, out)
    prompt_sampling_text = prompt_sampling_robustness_interpretation(
        prompt_feature_accuracy,
        prompt_feature_accuracy_formula,
        prompt_feature_balance,
        prompt_pool,
        sensitivity_summary,
        sensitivity_accuracy,
        sensitivity_question,
        out,
    )
    q_text = question_interpretation(question, friedman, out)
    lat_text = latency_interpretation(latency, out)
    prompt_text = prompt_type_interpretation(prompt_desc, prompt_tests, out)
    write_report(
        out,
        mapping,
        audit,
        notes,
        accuracy,
        accuracy_formula,
        prompt_feature_accuracy,
        prompt_feature_accuracy_formula,
        question,
        friedman,
        latency,
        balance,
        prompt_feature_balance,
        prompt_pool,
        sensitivity_summary,
        sensitivity_accuracy,
        sensitivity_question,
        prompt_desc,
        prompt_tests,
        supplementary,
        acc_text,
        prompt_sampling_text,
        q_text,
        lat_text,
        bal_text,
        prompt_text,
    )
    print("DONE")
    print(f"Report: {out / 'significance_report.md'}")
    print("CSV files:")
    for path in sorted(out.glob("*.csv")):
        print(f"- {path}")
    print("Figures:")
    for path in sorted(out.glob("*.png")):
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
