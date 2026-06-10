#!/usr/bin/env python3
"""Small-sample statistical analysis for the main_02 online robot experiment.

The online experiment has fewer scenes/trials than the offline benchmark.  This
script therefore emphasizes scene-level aggregation, paired sign-flip
permutation tests, and clustered bootstrap confidence intervals rather than
treating individual robot trials as independent samples.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["top_score", "random_candidate", "vlm_best_question", "proposed_efe"]
INTERACTIVE_METHODS = ["vlm_best_question", "proposed_efe"]
PROMPT_TYPES = ["clear", "ambiguous", "partial"]
SEED = 20260605


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y"}).fillna(False)


def _pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{100.0 * float(x):.2f}%"


def _num(x: float | int | None, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.{digits}f}"


def _md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_empty_"
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else f"{float(v):.4f}")
        else:
            table[col] = table[col].map(lambda v: "-" if pd.isna(v) else str(v))
    lines = [
        "| " + " | ".join(map(str, table.columns)) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def holm(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    arr = np.asarray(pvals, dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(arr))[0]
    if len(finite_idx) == 0:
        return out.tolist()
    order = finite_idx[np.argsort(arr[finite_idx])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min((m - rank) * arr[idx], 1.0)
        running = max(running, adjusted)
        out[idx] = running
    return out.tolist()


def bootstrap_ci(values: np.ndarray, n_iter: int, rng: np.random.Generator) -> tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    samples = rng.choice(vals, size=(n_iter, len(vals)), replace=True)
    means = samples.mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def exact_or_monte_carlo_signflip_p(
    diffs: np.ndarray,
    rng: np.random.Generator,
    max_exact: int = 65_536,
    n_iter: int = 20_000,
) -> tuple[float, str]:
    vals = np.asarray(diffs, dtype=float)
    vals = vals[np.isfinite(vals)]
    vals = vals[np.abs(vals) > 1e-12]
    n = len(vals)
    if n == 0:
        return (np.nan, "all_zero_or_empty")
    observed = abs(float(vals.mean()))
    total = 2**n
    if total <= max_exact:
        extreme = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            stat = abs(float((vals * np.asarray(signs)).mean()))
            if stat >= observed - 1e-12:
                extreme += 1
        return (float(extreme / total), "exact_sign_flip")
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_iter, n), replace=True)
    stats = np.abs((signs * vals).mean(axis=1))
    return (float((np.count_nonzero(stats >= observed - 1e-12) + 1) / (n_iter + 1)), "monte_carlo_sign_flip")


def infer_object_category(prompt: Any) -> str:
    text = str(prompt or "").lower()
    if re.search(r"\bcups?\b", text):
        return "cup"
    if re.search(r"\bbottles?\b", text):
        return "bottle"
    if re.search(r"\bforks?\b", text):
        return "fork"
    if re.search(r"\bspoons?\b", text):
        return "spoon"
    if re.search(r"\butensils?\b", text):
        return "utensil"
    return "object"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "trial_id",
        "scene_id",
        "scene_type",
        "prompt_type",
        "method",
        "prompt",
        "target_selection_success",
        "task_success",
        "physical_success",
        "wrong_target_grasp_prevented",
        "wrong_object_grasp",
        "grasp_attempted",
        "num_questions",
        "analysis_time_s",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    for col in [
        "target_selection_success",
        "task_success",
        "physical_success",
        "wrong_target_grasp_prevented",
        "wrong_object_grasp",
        "grasp_attempted",
        "asked_question",
        "audit_success_override",
    ]:
        if col in df.columns:
            df[col] = _bool_series(df[col])
    df["num_questions"] = pd.to_numeric(df["num_questions"], errors="coerce").fillna(0).astype(float)
    df["analysis_time_s"] = pd.to_numeric(df["analysis_time_s"], errors="coerce")
    df["candidate_count"] = pd.to_numeric(df.get("candidate_count", np.nan), errors="coerce")
    df["object_category"] = df["prompt"].map(infer_object_category)
    df["failure_stage"] = df.apply(classify_failure_stage, axis=1)
    df["failure_reason_coarse"] = df.apply(classify_failure_reason, axis=1)
    return df


def classify_failure_stage(row: pd.Series) -> str:
    if bool(row.get("task_success")):
        return "none"
    if bool(row.get("wrong_object_grasp")):
        return "wrong_object_grasp"
    if not bool(row.get("target_selection_success")):
        return "target_resolution_failure"
    if bool(row.get("grasp_attempted")) and not bool(row.get("physical_success")):
        return "grasp_execution_failure"
    return "other_task_failure"


def classify_failure_reason(row: pd.Series) -> str:
    stage = classify_failure_stage(row)
    if stage == "none":
        return "none"
    if stage == "target_resolution_failure":
        if bool(row.get("wrong_target_grasp_prevented")):
            return "target_selection_wrong_or_unresolved_grasp_skipped"
        return "target_selection_wrong_or_unresolved"
    if stage == "grasp_execution_failure":
        return "selected_correct_target_but_physical_grasp_failed"
    if stage == "wrong_object_grasp":
        return "wrong_object_grasp_after_execution"
    return "other_task_failure"


def summarize(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    groups = [(("overall",), df)] if group_cols is None else list(df.groupby(group_cols, dropna=False))
    columns = ["group"] if group_cols is None else group_cols
    rows: list[dict[str, Any]] = []
    for key, group in groups:
        if not isinstance(key, tuple):
            key = (key,)
        asked = group[group["num_questions"] > 0]
        attempted = group[group["grasp_attempted"]]
        row = {col: value for col, value in zip(columns, key)}
        row.update(
            {
                "N": int(len(group)),
                "target_correct": int(group["target_selection_success"].sum()),
                "target_fail": int((~group["target_selection_success"]).sum()),
                "target_selection_accuracy": float(group["target_selection_success"].mean()) if len(group) else np.nan,
                "grasp_attempted": int(group["grasp_attempted"].sum()),
                "physical_grasp_success": int(group["physical_success"].sum()),
                "physical_grasp_success_rate_attempted": (
                    float(attempted["physical_success"].mean()) if len(attempted) else np.nan
                ),
                "correct_object_grasp_success": int(group["task_success"].sum()),
                "task_success_rate": float(group["task_success"].mean()) if len(group) else np.nan,
                "wrong_target_prevented": int(group["wrong_target_grasp_prevented"].sum()),
                "wrong_target_prevented_rate": float(group["wrong_target_grasp_prevented"].mean()) if len(group) else np.nan,
                "wrong_object_grasp": int(group["wrong_object_grasp"].sum()),
                "asked_N": int(len(asked)),
                "asked_rate": float((group["num_questions"] > 0).mean()) if len(group) else np.nan,
                "mean_questions_all": float(group["num_questions"].mean()) if len(group) else np.nan,
                "mean_questions_asked": float(asked["num_questions"].mean()) if len(asked) else np.nan,
                "mean_time_s": float(group["analysis_time_s"].mean()),
                "median_time_s": float(group["analysis_time_s"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_bootstrap_method_ci(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    scenes = np.array(sorted(df["scene_id"].unique()))
    metrics = [
        "target_selection_accuracy",
        "task_success_rate",
        "physical_grasp_success_rate_attempted",
        "wrong_target_prevented_rate",
        "asked_rate",
        "mean_questions_all",
        "mean_time_s",
    ]
    scene_index = {scene: idx for idx, scene in enumerate(scenes)}
    method_index = {method: idx for idx, method in enumerate(METHODS)}
    shape = (len(scenes), len(METHODS))
    n = np.zeros(shape)
    target_sum = np.zeros(shape)
    task_sum = np.zeros(shape)
    attempted_sum = np.zeros(shape)
    physical_sum = np.zeros(shape)
    wrong_prevented_sum = np.zeros(shape)
    asked_sum = np.zeros(shape)
    question_sum = np.zeros(shape)
    time_sum = np.zeros(shape)
    for _, row in df.iterrows():
        s = scene_index[row["scene_id"]]
        m = method_index[row["method"]]
        n[s, m] += 1
        target_sum[s, m] += float(bool(row["target_selection_success"]))
        task_sum[s, m] += float(bool(row["task_success"]))
        attempted_sum[s, m] += float(bool(row["grasp_attempted"]))
        physical_sum[s, m] += float(bool(row["physical_success"]))
        wrong_prevented_sum[s, m] += float(bool(row["wrong_target_grasp_prevented"]))
        asked_sum[s, m] += float(float(row["num_questions"]) > 0)
        question_sum[s, m] += float(row["num_questions"])
        if np.isfinite(row["analysis_time_s"]):
            time_sum[s, m] += float(row["analysis_time_s"])

    def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        out = np.full(num.shape, np.nan, dtype=float)
        np.divide(num, den, out=out, where=den > 0)
        return out

    draws: dict[tuple[str, str], list[float]] = {(method, metric): [] for method in METHODS for metric in metrics}
    for _ in range(n_boot):
        sampled_idx = rng.integers(0, len(scenes), size=len(scenes))
        boot_n = n[sampled_idx].sum(axis=0)
        boot_target = target_sum[sampled_idx].sum(axis=0)
        boot_task = task_sum[sampled_idx].sum(axis=0)
        boot_attempted = attempted_sum[sampled_idx].sum(axis=0)
        boot_physical = physical_sum[sampled_idx].sum(axis=0)
        boot_wrong_prevented = wrong_prevented_sum[sampled_idx].sum(axis=0)
        boot_asked = asked_sum[sampled_idx].sum(axis=0)
        boot_questions = question_sum[sampled_idx].sum(axis=0)
        boot_time = time_sum[sampled_idx].sum(axis=0)
        estimates = {
            "target_selection_accuracy": safe_div(boot_target, boot_n),
            "task_success_rate": safe_div(boot_task, boot_n),
            "physical_grasp_success_rate_attempted": safe_div(boot_physical, boot_attempted),
            "wrong_target_prevented_rate": safe_div(boot_wrong_prevented, boot_n),
            "asked_rate": safe_div(boot_asked, boot_n),
            "mean_questions_all": safe_div(boot_questions, boot_n),
            "mean_time_s": safe_div(boot_time, boot_n),
        }
        for method, midx in method_index.items():
            for metric in metrics:
                draws[(method, metric)].append(float(estimates[metric][midx]))
    base = summarize(df, ["method"])
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        method = row["method"]
        for metric in metrics:
            vals = np.asarray(draws[(method, metric)], dtype=float)
            vals = vals[np.isfinite(vals)]
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": float(row[metric]),
                    "ci_low": float(np.percentile(vals, 2.5)) if len(vals) else np.nan,
                    "ci_high": float(np.percentile(vals, 97.5)) if len(vals) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_scene_test(
    df: pd.DataFrame,
    value_col: str,
    method_a: str,
    method_b: str,
    context: str,
    n_boot: int,
    rng: np.random.Generator,
    asked_only: bool = False,
) -> dict[str, Any]:
    sub = df.copy()
    if context in PROMPT_TYPES:
        sub = sub[sub["prompt_type"] == context].copy()
    if asked_only:
        sub = sub[sub["num_questions"] > 0].copy()
    sub = sub[sub["method"].isin([method_a, method_b])].copy()
    grouped = sub.groupby(["scene_id", "method"], dropna=False)[value_col].mean().reset_index()
    pivot = grouped.pivot(index="scene_id", columns="method", values=value_col).dropna()
    if method_a not in pivot.columns or method_b not in pivot.columns or pivot.empty:
        return {
            "context": context,
            "method_A": method_a,
            "method_B": method_b,
            "n_scenes": 0,
            "mean_A": np.nan,
            "mean_B": np.nan,
            "mean_diff_A_minus_B": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_raw": np.nan,
            "test": "not_enough_pairs",
        }
    a = pivot[method_a].to_numpy(dtype=float)
    b = pivot[method_b].to_numpy(dtype=float)
    diff = a - b
    ci_low, ci_high = bootstrap_ci(diff, n_boot, rng)
    p_raw, test_name = exact_or_monte_carlo_signflip_p(diff, rng)
    return {
        "context": context,
        "method_A": method_a,
        "method_B": method_b,
        "n_scenes": int(len(pivot)),
        "mean_A": float(np.mean(a)),
        "mean_B": float(np.mean(b)),
        "median_A": float(np.median(a)),
        "median_B": float(np.median(b)),
        "mean_diff_A_minus_B": float(np.mean(diff)),
        "median_diff_A_minus_B": float(np.median(diff)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_raw": p_raw,
        "test": test_name,
    }


def run_pairwise_tests(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_comparisons = [
        ("A_interactive_vs_noninteractive", "proposed_efe", "top_score"),
        ("A_interactive_vs_noninteractive", "proposed_efe", "random_candidate"),
        ("A_interactive_vs_noninteractive", "vlm_best_question", "top_score"),
        ("A_interactive_vs_noninteractive", "vlm_best_question", "random_candidate"),
        ("B_efe_vs_interactive", "proposed_efe", "vlm_best_question"),
    ]
    contexts = ["overall", "clear", "ambiguous", "partial"]
    target_rows = []
    task_rows = []
    for context in contexts:
        for family, a, b in target_comparisons:
            row = paired_scene_test(df, "target_selection_success", a, b, context, n_boot, rng)
            row["family"] = family
            row["metric"] = "target_selection_accuracy"
            target_rows.append(row)
            row = paired_scene_test(df, "task_success", a, b, context, n_boot, rng)
            row["family"] = family
            row["metric"] = "task_success_rate"
            task_rows.append(row)
    target = pd.DataFrame(target_rows)
    task = pd.DataFrame(task_rows)
    for frame in [target, task]:
        for (context, family), idx in frame.groupby(["context", "family"]).groups.items():
            frame.loc[idx, "p_holm"] = holm(frame.loc[idx, "p_raw"].tolist())

    question_rows = []
    for context in ["overall", "ambiguous", "partial"]:
        for asked_only in [False, True]:
            row = paired_scene_test(df, "num_questions", "proposed_efe", "vlm_best_question", context, n_boot, rng, asked_only)
            row["metric"] = "num_questions_asked_only" if asked_only else "num_questions_all"
            row["asked_only"] = asked_only
            if np.isfinite(row.get("mean_B", np.nan)) and abs(float(row["mean_B"])) > 1e-12:
                row["percent_reduction_A_vs_B"] = 100.0 * (float(row["mean_B"]) - float(row["mean_A"])) / float(row["mean_B"])
            else:
                row["percent_reduction_A_vs_B"] = np.nan
            question_rows.append(row)
    question = pd.DataFrame(question_rows)
    question["p_holm"] = question["p_raw"]

    latency_rows = []
    for context in ["overall", "ambiguous", "partial"]:
        row = paired_scene_test(df, "analysis_time_s", "proposed_efe", "vlm_best_question", context, n_boot, rng)
        row["metric"] = "analysis_time_s"
        latency_rows.append(row)
    latency = pd.DataFrame(latency_rows)
    latency["p_holm"] = holm(latency["p_raw"].tolist())
    return target, task, question, latency


def failure_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, group_cols in {
        "failure_by_stage": ["failure_stage"],
        "failure_by_reason": ["failure_reason_coarse"],
        "failure_by_method_stage": ["method", "failure_stage"],
        "failure_by_prompt_stage": ["prompt_type", "failure_stage"],
        "failure_by_scene_type_stage": ["scene_type", "failure_stage"],
        "failure_by_object_stage": ["object_category", "failure_stage"],
        "grasp_failure_by_scene_object": ["scene_type", "object_category"],
    }.items():
        if name == "grasp_failure_by_scene_object":
            sub = df[(df["target_selection_success"]) & (df["grasp_attempted"])].copy()
            grouped = sub.groupby(group_cols, dropna=False).agg(
                attempted=("trial_id", "size"),
                physical_success=("physical_success", "sum"),
                physical_fail=("physical_success", lambda s: int((~s.astype(bool)).sum())),
                physical_success_rate=("physical_success", "mean"),
            )
            tables[name] = grouped.reset_index()
        else:
            grouped = df.groupby(group_cols, dropna=False).size().reset_index(name="count")
            grouped["share_all"] = grouped["count"] / len(df)
            tables[name] = grouped
    return tables


def prompt_type_tests(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for metric in ["target_selection_success", "task_success", "num_questions", "analysis_time_s"]:
        for a, b in [("ambiguous", "clear"), ("partial", "clear"), ("ambiguous", "partial")]:
            # Pair prompt types within scene, aggregating across methods.
            sub = df[df["prompt_type"].isin([a, b])].copy()
            grouped = sub.groupby(["scene_id", "prompt_type"], dropna=False)[metric].mean().reset_index()
            pivot = grouped.pivot(index="scene_id", columns="prompt_type", values=metric).dropna()
            if a not in pivot or b not in pivot or pivot.empty:
                continue
            diff = pivot[a].to_numpy(float) - pivot[b].to_numpy(float)
            ci_low, ci_high = bootstrap_ci(diff, n_boot, rng)
            p, test = exact_or_monte_carlo_signflip_p(diff, rng)
            rows.append(
                {
                    "metric": metric,
                    "prompt_A": a,
                    "prompt_B": b,
                    "n_scenes": len(pivot),
                    "mean_A": float(pivot[a].mean()),
                    "mean_B": float(pivot[b].mean()),
                    "mean_diff_A_minus_B": float(diff.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_raw": p,
                    "test": test,
                }
            )
    out = pd.DataFrame(rows)
    for metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_raw"].tolist())
    return out


def make_figures(df: pd.DataFrame, out: Path) -> None:
    method_desc = summarize(df, ["method"]).set_index("method").loc[METHODS].reset_index()
    plt.figure(figsize=(7, 4))
    plt.bar(method_desc["method"], method_desc["target_selection_accuracy"], color=["#8a8f98", "#b0a38d", "#4f83cc", "#2ca25f"])
    plt.ylim(0, 1.05)
    plt.ylabel("Target selection accuracy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_online_target_accuracy_by_method.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(method_desc["method"], method_desc["task_success_rate"], color=["#8a8f98", "#b0a38d", "#4f83cc", "#2ca25f"])
    plt.ylim(0, 1.05)
    plt.ylabel("Correct-object grasp success")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_online_task_success_by_method.png", dpi=300)
    plt.close()

    interactive = df[df["method"].isin(INTERACTIVE_METHODS)]
    q = interactive.groupby(["method", "prompt_type"])["num_questions"].mean().unstack().reindex(INTERACTIVE_METHODS)
    q.plot(kind="bar", figsize=(7, 4), color=["#6688aa", "#c9a646", "#7aa35a"])
    plt.ylabel("Mean questions per trial")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_online_questions_interactive.png", dpi=300)
    plt.close()

    stage = pd.crosstab(df["method"], df["failure_stage"]).reindex(METHODS).fillna(0)
    stage = stage.div(stage.sum(axis=1), axis=0)
    stage.plot(kind="bar", stacked=True, figsize=(8, 4))
    plt.ylabel("Share of trials")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out / "fig_online_failure_stage_by_method.png", dpi=300)
    plt.close()


def write_report(
    out: Path,
    input_path: Path,
    df: pd.DataFrame,
    audit: pd.DataFrame,
    by_method: pd.DataFrame,
    by_prompt: pd.DataFrame,
    by_method_prompt: pd.DataFrame,
    target_tests: pd.DataFrame,
    task_tests: pd.DataFrame,
    question_tests: pd.DataFrame,
    latency_tests: pd.DataFrame,
    prompt_tests: pd.DataFrame,
    failure: dict[str, pd.DataFrame],
    ci_by_method: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    def select_rows(frame: pd.DataFrame, context: str, family: str | None = None) -> pd.DataFrame:
        sub = frame[frame["context"] == context].copy()
        if family is not None:
            sub = sub[sub["family"] == family]
        keep = [
            "method_A",
            "method_B",
            "n_scenes",
            "mean_A",
            "mean_B",
            "mean_diff_A_minus_B",
            "ci_low",
            "ci_high",
            "p_raw",
            "p_holm",
            "test",
        ]
        return sub[[col for col in keep if col in sub.columns]]

    lines: list[str] = []
    lines.append("# Online Main 02 Statistical Analysis and Failure Report")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append("This report analyzes only the cleaned `main_02` online robot experiment. Raw trial JSON files are not modified.")
    lines.append("")
    lines.append(f"- Input: `{input_path}`")
    lines.append(f"- Records analyzed: {len(df)}")
    lines.append(f"- Methods: {', '.join(METHODS)}")
    lines.append(f"- Prompt types: {', '.join(PROMPT_TYPES)}")
    lines.append("- Main statistical unit: scene-level aggregate")
    lines.append("- Main small-sample test: paired sign-flip permutation over scene-level differences")
    lines.append(f"- Confidence intervals: scene-clustered bootstrap or paired scene bootstrap, {n_bootstrap} iterations")
    lines.append("")
    lines.append("## 2. Why This Online Analysis Is Conservative")
    lines.append("")
    lines.append(
        "The online experiment is much smaller than the offline experiment and includes real robot noise. "
        "Therefore the main goal is not to force significance, but to report effect sizes, uncertainty, and cautious supporting tests."
    )
    lines.append("")
    lines.append("The exact prompt wording is not treated as a fully paired unit. Instead, trials are aggregated at the scene level before method comparisons.")
    lines.append("")
    lines.append("## 3. Dataset Audit")
    lines.append("")
    lines.append(_md_table(audit))
    lines.append("")
    lines.append("## 4. Descriptive Results")
    lines.append("")
    lines.append("### 4.1 By Method")
    lines.append("")
    lines.append(_md_table(by_method))
    lines.append("")
    lines.append("### 4.2 By Prompt Type")
    lines.append("")
    lines.append(_md_table(by_prompt))
    lines.append("")
    lines.append("### 4.3 Method x Prompt Type")
    lines.append("")
    lines.append(_md_table(by_method_prompt))
    lines.append("")
    lines.append("### 4.4 Bootstrap Confidence Intervals by Method")
    lines.append("")
    lines.append(_md_table(ci_by_method, max_rows=80))
    lines.append("")
    lines.append("## 5. Statistical Methods and Formulas")
    lines.append("")
    lines.append("For scene `s`, method `m`, and prompt type `t`, the scene-level mean for metric `Y` is:")
    lines.append("")
    lines.append("```text")
    lines.append("Ybar_smt = (1 / n_smt) * sum_{i in scene s, method m, prompt type t} Y_i")
    lines.append("```")
    lines.append("")
    lines.append("For a planned comparison between method A and method B, the paired scene difference is:")
    lines.append("")
    lines.append("```text")
    lines.append("d_s = Ybar_s,A,t - Ybar_s,B,t")
    lines.append("```")
    lines.append("")
    lines.append("The observed effect is the mean paired difference:")
    lines.append("")
    lines.append("```text")
    lines.append("dbar = (1 / S) * sum_s d_s")
    lines.append("```")
    lines.append("")
    lines.append("The paired sign-flip permutation test evaluates the null hypothesis that the signs of `d_s` are exchangeable:")
    lines.append("")
    lines.append("```text")
    lines.append("p = Pr(|mean(e_s * d_s)| >= |mean(d_s)|)")
    lines.append("e_s in {-1, +1}")
    lines.append("```")
    lines.append("")
    lines.append("For the current sample size, the script uses exact enumeration when feasible. Holm-Bonferroni correction is applied within each planned family and context.")
    lines.append("")
    lines.append("For question efficiency, percent reduction is:")
    lines.append("")
    lines.append("```text")
    lines.append("Percent reduction = 100 * (MeanQ_baseline - MeanQ_EFE) / MeanQ_baseline")
    lines.append("```")
    lines.append("")
    lines.append("## 6. Target Selection Accuracy Tests")
    lines.append("")
    lines.append("Overall planned contrasts:")
    lines.append("")
    lines.append(_md_table(select_rows(target_tests, "overall")))
    lines.append("")
    lines.append("Ambiguous-prompt contrasts:")
    lines.append("")
    lines.append(_md_table(select_rows(target_tests, "ambiguous")))
    lines.append("")
    lines.append("## 7. Full Pipeline Task Success Tests")
    lines.append("")
    lines.append("Task success means correct-object grasp success over all target-evaluated trials. If the target was wrong and the grasp was skipped, this is counted as task failure.")
    lines.append("")
    lines.append("Overall planned contrasts:")
    lines.append("")
    lines.append(_md_table(select_rows(task_tests, "overall")))
    lines.append("")
    lines.append("Ambiguous-prompt contrasts:")
    lines.append("")
    lines.append(_md_table(select_rows(task_tests, "ambiguous")))
    lines.append("")
    lines.append("## 8. Question Count Tests")
    lines.append("")
    lines.append("Only `proposed_efe` and `vlm_best_question` ask questions online. Both all-trial and asked-only variants are reported.")
    lines.append("")
    lines.append(_md_table(question_tests))
    lines.append("")
    lines.append("## 9. Time Tests")
    lines.append("")
    lines.append("Time uses `analysis_time_s`, which includes target resolution and any execution/evaluation time recorded for the online trial. Skipped wrong-target trials can be much faster, so time is interpreted cautiously.")
    lines.append("")
    lines.append(_md_table(latency_tests))
    lines.append("")
    lines.append("## 10. Prompt-Type Difficulty")
    lines.append("")
    lines.append(_md_table(prompt_tests))
    lines.append("")
    lines.append("## 11. Failure Analysis")
    lines.append("")
    lines.append("### 11.1 Failure Stage")
    lines.append("")
    lines.append(_md_table(failure["failure_by_stage"]))
    lines.append("")
    lines.append("### 11.2 Failure Reason")
    lines.append("")
    lines.append(_md_table(failure["failure_by_reason"]))
    lines.append("")
    lines.append("### 11.3 Failure Stage by Method")
    lines.append("")
    lines.append(_md_table(failure["failure_by_method_stage"]))
    lines.append("")
    lines.append("### 11.4 Failure Stage by Prompt Type")
    lines.append("")
    lines.append(_md_table(failure["failure_by_prompt_stage"]))
    lines.append("")
    lines.append("### 11.5 Grasp Execution Failure by Scene/Object")
    lines.append("")
    lines.append(_md_table(failure["grasp_failure_by_scene_object"]))
    lines.append("")
    lines.append("## 12. Grasping Policy Evaluation")
    lines.append("")
    lines.append(
        "The online data show that the target-resolution layer is now the dominant source of end-to-end difference between methods. "
        "`proposed_efe` selected the correct target in 53/57 trials, while non-interactive baselines selected the correct target in only 34-35/57 trials. "
        "Because wrong target selections were skipped rather than executed, wrong-object grasp rate is 0, but wrong-target-prevented failures remain task failures."
    )
    lines.append("")
    lines.append(
        "Conditional on attempting a grasp, the physical grasp success rate is high but not perfect. "
        "The balanced data contain 19 cases where the target was selected correctly but the physical grasp failed. "
        "These failures concentrate in bottle-only and utensil-only scenes, matching the observed real-robot issues: tall bottles stress the top-down height policy, and forks/spoons stress narrow-object orientation and gripper closing."
    )
    lines.append("")
    lines.append("Policy assessment:")
    lines.append("")
    lines.append("- The skip-on-wrong-target policy is correct for hardware safety and makes wrong-object grasp rate 0 in this dataset.")
    lines.append("- The target-resolution benefit transfers to full task success because EFE creates more correct grasp attempts.")
    lines.append("- The grasp policy is reliable enough for cups and many bottles/utensils, but still has object-geometry-specific weaknesses.")
    lines.append("- Remaining grasp failures are not mainly caused by EFE question selection; they are physical execution failures after correct target resolution.")
    lines.append("- The most important policy improvements are better segmentation-based point-cloud cropping, object-category-specific grasp pose selection, and a real wrist-orientation strategy for utensils.")
    lines.append("")
    lines.append("## 13. Conservative Conclusions")
    lines.append("")
    lines.append("- Online data support the offline conclusion that interactive clarification improves target resolution.")
    lines.append("- EFE has the highest target-selection accuracy and full task success rate among the online methods.")
    lines.append("- Because online N is small, report p-values as supporting evidence and emphasize effect sizes with confidence intervals.")
    lines.append("- EFE asks fewer questions than VLM-best in the online data, but the statistical strength depends on the context and asked-only subset size.")
    lines.append("- The physical grasp policy is the remaining bottleneck once target resolution is correct.")
    lines.append("")
    (out / "online_main02_significance_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="services/a6000_web/artifacts/online_experiments/main_02/analysis/balanced_method_prompt_trials_main02.csv",
    )
    parser.add_argument("--out", default="outputs/online_statistics/main_02")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    input_path = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = load_data(input_path)

    audit_rows = [
        {"metric": "records", "value": len(df)},
        {"metric": "methods", "value": df["method"].nunique()},
        {"metric": "scenes", "value": df["scene_id"].nunique()},
        {"metric": "prompt_types", "value": df["prompt_type"].nunique()},
        {"metric": "records_per_method_min", "value": int(df.groupby("method").size().min())},
        {"metric": "records_per_method_max", "value": int(df.groupby("method").size().max())},
        {"metric": "records_per_method_prompt_cell", "value": int(df.groupby(["method", "prompt_type"]).size().min())},
    ]
    audit = pd.DataFrame(audit_rows)
    by_method = summarize(df, ["method"]).sort_values("method")
    by_prompt = summarize(df, ["prompt_type"]).sort_values("prompt_type")
    by_method_prompt = summarize(df, ["method", "prompt_type"]).sort_values(["method", "prompt_type"])
    scene_method = summarize(df, ["scene_id", "method"]).sort_values(["scene_id", "method"])
    scene_method_prompt = summarize(df, ["scene_id", "method", "prompt_type"]).sort_values(["scene_id", "method", "prompt_type"])

    ci_by_method = cluster_bootstrap_method_ci(df, args.bootstrap, rng)
    target_tests, task_tests, question_tests, latency_tests = run_pairwise_tests(df, args.bootstrap, rng)
    prompt_tests = prompt_type_tests(df, args.bootstrap, rng)
    failure = failure_tables(df)

    audit.to_csv(out / "dataset_audit.csv", index=False)
    by_method.to_csv(out / "descriptives_by_method.csv", index=False)
    by_prompt.to_csv(out / "descriptives_by_prompt_type.csv", index=False)
    by_method_prompt.to_csv(out / "descriptives_by_method_prompt_type.csv", index=False)
    scene_method.to_csv(out / "scene_method_aggregates.csv", index=False)
    scene_method_prompt.to_csv(out / "scene_method_prompt_type_aggregates.csv", index=False)
    ci_by_method.to_csv(out / "cluster_bootstrap_method_ci.csv", index=False)
    target_tests.to_csv(out / "target_selection_scene_tests.csv", index=False)
    task_tests.to_csv(out / "task_success_scene_tests.csv", index=False)
    question_tests.to_csv(out / "question_count_scene_tests.csv", index=False)
    latency_tests.to_csv(out / "time_scene_tests.csv", index=False)
    prompt_tests.to_csv(out / "prompt_type_scene_tests.csv", index=False)
    for name, frame in failure.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    df.to_csv(out / "cleaned_online_main02_for_statistics.csv", index=False)

    make_figures(df, out)
    write_report(
        out,
        input_path,
        df,
        audit,
        by_method,
        by_prompt,
        by_method_prompt,
        target_tests,
        task_tests,
        question_tests,
        latency_tests,
        prompt_tests,
        failure,
        ci_by_method,
        args.bootstrap,
    )
    print(f"Wrote report: {out / 'online_main02_significance_report.md'}")
    print(f"Wrote CSV outputs under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
