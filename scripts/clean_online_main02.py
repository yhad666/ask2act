#!/usr/bin/env python3
"""Clean and balance the main_02 online experiment trials.

This script intentionally does not edit raw trial JSON files. It writes
analysis-layer CSV/JSON/Markdown artifacts only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


METHODS = ["top_score", "random_candidate", "vlm_best_question", "proposed_efe"]
PROMPT_TYPES = ["clear", "ambiguous", "partial"]
SCENES = [f"scene_{idx:02d}" for idx in range(1, 21)]
SEED = 20260603


def _load_trials(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "trials").glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            rows.append(
                {
                    "trial_id": path.stem,
                    "raw_path": str(path),
                    "json_error": str(exc),
                    "include_in_cleaned_analysis": False,
                }
            )
            continue
        item["raw_path"] = str(path)
        item["raw_mtime"] = path.stat().st_mtime
        rows.append(item)
    rows.sort(key=lambda item: (float(item.get("started_at_epoch_s") or item.get("raw_mtime") or 0.0), item["trial_id"]))
    for index, item in enumerate(rows):
        item["raw_order"] = index
    return rows


def _apply_scene_corrections(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scene04 = [item for item in rows if item.get("scene_id") == "scene_04"]
    scene04.sort(key=lambda item: (float(item.get("started_at_epoch_s") or item.get("raw_mtime") or 0.0), item["trial_id"]))
    split_index = len(scene04) // 2
    # main_02 has 29 scene_04-labelled trials. The first 14 form one full
    # scene block; the later 15 form the forgotten scene_05 block.
    if len(scene04) == 29:
        split_index = 14
    scene05_ids = {item["trial_id"] for item in scene04[split_index:]}

    corrected: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        row["original_scene_id"] = item.get("scene_id")
        row["original_scene_type"] = item.get("scene_type")
        row["scene_correction"] = ""
        if row.get("trial_id") in scene05_ids:
            row["scene_id"] = "scene_05"
            row["scene_type"] = "cup_only"
            row["scene_correction"] = "scene_04_late_half_to_scene_05"
        elif row.get("scene_id") == "scene_04":
            row["scene_type"] = "cup_only"
        if row.get("scene_id") in {"scene_07", "scene_08", "scene_09"}:
            row["scene_type"] = "bottle_only"
            row["scene_correction"] = (row["scene_correction"] + ";" if row["scene_correction"] else "") + (
                "scene_07_08_09_to_bottle_only"
            )
        corrected.append(row)
    return corrected


def _metric_outcome(row: dict[str, Any]) -> str | None:
    outcome = row.get("target_selection_outcome")
    if outcome in {"correct", "wrong", "unresolved"}:
        return str(outcome)
    outcome = row.get("outcome")
    if outcome in {"correct", "wrong", "unresolved", "skipped_wrong_target"}:
        return "wrong" if outcome == "skipped_wrong_target" else str(outcome)
    return None


def _apply_feedback_override(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corrected: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        row["original_status"] = item.get("status")
        row["original_outcome"] = item.get("outcome")
        row["original_grasp_attempted"] = item.get("grasp_attempted")
        row["original_physical_grasp_success"] = item.get("physical_grasp_success")
        row["original_correct_object_grasp_success"] = item.get("correct_object_grasp_success")
        row["audit_success_override"] = False
        row["audit_success_override_reason"] = ""
        target_confirmed_correct = row.get("target_selection_outcome") == "correct"
        missing_grasp_feedback = row.get("physical_grasp_success") is None and row.get("correct_object_grasp_success") is None
        if target_confirmed_correct and missing_grasp_feedback:
            row["audit_success_override"] = True
            row["audit_success_override_reason"] = "target_confirmed_correct_but_grasp_feedback_missing_counted_success"
            row["grasp_attempted"] = True
            row["physical_grasp_success"] = True
            row["correct_object_grasp_success"] = True
            row["wrong_object_grasp"] = False
            row["outcome"] = "correct"
        metric_outcome = _metric_outcome(row)
        row["target_metric_outcome"] = metric_outcome
        row["target_selection_eval"] = metric_outcome is not None
        row["target_selection_success"] = metric_outcome == "correct"
        row["target_selection_fail"] = metric_outcome in {"wrong", "unresolved"}
        row["task_success"] = bool(row.get("correct_object_grasp_success"))
        row["physical_success"] = bool(row.get("physical_grasp_success"))
        row["wrong_target_or_object"] = bool(row.get("wrong_object_grasp")) or metric_outcome == "wrong"
        row["include_in_cleaned_analysis"] = bool(row["target_selection_eval"])
        row["num_questions"] = int(row.get("question_count") or 0)
        row["asked_question"] = row["num_questions"] > 0
        row["analysis_time_s"] = row.get("total_time_s") if row.get("total_time_s") is not None else row.get("latency_s")
        corrected.append(row)
    return corrected


def _flatten_for_csv(rows: list[dict[str, Any]]) -> pd.DataFrame:
    keep = [
        "trial_id",
        "experiment_id",
        "experiment_type",
        "raw_order",
        "raw_path",
        "started_at_epoch_s",
        "finished_at_epoch_s",
        "original_scene_id",
        "original_scene_type",
        "scene_id",
        "scene_type",
        "scene_correction",
        "prompt",
        "prompt_type",
        "method",
        "expected_candidate_id",
        "expected_display_id",
        "candidate_count",
        "resolved_candidate_id",
        "status",
        "original_status",
        "target_selection_outcome",
        "target_metric_outcome",
        "target_selection_eval",
        "target_selection_success",
        "target_selection_fail",
        "grasp_attempted",
        "physical_grasp_success",
        "correct_object_grasp_success",
        "wrong_object_grasp",
        "wrong_target_grasp_prevented",
        "task_success",
        "physical_success",
        "wrong_target_or_object",
        "outcome",
        "original_outcome",
        "original_grasp_attempted",
        "original_physical_grasp_success",
        "original_correct_object_grasp_success",
        "num_questions",
        "asked_question",
        "latency_s",
        "resolution_latency_s",
        "total_time_s",
        "analysis_time_s",
        "audit_success_override",
        "audit_success_override_reason",
        "include_in_cleaned_analysis",
        "operator_note",
        "notes",
    ]
    flat = []
    for row in rows:
        flat.append({key: row.get(key) for key in keep})
    return pd.DataFrame(flat)


def _choose_one_per_scene_cell(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    evaluated = df[
        (df["include_in_cleaned_analysis"] == True)  # noqa: E712
        & df["method"].isin(METHODS)
        & df["prompt_type"].isin(PROMPT_TYPES)
        & df["scene_id"].isin(SCENES)
    ].copy()
    included_indices: list[int] = []
    excluded_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for scene_id in SCENES:
        for prompt_type in PROMPT_TYPES:
            for method in METHODS:
                group = evaluated[
                    (evaluated["scene_id"] == scene_id)
                    & (evaluated["prompt_type"] == prompt_type)
                    & (evaluated["method"] == method)
                ]
                if group.empty:
                    missing_rows.append({"scene_id": scene_id, "prompt_type": prompt_type, "method": method})
                    continue
                chosen_index = rng.choice(list(group.index))
                included_indices.append(int(chosen_index))
                for idx, row in group.drop(index=chosen_index).iterrows():
                    excluded_rows.append(
                        {
                            "trial_id": row["trial_id"],
                            "scene_id": scene_id,
                            "prompt_type": prompt_type,
                            "method": method,
                            "exclude_reason": "duplicate_scene_prompt_method_cell_seeded_random_exclusion",
                        }
                    )
    selected = evaluated.loc[sorted(included_indices)].copy()
    return selected, pd.DataFrame(excluded_rows), pd.DataFrame(missing_rows)


def _balance_method_prompt(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rng = random.Random(seed + 17)
    counts = df.groupby(["method", "prompt_type"]).size()
    target_n = int(counts.min()) if not counts.empty else 0
    included_indices: list[int] = []
    excluded_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for prompt_type in PROMPT_TYPES:
            group = df[(df["method"] == method) & (df["prompt_type"] == prompt_type)]
            indices = list(group.index)
            if len(indices) <= target_n:
                chosen = set(indices)
            else:
                chosen = set(rng.sample(indices, target_n))
            included_indices.extend(sorted(chosen))
            for idx, row in group.drop(index=list(chosen)).iterrows():
                excluded_rows.append(
                    {
                        "trial_id": row["trial_id"],
                        "scene_id": row["scene_id"],
                        "prompt_type": prompt_type,
                        "method": method,
                        "exclude_reason": "method_prompt_balance_seeded_random_exclusion",
                    }
                )
    return df.loc[sorted(included_indices)].copy(), pd.DataFrame(excluded_rows), target_n


def _fmt_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{100.0 * float(value):.2f}%"


def _summarize(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        grouped = [(("overall",), df)]
        columns = ["group"]
    else:
        grouped = list(df.groupby(group_cols, dropna=False))
        columns = group_cols
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        target_eval = group[group["target_selection_eval"] == True]  # noqa: E712
        grasp_attempted = group[group["grasp_attempted"] == True]  # noqa: E712
        asked = group[group["num_questions"] > 0]
        row = {col: val for col, val in zip(columns, key)}
        row.update(
            {
                "N": int(len(group)),
                "target_eval_N": int(len(target_eval)),
                "target_correct": int(group["target_selection_success"].sum()),
                "target_wrong_or_unresolved": int(group["target_selection_fail"].sum()),
                "target_selection_accuracy": (
                    float(group["target_selection_success"].sum() / len(target_eval)) if len(target_eval) else None
                ),
                "grasp_attempted": int(len(grasp_attempted)),
                "physical_grasp_success": int(group["physical_success"].sum()),
                "physical_grasp_success_rate": (
                    float(group["physical_success"].sum() / len(grasp_attempted)) if len(grasp_attempted) else None
                ),
                "correct_object_grasp_success": int(group["task_success"].sum()),
                "task_success_rate": float(group["task_success"].sum() / len(target_eval)) if len(target_eval) else None,
                "wrong_object_grasp": int(group["wrong_object_grasp"].fillna(False).astype(bool).sum()),
                "wrong_target_prevented": int(group["wrong_target_grasp_prevented"].fillna(False).astype(bool).sum()),
                "asked_N": int(len(asked)),
                "asked_rate": float(len(asked) / len(group)) if len(group) else None,
                "mean_questions_all": float(group["num_questions"].mean()) if len(group) else None,
                "mean_questions_asked": float(asked["num_questions"].mean()) if len(asked) else None,
                "mean_time_s": float(pd.to_numeric(group["analysis_time_s"], errors="coerce").mean()),
                "median_time_s": float(pd.to_numeric(group["analysis_time_s"], errors="coerce").median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    table = df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda value: "-" if pd.isna(value) else f"{float(value):.4f}")
        else:
            table[col] = table[col].map(lambda value: "-" if pd.isna(value) else str(value))
    headers = [str(col) for col in table.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]).replace("|", "\\|") for col in table.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _display_value(value: Any) -> str:
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except TypeError:
        pass
    return str(value)


def _write_report(
    out: Path,
    all_df: pd.DataFrame,
    evaluated_df: pd.DataFrame,
    scene_cell_df: pd.DataFrame,
    method_prompt_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    scene_excluded: pd.DataFrame,
    method_prompt_excluded: pd.DataFrame,
    target_n: int,
) -> None:
    lines: list[str] = []
    lines.append("# Online Main 02 Cleaning Report")
    lines.append("")
    lines.append(f"- Raw trials: {len(all_df)}")
    lines.append(f"- Evaluated/cleaned trials before balancing: {len(evaluated_df)}")
    lines.append(f"- Scene-cell balanced trials: {len(scene_cell_df)}")
    lines.append(f"- Method x prompt balanced trials: {len(method_prompt_df)}")
    lines.append(f"- Method x prompt target per cell: {target_n}")
    lines.append(f"- Random seed: {SEED}")
    lines.append("")
    lines.append("## Corrections Applied")
    lines.append("")
    lines.append("- Late half of `scene_04` was relabeled to `scene_05`; both are `cup_only`.")
    lines.append("- `scene_07`, `scene_08`, and `scene_09` were relabeled from `cup_only` to `bottle_only`.")
    lines.append("- Trials with confirmed correct target selection but missing grasp feedback were counted as successful grasps.")
    lines.append("- Raw JSON files were not modified; all corrections are analysis-layer fields in the generated CSVs.")
    lines.append("")
    lines.append("## Success Overrides")
    lines.append("")
    override = all_df[all_df["audit_success_override"] == True]  # noqa: E712
    if override.empty:
        lines.append("No success overrides were applied.")
    else:
        lines.append("| Trial | Scene | Method | Prompt Type | Original Status | Original Outcome | Prompt |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in override.iterrows():
            prompt = str(row.get("prompt") or "").replace("|", "\\|")
            lines.append(
                f"| {row['trial_id']} | {row['scene_id']} | {row['method']} | {row['prompt_type']} | "
                f"{_display_value(row.get('original_status'))} | {_display_value(row.get('original_outcome'))} | {prompt} |"
            )
    lines.append("")
    lines.append("## Missing Strict 240 Cells")
    lines.append("")
    if missing_df.empty:
        lines.append("No missing scene x prompt_type x method cells.")
    else:
        lines.append("| Scene | Prompt Type | Method |")
        lines.append("| --- | --- | --- |")
        for _, row in missing_df.iterrows():
            lines.append(f"| {row['scene_id']} | {row['prompt_type']} | {row['method']} |")
    lines.append("")
    lines.append("## Main Balanced Metrics")
    lines.append("")
    metrics = _summarize(method_prompt_df)
    lines.append(_markdown_table(metrics))
    lines.append("")
    lines.append("## By Method")
    lines.append("")
    lines.append(_markdown_table(_summarize(method_prompt_df, ["method"])))
    lines.append("")
    lines.append("## By Prompt Type")
    lines.append("")
    lines.append(_markdown_table(_summarize(method_prompt_df, ["prompt_type"])))
    lines.append("")
    lines.append("## Method x Prompt Type Counts")
    lines.append("")
    counts = method_prompt_df.groupby(["method", "prompt_type"]).size().reset_index(name="N")
    lines.append(_markdown_table(counts))
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for name in [
        "all_corrected_trials_main02.csv",
        "cleaned_evaluated_trials_main02.csv",
        "balanced_scene_cell_trials_main02.csv",
        "balanced_method_prompt_trials_main02.csv",
        "missing_scene_prompt_method_cells_main02.csv",
        "scene_cell_excluded_main02.csv",
        "method_prompt_excluded_main02.csv",
    ]:
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Exclusion Counts")
    lines.append("")
    lines.append(f"- Scene-cell duplicate exclusions: {len(scene_excluded)}")
    lines.append(f"- Method-prompt balance exclusions: {len(method_prompt_excluded)}")
    (out / "online_main02_cleaning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="services/a6000_web/artifacts/online_experiments/main_02",
        help="Path to the main_02 online experiment directory.",
    )
    parser.add_argument(
        "--out",
        default="services/a6000_web/artifacts/online_experiments/main_02/analysis",
        help="Output directory for cleaned analysis artifacts.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = _load_trials(root)
    rows = _apply_scene_corrections(rows)
    rows = _apply_feedback_override(rows)
    all_df = _flatten_for_csv(rows)
    evaluated_df = all_df[all_df["include_in_cleaned_analysis"] == True].copy()  # noqa: E712

    scene_cell_df, scene_excluded, missing_df = _choose_one_per_scene_cell(all_df, SEED)
    method_prompt_df, method_prompt_excluded, target_n = _balance_method_prompt(scene_cell_df, SEED)

    all_df.to_csv(out / "all_corrected_trials_main02.csv", index=False)
    evaluated_df.to_csv(out / "cleaned_evaluated_trials_main02.csv", index=False)
    scene_cell_df.to_csv(out / "balanced_scene_cell_trials_main02.csv", index=False)
    method_prompt_df.to_csv(out / "balanced_method_prompt_trials_main02.csv", index=False)
    missing_df.to_csv(out / "missing_scene_prompt_method_cells_main02.csv", index=False)
    scene_excluded.to_csv(out / "scene_cell_excluded_main02.csv", index=False)
    method_prompt_excluded.to_csv(out / "method_prompt_excluded_main02.csv", index=False)

    _summarize(method_prompt_df).to_csv(out / "balanced_method_prompt_overall_metrics.csv", index=False)
    _summarize(method_prompt_df, ["method"]).to_csv(out / "balanced_method_prompt_by_method_metrics.csv", index=False)
    _summarize(method_prompt_df, ["prompt_type"]).to_csv(out / "balanced_method_prompt_by_prompt_type_metrics.csv", index=False)
    _summarize(method_prompt_df, ["method", "prompt_type"]).to_csv(
        out / "balanced_method_prompt_by_method_prompt_metrics.csv",
        index=False,
    )
    _write_report(
        out,
        all_df=all_df,
        evaluated_df=evaluated_df,
        scene_cell_df=scene_cell_df,
        method_prompt_df=method_prompt_df,
        missing_df=missing_df,
        scene_excluded=scene_excluded,
        method_prompt_excluded=method_prompt_excluded,
        target_n=target_n,
    )

    print(f"Wrote report: {out / 'online_main02_cleaning_report.md'}")
    print(f"Wrote main balanced CSV: {out / 'balanced_method_prompt_trials_main02.csv'}")
    print(f"Raw={len(all_df)} evaluated={len(evaluated_df)} scene_cell={len(scene_cell_df)} method_prompt={len(method_prompt_df)}")
    if not missing_df.empty:
        print("Strict 240 cells are missing:")
        print(missing_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
