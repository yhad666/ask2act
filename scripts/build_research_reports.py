from __future__ import annotations

import argparse
import datetime as dt
import math
import textwrap
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def md_path(path: Path) -> str:
    return f"`{rel(path)}`"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame([{"missing_file": rel(path)}])
    return pd.read_csv(path)


def fmt_scalar(value) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.4g}"
        return str(value)
    return str(value)


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value) * 100.0:.2f}%"


def df_to_md(df: pd.DataFrame, *, max_rows: int | None = None, float_digits: int = 2) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    if max_rows is not None and len(out) > max_rows:
        out = out.head(max_rows).copy()
        out.loc[len(out)] = {col: "..." for col in out.columns}
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "-" if pd.isna(x) else f"{float(x):.{float_digits}f}")
    out = out.fillna("-").astype(str)
    escaped = out.copy()
    for col in escaped.columns:
        escaped[col] = escaped[col].map(lambda value: value.replace("|", "\\|").replace("\n", "<br>"))
    headers = list(escaped.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in escaped.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def csv_section(title: str, path: Path, *, max_rows: int | None = None, float_digits: int = 2) -> str:
    df = read_csv(path)
    return "\n".join(
        [
            f"## {title}",
            "",
            f"Source: {md_path(path)}",
            "",
            df_to_md(df, max_rows=max_rows, float_digits=float_digits),
            "",
        ]
    )


def select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [col for col in columns if col in df.columns]
    return df[cols].copy() if cols else df.copy()


def order_by_method(df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    if "method" not in df.columns:
        return df
    ordered = pd.Categorical(df["method"], categories=methods, ordered=True)
    out = df.assign(_method_order=ordered)
    return out.sort_values(["_method_order", "method"], na_position="last").drop(columns=["_method_order"])


def summarize_offline_failure(cleaned_path: Path) -> dict[str, pd.DataFrame]:
    df = read_csv(cleaned_path)
    if df.empty or "correct" not in df.columns:
        return {"offline_failure_overall": df}
    fail = df[df["correct"].astype(int) == 0].copy()
    if "audit_failure_reason" not in fail.columns:
        fail["audit_failure_reason"] = "none"
    fail["failure_reason"] = fail["audit_failure_reason"].fillna("").replace("", "none")
    overall = fail.groupby("failure_reason").size().reset_index(name="count")
    overall["failure_share"] = overall["count"] / max(len(fail), 1)
    overall["all_share"] = overall["count"] / max(len(df), 1)
    overall = overall.sort_values("count", ascending=False)
    method = fail.groupby(["method", "failure_reason"]).size().unstack(fill_value=0).reset_index()
    prompt = fail.groupby(["prompt_type", "failure_reason"]).size().unstack(fill_value=0).reset_index()
    scene = fail.groupby(["scene_type", "failure_reason"]).size().unstack(fill_value=0).reset_index()
    return {
        "offline_failure_overall": overall,
        "offline_failure_by_method": method,
        "offline_failure_by_prompt_type": prompt,
        "offline_failure_by_scene_type": scene,
    }


def make_online_method_paper_table(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    rename = {
        "target_selection_accuracy": "Target Acc",
        "task_success_rate": "Task Success",
        "physical_grasp_success_rate_attempted": "Physical Success Given Attempt",
        "mean_time_s": "Mean Time s",
        "median_time_s": "Median Time s",
        "target_correct": "Target Correct",
        "target_fail": "Target Fail",
        "grasp_attempted": "Grasp Attempted",
        "physical_grasp_success": "Physical Success",
        "correct_object_grasp_success": "Correct-Object Success",
        "wrong_target_prevented": "Wrong Target Prevented",
        "wrong_object_grasp": "Wrong Object Grasp",
        "asked_N": "Asked N",
        "asked_rate": "Asked Rate",
        "mean_questions_all": "Mean Q All",
        "mean_questions_asked": "Mean Q Asked",
    }
    out = df.rename(columns=rename)
    cols = [
        "method",
        "N",
        "Target Correct",
        "Target Acc",
        "Task Success",
        "Grasp Attempted",
        "Physical Success",
        "Physical Success Given Attempt",
        "Correct-Object Success",
        "Wrong Target Prevented",
        "Wrong Object Grasp",
        "Asked N",
        "Asked Rate",
        "Mean Q All",
        "Mean Q Asked",
        "Mean Time s",
        "Median Time s",
    ]
    out = select_columns(out, cols)
    for col in ("Target Acc", "Task Success", "Physical Success Given Attempt", "Asked Rate"):
        if col in out.columns:
            out[col] = out[col].map(pct)
    return order_by_method(out, ["top_score", "random_candidate", "vlm_best_question", "proposed_efe"])


def make_offline_method_paper_table(path: Path) -> pd.DataFrame:
    df = read_csv(path)
    if df.empty:
        return df
    out = df.rename(
        columns={
            "Correct": "Correct",
            "Acc": "Accuracy",
            "Wrong": "Wrong",
            "Unresolved": "Unresolved",
            "Asked_N": "Asked N",
            "Avg_Q_Asked": "Avg Q Asked",
            "Acc_pct": "Acc %",
        }
    ).copy()
    if "Wrong" in out.columns and "Unresolved" in out.columns:
        out["Fail"] = out["Wrong"].fillna(0).astype(int) + out["Unresolved"].fillna(0).astype(int)
    if "N" in out.columns and "Fail" in out.columns:
        out["Fail Rate"] = out["Fail"] / out["N"].replace(0, pd.NA)
    cols = ["method", "N", "Correct", "Wrong", "Unresolved", "Fail", "Accuracy", "Fail Rate", "Asked N", "Avg Q Asked", "Acc %"]
    out = select_columns(out, cols)
    for col in ("Accuracy", "Fail Rate"):
        if col in out.columns:
            out[col] = out[col].map(pct)
    return order_by_method(
        out,
        [
            "top_score",
            "random_candidate",
            "vlm_direct",
            "first_question",
            "random_question",
            "vlm_best_question",
            "proposed_efe",
        ],
    )


def build_table_summary(out_dir: Path) -> Path:
    stats = ROOT / "outputs" / "statistics"
    online = ROOT / "outputs" / "online_statistics" / "main_02"
    final_metrics_md = (
        ROOT
        / "services"
        / "a6000_web"
        / "artifacts"
        / "offline_experiments"
        / "top_cups_01"
        / "analysis"
        / "final_audit_metrics_after_replacement_seed20260520.md"
    )
    online_cleaning_md = (
        ROOT
        / "services"
        / "a6000_web"
        / "artifacts"
        / "online_experiments"
        / "main_02"
        / "analysis"
        / "online_main02_cleaning_report.md"
    )

    lines: list[str] = []
    lines += [
        "# Offline and Online Experiment Table Summary",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This report collects the table-level results from the cleaned offline and online experiments. It intentionally keeps interpretation light; detailed statistical and policy discussion stays in the dedicated reports.",
        "",
        "## Source Reports",
        "",
        f"- Offline final audit summary: {md_path(final_metrics_md)}",
        f"- Offline significance report: {md_path(stats / 'offline_significance_detailed_methods_report.md')}",
        f"- Online cleaning report: {md_path(online_cleaning_md)}",
        f"- Online significance report: {md_path(online / 'online_main02_significance_report.md')}",
        f"- Online grasp policy evaluation: {md_path(online / 'grasp_policy_evaluation.md')}",
        "",
        "# Offline Tables",
        "",
    ]

    offline_by_method = make_offline_method_paper_table(stats / "main_data_consistency_by_method.csv")
    lines += ["## Offline By Method", "", df_to_md(offline_by_method, float_digits=2), ""]
    lines.append(csv_section("Offline By Prompt Type", stats / "main_data_consistency_by_prompt_type.csv", float_digits=2))
    lines.append(
        csv_section(
            "Offline Method x Prompt Type",
            stats / "main_data_consistency_method_prompt_type.csv",
            float_digits=2,
        )
    )
    lines.append(csv_section("Offline Accuracy GEE Contrasts", stats / "accuracy_prompt_feature_gee_contrasts.csv", float_digits=4))
    lines.append(csv_section("Offline Accuracy Paper Table", stats / "accuracy_contrast_paper_table.csv", float_digits=4))
    lines.append(
        csv_section(
            "Offline Question Count Scene-Level Tests",
            stats / "question_count_asked_only_scene_level_tests.csv",
            float_digits=4,
        )
    )
    lines.append(
        csv_section(
            "Offline Ambiguous Asked-Only Question Paper Table",
            stats / "question_count_ambiguous_asked_only_paper_table.csv",
            float_digits=4,
        )
    )
    lines.append(csv_section("Offline Latency Scene-Level Tests", stats / "latency_scene_level_tests.csv", float_digits=4))
    lines.append(csv_section("Offline Prompt-Type Descriptives", stats / "prompt_type_descriptives.csv", float_digits=4))
    lines.append(csv_section("Offline Prompt-Type GEE Tests", stats / "prompt_type_gee_tests.csv", float_digits=4))

    failures = summarize_offline_failure(stats / "cleaned_trial_level_for_statistics.csv")
    for title, frame in (
        ("Offline Failure Reasons Overall", failures["offline_failure_overall"]),
        ("Offline Failure Reasons By Method", failures["offline_failure_by_method"]),
        ("Offline Failure Reasons By Prompt Type", failures["offline_failure_by_prompt_type"]),
        ("Offline Failure Reasons By Scene Type", failures["offline_failure_by_scene_type"]),
    ):
        if "failure_share" in frame.columns:
            frame = frame.copy()
            frame["failure_share"] = frame["failure_share"].map(pct)
            frame["all_share"] = frame["all_share"].map(pct)
        lines += [f"## {title}", "", df_to_md(frame, float_digits=2), ""]

    lines += ["# Online Tables", ""]
    lines.append(csv_section("Online Dataset Audit", online / "dataset_audit.csv", float_digits=2))
    online_method = make_online_method_paper_table(online / "descriptives_by_method.csv")
    lines += ["## Online By Method", "", df_to_md(online_method, float_digits=2), ""]
    lines.append(csv_section("Online By Prompt Type", online / "descriptives_by_prompt_type.csv", float_digits=4))
    lines.append(csv_section("Online Method x Prompt Type", online / "descriptives_by_method_prompt_type.csv", float_digits=4))
    lines.append(csv_section("Online Target Selection Scene Tests", online / "target_selection_scene_tests.csv", float_digits=4))
    lines.append(csv_section("Online Task Success Scene Tests", online / "task_success_scene_tests.csv", float_digits=4))
    lines.append(csv_section("Online Question Count Scene Tests", online / "question_count_scene_tests.csv", float_digits=4))
    lines.append(csv_section("Online Time Scene Tests", online / "time_scene_tests.csv", float_digits=4))
    lines.append(csv_section("Online Prompt-Type Scene Tests", online / "prompt_type_scene_tests.csv", float_digits=4))
    lines.append(csv_section("Online Failure By Stage", online / "failure_by_stage.csv", float_digits=4))
    lines.append(csv_section("Online Failure By Method Stage", online / "failure_by_method_stage.csv", float_digits=4))
    lines.append(csv_section("Online Failure By Prompt Stage", online / "failure_by_prompt_stage.csv", float_digits=4))
    lines.append(csv_section("Online Failure By Object Stage", online / "failure_by_object_stage.csv", float_digits=4))
    lines.append(csv_section("Online Failure By Reason", online / "failure_by_reason.csv", float_digits=4))
    lines.append(csv_section("Online Grasp Failure By Scene/Object", online / "grasp_failure_by_scene_object.csv", float_digits=4))

    lines += [
        "# File Inventory",
        "",
        "## Offline CSV Outputs",
        "",
        *[f"- {md_path(path)}" for path in sorted(stats.glob("*.csv"))],
        "",
        "## Online CSV Outputs",
        "",
        *[f"- {md_path(path)}" for path in sorted(online.glob("*.csv"))],
        "",
        "## Figure Outputs",
        "",
        *[f"- {md_path(path)}" for path in sorted(list(stats.glob("*.png")) + list(online.glob("*.png")))],
        "",
    ]

    out_path = out_dir / "offline_online_table_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"[missing: {rel(path)}]"


def code_excerpt(path: Path, start: int, end: int) -> str:
    text = read_text(path).splitlines()
    selected = text[max(start - 1, 0) : min(end, len(text))]
    numbered = [f"{idx + start:04d}: {line}" for idx, line in enumerate(selected)]
    return "\n".join(numbered)


def build_technical_report(out_dir: Path) -> Path:
    system_prompt_path = ROOT / "services" / "a6000_web" / "prompts" / "system_prompt.txt"
    detection_path = ROOT / "services" / "a6000_web" / "detection.py"
    clarification_path = ROOT / "services" / "a6000_web" / "clarification.py"
    server_path = ROOT / "services" / "a6000_web" / "server.py"
    schemas_path = ROOT / "services" / "a6000_web" / "schemas.py"
    grasp_runtime_path = ROOT / "services" / "a6000_web" / "grasp_runtime.py"
    motion_planner_path = ROOT / "simulation" / "ask2act_grasp" / "planning" / "motion_planner.py"
    robot_dispatch_path = ROOT / "real" / "stretch_transport" / "scripts" / "dispatch_grasp.py"
    cp_root = Path("/home/haoandong/workspace/project/cp_gate")
    cp_compute = cp_root / "compute_tau_balanced.py"
    cp_infer = cp_root / "infer_with_tau.py"
    cp_eval = cp_root / "evalue_tau_test.py"

    system_prompt = read_text(system_prompt_path).strip()

    lines = [
        "# Ask2Act Offline/Online Experimental Protocol and System Implementation",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This report explains how the offline and online experiments were implemented technically, how the comparison methods differ, and how the robot system turns a natural-language command into a grasp attempt.",
        "",
        "> Note: the full prompt shown here is the project prompt sent to the VLM by `services/a6000_web`. It is not the hidden system prompt of this Codex session.",
        "",
        "## 1. Code and Data Map",
        "",
        "| Component | Path | Role |",
        "| --- | --- | --- |",
        f"| Web service/API | {md_path(server_path)} | FastAPI endpoints for sessions, offline trials, online trials, audit UI, and robot execution. |",
        f"| Detection | {md_path(detection_path)} | GroundingDINO prompt construction, detection, NMS, overlay rendering, recall fallback. |",
        f"| VLM clarification | {md_path(clarification_path)} | VLM message construction, JSON extraction/repair, candidate state, question ranking, direct selection. |",
        f"| Schemas | {md_path(schemas_path)} | Trial/method/prompt/online execution request schemas and candidate/result objects. |",
        f"| Offline/online storage | {md_path(ROOT / 'services/a6000_web/offline_experiments.py')} | Experiment, scene, trial JSON storage and metrics. |",
        f"| Real grasp runtime | {md_path(grasp_runtime_path)} | RGB-D loading, SAM mask, point cloud, geometric grasp, dispatch payload. |",
        f"| Motion planner | {md_path(motion_planner_path)} | Stretch top-down IK, wrist yaw, compensation, waypoint plan. |",
        f"| Robot dispatch | {md_path(robot_dispatch_path)} | Runs Stretch waypoints on the robot server side. |",
        f"| Older CP gate | `{cp_root}` | Historical conformal/calibrated threshold scripts for DINO score filtering. |",
        "",
        "## 2. Offline Experiment Protocol",
        "",
        "The offline experiment used saved tabletop observations rather than live robot execution. Each trial followed this path:",
        "",
        "1. Load a saved scene image from `services/a6000_web/artifacts/offline_experiments/top_cups_01/scenes/<scene_id>/observation.jpg`.",
        "2. Send the prompt to the same detection and VLM resolution pipeline used by the web service.",
        "3. Record target-resolution outcome as `correct`, `wrong`, or `unresolved`.",
        "4. Audit and clean the trial-level data, then balance the final main analysis to 1512 records: 216 records for each of 7 methods.",
        "",
        "Offline methods:",
        "",
        "| Method | Interaction | Implementation |",
        "| --- | --- | --- |",
        "| `top_score` | no | Selects the highest GroundingDINO score candidate. |",
        "| `random_candidate` | no | Selects one candidate uniformly using a seeded trial id. |",
        "| `vlm_direct` | no | Asks the VLM to directly output a final `decision` JSON without questions. |",
        "| `first_question` | yes | Uses the first legal VLM question as the clarification question. |",
        "| `random_question` | yes | Randomly chooses one legal VLM question from the generated four. |",
        "| `vlm_best_question` | yes | Uses the VLM's self-ranked best question, id 1. |",
        "| `proposed_efe` | yes | Backend ranks the four legal questions by balanced split plus diversity penalties. |",
        "",
        "The common method dispatcher is implemented in `server.py`:",
        "",
        "```python",
        code_excerpt(server_path, 1107, 1223),
        "```",
        "",
        "Offline trial startup is implemented as:",
        "",
        "```python",
        code_excerpt(server_path, 3481, 3550),
        "```",
        "",
        "## 3. Online Experiment Protocol",
        "",
        "The online main experiment used live Stretch observations and physical grasp execution. The cleaned main dataset is `main_02`, balanced to 228 records: 57 trials per method and 19 trials per method × prompt type.",
        "",
        "Online methods:",
        "",
        "| Method | Uses questions | Robot execution gate |",
        "| --- | ---: | --- |",
        "| `top_score` | no | Execute only if operator confirms the selected target is correct. |",
        "| `random_candidate` | no | Execute only if operator confirms the selected target is correct. |",
        "| `vlm_best_question` | yes | Execute only after resolved target is confirmed correct. |",
        "| `proposed_efe` | yes | Execute only after resolved target is confirmed correct. |",
        "",
        "The online gate is intentionally conservative: if the resolved target is wrong, no physical grasp is executed. This saves robot wear and makes wrong-object grasp rate zero in the cleaned main experiment.",
        "",
        "Online trial startup and execution gate:",
        "",
        "```python",
        code_excerpt(server_path, 2953, 3136),
        "```",
        "",
        "Online confirmation converts physical feedback into task outcome:",
        "",
        "```python",
        code_excerpt(server_path, 3198, 3228),
        "```",
        "",
        "## 4. CP / Calibrated DINO Thresholding From the Earlier Project",
        "",
        "The older `~/workspace/project/cp_gate` code implements a calibrated score threshold for GroundingDINO candidates. It is best understood as a score-filtering/calibration module around DINO rather than the VLM clarification method itself.",
        "",
        "Data collection and inference used a broad DINO prompt:",
        "",
        "```text",
        "cup . bottle . spoon . fork . knife . plate .",
        "```",
        "",
        "The CP/calibration scripts use low baseline DINO thresholds to collect candidate scores, annotate false positives, then choose a final score threshold `tau_final`.",
        "",
        "Formula used by `compute_tau_balanced.py`:",
        "",
        "```text",
        "S_FP = scores of detections labeled as false positives",
        "S_TP = scores of all other detections",
        "tau_fp = Quantile_{1 - eps_fp}(S_FP)",
        "tau_tp = Quantile_{eps_tp}(S_TP)",
        "tau_final = max(tau_fp, tau_tp)",
        "keep detection iff score >= tau_final",
        "```",
        "",
        "Core implementation:",
        "",
        "```python",
        code_excerpt(cp_compute, 12, 78),
        "```",
        "",
        "Inference with `tau_final`:",
        "",
        "```python",
        code_excerpt(cp_infer, 108, 169),
        "```",
        "",
        "Evaluation computes object recall, precision, F1, and false positives per image:",
        "",
        "```python",
        code_excerpt(cp_eval, 83, 131),
        "```",
        "",
        "In the current A6000 runtime, GroundingDINO thresholds are controlled through environment variables such as `ASK2ACT_DINO_BOX_THRESHOLD` and `ASK2ACT_DINO_TEXT_THRESHOLD`; the old CP scripts document the calibration method and can be reused to set those thresholds or reintroduce a separate calibrated post-filter.",
        "",
        "## 5. GroundingDINO Candidate Generation",
        "",
        "Detection begins by extracting noun phrases from the instruction. The resulting terms are joined with periods because GroundingDINO expects category-like phrases separated by `.`.",
        "",
        "Core prompt construction:",
        "",
        "```python",
        code_excerpt(detection_path, 82, 115),
        "```",
        "",
        "Phrase extraction returns either spaCy noun phrases or a fallback keyword phrase:",
        "",
        "```python",
        code_excerpt(ROOT / 'services/a6000_web/phrase_extractor.py', 240, 268),
        "```",
        "",
        "Detection fallbacks were added to handle cases such as `yellow cup`, `left fork`, or a partial prompt where the initial DINO result missed matching candidates:",
        "",
        "```python",
        code_excerpt(detection_path, 360, 472),
        "```",
        "",
        "Every candidate is represented as:",
        "",
        "```python",
        code_excerpt(schemas_path, 9, 18),
        "```",
        "",
        "The candidate overlay uses `display_id` only for humans. The VLM is forbidden from asking about display ids, candidate ids, tags, or numbers.",
        "",
        "## 6. VLM JSON Protocol",
        "",
        "The VLM sees the annotated candidate overlay image plus one JSON payload. The payload contains:",
        "",
        "- `Head`: `start` or `answer`.",
        "- `Task_ID` and `Round`.",
        "- `Target_Instruction`.",
        "- `Prompt_Type`.",
        "- `Question_Mode`.",
        "- `Candidate_State` with plausible/eliminated ids.",
        "- `Candidates`, each containing `display_id`, `visual_tag`, `label`, `bbox`, ranks, and score.",
        "- `Asked` history after each human answer.",
        "",
        "Payload construction:",
        "",
        "```python",
        code_excerpt(clarification_path, 436, 589),
        "```",
        "",
        "The output must end in exactly one JSON object. The backend extracts the final valid protocol object even if the model first emits optional `<think>` reasoning:",
        "",
        "```python",
        code_excerpt(clarification_path, 31, 86),
        "```",
        "",
        "If the VLM output is truncated or illegal, the backend performs bounded repair calls and finally falls back to a legal backend protocol rather than letting robot control depend on invalid text:",
        "",
        "```python",
        code_excerpt(clarification_path, 340, 432),
        "```",
        "",
        "Question ranking for `proposed_efe`:",
        "",
        "```python",
        code_excerpt(clarification_path, 1040, 1109),
        "```",
        "",
        "The rank key combines split balance and small penalties:",
        "",
        "```text",
        "balance = |y - n| / total",
        "rank_score = balance + category_prior + repeat_category_penalty + duplicate_penalty + no_split_penalty + isolate_penalty",
        "smaller rank_score is better",
        "```",
        "",
        "This approximates the EFE intuition: a good question should reduce uncertainty by splitting the plausible candidates evenly, while also staying natural and diverse.",
        "",
        "## 7. Full Project VLM System Prompt",
        "",
        "```text",
        system_prompt,
        "```",
        "",
        "## 8. Real-Robot Grasping Pipeline",
        "",
        "After target resolution, online execution uses the selected candidate bbox and the current RGB-D observation.",
        "",
        "Pipeline:",
        "",
        "1. Load RGB, depth, intrinsics, and camera extrinsics from Stretch observation metadata.",
        "2. Map the selected candidate bbox to the depth frame, including rotated-image handling.",
        "3. Optionally predict a Segment Anything mask from the bbox.",
        "4. Generate a target point cloud using the mask; if mask point cloud has too few points, fall back to bbox crop.",
        "5. Estimate a geometric grasp from target points.",
        "6. Convert the grasp to Stretch top-down motion targets using SimpleIK when available.",
        "7. Dispatch the waypoint trajectory to the Stretch transport server.",
        "",
        "SAM/mask and point-cloud selection path:",
        "",
        "```python",
        code_excerpt(grasp_runtime_path, 883, 976),
        "```",
        "",
        "Fallback geometric grasp estimate:",
        "",
        "```python",
        code_excerpt(grasp_runtime_path, 760, 824),
        "```",
        "",
        "Top-down target solving applies side biases, rubber contact offsets, wrist yaw, SimpleIK, and FK validation:",
        "",
        "```python",
        code_excerpt(motion_planner_path, 480, 650),
        "```",
        "",
        "Tall-object top-down height is capped by the top-delta rule:",
        "",
        "```text",
        "grasp_z = max(center_z, object_top_z - max_top_grasp_delta_m)",
        "```",
        "",
        "Utensil/fork/spoon behavior depends on whether the geometric grasp is classified as slender or the requested open width is below the wrist-yaw threshold. In that case the planner tries to align the wrist yaw to the geometric grip angle instead of using a cup-like default.",
        "",
        "## 9. End-to-End System Summary",
        "",
        "The whole system is a target-resolution-first robot pipeline:",
        "",
        "```text",
        "camera/RGB-D observation",
        "  -> optional CP-calibrated / thresholded GroundingDINO candidate set",
        "  -> annotated candidate overlay",
        "  -> VLM JSON protocol for direct selection or clarification questions",
        "  -> backend candidate-state update and EFE-style question ranking",
        "  -> operator answers / target confirmation",
        "  -> skip execution if target is wrong",
        "  -> SAM/bbox point-cloud crop for confirmed target",
        "  -> geometric grasp estimate",
        "  -> Stretch top-down motion plan",
        "  -> physical grasp execution",
        "  -> operator grasp evaluation",
        "```",
        "",
        "The offline experiment isolates target-resolution accuracy and question efficiency. The online experiment measures whether those target-resolution gains transfer to full physical correct-object grasp success while protecting the robot from unnecessary wrong-target grasps.",
        "",
    ]

    out_path = out_dir / "experimental_protocol_and_system_implementation.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/research_reports")
    args = parser.parse_args()
    out_dir = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    table_report = build_table_summary(out_dir)
    technical_report = build_technical_report(out_dir)
    print(f"Wrote {table_report}")
    print(f"Wrote {technical_report}")


if __name__ == "__main__":
    main()
