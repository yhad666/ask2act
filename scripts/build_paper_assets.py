from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_PROJECT = Path("/home/haoandong/workspace/project")
CP_GATE = WORKSPACE_PROJECT / "cp_gate"
CALIB_DATA = WORKSPACE_PROJECT / "calib_data"
PAPER_ASSETS = ROOT / "paper_assets"
REPORTS = ROOT / "outputs" / "research_reports"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> Path | None:
    if not src.exists() or not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path: Path, *, kind: str, role: str, source: str = "", notes: str = "") -> dict[str, str]:
    return {
        "path": rel(path),
        "kind": kind,
        "role": role,
        "source": source,
        "size_bytes": str(path.stat().st_size if path.exists() else ""),
        "sha256": file_sha256(path) if path.exists() and path.is_file() else "",
        "notes": notes,
    }


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def md_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = str(row.get(col, "")).replace("|", "\\|").replace("\n", "<br>")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def list_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for pattern in patterns:
        out.extend(root.rglob(pattern))
    return sorted({path for path in out if path.is_file()})


def copy_cp_assets(manifest: list[dict[str, str]]) -> None:
    cp_dst = PAPER_ASSETS / "cp_gate"
    source_dst = cp_dst / "source"
    results_dst = cp_dst / "calibration_results"
    sample_dst = cp_dst / "sample_images"

    for src in sorted(CP_GATE.glob("*.py")):
        dst = copy_file(src, source_dst / src.name)
        if dst:
            manifest.append(
                file_row(
                    dst,
                    kind="cp_code",
                    role="CP / calibrated GroundingDINO thresholding script",
                    source=str(src),
                    notes="Copied from the earlier ~/workspace/project/cp_gate project.",
                )
            )

    result_paths = [
        CALIB_DATA / "work" / "tau" / "tau_balanced.json",
        CALIB_DATA / "work" / "tau" / "tau_grid.csv",
        CALIB_DATA / "work" / "tau" / "tau_grid.json",
        CALIB_DATA / "work" / "eval" / "curve_single.csv",
        CALIB_DATA / "work" / "eval" / "curve_single.json",
        CALIB_DATA / "work" / "exports" / "detections.csv",
        CALIB_DATA / "work" / "exports" / "test_single_raw.csv",
        CALIB_DATA / "work" / "annotations" / "detections_labeled.csv",
        CALIB_DATA / "work" / "annotations" / "test_single_labeled_tp.csv",
        CALIB_DATA / "test_dino" / "detections.csv",
    ]
    for src in result_paths:
        dst = copy_file(src, results_dst / src.name)
        if dst:
            manifest.append(
                file_row(
                    dst,
                    kind="cp_data",
                    role="CP calibration / evaluation table",
                    source=str(src),
                    notes="Lightweight numeric calibration artifact; full calib_data images remain external.",
                )
            )

    selected_images = [
        CALIB_DATA / "bottle" / "bottle_000000.png",
        CALIB_DATA / "cup" / "cup_000000.png",
        CALIB_DATA / "kitchen" / "kitchen_000000.png",
        CALIB_DATA / "mix" / "mix_000000.png",
        CALIB_DATA / "mix_bc" / "mix_bc_000000.png",
        CALIB_DATA / "mix_n" / "mix_n_000000.png",
        CALIB_DATA / "test" / "test_000000.png",
        CALIB_DATA / "test" / "test_000050.png",
        CALIB_DATA / "test_dino" / "vis" / "test_000000.png",
        CALIB_DATA / "test_dino" / "vis" / "test_000050.png",
        CALIB_DATA / "work" / "exports" / "vis" / "bottle" / "bottle_000000.png",
        CALIB_DATA / "work" / "exports" / "vis" / "cup" / "cup_000000.png",
    ]
    for src in selected_images:
        if not src.exists():
            continue
        category = src.parent.name
        if src.parent.name == "vis":
            category = "test_dino_vis"
        dst = copy_file(src, sample_dst / category / src.name)
        if dst:
            manifest.append(
                file_row(
                    dst,
                    kind="cp_image",
                    role="Representative CP / DINO calibration image",
                    source=str(src),
                    notes="Representative sample only; full calib_data is 5.7G and intentionally not committed.",
                )
            )


def copy_scene_assets(manifest: list[dict[str, str]]) -> None:
    offline_root = ROOT / "services" / "a6000_web" / "artifacts" / "offline_experiments" / "top_cups_01" / "scenes"
    offline_dst = PAPER_ASSETS / "offline" / "sample_scenes"
    for scene_dir in sorted(offline_root.glob("scene_*")):
        for name in ("observation.jpg", "scene.json"):
            src = scene_dir / name
            dst = copy_file(src, offline_dst / scene_dir.name / name)
            if dst:
                manifest.append(
                    file_row(
                        dst,
                        kind="offline_scene_asset",
                        role="Offline tabletop observation / scene metadata",
                        source=rel(src),
                        notes="All 36 offline scene observations are included because they are compact.",
                    )
                )

    online_root = ROOT / "services" / "a6000_web" / "artifacts" / "online_experiments" / "main_02" / "scenes"
    online_dst = PAPER_ASSETS / "online" / "main_02_sample_scenes"
    for scene_dir in sorted(online_root.glob("scene_*")):
        for name in ("observation.jpg", "scene.json"):
            src = scene_dir / name
            dst = copy_file(src, online_dst / scene_dir.name / name)
            if dst:
                manifest.append(
                    file_row(
                        dst,
                        kind="online_scene_asset",
                        role="Online main_02 reference observation / scene metadata",
                        source=rel(src),
                        notes="Reference scene assets for paper figures and experiment description.",
                    )
                )


def copy_prompt_assets(manifest: list[dict[str, str]]) -> None:
    prompt_src = ROOT / "services" / "a6000_web" / "prompts" / "system_prompt.txt"
    prompt_dst = PAPER_ASSETS / "prompts" / "vlm_system_prompt.txt"
    dst = copy_file(prompt_src, prompt_dst)
    if dst:
        manifest.append(
            file_row(
                dst,
                kind="prompt",
                role="Full VLM system prompt used by Ask2Act",
                source=rel(prompt_src),
                notes="Project prompt sent to the VLM, not Codex's hidden system prompt.",
            )
        )


def build_code_map() -> list[dict[str, str]]:
    rows = [
        {
            "component": "CP calibration",
            "path": "paper_assets/cp_gate/source/compute_tau_balanced.py",
            "role": "Computes tau_final from labeled false-positive and true-positive DINO score distributions.",
            "paper_section": "GroundingDINO calibration / CP gate",
        },
        {
            "component": "CP inference",
            "path": "paper_assets/cp_gate/source/infer_with_tau.py",
            "role": "Runs GroundingDINO and filters detections using tau_final.",
            "paper_section": "GroundingDINO calibration / CP gate",
        },
        {
            "component": "Phrase extraction",
            "path": "services/a6000_web/phrase_extractor.py",
            "role": "Extracts noun-like phrases and object terms from natural-language instructions.",
            "paper_section": "Language-to-detection prompt construction",
        },
        {
            "component": "GroundingDINO runtime",
            "path": "services/a6000_web/detection.py",
            "role": "Builds dot-separated DINO prompts, runs detection, applies thresholds/NMS/fallbacks, and renders candidate overlays.",
            "paper_section": "Candidate generation",
        },
        {
            "component": "Candidate schema",
            "path": "services/a6000_web/schemas.py",
            "role": "Defines Candidate, ScoredQuestion, ResolvedTarget, trial requests, and session state.",
            "paper_section": "System representation",
        },
        {
            "component": "VLM protocol",
            "path": "services/a6000_web/clarification.py",
            "role": "Constructs VLM JSON payloads, extracts/repairs final JSON, maintains candidate state, and scores questions.",
            "paper_section": "VLM clarification protocol",
        },
        {
            "component": "VLM prompt",
            "path": "paper_assets/prompts/vlm_system_prompt.txt",
            "role": "Full system prompt defining allowed outputs, Candidate_State, yes/no splits, and forbidden question topics.",
            "paper_section": "Prompt engineering",
        },
        {
            "component": "EFE question ranking",
            "path": "services/a6000_web/clarification.py",
            "role": "Ranks proposed questions by balanced yes/no split and small diversity/repetition penalties.",
            "paper_section": "Proposed method",
        },
        {
            "component": "Offline/online API",
            "path": "services/a6000_web/server.py",
            "role": "Implements trial startup, method dispatch, online confirmation gate, auditing endpoints, and experiment UI APIs.",
            "paper_section": "Experimental platform",
        },
        {
            "component": "Experiment storage",
            "path": "services/a6000_web/offline_experiments.py",
            "role": "Stores scene/trial JSON and metrics for offline and online experiments.",
            "paper_section": "Data collection",
        },
        {
            "component": "Offline statistics",
            "path": "scripts/run_significance_tests.py",
            "role": "Runs scene-clustered GEE accuracy tests and scene-level question/latency tests.",
            "paper_section": "Offline evaluation",
        },
        {
            "component": "Online cleaning",
            "path": "scripts/clean_online_main02.py",
            "role": "Applies online main_02 cleaning rules and balances method x prompt-type cells.",
            "paper_section": "Online evaluation",
        },
        {
            "component": "Online statistics",
            "path": "scripts/run_online_main02_statistics.py",
            "role": "Computes online target, grasp, task-success, failure, and scene-level significance tables.",
            "paper_section": "Online evaluation",
        },
        {
            "component": "Real grasp runtime",
            "path": "services/a6000_web/grasp_runtime.py",
            "role": "Converts selected target bbox into SAM/bbox point cloud, geometric grasp, and dispatch payload.",
            "paper_section": "Robot execution",
        },
        {
            "component": "Point cloud generation",
            "path": "simulation/ask2act_grasp/perception/point_cloud_gen.py",
            "role": "Back-projects RGB-D/depth into target point clouds from bbox or segmentation mask.",
            "paper_section": "Robot perception",
        },
        {
            "component": "Motion planner",
            "path": "simulation/ask2act_grasp/planning/motion_planner.py",
            "role": "Solves Stretch top-down grasp motion with SimpleIK, wrist yaw, and tuning offsets.",
            "paper_section": "Robot manipulation",
        },
        {
            "component": "Robot dispatch",
            "path": "real/stretch_transport/scripts/dispatch_grasp.py",
            "role": "Executes planned waypoints on Stretch through the transport server.",
            "paper_section": "Robot deployment",
        },
    ]
    return rows


def build_data_map() -> list[dict[str, str]]:
    rows = [
        {
            "artifact": "Offline cleaned trial-level dataset",
            "path": "outputs/statistics/cleaned_trial_level_for_statistics.csv",
            "role": "Main cleaned balanced offline data used for statistics.",
            "paper_use": "Offline result tables and significance tests.",
        },
        {
            "artifact": "Offline significance report",
            "path": "outputs/statistics/significance_report.md",
            "role": "Scene-clustered/prompt-feature adjusted statistical report.",
            "paper_use": "Statistical methods/results section.",
        },
        {
            "artifact": "Offline detailed methods report",
            "path": "outputs/statistics/offline_significance_detailed_methods_report.md",
            "role": "Detailed formulas, rationale, implementation notes, and failure analysis.",
            "paper_use": "Methods appendix / reviewer response.",
        },
        {
            "artifact": "Offline table summary",
            "path": "outputs/research_reports/offline_online_table_summary.md",
            "role": "Consolidated offline and online tables.",
            "paper_use": "Fast table lookup for manuscript drafting.",
        },
        {
            "artifact": "Online cleaned trial-level dataset",
            "path": "outputs/online_statistics/main_02/cleaned_online_main02_for_statistics.csv",
            "role": "Main cleaned balanced online dataset, N=228.",
            "paper_use": "Online result tables and significance tests.",
        },
        {
            "artifact": "Online significance report",
            "path": "outputs/online_statistics/main_02/online_main02_significance_report.md",
            "role": "Online target/grasp/task success significance analysis.",
            "paper_use": "Online results section.",
        },
        {
            "artifact": "Online grasp policy evaluation",
            "path": "outputs/online_statistics/main_02/grasp_policy_evaluation.md",
            "role": "Failure analysis and policy-level interpretation of grasping results.",
            "paper_use": "Robot policy limitations / discussion.",
        },
        {
            "artifact": "CP calibration outputs",
            "path": "paper_assets/cp_gate/calibration_results/",
            "role": "Tau, detection tables, labeled annotations, and evaluation curves.",
            "paper_use": "Candidate generation calibration section.",
        },
        {
            "artifact": "CP sample images",
            "path": "paper_assets/cp_gate/sample_images/",
            "role": "Representative raw and DINO-visualized calibration images.",
            "paper_use": "Detection / calibration figures.",
        },
        {
            "artifact": "Offline scene observations",
            "path": "paper_assets/offline/sample_scenes/",
            "role": "Compact copy of all 36 offline observation images and metadata.",
            "paper_use": "Scene composition figures and dataset description.",
        },
        {
            "artifact": "Online main_02 scene observations",
            "path": "paper_assets/online/main_02_sample_scenes/",
            "role": "Reference scene observations and metadata for online experiment.",
            "paper_use": "Real-robot experiment figure examples.",
        },
    ]
    return rows


def summarize_directory(root: Path) -> dict[str, str]:
    if not root.exists():
        return {"root": str(root), "files": "0", "size_bytes": "0"}
    count = 0
    size = 0
    for path in root.rglob("*"):
        if path.is_file():
            count += 1
            size += path.stat().st_size
    return {"root": str(root), "files": str(count), "size_bytes": str(size)}


def build_report(manifest: list[dict[str, str]], code_rows: list[dict[str, str]], data_rows: list[dict[str, str]]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "paper_writing_foundation_asset_report.md"

    component_groups = [
        {
            "component": "CP / calibrated GroundingDINO gate",
            "what_to_use": "paper_assets/cp_gate/source plus paper_assets/cp_gate/calibration_results",
            "writing_angle": "Describe how DINO score thresholds were calibrated from labeled false positives and retained true positives.",
        },
        {
            "component": "Vocabulary extraction and DINO prompt construction",
            "what_to_use": "services/a6000_web/phrase_extractor.py and services/a6000_web/detection.py",
            "writing_angle": "Explain noun/attribute phrase extraction, dot-separated GroundingDINO prompts, and fallback broad-category recall.",
        },
        {
            "component": "VLM JSON protocol and prompt",
            "what_to_use": "paper_assets/prompts/vlm_system_prompt.txt and services/a6000_web/clarification.py",
            "writing_angle": "Explain strict JSON schema, Candidate_State, yes/no candidate partitions, repair calls, and forbidden candidate-id questions.",
        },
        {
            "component": "EFE question selection",
            "what_to_use": "ClarificationEngine.rank_questions_for_mode and _rank_question_key in services/a6000_web/clarification.py",
            "writing_angle": "Frame the method as selecting informative clarification questions by balanced split plus diversity/repetition penalties.",
        },
        {
            "component": "Offline experiment",
            "what_to_use": "outputs/statistics and paper_assets/offline/sample_scenes",
            "writing_angle": "Use cleaned 1512-record dataset, 7 methods, scene-clustered statistics, and offline failure analysis.",
        },
        {
            "component": "Online experiment",
            "what_to_use": "outputs/online_statistics/main_02 and paper_assets/online/main_02_sample_scenes",
            "writing_angle": "Use cleaned 228-record main_02 data, target-confirmation gate, and task/grasp success metrics.",
        },
        {
            "component": "Real robot grasping",
            "what_to_use": "services/a6000_web/grasp_runtime.py, simulation/ask2act_grasp, and real/stretch_transport",
            "writing_angle": "Explain selected target to SAM/bbox point cloud, geometric grasp, top-down Stretch planning, and execution feedback.",
        },
    ]

    dir_summaries = [
        summarize_directory(PAPER_ASSETS),
        summarize_directory(ROOT / "outputs" / "statistics"),
        summarize_directory(ROOT / "outputs" / "online_statistics" / "main_02"),
        summarize_directory(CALIB_DATA),
    ]

    lines = [
        "# Paper Writing Foundation Asset Report",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This report is a map of the important code, data, prompts, figures, and representative images for writing the Ask2Act paper. It intentionally separates lightweight committed assets from large raw runtime artifacts.",
        "",
        "## Key Deliverables",
        "",
        "- `paper_assets/asset_manifest.csv`: checksum-level inventory of copied paper assets.",
        "- `paper_assets/code_map.csv`: important implementation files and their paper roles.",
        "- `paper_assets/data_products.csv`: cleaned datasets, statistics, reports, and figure outputs.",
        "- `paper_assets/prompts/vlm_system_prompt.txt`: full project prompt sent to the VLM.",
        "- `paper_assets/cp_gate/`: CP calibration scripts, calibration tables, and representative images from the earlier project.",
        "- `outputs/research_reports/offline_online_table_summary.md`: compact table summary.",
        "- `outputs/research_reports/experimental_protocol_and_system_implementation.md`: detailed system/protocol explanation.",
        "",
        "## Directory Scale",
        "",
        md_table(dir_summaries, ["root", "files", "size_bytes"]),
        "",
        "The full historical `calib_data` directory is intentionally not copied because it is large. The committed CP package contains core scripts, numeric outputs, and representative images; the manifest records the original source paths.",
        "",
        "## Component Map",
        "",
        md_table(component_groups, ["component", "what_to_use", "writing_angle"]),
        "",
        "## Important Code Map",
        "",
        md_table(code_rows, ["component", "path", "role", "paper_section"]),
        "",
        "## Important Data and Figures",
        "",
        md_table(data_rows, ["artifact", "path", "role", "paper_use"]),
        "",
        "## Copied Asset Summary",
        "",
    ]

    by_kind: dict[str, dict[str, str]] = {}
    for row in manifest:
        kind = row["kind"]
        entry = by_kind.setdefault(kind, {"kind": kind, "files": "0", "size_bytes": "0"})
        entry["files"] = str(int(entry["files"]) + 1)
        entry["size_bytes"] = str(int(entry["size_bytes"]) + int(row.get("size_bytes") or 0))
    lines += [md_table(sorted(by_kind.values(), key=lambda item: item["kind"]), ["kind", "files", "size_bytes"]), ""]

    lines += [
        "## How To Use This In The Paper",
        "",
        "1. Use `outputs/research_reports/offline_online_table_summary.md` for result tables.",
        "2. Use `outputs/statistics/offline_significance_detailed_methods_report.md` for exact offline statistical formulas and implementation details.",
        "3. Use `outputs/online_statistics/main_02/online_main02_significance_report.md` for online statistical evidence.",
        "4. Use `outputs/online_statistics/main_02/grasp_policy_evaluation.md` for grasping-policy limitations and failure analysis.",
        "5. Use `paper_assets/code_map.csv` when writing the methods section and deciding which code to cite or inspect.",
        "6. Use `paper_assets/asset_manifest.csv` when you need exact file provenance, sizes, and checksums.",
        "",
        "## Large Artifacts Not Copied",
        "",
        "- Full CP calibration image tree: `/home/haoandong/workspace/project/calib_data`.",
        "- Full offline raw trial JSON and overlays under `services/a6000_web/artifacts/offline_experiments/top_cups_01/trials`.",
        "- Full online raw trial JSON under `services/a6000_web/artifacts/online_experiments/main_02/trials`.",
        "",
        "These raw sources remain useful for forensic debugging, but the committed cleaned CSVs, statistics, reports, scene observations, and sample images are the stable paper-writing base.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    ensure_clean_dir(PAPER_ASSETS)
    manifest: list[dict[str, str]] = []

    copy_cp_assets(manifest)
    copy_scene_assets(manifest)
    copy_prompt_assets(manifest)

    code_rows = build_code_map()
    data_rows = build_data_map()

    write_csv(PAPER_ASSETS / "code_map.csv", code_rows, ["component", "path", "role", "paper_section"])
    write_csv(PAPER_ASSETS / "data_products.csv", data_rows, ["artifact", "path", "role", "paper_use"])

    readme_lines = [
        "# Ask2Act Paper Assets",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This directory contains curated, lightweight assets for paper writing.",
        "",
        "Start with:",
        "",
        "- `asset_manifest.csv` for the full copied-file inventory.",
        "- `code_map.csv` for important implementation files.",
        "- `data_products.csv` for cleaned datasets and result tables.",
        "- `prompts/vlm_system_prompt.txt` for the full VLM protocol prompt.",
        "- `cp_gate/` for the earlier CP / calibrated DINO thresholding materials.",
        "",
        "Large raw artifacts are intentionally not mirrored here. See `outputs/research_reports/paper_writing_foundation_asset_report.md` for details.",
        "",
    ]
    (PAPER_ASSETS / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    index_files = [
        (PAPER_ASSETS / "README.md", "Top-level paper asset README"),
        (PAPER_ASSETS / "code_map.csv", "Important implementation file map"),
        (PAPER_ASSETS / "data_products.csv", "Cleaned data, statistics, and figure map"),
    ]
    for index_path, role in index_files:
        manifest.append(
            file_row(
                index_path,
                kind="index",
                role=role,
                source="generated",
                notes="Generated by scripts/build_paper_assets.py",
            )
        )
    report_path = build_report(manifest, code_rows, data_rows)
    manifest.append(
        file_row(
            report_path,
            kind="index",
            role="Human-readable paper writing foundation report",
            source="generated",
            notes="Generated by scripts/build_paper_assets.py",
        )
    )
    write_csv(
        PAPER_ASSETS / "asset_manifest.csv",
        manifest,
        ["path", "kind", "role", "source", "size_bytes", "sha256", "notes"],
    )
    print(f"Wrote {rel(report_path)}")
    print(f"Wrote {rel(PAPER_ASSETS / 'asset_manifest.csv')}")
    print(f"Wrote {rel(PAPER_ASSETS / 'code_map.csv')}")
    print(f"Wrote {rel(PAPER_ASSETS / 'data_products.csv')}")
    print(f"Copied {len(manifest)} curated assets.")


if __name__ == "__main__":
    main()
