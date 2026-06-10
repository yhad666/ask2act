# Paper Writing Foundation Asset Report

Generated: 2026-06-11T00:53:22

This report is a map of the important code, data, prompts, figures, and representative images for writing the Ask2Act paper. It intentionally separates lightweight committed assets from large raw runtime artifacts.

## Key Deliverables

- `paper_assets/asset_manifest.csv`: checksum-level inventory of copied paper assets.
- `paper_assets/code_map.csv`: important implementation files and their paper roles.
- `paper_assets/data_products.csv`: cleaned datasets, statistics, reports, and figure outputs.
- `paper_assets/prompts/vlm_system_prompt.txt`: full project prompt sent to the VLM.
- `paper_assets/cp_gate/`: CP calibration scripts, calibration tables, and representative images from the earlier project.
- `outputs/research_reports/offline_online_table_summary.md`: compact table summary.
- `outputs/research_reports/experimental_protocol_and_system_implementation.md`: detailed system/protocol explanation.

## Directory Scale

| root | files | size_bytes |
| --- | --- | --- |
| /home/haoandong/ask2act/paper_assets | 141 | 36854583 |
| /home/haoandong/ask2act/outputs/statistics | 68 | 2231621 |
| /home/haoandong/ask2act/outputs/online_statistics/main_02 | 26 | 684514 |
| /home/haoandong/workspace/project/calib_data | 4715 | 6020487749 |

The full historical `calib_data` directory is intentionally not copied because it is large. The committed CP package contains core scripts, numeric outputs, and representative images; the manifest records the original source paths.

## Component Map

| component | what_to_use | writing_angle |
| --- | --- | --- |
| CP / calibrated GroundingDINO gate | paper_assets/cp_gate/source plus paper_assets/cp_gate/calibration_results | Describe how DINO score thresholds were calibrated from labeled false positives and retained true positives. |
| Vocabulary extraction and DINO prompt construction | services/a6000_web/phrase_extractor.py and services/a6000_web/detection.py | Explain noun/attribute phrase extraction, dot-separated GroundingDINO prompts, and fallback broad-category recall. |
| VLM JSON protocol and prompt | paper_assets/prompts/vlm_system_prompt.txt and services/a6000_web/clarification.py | Explain strict JSON schema, Candidate_State, yes/no candidate partitions, repair calls, and forbidden candidate-id questions. |
| EFE question selection | ClarificationEngine.rank_questions_for_mode and _rank_question_key in services/a6000_web/clarification.py | Frame the method as selecting informative clarification questions by balanced split plus diversity/repetition penalties. |
| Offline experiment | outputs/statistics and paper_assets/offline/sample_scenes | Use cleaned 1512-record dataset, 7 methods, scene-clustered statistics, and offline failure analysis. |
| Online experiment | outputs/online_statistics/main_02 and paper_assets/online/main_02_sample_scenes | Use cleaned 228-record main_02 data, target-confirmation gate, and task/grasp success metrics. |
| Real robot grasping | services/a6000_web/grasp_runtime.py, simulation/ask2act_grasp, and real/stretch_transport | Explain selected target to SAM/bbox point cloud, geometric grasp, top-down Stretch planning, and execution feedback. |

## Important Code Map

| component | path | role | paper_section |
| --- | --- | --- | --- |
| CP calibration | paper_assets/cp_gate/source/compute_tau_balanced.py | Computes tau_final from labeled false-positive and true-positive DINO score distributions. | GroundingDINO calibration / CP gate |
| CP inference | paper_assets/cp_gate/source/infer_with_tau.py | Runs GroundingDINO and filters detections using tau_final. | GroundingDINO calibration / CP gate |
| Phrase extraction | services/a6000_web/phrase_extractor.py | Extracts noun-like phrases and object terms from natural-language instructions. | Language-to-detection prompt construction |
| GroundingDINO runtime | services/a6000_web/detection.py | Builds dot-separated DINO prompts, runs detection, applies thresholds/NMS/fallbacks, and renders candidate overlays. | Candidate generation |
| Candidate schema | services/a6000_web/schemas.py | Defines Candidate, ScoredQuestion, ResolvedTarget, trial requests, and session state. | System representation |
| VLM protocol | services/a6000_web/clarification.py | Constructs VLM JSON payloads, extracts/repairs final JSON, maintains candidate state, and scores questions. | VLM clarification protocol |
| VLM prompt | paper_assets/prompts/vlm_system_prompt.txt | Full system prompt defining allowed outputs, Candidate_State, yes/no splits, and forbidden question topics. | Prompt engineering |
| EFE question ranking | services/a6000_web/clarification.py | Ranks proposed questions by balanced yes/no split and small diversity/repetition penalties. | Proposed method |
| Offline/online API | services/a6000_web/server.py | Implements trial startup, method dispatch, online confirmation gate, auditing endpoints, and experiment UI APIs. | Experimental platform |
| Experiment storage | services/a6000_web/offline_experiments.py | Stores scene/trial JSON and metrics for offline and online experiments. | Data collection |
| Offline statistics | scripts/run_significance_tests.py | Runs scene-clustered GEE accuracy tests and scene-level question/latency tests. | Offline evaluation |
| Online cleaning | scripts/clean_online_main02.py | Applies online main_02 cleaning rules and balances method x prompt-type cells. | Online evaluation |
| Online statistics | scripts/run_online_main02_statistics.py | Computes online target, grasp, task-success, failure, and scene-level significance tables. | Online evaluation |
| Real grasp runtime | services/a6000_web/grasp_runtime.py | Converts selected target bbox into SAM/bbox point cloud, geometric grasp, and dispatch payload. | Robot execution |
| Point cloud generation | simulation/ask2act_grasp/perception/point_cloud_gen.py | Back-projects RGB-D/depth into target point clouds from bbox or segmentation mask. | Robot perception |
| Motion planner | simulation/ask2act_grasp/planning/motion_planner.py | Solves Stretch top-down grasp motion with SimpleIK, wrist yaw, and tuning offsets. | Robot manipulation |
| Robot dispatch | real/stretch_transport/scripts/dispatch_grasp.py | Executes planned waypoints on Stretch through the transport server. | Robot deployment |

## Important Data and Figures

| artifact | path | role | paper_use |
| --- | --- | --- | --- |
| Offline cleaned trial-level dataset | outputs/statistics/cleaned_trial_level_for_statistics.csv | Main cleaned balanced offline data used for statistics. | Offline result tables and significance tests. |
| Offline significance report | outputs/statistics/significance_report.md | Scene-clustered/prompt-feature adjusted statistical report. | Statistical methods/results section. |
| Offline detailed methods report | outputs/statistics/offline_significance_detailed_methods_report.md | Detailed formulas, rationale, implementation notes, and failure analysis. | Methods appendix / reviewer response. |
| Offline table summary | outputs/research_reports/offline_online_table_summary.md | Consolidated offline and online tables. | Fast table lookup for manuscript drafting. |
| Online cleaned trial-level dataset | outputs/online_statistics/main_02/cleaned_online_main02_for_statistics.csv | Main cleaned balanced online dataset, N=228. | Online result tables and significance tests. |
| Online significance report | outputs/online_statistics/main_02/online_main02_significance_report.md | Online target/grasp/task success significance analysis. | Online results section. |
| Online grasp policy evaluation | outputs/online_statistics/main_02/grasp_policy_evaluation.md | Failure analysis and policy-level interpretation of grasping results. | Robot policy limitations / discussion. |
| CP calibration outputs | paper_assets/cp_gate/calibration_results/ | Tau, detection tables, labeled annotations, and evaluation curves. | Candidate generation calibration section. |
| CP sample images | paper_assets/cp_gate/sample_images/ | Representative raw and DINO-visualized calibration images. | Detection / calibration figures. |
| Offline scene observations | paper_assets/offline/sample_scenes/ | Compact copy of all 36 offline observation images and metadata. | Scene composition figures and dataset description. |
| Online main_02 scene observations | paper_assets/online/main_02_sample_scenes/ | Reference scene observations and metadata for online experiment. | Real-robot experiment figure examples. |

## Copied Asset Summary

| kind | files | size_bytes |
| --- | --- | --- |
| cp_code | 9 | 51296 |
| cp_data | 10 | 5522773 |
| cp_image | 12 | 18269318 |
| index | 3 | 5570 |
| offline_scene_asset | 72 | 11463894 |
| online_scene_asset | 37 | 5664228 |
| prompt | 1 | 17085 |

## How To Use This In The Paper

1. Use `outputs/research_reports/offline_online_table_summary.md` for result tables.
2. Use `outputs/statistics/offline_significance_detailed_methods_report.md` for exact offline statistical formulas and implementation details.
3. Use `outputs/online_statistics/main_02/online_main02_significance_report.md` for online statistical evidence.
4. Use `outputs/online_statistics/main_02/grasp_policy_evaluation.md` for grasping-policy limitations and failure analysis.
5. Use `paper_assets/code_map.csv` when writing the methods section and deciding which code to cite or inspect.
6. Use `paper_assets/asset_manifest.csv` when you need exact file provenance, sizes, and checksums.

## Large Artifacts Not Copied

- Full CP calibration image tree: `/home/haoandong/workspace/project/calib_data`.
- Full offline raw trial JSON and overlays under `services/a6000_web/artifacts/offline_experiments/top_cups_01/trials`.
- Full online raw trial JSON under `services/a6000_web/artifacts/online_experiments/main_02/trials`.

These raw sources remain useful for forensic debugging, but the committed cleaned CSVs, statistics, reports, scene observations, and sample images are the stable paper-writing base.
