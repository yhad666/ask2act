# Offline and Online Experiment Table Summary

Generated: 2026-06-08T21:40:45

This report collects the table-level results from the cleaned offline and online experiments. It intentionally keeps interpretation light; detailed statistical and policy discussion stays in the dedicated reports.

## Source Reports

- Offline final audit summary: `services/a6000_web/artifacts/offline_experiments/top_cups_01/analysis/final_audit_metrics_after_replacement_seed20260520.md`
- Offline significance report: `outputs/statistics/offline_significance_detailed_methods_report.md`
- Online cleaning report: `services/a6000_web/artifacts/online_experiments/main_02/analysis/online_main02_cleaning_report.md`
- Online significance report: `outputs/online_statistics/main_02/online_main02_significance_report.md`
- Online grasp policy evaluation: `outputs/online_statistics/main_02/grasp_policy_evaluation.md`

# Offline Tables

## Offline By Method

| method | N | Correct | Wrong | Unresolved | Fail | Accuracy | Fail Rate | Asked N | Avg Q Asked | Acc % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_score | 216 | 116 | 83 | 17 | 100 | 53.70% | 46.30% | 0 | - | 53.70 |
| random_candidate | 216 | 115 | 91 | 10 | 101 | 53.24% | 46.76% | 0 | - | 53.24 |
| vlm_direct | 216 | 116 | 91 | 9 | 100 | 53.70% | 46.30% | 0 | - | 53.70 |
| first_question | 216 | 197 | 5 | 14 | 19 | 91.20% | 8.80% | 101 | 2.37 | 91.20 |
| random_question | 216 | 196 | 8 | 12 | 20 | 90.74% | 9.26% | 101 | 2.39 | 90.74 |
| vlm_best_question | 216 | 199 | 5 | 12 | 17 | 92.13% | 7.87% | 102 | 2.30 | 92.13 |
| proposed_efe | 216 | 199 | 9 | 8 | 17 | 92.13% | 7.87% | 104 | 1.91 | 92.13 |

## Offline By Prompt Type

Source: `outputs/statistics/main_data_consistency_by_prompt_type.csv`

| prompt_type | N | Correct | Acc | Wrong | Unresolved | Asked_N | Avg_Q_Asked | Acc_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | 506 | 306 | 0.60 | 170 | 30 | 265 | 2.75 | 60.47 |
| clear | 503 | 483 | 0.96 | 19 | 1 | 9 | 1.22 | 96.02 |
| partial | 503 | 349 | 0.69 | 103 | 51 | 134 | 1.30 | 69.38 |

## Offline Method x Prompt Type

Source: `outputs/statistics/main_data_consistency_method_prompt_type.csv`

| method | prompt_type | N | Correct | Acc | Asked_N | Avg_Q_Asked | Acc_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| first_question | ambiguous | 72 | 67 | 0.93 | 67 | 2.88 | 93.06 |
| first_question | clear | 72 | 71 | 0.99 | 3 | 1.67 | 98.61 |
| first_question | partial | 72 | 59 | 0.82 | 31 | 1.32 | 81.94 |
| proposed_efe | ambiguous | 72 | 66 | 0.92 | 66 | 2.26 | 91.67 |
| proposed_efe | clear | 71 | 70 | 0.99 | 0 | - | 98.59 |
| proposed_efe | partial | 73 | 63 | 0.86 | 38 | 1.32 | 86.30 |
| random_candidate | ambiguous | 73 | 13 | 0.18 | 0 | - | 17.81 |
| random_candidate | clear | 71 | 66 | 0.93 | 0 | - | 92.96 |
| random_candidate | partial | 72 | 36 | 0.50 | 0 | - | 50.00 |
| random_question | ambiguous | 72 | 66 | 0.92 | 66 | 2.94 | 91.67 |
| random_question | clear | 73 | 72 | 0.99 | 2 | 1.00 | 98.63 |
| random_question | partial | 71 | 58 | 0.82 | 33 | 1.36 | 81.69 |
| top_score | ambiguous | 73 | 13 | 0.18 | 0 | - | 17.81 |
| top_score | clear | 72 | 67 | 0.93 | 0 | - | 93.06 |
| top_score | partial | 71 | 36 | 0.51 | 0 | - | 50.70 |
| vlm_best_question | ambiguous | 72 | 67 | 0.93 | 66 | 2.92 | 93.06 |
| vlm_best_question | clear | 72 | 71 | 0.99 | 4 | 1.00 | 98.61 |
| vlm_best_question | partial | 72 | 61 | 0.85 | 32 | 1.19 | 84.72 |
| vlm_direct | ambiguous | 72 | 14 | 0.19 | 0 | - | 19.44 |
| vlm_direct | clear | 72 | 66 | 0.92 | 0 | - | 91.67 |
| vlm_direct | partial | 72 | 36 | 0.50 | 0 | - | 50.00 |

## Offline Accuracy GEE Contrasts

Source: `outputs/statistics/accuracy_prompt_feature_gee_contrasts.csv`

| family | method_A | method_B | log_odds_diff_A_minus_B | se | odds_ratio | or_ci95_low | or_ci95_high | p_raw | descriptive_acc_A | descriptive_acc_B | descriptive_acc_diff_A_minus_B | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | proposed_efe | top_score | 2.7483 | 0.2713 | 15.6154 | 9.1751 | 26.5763 | 0.0000 | 0.9213 | 0.5370 | 0.3843 | 0.0000 |
| A | proposed_efe | random_candidate | 2.8306 | 0.3092 | 16.9550 | 9.2488 | 31.0822 | 0.0000 | 0.9213 | 0.5324 | 0.3889 | 0.0000 |
| A | proposed_efe | vlm_direct | 2.9268 | 0.3450 | 18.6679 | 9.4927 | 36.7118 | 0.0000 | 0.9213 | 0.5370 | 0.3843 | 0.0000 |
| A | vlm_best_question | vlm_direct | 2.8876 | 0.3051 | 17.9507 | 9.8705 | 32.6456 | 0.0000 | 0.9213 | 0.5370 | 0.3843 | 0.0000 |
| B | proposed_efe | first_question | 0.1111 | 0.1780 | 1.1175 | 0.7884 | 1.5840 | 0.5325 | 0.9213 | 0.9120 | 0.0093 | 1.0000 |
| B | proposed_efe | random_question | 0.1938 | 0.1805 | 1.2139 | 0.8522 | 1.7291 | 0.2829 | 0.9213 | 0.9074 | 0.0139 | 0.8488 |
| B | proposed_efe | vlm_best_question | 0.0392 | 0.1889 | 1.0400 | 0.7182 | 1.5059 | 0.8357 | 0.9213 | 0.9213 | 0.0000 | 1.0000 |

## Offline Accuracy Paper Table

Source: `outputs/statistics/accuracy_contrast_paper_table.csv`

| Contrast | Acc A (%) | Acc B (%) | Raw Acc Diff (pp) | Adjusted OR (GEE) | 95% OR CI | Holm p | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe vs top_score | 92.1300 | 53.7000 | 38.4300 | 15.5200 | [9.14, 26.37] | 0.0000 | significant |
| proposed_efe vs random_candidate | 92.1300 | 53.2400 | 38.8900 | 15.8800 | [9.30, 27.10] | 0.0000 | significant |
| proposed_efe vs vlm_direct | 92.1300 | 53.7000 | 38.4300 | 15.8100 | [8.57, 29.15] | 0.0000 | significant |
| vlm_best_question vs vlm_direct | 92.1300 | 53.7000 | 38.4300 | 15.7100 | [8.70, 28.36] | 0.0000 | significant |
| proposed_efe vs first_question | 92.1300 | 91.2000 | 0.9300 | 1.1700 | [0.80, 1.69] | 0.8320 | comparable / n.s. |
| proposed_efe vs random_question | 92.1300 | 90.7400 | 1.3900 | 1.2200 | [0.86, 1.74] | 0.7770 | comparable / n.s. |
| proposed_efe vs vlm_best_question | 92.1300 | 92.1300 | 0.0000 | 1.0100 | [0.71, 1.43] | 0.9710 | comparable / n.s. |

## Offline Question Count Scene-Level Tests

Source: `outputs/statistics/question_count_asked_only_scene_level_tests.csv`

| context | method_A | method_B | n_paired_scenes_with_asked_trials | scene_mean_Q_EFE_asked_only | scene_mean_Q_baseline_asked_only | scene_median_Q_EFE_asked_only | scene_median_Q_baseline_asked_only | scene_level_paired_mean_diff_EFE_minus_B | scene_level_paired_median_diff_EFE_minus_B | percent_reduction_mean_Q | ci95_low | ci95_high | wilcoxon_statistic | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall_asked_only | proposed_efe | first_question | 36 | 1.9042 | 2.3639 | 1.7500 | 2.0000 | -0.4597 | -0.3167 | 19.4477 | -0.7306 | -0.2278 | 41.0000 | 0.0004 | 0.0011 |
| overall_asked_only | proposed_efe | random_question | 35 | 1.8443 | 2.2971 | 1.7500 | 2.0000 | -0.4529 | -0.3333 | 19.7139 | -0.6843 | -0.2319 | 53.5000 | 0.0007 | 0.0013 |
| overall_asked_only | proposed_efe | vlm_best_question | 36 | 1.9042 | 2.2528 | 1.7500 | 2.0000 | -0.3486 | -0.2500 | 15.4747 | -0.5875 | -0.1000 | 85.5000 | 0.0042 | 0.0042 |
| ambiguous_asked_only | proposed_efe | first_question | 35 | 2.1952 | 2.8476 | 2.0000 | 2.5000 | -0.6524 | -0.5000 | 22.9097 | -0.9476 | -0.3905 | 5.5000 | 0.0001 | 0.0002 |
| ambiguous_asked_only | proposed_efe | random_question | 34 | 2.1569 | 2.8333 | 2.0000 | 2.5000 | -0.6765 | -0.5000 | 23.8754 | -0.9510 | -0.4216 | 11.0000 | 0.0001 | 0.0002 |
| ambiguous_asked_only | proposed_efe | vlm_best_question | 33 | 2.2222 | 2.8283 | 2.0000 | 2.5000 | -0.6061 | -0.5000 | 21.4286 | -0.9040 | -0.3283 | 25.5000 | 0.0006 | 0.0006 |
| partial_asked_only | proposed_efe | first_question | 20 | 1.3500 | 1.2750 | 1.0000 | 1.0000 | 0.0750 | 0.0000 | -5.8824 | -0.0750 | 0.2250 | 4.5000 | 0.4076 | 0.8153 |
| partial_asked_only | proposed_efe | random_question | 22 | 1.3182 | 1.3636 | 1.0000 | 1.0000 | -0.0455 | 0.0000 | 3.3333 | -0.2500 | 0.1591 | 14.0000 | 0.5657 | 0.8153 |
| partial_asked_only | proposed_efe | vlm_best_question | 21 | 1.2619 | 1.1905 | 1.0000 | 1.0000 | 0.0714 | 0.0000 | -6.0000 | 0.0000 | 0.1905 | 0.0000 | 0.1797 | 0.5391 |

## Offline Ambiguous Asked-Only Question Paper Table

Source: `outputs/statistics/question_count_ambiguous_asked_only_paper_table.csv`

| Comparison | Mean Q EFE (trial asked-only) | Mean Q Baseline (trial asked-only) | Reduction (%) | Scene-level Holm p | Scene-level mean Q EFE | Scene-level mean Q Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| EFE vs first_question | 2.2600 | 2.8800 | 21.6300 | 0.0002 | 2.2000 | 2.8500 |
| EFE vs random_question | 2.2600 | 2.9400 | 23.2000 | 0.0002 | 2.1600 | 2.8300 |
| EFE vs vlm_best_question | 2.2600 | 2.9200 | 22.8000 | 0.0006 | 2.2200 | 2.8300 |

## Offline Latency Scene-Level Tests

Source: `outputs/statistics/latency_scene_level_tests.csv`

| context | method_A | method_B | n_scenes | mean_latency_EFE | median_latency_EFE | iqr_latency_EFE | mean_latency_baseline | median_latency_baseline | iqr_latency_baseline | paired_mean_diff_EFE_minus_B | mean_diff_ci95_low | mean_diff_ci95_high | paired_median_diff_EFE_minus_B | median_diff_ci95_low | median_diff_ci95_high | wilcoxon_statistic | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | proposed_efe | first_question | 33 | 141.7113 | 118.9708 | 37.7752 | 177.0751 | 141.1843 | 46.7060 | -35.3638 | -85.7474 | 10.1308 | -21.4883 | -53.6528 | 5.2731 | 216.0000 | 0.2565 | 0.5131 |
| ambiguous | proposed_efe | random_question | 33 | 141.7113 | 118.9708 | 37.7752 | 188.5758 | 131.0208 | 65.6984 | -46.8645 | -91.8774 | -8.8347 | -25.4686 | -50.4719 | -11.3248 | 155.0000 | 0.0241 | 0.0724 |
| ambiguous | proposed_efe | vlm_best_question | 33 | 141.7113 | 118.9708 | 37.7752 | 162.3159 | 130.3849 | 42.7510 | -20.6046 | -63.2712 | 23.1777 | -9.4189 | -40.9476 | 13.9602 | 232.0000 | 0.3960 | 0.5131 |
| overall | proposed_efe | first_question | 36 | 76.7671 | 62.6865 | 23.2658 | 86.7665 | 37.8690 | 28.6439 | -9.9994 | -31.2614 | 8.5536 | 7.7529 | -2.0276 | 21.1097 | 301.0000 | 0.6247 | 1.0000 |
| overall | proposed_efe | random_question | 36 | 76.7671 | 62.6865 | 23.2658 | 89.1967 | 58.2893 | 29.2955 | -12.4296 | -28.7596 | 3.7388 | -2.8239 | -17.9234 | 1.7791 | 239.0000 | 0.1433 | 0.4299 |
| overall | proposed_efe | vlm_best_question | 36 | 76.7671 | 62.6865 | 23.2658 | 77.0734 | 39.1545 | 25.2944 | -0.3063 | -18.4408 | 16.9894 | 8.1905 | -0.0500 | 14.1796 | 315.0000 | 0.7859 | 1.0000 |
| partial | proposed_efe | first_question | 32 | 78.5300 | 52.1580 | 22.0133 | 67.7650 | 33.5806 | 25.3122 | 10.7651 | -25.7447 | 38.2953 | 13.0995 | 2.9437 | 20.0501 | 111.0000 | 0.0034 | 0.0101 |
| partial | proposed_efe | random_question | 32 | 78.5300 | 52.1580 | 22.0133 | 78.3965 | 61.8666 | 22.5647 | 0.1336 | -21.3024 | 22.7520 | -1.0226 | -11.9907 | 2.3757 | 247.0000 | 0.7609 | 0.7609 |
| partial | proposed_efe | vlm_best_question | 32 | 78.5300 | 52.1580 | 22.0133 | 65.3867 | 36.9707 | 28.1480 | 13.1434 | -8.7629 | 31.1826 | 7.2580 | -0.2960 | 14.9698 | 138.0000 | 0.0175 | 0.0349 |

## Offline Prompt-Type Descriptives

Source: `outputs/statistics/prompt_type_descriptives.csv`

| prompt_type | N | accuracy | failure_rate | asked_rate | mean_Q | median_Q | mean_latency | median_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | 506 | 0.6047 | 0.3953 | 0.5237 | 1.4407 | 1.0000 | 109.7196 | 75.4907 |
| clear | 503 | 0.9602 | 0.0398 | 0.0179 | 0.0219 | 0.0000 | 8.0890 | 3.7722 |
| partial | 503 | 0.6938 | 0.3062 | 0.2664 | 0.3459 | 0.0000 | 46.6032 | 18.3558 |

## Offline Prompt-Type GEE Tests

Source: `outputs/statistics/prompt_type_gee_tests.csv`

| contrast | log_odds_diff | se | odds_ratio | or_ci95_low | or_ci95_high | p_raw | accuracy_A | accuracy_B | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_vs_clear | -3.3359 | 0.3298 | 0.0356 | 0.0186 | 0.0679 | 0.0000 | 0.6047 | 0.9602 | 0.0000 |
| partial_vs_clear | -2.7747 | 0.3673 | 0.0624 | 0.0304 | 0.1281 | 0.0000 | 0.6938 | 0.9602 | 0.0000 |
| ambiguous_vs_partial | -0.5612 | 0.3103 | 0.5705 | 0.3106 | 1.0481 | 0.0705 | 0.6047 | 0.6938 | 0.0705 |

## Offline Failure Reasons Overall

| failure_reason | count | failure_share | all_share |
| --- | --- | --- | --- |
| ambiguity_not_solved | 200 | 53.48% | 13.23% |
| gd_missing_target_candidate | 79 | 21.12% | 5.22% |
| wrong_object_selected | 65 | 17.38% | 4.30% |
| none | 26 | 6.95% | 1.72% |
| vlm_error | 3 | 0.80% | 0.20% |
| gd_label_error | 1 | 0.27% | 0.07% |

## Offline Failure Reasons By Method

| method | ambiguity_not_solved | gd_label_error | gd_missing_target_candidate | none | vlm_error | wrong_object_selected |
| --- | --- | --- | --- | --- | --- | --- |
| first_question | 1 | 0 | 11 | 1 | 0 | 6 |
| proposed_efe | 2 | 0 | 7 | 1 | 0 | 7 |
| random_candidate | 61 | 0 | 15 | 11 | 0 | 14 |
| random_question | 1 | 0 | 8 | 1 | 2 | 8 |
| top_score | 69 | 1 | 13 | 5 | 0 | 12 |
| vlm_best_question | 0 | 0 | 9 | 0 | 1 | 7 |
| vlm_direct | 66 | 0 | 16 | 7 | 0 | 11 |

## Offline Failure Reasons By Prompt Type

| prompt_type | ambiguity_not_solved | gd_label_error | gd_missing_target_candidate | none | vlm_error | wrong_object_selected |
| --- | --- | --- | --- | --- | --- | --- |
| ambiguous | 156 | 0 | 28 | 12 | 1 | 3 |
| clear | 7 | 1 | 0 | 2 | 0 | 10 |
| partial | 37 | 0 | 51 | 12 | 2 | 52 |

## Offline Failure Reasons By Scene Type

| scene_type | ambiguity_not_solved | gd_label_error | gd_missing_target_candidate | none | vlm_error | wrong_object_selected |
| --- | --- | --- | --- | --- | --- | --- |
| bottle_only | 28 | 0 | 0 | 1 | 0 | 28 |
| cup_only | 71 | 1 | 35 | 11 | 2 | 12 |
| mixed | 40 | 0 | 15 | 9 | 1 | 9 |
| utensil_only | 61 | 0 | 29 | 5 | 0 | 16 |

# Online Tables

## Online Dataset Audit

Source: `outputs/online_statistics/main_02/dataset_audit.csv`

| metric | value |
| --- | --- |
| records | 228 |
| methods | 4 |
| scenes | 20 |
| prompt_types | 3 |
| records_per_method_min | 57 |
| records_per_method_max | 57 |
| records_per_method_prompt_cell | 19 |

## Online By Method

| method | N | Target Correct | Target Acc | Task Success | Grasp Attempted | Physical Success | Physical Success Given Attempt | Correct-Object Success | Wrong Target Prevented | Wrong Object Grasp | Asked N | Asked Rate | Mean Q All | Mean Q Asked | Mean Time s | Median Time s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_score | 57 | 34 | 59.65% | 52.63% | 34 | 30 | 88.24% | 30 | 23 | 0 | 0 | 0.00% | 0.00 | - | 33.32 | 40.95 |
| random_candidate | 57 | 35 | 61.40% | 50.88% | 35 | 29 | 82.86% | 29 | 22 | 0 | 0 | 0.00% | 0.00 | - | 40.24 | 46.50 |
| vlm_best_question | 57 | 52 | 91.23% | 80.70% | 52 | 46 | 88.46% | 46 | 5 | 0 | 20 | 35.09% | 0.79 | 2.25 | 97.33 | 76.69 |
| proposed_efe | 57 | 53 | 92.98% | 87.72% | 53 | 50 | 94.34% | 50 | 4 | 0 | 19 | 33.33% | 0.61 | 1.84 | 80.24 | 60.34 |

## Online By Prompt Type

Source: `outputs/online_statistics/main_02/descriptives_by_prompt_type.csv`

| prompt_type | N | target_correct | target_fail | target_selection_accuracy | grasp_attempted | physical_grasp_success | physical_grasp_success_rate_attempted | correct_object_grasp_success | task_success_rate | wrong_target_prevented | wrong_target_prevented_rate | wrong_object_grasp | asked_N | asked_rate | mean_questions_all | mean_questions_asked | mean_time_s | median_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | 76 | 47 | 29 | 0.6184 | 47 | 44 | 0.9362 | 44 | 0.5789 | 29 | 0.3816 | 0 | 35 | 0.4605 | 1.0000 | 2.1714 | 85.0952 | 56.8869 |
| clear | 76 | 75 | 1 | 0.9868 | 75 | 68 | 0.9067 | 68 | 0.8947 | 1 | 0.0132 | 0 | 1 | 0.0132 | 0.0132 | 1.0000 | 53.5303 | 48.7368 |
| partial | 76 | 52 | 24 | 0.6842 | 52 | 43 | 0.8269 | 43 | 0.5658 | 24 | 0.3158 | 0 | 3 | 0.0395 | 0.0395 | 1.0000 | 49.7251 | 51.0438 |

## Online Method x Prompt Type

Source: `outputs/online_statistics/main_02/descriptives_by_method_prompt_type.csv`

| method | prompt_type | N | target_correct | target_fail | target_selection_accuracy | grasp_attempted | physical_grasp_success | physical_grasp_success_rate_attempted | correct_object_grasp_success | task_success_rate | wrong_target_prevented | wrong_target_prevented_rate | wrong_object_grasp | asked_N | asked_rate | mean_questions_all | mean_questions_asked | mean_time_s | median_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe | ambiguous | 19 | 18 | 1 | 0.9474 | 18 | 17 | 0.9444 | 17 | 0.8947 | 1 | 0.0526 | 0 | 17 | 0.8947 | 1.7368 | 1.9412 | 130.9127 | 126.7930 |
| proposed_efe | clear | 19 | 19 | 0 | 1.0000 | 19 | 18 | 0.9474 | 18 | 0.9474 | 0 | 0.0000 | 0 | 1 | 0.0526 | 0.0526 | 1.0000 | 52.5161 | 46.6464 |
| proposed_efe | partial | 19 | 16 | 3 | 0.8421 | 16 | 15 | 0.9375 | 15 | 0.7895 | 3 | 0.1579 | 0 | 1 | 0.0526 | 0.0526 | 1.0000 | 57.3046 | 57.9843 |
| random_candidate | ambiguous | 19 | 7 | 12 | 0.3684 | 7 | 7 | 1.0000 | 7 | 0.3684 | 12 | 0.6316 | 0 | 0 | 0.0000 | 0.0000 | - | 35.8088 | 17.0469 |
| random_candidate | clear | 19 | 19 | 0 | 1.0000 | 19 | 16 | 0.8421 | 16 | 0.8421 | 0 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | - | 50.7805 | 51.5394 |
| random_candidate | partial | 19 | 9 | 10 | 0.4737 | 9 | 6 | 0.6667 | 6 | 0.3158 | 10 | 0.5263 | 0 | 0 | 0.0000 | 0.0000 | - | 34.1271 | 42.9458 |
| top_score | ambiguous | 19 | 4 | 15 | 0.2105 | 4 | 4 | 1.0000 | 4 | 0.2105 | 15 | 0.7895 | 0 | 0 | 0.0000 | 0.0000 | - | 20.8550 | 8.7343 |
| top_score | clear | 19 | 19 | 0 | 1.0000 | 19 | 18 | 0.9474 | 18 | 0.9474 | 0 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | - | 46.4479 | 46.9819 |
| top_score | partial | 19 | 11 | 8 | 0.5789 | 11 | 8 | 0.7273 | 8 | 0.4211 | 8 | 0.4211 | 0 | 0 | 0.0000 | 0.0000 | - | 32.6558 | 38.9033 |
| vlm_best_question | ambiguous | 19 | 18 | 1 | 0.9474 | 18 | 16 | 0.8889 | 16 | 0.8421 | 1 | 0.0526 | 0 | 18 | 0.9474 | 2.2632 | 2.3889 | 152.8044 | 147.0920 |
| vlm_best_question | clear | 19 | 18 | 1 | 0.9474 | 18 | 16 | 0.8889 | 16 | 0.8421 | 1 | 0.0526 | 0 | 0 | 0.0000 | 0.0000 | - | 64.3767 | 49.2460 |
| vlm_best_question | partial | 19 | 16 | 3 | 0.8421 | 16 | 14 | 0.8750 | 14 | 0.7368 | 3 | 0.1579 | 0 | 2 | 0.1053 | 0.1053 | 1.0000 | 74.8128 | 74.5450 |

## Online Target Selection Scene Tests

Source: `outputs/online_statistics/main_02/target_selection_scene_tests.csv`

| context | method_A | method_B | n_scenes | mean_A | mean_B | median_A | median_B | mean_diff_A_minus_B | median_diff_A_minus_B | ci_low | ci_high | p_raw | test | family | metric | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | proposed_efe | top_score | 20 | 0.9000 | 0.5833 | 1.0000 | 0.6667 | 0.3167 | 0.3333 | 0.2250 | 0.4083 | 0.0001 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.0002 |
| overall | proposed_efe | random_candidate | 20 | 0.9000 | 0.5917 | 1.0000 | 0.6667 | 0.3083 | 0.3333 | 0.1583 | 0.4500 | 0.0011 | monte_carlo_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.0011 |
| overall | vlm_best_question | top_score | 20 | 0.9083 | 0.5833 | 1.0000 | 0.6667 | 0.3250 | 0.3333 | 0.2417 | 0.4083 | 0.0000 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.0001 |
| overall | vlm_best_question | random_candidate | 20 | 0.9083 | 0.5917 | 1.0000 | 0.6667 | 0.3167 | 0.3333 | 0.2000 | 0.4417 | 0.0003 | monte_carlo_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.0006 |
| overall | proposed_efe | vlm_best_question | 20 | 0.9000 | 0.9083 | 1.0000 | 1.0000 | -0.0083 | 0.0000 | -0.0750 | 0.0500 | 1.0000 | exact_sign_flip | B_efe_vs_interactive | target_selection_accuracy | 1.0000 |
| clear | proposed_efe | top_score | 19 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | - | all_zero_or_empty | A_interactive_vs_noninteractive | target_selection_accuracy | - |
| clear | proposed_efe | random_candidate | 18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | - | all_zero_or_empty | A_interactive_vs_noninteractive | target_selection_accuracy | - |
| clear | vlm_best_question | top_score | 19 | 0.9474 | 1.0000 | 1.0000 | 1.0000 | -0.0526 | 0.0000 | -0.1579 | 0.0000 | 1.0000 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 1.0000 |
| clear | vlm_best_question | random_candidate | 18 | 0.9444 | 1.0000 | 1.0000 | 1.0000 | -0.0556 | 0.0000 | -0.1667 | 0.0000 | 1.0000 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 1.0000 |
| clear | proposed_efe | vlm_best_question | 19 | 1.0000 | 0.9474 | 1.0000 | 1.0000 | 0.0526 | 0.0000 | 0.0000 | 0.1579 | 1.0000 | exact_sign_flip | B_efe_vs_interactive | target_selection_accuracy | 1.0000 |
| ambiguous | proposed_efe | top_score | 18 | 0.9444 | 0.2222 | 1.0000 | 0.0000 | 0.7222 | 1.0000 | 0.5000 | 0.8889 | 0.0002 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.0010 |
| ambiguous | proposed_efe | random_candidate | 18 | 0.9444 | 0.3889 | 1.0000 | 0.0000 | 0.5556 | 1.0000 | 0.2778 | 0.8333 | 0.0063 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.0127 |
| ambiguous | vlm_best_question | top_score | 18 | 0.9444 | 0.2222 | 1.0000 | 0.0000 | 0.7222 | 1.0000 | 0.5000 | 0.8889 | 0.0002 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.0010 |
| ambiguous | vlm_best_question | random_candidate | 18 | 0.9444 | 0.3889 | 1.0000 | 0.0000 | 0.5556 | 1.0000 | 0.2778 | 0.8333 | 0.0063 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.0127 |
| ambiguous | proposed_efe | vlm_best_question | 18 | 0.9444 | 0.9444 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | - | all_zero_or_empty | B_efe_vs_interactive | target_selection_accuracy | - |
| partial | proposed_efe | top_score | 18 | 0.8333 | 0.6111 | 1.0000 | 1.0000 | 0.2222 | 0.0000 | 0.0556 | 0.4444 | 0.1250 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.2500 |
| partial | proposed_efe | random_candidate | 18 | 0.8333 | 0.5000 | 1.0000 | 0.5000 | 0.3333 | 0.0000 | 0.1111 | 0.5556 | 0.0312 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.1250 |
| partial | vlm_best_question | top_score | 18 | 0.8333 | 0.6111 | 1.0000 | 1.0000 | 0.2222 | 0.0000 | 0.0556 | 0.4444 | 0.1250 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.2500 |
| partial | vlm_best_question | random_candidate | 18 | 0.8333 | 0.5000 | 1.0000 | 0.5000 | 0.3333 | 0.0000 | 0.1111 | 0.5556 | 0.0312 | exact_sign_flip | A_interactive_vs_noninteractive | target_selection_accuracy | 0.1250 |
| partial | proposed_efe | vlm_best_question | 18 | 0.8333 | 0.8333 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | - | all_zero_or_empty | B_efe_vs_interactive | target_selection_accuracy | - |

## Online Task Success Scene Tests

Source: `outputs/online_statistics/main_02/task_success_scene_tests.csv`

| context | method_A | method_B | n_scenes | mean_A | mean_B | median_A | median_B | mean_diff_A_minus_B | median_diff_A_minus_B | ci_low | ci_high | p_raw | test | family | metric | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | proposed_efe | top_score | 20 | 0.8500 | 0.5167 | 1.0000 | 0.5000 | 0.3333 | 0.3333 | 0.2000 | 0.4669 | 0.0005 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0022 |
| overall | proposed_efe | random_candidate | 20 | 0.8500 | 0.4917 | 1.0000 | 0.4167 | 0.3583 | 0.3333 | 0.2000 | 0.5085 | 0.0005 | monte_carlo_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0022 |
| overall | vlm_best_question | top_score | 20 | 0.8000 | 0.5167 | 0.8333 | 0.5000 | 0.2833 | 0.3333 | 0.1500 | 0.4250 | 0.0017 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0034 |
| overall | vlm_best_question | random_candidate | 20 | 0.8000 | 0.4917 | 0.8333 | 0.4167 | 0.3083 | 0.3333 | 0.1500 | 0.4583 | 0.0017 | monte_carlo_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0034 |
| overall | proposed_efe | vlm_best_question | 20 | 0.8500 | 0.8000 | 1.0000 | 0.8333 | 0.0500 | 0.0000 | -0.0502 | 0.1417 | 0.4062 | exact_sign_flip | B_efe_vs_interactive | task_success_rate | 0.4062 |
| clear | proposed_efe | top_score | 19 | 0.9474 | 0.9474 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | -0.1579 | 0.1579 | 1.0000 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 1.0000 |
| clear | proposed_efe | random_candidate | 18 | 0.9444 | 0.8333 | 1.0000 | 1.0000 | 0.1111 | 0.0000 | -0.1111 | 0.3333 | 0.6250 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 1.0000 |
| clear | vlm_best_question | top_score | 19 | 0.8421 | 0.9474 | 1.0000 | 1.0000 | -0.1053 | 0.0000 | -0.3158 | 0.1053 | 0.6250 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 1.0000 |
| clear | vlm_best_question | random_candidate | 18 | 0.8333 | 0.8333 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | -0.2222 | 0.2222 | 1.0000 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 1.0000 |
| clear | proposed_efe | vlm_best_question | 19 | 0.9474 | 0.8421 | 1.0000 | 1.0000 | 0.1053 | 0.0000 | -0.1053 | 0.3158 | 0.6250 | exact_sign_flip | B_efe_vs_interactive | task_success_rate | 0.6250 |
| ambiguous | proposed_efe | top_score | 18 | 0.8889 | 0.2222 | 1.0000 | 0.0000 | 0.6667 | 1.0000 | 0.4444 | 0.8889 | 0.0005 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0020 |
| ambiguous | proposed_efe | random_candidate | 18 | 0.8889 | 0.3889 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.2222 | 0.7778 | 0.0117 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0234 |
| ambiguous | vlm_best_question | top_score | 18 | 0.8333 | 0.2222 | 1.0000 | 0.0000 | 0.6111 | 1.0000 | 0.3889 | 0.8333 | 0.0010 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0029 |
| ambiguous | vlm_best_question | random_candidate | 18 | 0.8889 | 0.3889 | 1.0000 | 0.0000 | 0.5000 | 1.0000 | 0.1667 | 0.7778 | 0.0225 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0234 |
| ambiguous | proposed_efe | vlm_best_question | 18 | 0.9444 | 0.8333 | 1.0000 | 1.0000 | 0.1111 | 0.0000 | 0.0000 | 0.2778 | 0.5000 | exact_sign_flip | B_efe_vs_interactive | task_success_rate | 0.5000 |
| partial | proposed_efe | top_score | 18 | 0.7778 | 0.4444 | 1.0000 | 0.0000 | 0.3333 | 0.0000 | 0.0556 | 0.6111 | 0.0703 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.1406 |
| partial | proposed_efe | random_candidate | 18 | 0.7778 | 0.3333 | 1.0000 | 0.0000 | 0.4444 | 0.0000 | 0.2222 | 0.6667 | 0.0078 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0312 |
| partial | vlm_best_question | top_score | 18 | 0.7222 | 0.4444 | 1.0000 | 0.0000 | 0.2778 | 0.0000 | 0.0000 | 0.5556 | 0.1250 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.1406 |
| partial | vlm_best_question | random_candidate | 18 | 0.7222 | 0.3333 | 1.0000 | 0.0000 | 0.3889 | 0.0000 | 0.1667 | 0.6111 | 0.0156 | exact_sign_flip | A_interactive_vs_noninteractive | task_success_rate | 0.0469 |
| partial | proposed_efe | vlm_best_question | 18 | 0.7778 | 0.7222 | 1.0000 | 1.0000 | 0.0556 | 0.0000 | 0.0000 | 0.1667 | 1.0000 | exact_sign_flip | B_efe_vs_interactive | task_success_rate | 1.0000 |

## Online Question Count Scene Tests

Source: `outputs/online_statistics/main_02/question_count_scene_tests.csv`

| context | method_A | method_B | n_scenes | mean_A | mean_B | median_A | median_B | mean_diff_A_minus_B | median_diff_A_minus_B | ci_low | ci_high | p_raw | test | metric | asked_only | percent_reduction_A_vs_B | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | proposed_efe | vlm_best_question | 20 | 0.6000 | 0.7917 | 0.6667 | 0.6667 | -0.1917 | 0.0000 | -0.3917 | -0.0165 | 0.1079 | exact_sign_flip | num_questions_all | False | 24.2105 | 0.1079 |
| overall | proposed_efe | vlm_best_question | 16 | 1.9375 | 2.2812 | 2.0000 | 2.0000 | -0.3438 | 0.0000 | -0.7500 | 0.0312 | 0.1562 | exact_sign_flip | num_questions_asked_only | True | 15.0685 | 0.1562 |
| ambiguous | proposed_efe | vlm_best_question | 18 | 1.7222 | 2.2222 | 2.0000 | 2.0000 | -0.5000 | 0.0000 | -0.8333 | -0.1667 | 0.0312 | exact_sign_flip | num_questions_all | False | 22.5000 | 0.0312 |
| ambiguous | proposed_efe | vlm_best_question | 16 | 1.9375 | 2.4375 | 2.0000 | 2.0000 | -0.5000 | 0.0000 | -0.8750 | -0.1250 | 0.0547 | exact_sign_flip | num_questions_asked_only | True | 20.5128 | 0.0547 |
| partial | proposed_efe | vlm_best_question | 18 | 0.0556 | 0.1111 | 0.0000 | 0.0000 | -0.0556 | 0.0000 | -0.2778 | 0.1111 | 1.0000 | exact_sign_flip | num_questions_all | False | 50.0000 | 1.0000 |
| partial | proposed_efe | vlm_best_question | 0 | - | - | - | - | - | - | - | - | - | not_enough_pairs | num_questions_asked_only | True | - | - |

## Online Time Scene Tests

Source: `outputs/online_statistics/main_02/time_scene_tests.csv`

| context | method_A | method_B | n_scenes | mean_A | mean_B | median_A | median_B | mean_diff_A_minus_B | median_diff_A_minus_B | ci_low | ci_high | p_raw | test | metric | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | proposed_efe | vlm_best_question | 20 | 78.9421 | 98.3166 | 83.5772 | 100.7713 | -19.3745 | -15.5502 | -33.5578 | -5.9528 | 0.0129 | monte_carlo_sign_flip | analysis_time_s | 0.0334 |
| ambiguous | proposed_efe | vlm_best_question | 18 | 129.8366 | 150.4702 | 125.1627 | 143.5182 | -20.6335 | -27.9547 | -40.8244 | -1.1955 | 0.0662 | monte_carlo_sign_flip | analysis_time_s | 0.0662 |
| partial | proposed_efe | vlm_best_question | 18 | 55.7003 | 75.0992 | 57.0878 | 75.4540 | -19.3989 | -12.8026 | -32.1493 | -6.7814 | 0.0111 | monte_carlo_sign_flip | analysis_time_s | 0.0334 |

## Online Prompt-Type Scene Tests

Source: `outputs/online_statistics/main_02/prompt_type_scene_tests.csv`

| metric | prompt_A | prompt_B | n_scenes | mean_A | mean_B | mean_diff_A_minus_B | ci_low | ci_high | p_raw | test | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target_selection_success | ambiguous | clear | 20 | 0.6167 | 0.9875 | -0.3708 | -0.4625 | -0.2833 | 0.0000 | monte_carlo_sign_flip | 0.0001 |
| target_selection_success | partial | clear | 20 | 0.6833 | 0.9875 | -0.3042 | -0.4667 | -0.1500 | 0.0020 | exact_sign_flip | 0.0039 |
| target_selection_success | ambiguous | partial | 20 | 0.6167 | 0.6833 | -0.0667 | -0.2458 | 0.1084 | 0.5038 | monte_carlo_sign_flip | 0.5038 |
| task_success | ambiguous | clear | 20 | 0.5708 | 0.9000 | -0.3292 | -0.4583 | -0.2125 | 0.0001 | exact_sign_flip | 0.0002 |
| task_success | partial | clear | 20 | 0.5708 | 0.9000 | -0.3292 | -0.5000 | -0.1625 | 0.0028 | exact_sign_flip | 0.0056 |
| task_success | ambiguous | partial | 20 | 0.5708 | 0.5708 | 0.0000 | -0.1708 | 0.1542 | 1.0000 | exact_sign_flip | 1.0000 |
| num_questions | ambiguous | clear | 20 | 0.9958 | 0.0125 | 0.9833 | 0.7708 | 1.2125 | 0.0000 | monte_carlo_sign_flip | 0.0001 |
| num_questions | partial | clear | 20 | 0.0500 | 0.0125 | 0.0375 | -0.0250 | 0.1125 | 0.5000 | exact_sign_flip | 0.5000 |
| num_questions | ambiguous | partial | 20 | 0.9958 | 0.0500 | 0.9458 | 0.7250 | 1.1708 | 0.0000 | monte_carlo_sign_flip | 0.0001 |
| analysis_time_s | ambiguous | clear | 20 | 84.6971 | 53.4960 | 31.2011 | 13.6453 | 48.9083 | 0.0039 | monte_carlo_sign_flip | 0.0078 |
| analysis_time_s | partial | clear | 20 | 50.6996 | 53.4960 | -2.7964 | -14.3954 | 9.2141 | 0.6674 | monte_carlo_sign_flip | 0.6674 |
| analysis_time_s | ambiguous | partial | 20 | 84.6971 | 50.6996 | 33.9975 | 15.6375 | 52.2273 | 0.0013 | monte_carlo_sign_flip | 0.0039 |

## Online Failure By Stage

Source: `outputs/online_statistics/main_02/failure_by_stage.csv`

| failure_stage | count | share_all |
| --- | --- | --- |
| grasp_execution_failure | 19 | 0.0833 |
| none | 155 | 0.6798 |
| target_resolution_failure | 54 | 0.2368 |

## Online Failure By Method Stage

Source: `outputs/online_statistics/main_02/failure_by_method_stage.csv`

| method | failure_stage | count | share_all |
| --- | --- | --- | --- |
| proposed_efe | grasp_execution_failure | 3 | 0.0132 |
| proposed_efe | none | 50 | 0.2193 |
| proposed_efe | target_resolution_failure | 4 | 0.0175 |
| random_candidate | grasp_execution_failure | 6 | 0.0263 |
| random_candidate | none | 29 | 0.1272 |
| random_candidate | target_resolution_failure | 22 | 0.0965 |
| top_score | grasp_execution_failure | 4 | 0.0175 |
| top_score | none | 30 | 0.1316 |
| top_score | target_resolution_failure | 23 | 0.1009 |
| vlm_best_question | grasp_execution_failure | 6 | 0.0263 |
| vlm_best_question | none | 46 | 0.2018 |
| vlm_best_question | target_resolution_failure | 5 | 0.0219 |

## Online Failure By Prompt Stage

Source: `outputs/online_statistics/main_02/failure_by_prompt_stage.csv`

| prompt_type | failure_stage | count | share_all |
| --- | --- | --- | --- |
| ambiguous | grasp_execution_failure | 3 | 0.0132 |
| ambiguous | none | 44 | 0.1930 |
| ambiguous | target_resolution_failure | 29 | 0.1272 |
| clear | grasp_execution_failure | 7 | 0.0307 |
| clear | none | 68 | 0.2982 |
| clear | target_resolution_failure | 1 | 0.0044 |
| partial | grasp_execution_failure | 9 | 0.0395 |
| partial | none | 43 | 0.1886 |
| partial | target_resolution_failure | 24 | 0.1053 |

## Online Failure By Object Stage

Source: `outputs/online_statistics/main_02/failure_by_object_stage.csv`

| object_category | failure_stage | count | share_all |
| --- | --- | --- | --- |
| bottle | grasp_execution_failure | 9 | 0.0395 |
| bottle | none | 52 | 0.2281 |
| bottle | target_resolution_failure | 7 | 0.0307 |
| cup | grasp_execution_failure | 2 | 0.0088 |
| cup | none | 57 | 0.2500 |
| cup | target_resolution_failure | 21 | 0.0921 |
| fork | grasp_execution_failure | 5 | 0.0219 |
| fork | none | 21 | 0.0921 |
| fork | target_resolution_failure | 17 | 0.0746 |
| spoon | grasp_execution_failure | 3 | 0.0132 |
| spoon | none | 25 | 0.1096 |
| spoon | target_resolution_failure | 9 | 0.0395 |

## Online Failure By Reason

Source: `outputs/online_statistics/main_02/failure_by_reason.csv`

| failure_reason_coarse | count | share_all |
| --- | --- | --- |
| none | 155 | 0.6798 |
| selected_correct_target_but_physical_grasp_failed | 19 | 0.0833 |
| target_selection_wrong_or_unresolved_grasp_skipped | 54 | 0.2368 |

## Online Grasp Failure By Scene/Object

Source: `outputs/online_statistics/main_02/grasp_failure_by_scene_object.csv`

| scene_type | object_category | attempted | physical_success | physical_fail | physical_success_rate |
| --- | --- | --- | --- | --- | --- |
| bottle_only | bottle | 51 | 42 | 9 | 0.8235 |
| bottle_only | cup | 3 | 3 | 0 | 1.0000 |
| cup_only | cup | 42 | 40 | 2 | 0.9524 |
| mixed | bottle | 10 | 10 | 0 | 1.0000 |
| mixed | cup | 14 | 14 | 0 | 1.0000 |
| mixed | fork | 4 | 2 | 2 | 0.5000 |
| mixed | spoon | 9 | 9 | 0 | 1.0000 |
| utensil_only | fork | 22 | 19 | 3 | 0.8636 |
| utensil_only | spoon | 19 | 16 | 3 | 0.8421 |

# File Inventory

## Offline CSV Outputs

- `outputs/statistics/accuracy_contrast_paper_table.csv`
- `outputs/statistics/accuracy_gee_contrasts.csv`
- `outputs/statistics/accuracy_prompt_feature_gee_contrasts.csv`
- `outputs/statistics/balance_checks.csv`
- `outputs/statistics/cleaned_trial_level_for_statistics.csv`
- `outputs/statistics/column_mapping.csv`
- `outputs/statistics/dataset_audit.csv`
- `outputs/statistics/latency_scene_level_tests.csv`
- `outputs/statistics/main_data_consistency_by_method.csv`
- `outputs/statistics/main_data_consistency_by_prompt_type.csv`
- `outputs/statistics/main_data_consistency_method_prompt_type.csv`
- `outputs/statistics/prompt_balance_checks.csv`
- `outputs/statistics/prompt_feature_balance_by_method_prompt_type.csv`
- `outputs/statistics/prompt_pool_diversity_by_scene_prompt_type.csv`
- `outputs/statistics/prompt_pool_sensitivity_summary.csv`
- `outputs/statistics/prompt_type_descriptives.csv`
- `outputs/statistics/prompt_type_gee_tests.csv`
- `outputs/statistics/question_count_ambiguous_asked_only_paper_table.csv`
- `outputs/statistics/question_count_asked_only_descriptives.csv`
- `outputs/statistics/question_count_asked_only_friedman.csv`
- `outputs/statistics/question_count_asked_only_scene_level_aggregates.csv`
- `outputs/statistics/question_count_asked_only_scene_level_tests.csv`
- `outputs/statistics/question_count_scene_level_friedman.csv`
- `outputs/statistics/question_count_scene_level_tests.csv`
- `outputs/statistics/question_efficiency_80split_by_candidate_count.csv`
- `outputs/statistics/question_efficiency_80split_by_method.csv`
- `outputs/statistics/question_efficiency_80split_by_method_candidate_count.csv`
- `outputs/statistics/question_efficiency_80split_by_method_prompt_type.csv`
- `outputs/statistics/question_efficiency_80split_overall.csv`
- `outputs/statistics/question_efficiency_80split_trial_level.csv`
- `outputs/statistics/question_efficiency_asked_candidate_distribution.csv`
- `outputs/statistics/question_efficiency_by_candidate_count.csv`
- `outputs/statistics/question_efficiency_by_method.csv`
- `outputs/statistics/question_efficiency_by_method_candidate_count.csv`
- `outputs/statistics/question_efficiency_by_method_prompt_type.csv`
- `outputs/statistics/question_efficiency_overall.csv`
- `outputs/statistics/question_efficiency_trial_level.csv`
- `outputs/statistics/robustness_accuracy_old_vs_prompt_feature.csv`
- `outputs/statistics/robustness_question_old_vs_asked_only.csv`
- `outputs/statistics/scene_method_prompt_type_aggregates.csv`
- `outputs/statistics/sensitivity_accuracy_prompt_feature_gee_contrasts.csv`
- `outputs/statistics/sensitivity_question_count_scene_level_friedman.csv`
- `outputs/statistics/sensitivity_question_count_scene_level_tests.csv`
- `outputs/statistics/supplementary_exact_paired_tests.csv`

## Online CSV Outputs

- `outputs/online_statistics/main_02/cleaned_online_main02_for_statistics.csv`
- `outputs/online_statistics/main_02/cluster_bootstrap_method_ci.csv`
- `outputs/online_statistics/main_02/dataset_audit.csv`
- `outputs/online_statistics/main_02/descriptives_by_method.csv`
- `outputs/online_statistics/main_02/descriptives_by_method_prompt_type.csv`
- `outputs/online_statistics/main_02/descriptives_by_prompt_type.csv`
- `outputs/online_statistics/main_02/failure_by_method_stage.csv`
- `outputs/online_statistics/main_02/failure_by_object_stage.csv`
- `outputs/online_statistics/main_02/failure_by_prompt_stage.csv`
- `outputs/online_statistics/main_02/failure_by_reason.csv`
- `outputs/online_statistics/main_02/failure_by_scene_type_stage.csv`
- `outputs/online_statistics/main_02/failure_by_stage.csv`
- `outputs/online_statistics/main_02/grasp_failure_by_scene_object.csv`
- `outputs/online_statistics/main_02/prompt_type_scene_tests.csv`
- `outputs/online_statistics/main_02/question_count_scene_tests.csv`
- `outputs/online_statistics/main_02/scene_method_aggregates.csv`
- `outputs/online_statistics/main_02/scene_method_prompt_type_aggregates.csv`
- `outputs/online_statistics/main_02/target_selection_scene_tests.csv`
- `outputs/online_statistics/main_02/task_success_scene_tests.csv`
- `outputs/online_statistics/main_02/time_scene_tests.csv`

## Figure Outputs

- `outputs/online_statistics/main_02/fig_online_failure_stage_by_method.png`
- `outputs/online_statistics/main_02/fig_online_questions_interactive.png`
- `outputs/online_statistics/main_02/fig_online_target_accuracy_by_method.png`
- `outputs/online_statistics/main_02/fig_online_task_success_by_method.png`
- `outputs/statistics/fig_accuracy_by_method.png`
- `outputs/statistics/fig_accuracy_method_prompt_type.png`
- `outputs/statistics/fig_accuracy_vs_questions.png`
- `outputs/statistics/fig_latency_interactive.png`
- `outputs/statistics/fig_scene_level_questions_ambiguous.png`
