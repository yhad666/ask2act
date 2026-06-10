# Asked-Only Question Count Recalculation

This recalculation excludes trials where no question was asked (`num_questions == 0`). It does not modify raw data.
Scene-level tests aggregate only asked trials within each `scene_id + method` cell. Therefore paired sample sizes can differ from the main all-trial analysis.

## Trial-Level Asked-Only Descriptives
| context | method | asked_trials | n_scenes_with_asked_trials | trial_level_mean_Q_asked_only | trial_level_median_Q_asked_only | avg_candidate_count_asked_only |
| --- | --- | --- | --- | --- | --- | --- |
| overall_asked_only | first_question | 101 | 36 | 2.366 | 2 | 4.218 |
| overall_asked_only | random_question | 101 | 35 | 2.386 | 2 | 4.198 |
| overall_asked_only | vlm_best_question | 102 | 36 | 2.304 | 2 | 4.147 |
| overall_asked_only | proposed_efe | 104 | 36 | 1.913 | 2 | 4.106 |
| ambiguous_asked_only | first_question | 67 | 35 | 2.881 | 3 | 4.731 |
| ambiguous_asked_only | random_question | 66 | 34 | 2.939 | 3 | 4.879 |
| ambiguous_asked_only | vlm_best_question | 66 | 33 | 2.924 | 3 | 4.864 |
| ambiguous_asked_only | proposed_efe | 66 | 36 | 2.258 | 2 | 4.773 |
| partial_asked_only | first_question | 31 | 22 | 1.323 | 1 | 3.258 |
| partial_asked_only | random_question | 33 | 24 | 1.364 | 1 | 2.97 |
| partial_asked_only | vlm_best_question | 32 | 24 | 1.188 | 1 | 2.781 |
| partial_asked_only | proposed_efe | 38 | 24 | 1.316 | 1 | 2.947 |

## Friedman Tests
| context | n_complete_scenes | friedman_statistic | p_raw |
| --- | --- | --- | --- |
| overall_asked_only | 35 | 24.18 | 2.289e-05 |
| ambiguous_asked_only | 31 | 26.84 | 6.351e-06 |
| partial_asked_only | 19 | 2.114 | 0.5492 |

## Pairwise Scene-Level Wilcoxon Tests
| context | method_A | method_B | n_paired_scenes_with_asked_trials | scene_mean_Q_EFE_asked_only | scene_mean_Q_baseline_asked_only | scene_median_Q_EFE_asked_only | scene_median_Q_baseline_asked_only | scene_level_paired_mean_diff_EFE_minus_B | scene_level_paired_median_diff_EFE_minus_B | percent_reduction_mean_Q | ci95_low | ci95_high | wilcoxon_statistic | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall_asked_only | proposed_efe | first_question | 36 | 1.904 | 2.364 | 1.75 | 2 | -0.4597 | -0.3167 | 19.45 | -0.7306 | -0.2278 | 41 | 0.0003695 | 0.001108 |
| overall_asked_only | proposed_efe | random_question | 35 | 1.844 | 2.297 | 1.75 | 2 | -0.4529 | -0.3333 | 19.71 | -0.6843 | -0.2319 | 53.5 | 0.0006553 | 0.001311 |
| overall_asked_only | proposed_efe | vlm_best_question | 36 | 1.904 | 2.253 | 1.75 | 2 | -0.3486 | -0.25 | 15.47 | -0.5875 | -0.1 | 85.5 | 0.004218 | 0.004218 |
| ambiguous_asked_only | proposed_efe | first_question | 35 | 2.195 | 2.848 | 2 | 2.5 | -0.6524 | -0.5 | 22.91 | -0.9476 | -0.3905 | 5.5 | 7.51e-05 | 0.0001821 |
| ambiguous_asked_only | proposed_efe | random_question | 34 | 2.157 | 2.833 | 2 | 2.5 | -0.6765 | -0.5 | 23.88 | -0.951 | -0.4216 | 11 | 6.07e-05 | 0.0001821 |
| ambiguous_asked_only | proposed_efe | vlm_best_question | 33 | 2.222 | 2.828 | 2 | 2.5 | -0.6061 | -0.5 | 21.43 | -0.904 | -0.3283 | 25.5 | 0.0005549 | 0.0005549 |
| partial_asked_only | proposed_efe | first_question | 20 | 1.35 | 1.275 | 1 | 1 | 0.075 | 0 | -5.882 | -0.075 | 0.225 | 4.5 | 0.4076 | 0.8153 |
| partial_asked_only | proposed_efe | random_question | 22 | 1.318 | 1.364 | 1 | 1 | -0.04545 | 0 | 3.333 | -0.25 | 0.1591 | 14 | 0.5657 | 0.8153 |
| partial_asked_only | proposed_efe | vlm_best_question | 21 | 1.262 | 1.19 | 1 | 1 | 0.07143 | 0 | -6 | 0 | 0.1905 | 0 | 0.1797 | 0.5391 |

## Short Interpretation
- `overall_asked_only`: Friedman p = 2.289e-05 with 35 complete scenes.
  - EFE vs `first_question`: mean Q 1.90 vs 2.36, diff -0.46, Holm p = 0.001108 (significant after Holm).
  - EFE vs `random_question`: mean Q 1.84 vs 2.30, diff -0.45, Holm p = 0.001311 (significant after Holm).
  - EFE vs `vlm_best_question`: mean Q 1.90 vs 2.25, diff -0.35, Holm p = 0.004218 (significant after Holm).
- `partial_asked_only`: Friedman p = 0.5492 with 19 complete scenes.
  - EFE vs `first_question`: mean Q 1.35 vs 1.27, diff 0.07, Holm p = 0.8153 (not significant after Holm).
  - EFE vs `random_question`: mean Q 1.32 vs 1.36, diff -0.05, Holm p = 0.8153 (not significant after Holm).
  - EFE vs `vlm_best_question`: mean Q 1.26 vs 1.19, diff 0.07, Holm p = 0.5391 (not significant after Holm).
- `ambiguous_asked_only`: Friedman p = 6.351e-06 with 31 complete scenes.
  - EFE vs `first_question`: mean Q 2.20 vs 2.85, diff -0.65, Holm p = 0.0001821 (significant after Holm).
  - EFE vs `random_question`: mean Q 2.16 vs 2.83, diff -0.68, Holm p = 0.0001821 (significant after Holm).
  - EFE vs `vlm_best_question`: mean Q 2.22 vs 2.83, diff -0.61, Holm p = 0.0005549 (significant after Holm).

## Saved Files
- `outputs/statistics/question_count_asked_only_descriptives.csv`
- `outputs/statistics/question_count_asked_only_friedman.csv`
- `outputs/statistics/question_count_asked_only_scene_level_tests.csv`
- `outputs/statistics/question_count_asked_only_scene_level_aggregates.csv`
