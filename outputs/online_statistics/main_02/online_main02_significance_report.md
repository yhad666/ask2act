# Online Main 02 Statistical Analysis and Failure Report

## 1. Scope

This report analyzes only the cleaned `main_02` online robot experiment. Raw trial JSON files are not modified.

- Input: `services/a6000_web/artifacts/online_experiments/main_02/analysis/balanced_method_prompt_trials_main02.csv`
- Records analyzed: 228
- Methods: top_score, random_candidate, vlm_best_question, proposed_efe
- Prompt types: clear, ambiguous, partial
- Main statistical unit: scene-level aggregate
- Main small-sample test: paired sign-flip permutation over scene-level differences
- Confidence intervals: scene-clustered bootstrap or paired scene bootstrap, 2000 iterations

## 2. Why This Online Analysis Is Conservative

The online experiment is much smaller than the offline experiment and includes real robot noise. Therefore the main goal is not to force significance, but to report effect sizes, uncertainty, and cautious supporting tests.

The exact prompt wording is not treated as a fully paired unit. Instead, trials are aggregated at the scene level before method comparisons.

## 3. Dataset Audit

| metric | value |
| --- | --- |
| records | 228 |
| methods | 4 |
| scenes | 20 |
| prompt_types | 3 |
| records_per_method_min | 57 |
| records_per_method_max | 57 |
| records_per_method_prompt_cell | 19 |

## 4. Descriptive Results

### 4.1 By Method

| method | N | target_correct | target_fail | target_selection_accuracy | grasp_attempted | physical_grasp_success | physical_grasp_success_rate_attempted | correct_object_grasp_success | task_success_rate | wrong_target_prevented | wrong_target_prevented_rate | wrong_object_grasp | asked_N | asked_rate | mean_questions_all | mean_questions_asked | mean_time_s | median_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe | 57 | 53 | 4 | 0.9298 | 53 | 50 | 0.9434 | 50 | 0.8772 | 4 | 0.0702 | 0 | 19 | 0.3333 | 0.6140 | 1.8421 | 80.2445 | 60.3389 |
| random_candidate | 57 | 35 | 22 | 0.6140 | 35 | 29 | 0.8286 | 29 | 0.5088 | 22 | 0.3860 | 0 | 0 | 0.0000 | 0.0000 | - | 40.2388 | 46.4967 |
| top_score | 57 | 34 | 23 | 0.5965 | 34 | 30 | 0.8824 | 30 | 0.5263 | 23 | 0.4035 | 0 | 0 | 0.0000 | 0.0000 | - | 33.3195 | 40.9477 |
| vlm_best_question | 57 | 52 | 5 | 0.9123 | 52 | 46 | 0.8846 | 46 | 0.8070 | 5 | 0.0877 | 0 | 20 | 0.3509 | 0.7895 | 2.2500 | 97.3313 | 76.6857 |

### 4.2 By Prompt Type

| prompt_type | N | target_correct | target_fail | target_selection_accuracy | grasp_attempted | physical_grasp_success | physical_grasp_success_rate_attempted | correct_object_grasp_success | task_success_rate | wrong_target_prevented | wrong_target_prevented_rate | wrong_object_grasp | asked_N | asked_rate | mean_questions_all | mean_questions_asked | mean_time_s | median_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | 76 | 47 | 29 | 0.6184 | 47 | 44 | 0.9362 | 44 | 0.5789 | 29 | 0.3816 | 0 | 35 | 0.4605 | 1.0000 | 2.1714 | 85.0952 | 56.8869 |
| clear | 76 | 75 | 1 | 0.9868 | 75 | 68 | 0.9067 | 68 | 0.8947 | 1 | 0.0132 | 0 | 1 | 0.0132 | 0.0132 | 1.0000 | 53.5303 | 48.7368 |
| partial | 76 | 52 | 24 | 0.6842 | 52 | 43 | 0.8269 | 43 | 0.5658 | 24 | 0.3158 | 0 | 3 | 0.0395 | 0.0395 | 1.0000 | 49.7251 | 51.0438 |

### 4.3 Method x Prompt Type

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

### 4.4 Bootstrap Confidence Intervals by Method

| method | metric | estimate | ci_low | ci_high |
| --- | --- | --- | --- | --- |
| proposed_efe | target_selection_accuracy | 0.9298 | 0.8571 | 0.9831 |
| proposed_efe | task_success_rate | 0.8772 | 0.7963 | 0.9492 |
| proposed_efe | physical_grasp_success_rate_attempted | 0.9434 | 0.8823 | 1.0000 |
| proposed_efe | wrong_target_prevented_rate | 0.0702 | 0.0169 | 0.1429 |
| proposed_efe | asked_rate | 0.3333 | 0.2807 | 0.3833 |
| proposed_efe | mean_questions_all | 0.6140 | 0.4828 | 0.7544 |
| proposed_efe | mean_time_s | 80.2445 | 66.6669 | 92.8627 |
| random_candidate | target_selection_accuracy | 0.6140 | 0.5000 | 0.7241 |
| random_candidate | task_success_rate | 0.5088 | 0.3860 | 0.6429 |
| random_candidate | physical_grasp_success_rate_attempted | 0.8286 | 0.6774 | 0.9487 |
| random_candidate | wrong_target_prevented_rate | 0.3860 | 0.2759 | 0.5000 |
| random_candidate | asked_rate | 0.0000 | 0.0000 | 0.0000 |
| random_candidate | mean_questions_all | 0.0000 | 0.0000 | 0.0000 |
| random_candidate | mean_time_s | 40.2388 | 34.2393 | 47.3195 |
| top_score | target_selection_accuracy | 0.5965 | 0.4912 | 0.7018 |
| top_score | task_success_rate | 0.5263 | 0.4000 | 0.6552 |
| top_score | physical_grasp_success_rate_attempted | 0.8824 | 0.7419 | 1.0000 |
| top_score | wrong_target_prevented_rate | 0.4035 | 0.2982 | 0.5088 |
| top_score | asked_rate | 0.0000 | 0.0000 | 0.0000 |
| top_score | mean_questions_all | 0.0000 | 0.0000 | 0.0000 |
| top_score | mean_time_s | 33.3195 | 27.4293 | 38.8183 |
| vlm_best_question | target_selection_accuracy | 0.9123 | 0.8421 | 0.9667 |
| vlm_best_question | task_success_rate | 0.8070 | 0.7143 | 0.8966 |
| vlm_best_question | physical_grasp_success_rate_attempted | 0.8846 | 0.8039 | 0.9608 |
| vlm_best_question | wrong_target_prevented_rate | 0.0877 | 0.0333 | 0.1579 |
| vlm_best_question | asked_rate | 0.3509 | 0.2830 | 0.4167 |
| vlm_best_question | mean_questions_all | 0.7895 | 0.5833 | 1.0000 |
| vlm_best_question | mean_time_s | 97.3313 | 81.7864 | 112.2854 |

## 5. Statistical Methods and Formulas

For scene `s`, method `m`, and prompt type `t`, the scene-level mean for metric `Y` is:

```text
Ybar_smt = (1 / n_smt) * sum_{i in scene s, method m, prompt type t} Y_i
```

For a planned comparison between method A and method B, the paired scene difference is:

```text
d_s = Ybar_s,A,t - Ybar_s,B,t
```

The observed effect is the mean paired difference:

```text
dbar = (1 / S) * sum_s d_s
```

The paired sign-flip permutation test evaluates the null hypothesis that the signs of `d_s` are exchangeable:

```text
p = Pr(|mean(e_s * d_s)| >= |mean(d_s)|)
e_s in {-1, +1}
```

For the current sample size, the script uses exact enumeration when feasible. Holm-Bonferroni correction is applied within each planned family and context.

For question efficiency, percent reduction is:

```text
Percent reduction = 100 * (MeanQ_baseline - MeanQ_EFE) / MeanQ_baseline
```

## 6. Target Selection Accuracy Tests

Overall planned contrasts:

| method_A | method_B | n_scenes | mean_A | mean_B | mean_diff_A_minus_B | ci_low | ci_high | p_raw | p_holm | test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe | top_score | 20 | 0.9000 | 0.5833 | 0.3167 | 0.2250 | 0.4083 | 0.0001 | 0.0002 | exact_sign_flip |
| proposed_efe | random_candidate | 20 | 0.9000 | 0.5917 | 0.3083 | 0.1583 | 0.4500 | 0.0011 | 0.0011 | monte_carlo_sign_flip |
| vlm_best_question | top_score | 20 | 0.9083 | 0.5833 | 0.3250 | 0.2417 | 0.4083 | 0.0000 | 0.0001 | exact_sign_flip |
| vlm_best_question | random_candidate | 20 | 0.9083 | 0.5917 | 0.3167 | 0.2000 | 0.4417 | 0.0003 | 0.0006 | monte_carlo_sign_flip |
| proposed_efe | vlm_best_question | 20 | 0.9000 | 0.9083 | -0.0083 | -0.0750 | 0.0500 | 1.0000 | 1.0000 | exact_sign_flip |

Ambiguous-prompt contrasts:

| method_A | method_B | n_scenes | mean_A | mean_B | mean_diff_A_minus_B | ci_low | ci_high | p_raw | p_holm | test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe | top_score | 18 | 0.9444 | 0.2222 | 0.7222 | 0.5000 | 0.8889 | 0.0002 | 0.0010 | exact_sign_flip |
| proposed_efe | random_candidate | 18 | 0.9444 | 0.3889 | 0.5556 | 0.2778 | 0.8333 | 0.0063 | 0.0127 | exact_sign_flip |
| vlm_best_question | top_score | 18 | 0.9444 | 0.2222 | 0.7222 | 0.5000 | 0.8889 | 0.0002 | 0.0010 | exact_sign_flip |
| vlm_best_question | random_candidate | 18 | 0.9444 | 0.3889 | 0.5556 | 0.2778 | 0.8333 | 0.0063 | 0.0127 | exact_sign_flip |
| proposed_efe | vlm_best_question | 18 | 0.9444 | 0.9444 | 0.0000 | 0.0000 | 0.0000 | - | - | all_zero_or_empty |

## 7. Full Pipeline Task Success Tests

Task success means correct-object grasp success over all target-evaluated trials. If the target was wrong and the grasp was skipped, this is counted as task failure.

Overall planned contrasts:

| method_A | method_B | n_scenes | mean_A | mean_B | mean_diff_A_minus_B | ci_low | ci_high | p_raw | p_holm | test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe | top_score | 20 | 0.8500 | 0.5167 | 0.3333 | 0.2000 | 0.4669 | 0.0005 | 0.0022 | exact_sign_flip |
| proposed_efe | random_candidate | 20 | 0.8500 | 0.4917 | 0.3583 | 0.2000 | 0.5085 | 0.0005 | 0.0022 | monte_carlo_sign_flip |
| vlm_best_question | top_score | 20 | 0.8000 | 0.5167 | 0.2833 | 0.1500 | 0.4250 | 0.0017 | 0.0034 | exact_sign_flip |
| vlm_best_question | random_candidate | 20 | 0.8000 | 0.4917 | 0.3083 | 0.1500 | 0.4583 | 0.0017 | 0.0034 | monte_carlo_sign_flip |
| proposed_efe | vlm_best_question | 20 | 0.8500 | 0.8000 | 0.0500 | -0.0502 | 0.1417 | 0.4062 | 0.4062 | exact_sign_flip |

Ambiguous-prompt contrasts:

| method_A | method_B | n_scenes | mean_A | mean_B | mean_diff_A_minus_B | ci_low | ci_high | p_raw | p_holm | test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe | top_score | 18 | 0.8889 | 0.2222 | 0.6667 | 0.4444 | 0.8889 | 0.0005 | 0.0020 | exact_sign_flip |
| proposed_efe | random_candidate | 18 | 0.8889 | 0.3889 | 0.5000 | 0.2222 | 0.7778 | 0.0117 | 0.0234 | exact_sign_flip |
| vlm_best_question | top_score | 18 | 0.8333 | 0.2222 | 0.6111 | 0.3889 | 0.8333 | 0.0010 | 0.0029 | exact_sign_flip |
| vlm_best_question | random_candidate | 18 | 0.8889 | 0.3889 | 0.5000 | 0.1667 | 0.7778 | 0.0225 | 0.0234 | exact_sign_flip |
| proposed_efe | vlm_best_question | 18 | 0.9444 | 0.8333 | 0.1111 | 0.0000 | 0.2778 | 0.5000 | 0.5000 | exact_sign_flip |

## 8. Question Count Tests

Only `proposed_efe` and `vlm_best_question` ask questions online. Both all-trial and asked-only variants are reported.

| context | method_A | method_B | n_scenes | mean_A | mean_B | median_A | median_B | mean_diff_A_minus_B | median_diff_A_minus_B | ci_low | ci_high | p_raw | test | metric | asked_only | percent_reduction_A_vs_B | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | proposed_efe | vlm_best_question | 20 | 0.6000 | 0.7917 | 0.6667 | 0.6667 | -0.1917 | 0.0000 | -0.3917 | -0.0165 | 0.1079 | exact_sign_flip | num_questions_all | False | 24.2105 | 0.1079 |
| overall | proposed_efe | vlm_best_question | 16 | 1.9375 | 2.2812 | 2.0000 | 2.0000 | -0.3438 | 0.0000 | -0.7500 | 0.0312 | 0.1562 | exact_sign_flip | num_questions_asked_only | True | 15.0685 | 0.1562 |
| ambiguous | proposed_efe | vlm_best_question | 18 | 1.7222 | 2.2222 | 2.0000 | 2.0000 | -0.5000 | 0.0000 | -0.8333 | -0.1667 | 0.0312 | exact_sign_flip | num_questions_all | False | 22.5000 | 0.0312 |
| ambiguous | proposed_efe | vlm_best_question | 16 | 1.9375 | 2.4375 | 2.0000 | 2.0000 | -0.5000 | 0.0000 | -0.8750 | -0.1250 | 0.0547 | exact_sign_flip | num_questions_asked_only | True | 20.5128 | 0.0547 |
| partial | proposed_efe | vlm_best_question | 18 | 0.0556 | 0.1111 | 0.0000 | 0.0000 | -0.0556 | 0.0000 | -0.2778 | 0.1111 | 1.0000 | exact_sign_flip | num_questions_all | False | 50.0000 | 1.0000 |
| partial | proposed_efe | vlm_best_question | 0 | - | - | - | - | - | - | - | - | - | not_enough_pairs | num_questions_asked_only | True | - | - |

## 9. Time Tests

Time uses `analysis_time_s`, which includes target resolution and any execution/evaluation time recorded for the online trial. Skipped wrong-target trials can be much faster, so time is interpreted cautiously.

| context | method_A | method_B | n_scenes | mean_A | mean_B | median_A | median_B | mean_diff_A_minus_B | median_diff_A_minus_B | ci_low | ci_high | p_raw | test | metric | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | proposed_efe | vlm_best_question | 20 | 78.9421 | 98.3166 | 83.5772 | 100.7713 | -19.3745 | -15.5502 | -33.5578 | -5.9528 | 0.0129 | monte_carlo_sign_flip | analysis_time_s | 0.0334 |
| ambiguous | proposed_efe | vlm_best_question | 18 | 129.8366 | 150.4702 | 125.1627 | 143.5182 | -20.6335 | -27.9547 | -40.8244 | -1.1955 | 0.0662 | monte_carlo_sign_flip | analysis_time_s | 0.0662 |
| partial | proposed_efe | vlm_best_question | 18 | 55.7003 | 75.0992 | 57.0878 | 75.4540 | -19.3989 | -12.8026 | -32.1493 | -6.7814 | 0.0111 | monte_carlo_sign_flip | analysis_time_s | 0.0334 |

## 10. Prompt-Type Difficulty

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

## 11. Failure Analysis

### 11.1 Failure Stage

| failure_stage | count | share_all |
| --- | --- | --- |
| grasp_execution_failure | 19 | 0.0833 |
| none | 155 | 0.6798 |
| target_resolution_failure | 54 | 0.2368 |

### 11.2 Failure Reason

| failure_reason_coarse | count | share_all |
| --- | --- | --- |
| none | 155 | 0.6798 |
| selected_correct_target_but_physical_grasp_failed | 19 | 0.0833 |
| target_selection_wrong_or_unresolved_grasp_skipped | 54 | 0.2368 |

### 11.3 Failure Stage by Method

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

### 11.4 Failure Stage by Prompt Type

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

### 11.5 Grasp Execution Failure by Scene/Object

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

## 12. Grasping Policy Evaluation

The online data show that the target-resolution layer is now the dominant source of end-to-end difference between methods. `proposed_efe` selected the correct target in 53/57 trials, while non-interactive baselines selected the correct target in only 34-35/57 trials. Because wrong target selections were skipped rather than executed, wrong-object grasp rate is 0, but wrong-target-prevented failures remain task failures.

Conditional on attempting a grasp, the physical grasp success rate is high but not perfect. The balanced data contain 19 cases where the target was selected correctly but the physical grasp failed. These failures concentrate in bottle-only and utensil-only scenes, matching the observed real-robot issues: tall bottles stress the top-down height policy, and forks/spoons stress narrow-object orientation and gripper closing.

Policy assessment:

- The skip-on-wrong-target policy is correct for hardware safety and makes wrong-object grasp rate 0 in this dataset.
- The target-resolution benefit transfers to full task success because EFE creates more correct grasp attempts.
- The grasp policy is reliable enough for cups and many bottles/utensils, but still has object-geometry-specific weaknesses.
- Remaining grasp failures are not mainly caused by EFE question selection; they are physical execution failures after correct target resolution.
- The most important policy improvements are better segmentation-based point-cloud cropping, object-category-specific grasp pose selection, and a real wrist-orientation strategy for utensils.

## 13. Conservative Conclusions

- Online data support the offline conclusion that interactive clarification improves target resolution.
- EFE has the highest target-selection accuracy and full task success rate among the online methods.
- Because online N is small, report p-values as supporting evidence and emphasize effect sizes with confidence intervals.
- EFE asks fewer questions than VLM-best in the online data, but the statistical strength depends on the context and asked-only subset size.
- The physical grasp policy is the remaining bottleneck once target resolution is correct.

