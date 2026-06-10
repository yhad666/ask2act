# Online Main 02 Cleaning Report

- Raw trials: 266
- Evaluated/cleaned trials before balancing: 247
- Scene-cell balanced trials: 236
- Method x prompt balanced trials: 228
- Method x prompt target per cell: 19
- Random seed: 20260603

## Corrections Applied

- Late half of `scene_04` was relabeled to `scene_05`; both are `cup_only`.
- `scene_07`, `scene_08`, and `scene_09` were relabeled from `cup_only` to `bottle_only`.
- Trials with confirmed correct target selection but missing grasp feedback were counted as successful grasps.
- Raw JSON files were not modified; all corrections are analysis-layer fields in the generated CSVs.

## Success Overrides

| Trial | Scene | Method | Prompt Type | Original Status | Original Outcome | Prompt |
| --- | --- | --- | --- | --- | --- | --- |
| online_trial_20260521_194540_4ccde2 | scene_01 | vlm_best_question | partial | executed | - | Pick up the green cup. |
| online_trial_20260521_202740_d99b55 | scene_02 | random_candidate | clear | executed | - | Pick up the yellow cup. |
| online_trial_20260522_184149_b1920e | scene_06 | vlm_best_question | clear | finished | aborted | Pick up the green bottle. |
| online_trial_20260522_201827_ef0006 | scene_09 | proposed_efe | clear | executed | - | Pick up the right bottle. |
| online_trial_20260523_212636_625cf4 | scene_12 | proposed_efe | ambiguous | executed | - | Pick up my fork. |
| online_trial_20260523_214451_6b1291 | scene_13 | top_score | partial | finished | aborted | Pick up the right spoon. |
| online_trial_20260523_215338_7d2435 | scene_14 | proposed_efe | ambiguous | executed | - | Pick up my spoon. |
| online_trial_20260527_184207_cdc237 | scene_16 | top_score | clear | executed | - | Picky up the red bottle. |
| online_trial_20260527_204225_08e5cb | scene_15 | proposed_efe | partial | executed | - | Pick up the blue spoon. |

## Missing Strict 240 Cells

| Scene | Prompt Type | Method |
| --- | --- | --- |
| scene_06 | ambiguous | vlm_best_question |
| scene_20 | clear | top_score |
| scene_20 | clear | vlm_best_question |
| scene_20 | clear | proposed_efe |

## Main Balanced Metrics

| group | N | target_eval_N | target_correct | target_wrong_or_unresolved | target_selection_accuracy | grasp_attempted | physical_grasp_success | physical_grasp_success_rate | correct_object_grasp_success | task_success_rate | wrong_object_grasp | wrong_target_prevented | asked_N | asked_rate | mean_questions_all | mean_questions_asked | mean_time_s | median_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 228 | 228 | 174 | 54 | 0.7632 | 174 | 155 | 0.8908 | 155 | 0.6798 | 0 | 54 | 39 | 0.1711 | 0.3509 | 2.0513 | 62.7835 | 50.0764 |

## By Method

| method | N | target_eval_N | target_correct | target_wrong_or_unresolved | target_selection_accuracy | grasp_attempted | physical_grasp_success | physical_grasp_success_rate | correct_object_grasp_success | task_success_rate | wrong_object_grasp | wrong_target_prevented | asked_N | asked_rate | mean_questions_all | mean_questions_asked | mean_time_s | median_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe | 57 | 57 | 53 | 4 | 0.9298 | 53 | 50 | 0.9434 | 50 | 0.8772 | 0 | 4 | 19 | 0.3333 | 0.6140 | 1.8421 | 80.2445 | 60.3389 |
| random_candidate | 57 | 57 | 35 | 22 | 0.6140 | 35 | 29 | 0.8286 | 29 | 0.5088 | 0 | 22 | 0 | 0.0000 | 0.0000 | - | 40.2388 | 46.4967 |
| top_score | 57 | 57 | 34 | 23 | 0.5965 | 34 | 30 | 0.8824 | 30 | 0.5263 | 0 | 23 | 0 | 0.0000 | 0.0000 | - | 33.3195 | 40.9477 |
| vlm_best_question | 57 | 57 | 52 | 5 | 0.9123 | 52 | 46 | 0.8846 | 46 | 0.8070 | 0 | 5 | 20 | 0.3509 | 0.7895 | 2.2500 | 97.3313 | 76.6857 |

## By Prompt Type

| prompt_type | N | target_eval_N | target_correct | target_wrong_or_unresolved | target_selection_accuracy | grasp_attempted | physical_grasp_success | physical_grasp_success_rate | correct_object_grasp_success | task_success_rate | wrong_object_grasp | wrong_target_prevented | asked_N | asked_rate | mean_questions_all | mean_questions_asked | mean_time_s | median_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | 76 | 76 | 47 | 29 | 0.6184 | 47 | 44 | 0.9362 | 44 | 0.5789 | 0 | 29 | 35 | 0.4605 | 1.0000 | 2.1714 | 85.0952 | 56.8869 |
| clear | 76 | 76 | 75 | 1 | 0.9868 | 75 | 68 | 0.9067 | 68 | 0.8947 | 0 | 1 | 1 | 0.0132 | 0.0132 | 1.0000 | 53.5303 | 48.7368 |
| partial | 76 | 76 | 52 | 24 | 0.6842 | 52 | 43 | 0.8269 | 43 | 0.5658 | 0 | 24 | 3 | 0.0395 | 0.0395 | 1.0000 | 49.7251 | 51.0438 |

## Method x Prompt Type Counts

| method | prompt_type | N |
| --- | --- | --- |
| proposed_efe | ambiguous | 19 |
| proposed_efe | clear | 19 |
| proposed_efe | partial | 19 |
| random_candidate | ambiguous | 19 |
| random_candidate | clear | 19 |
| random_candidate | partial | 19 |
| top_score | ambiguous | 19 |
| top_score | clear | 19 |
| top_score | partial | 19 |
| vlm_best_question | ambiguous | 19 |
| vlm_best_question | clear | 19 |
| vlm_best_question | partial | 19 |

## Files

- `all_corrected_trials_main02.csv`
- `cleaned_evaluated_trials_main02.csv`
- `balanced_scene_cell_trials_main02.csv`
- `balanced_method_prompt_trials_main02.csv`
- `missing_scene_prompt_method_cells_main02.csv`
- `scene_cell_excluded_main02.csv`
- `method_prompt_excluded_main02.csv`

## Exclusion Counts

- Scene-cell duplicate exclusions: 11
- Method-prompt balance exclusions: 8
