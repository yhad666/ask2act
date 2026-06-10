# Significance Report

Generated at epoch `1780597930.05`.

## Why The Analysis Changed

Prompt texts are not fully identical across methods; exact prompt-level paired tests are therefore not valid as the main analysis. Accuracy is analyzed with scene-clustered GEE, while question count and latency are analyzed after scene-level aggregation.

## Column Mapping

| canonical | source |
| --- | --- |
| scene_id | scene_id |
| method | method |
| prompt_type | prompt_type |
| correct | outcome |
| num_questions | question_count |
| latency_sec | latency_s |
| scene_type | scene_type |
| candidate_set_size | candidate_set_size |
| prompt_text | prompt_text |
| target_category |  |
| attribute_type |  |
| trial_id | trial_id |
| prompt_id | prompt_text inferred from prompt text |

## Dataset Audit

| section | group | n |
| --- | --- | --- |
| overall | total_records | 1512 |
| overall | scenes | 36 |
| overall | methods | 7 |
| method | first_question | 216 |
| method | proposed_efe | 216 |
| method | random_candidate | 216 |
| method | random_question | 216 |
| method | top_score | 216 |
| method | vlm_best_question | 216 |
| method | vlm_direct | 216 |
| prompt_type | ambiguous | 506 |
| prompt_type | clear | 503 |
| prompt_type | partial | 503 |
| method_x_prompt_type | first_question|ambiguous | 72 |
| method_x_prompt_type | first_question|clear | 72 |
| method_x_prompt_type | first_question|partial | 72 |
| method_x_prompt_type | proposed_efe|ambiguous | 72 |
| method_x_prompt_type | proposed_efe|clear | 71 |
| method_x_prompt_type | proposed_efe|partial | 73 |
| method_x_prompt_type | random_candidate|ambiguous | 73 |
| method_x_prompt_type | random_candidate|clear | 71 |
| method_x_prompt_type | random_candidate|partial | 72 |
| method_x_prompt_type | random_question|ambiguous | 72 |
| method_x_prompt_type | random_question|clear | 73 |
| method_x_prompt_type | random_question|partial | 71 |
| method_x_prompt_type | top_score|ambiguous | 73 |
| method_x_prompt_type | top_score|clear | 72 |
| method_x_prompt_type | top_score|partial | 71 |
| method_x_prompt_type | vlm_best_question|ambiguous | 72 |
| method_x_prompt_type | vlm_best_question|clear | 72 |
| method_x_prompt_type | vlm_best_question|partial | 72 |
| method_x_prompt_type | vlm_direct|ambiguous | 72 |
| method_x_prompt_type | vlm_direct|clear | 72 |
| method_x_prompt_type | vlm_direct|partial | 72 |
| method_x_scene_id | first_question|scene_001 | 9 |
| method_x_scene_id | first_question|scene_002 | 8 |
| method_x_scene_id | first_question|scene_003 | 8 |
| method_x_scene_id | first_question|scene_004 | 9 |
| method_x_scene_id | first_question|scene_005 | 8 |
| method_x_scene_id | first_question|scene_006 | 8 |
| method_x_scene_id | first_question|scene_007 | 6 |
| method_x_scene_id | first_question|scene_008 | 5 |
| method_x_scene_id | first_question|scene_009 | 6 |
| method_x_scene_id | first_question|scene_010 | 6 |
| method_x_scene_id | first_question|scene_011 | 5 |
| method_x_scene_id | first_question|scene_012 | 6 |
| method_x_scene_id | first_question|scene_013 | 6 |
| method_x_scene_id | first_question|scene_014 | 5 |
| method_x_scene_id | first_question|scene_015 | 7 |
| method_x_scene_id | first_question|scene_016 | 6 |
| method_x_scene_id | first_question|scene_017 | 6 |
| method_x_scene_id | first_question|scene_018 | 6 |
| method_x_scene_id | first_question|scene_019 | 6 |
| method_x_scene_id | first_question|scene_020 | 6 |
| method_x_scene_id | first_question|scene_021 | 6 |
| method_x_scene_id | first_question|scene_022 | 4 |
| method_x_scene_id | first_question|scene_023 | 2 |
| method_x_scene_id | first_question|scene_024 | 6 |
| method_x_scene_id | first_question|scene_025 | 4 |
| method_x_scene_id | first_question|scene_026 | 6 |

## Validation / Balance Notes

- Method x scene counts are not exactly 6 for every cell (min=1, max=10); scene-clustered GEE and scene-level aggregation are used.

## Accuracy Results: Scene-Clustered GEE

Formula: `correct ~ C(method, Treatment(reference="proposed_efe")) + C(prompt_type, Treatment(reference="clear")) + C(scene_type) + candidate_set_size`

| family | method_A | method_B | log_odds_diff_A_minus_B | se | odds_ratio | or_ci95_low | or_ci95_high | p_raw | descriptive_acc_A | descriptive_acc_B | descriptive_acc_diff_A_minus_B | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | proposed_efe | top_score | 2.742 | 0.2703 | 15.52 | 9.139 | 26.37 | 3.485e-24 | 0.9213 | 0.537 | 0.3843 | 1.394e-23 |
| A | proposed_efe | random_candidate | 2.765 | 0.2728 | 15.88 | 9.3 | 27.1 | 3.914e-24 | 0.9213 | 0.5324 | 0.3889 | 1.394e-23 |
| A | proposed_efe | vlm_direct | 2.761 | 0.3122 | 15.81 | 8.574 | 29.15 | 9.33e-19 | 0.9213 | 0.537 | 0.3843 | 9.33e-19 |
| A | vlm_best_question | vlm_direct | 2.754 | 0.3014 | 15.71 | 8.701 | 28.36 | 6.376e-20 | 0.9213 | 0.537 | 0.3843 | 1.275e-19 |
| B | proposed_efe | first_question | 0.1539 | 0.1893 | 1.166 | 0.8048 | 1.69 | 0.4162 | 0.9213 | 0.912 | 0.009259 | 0.8324 |
| B | proposed_efe | random_question | 0.2021 | 0.1791 | 1.224 | 0.8617 | 1.739 | 0.2591 | 0.9213 | 0.9074 | 0.01389 | 0.7772 |
| B | proposed_efe | vlm_best_question | 0.006352 | 0.1776 | 1.006 | 0.7105 | 1.426 | 0.9715 | 0.9213 | 0.9213 | 0 | 0.9715 |

Accuracy interpretation:

- proposed_efe had higher adjusted accuracy than top_score (OR=15.52, Holm p=1.394e-23).
- proposed_efe had higher adjusted accuracy than random_candidate (OR=15.88, Holm p=1.394e-23).
- proposed_efe had higher adjusted accuracy than vlm_direct (OR=15.81, Holm p=9.33e-19).
- vlm_best_question had higher adjusted accuracy than vlm_direct (OR=15.71, Holm p=1.275e-19).
- EFE was not significantly different in accuracy from the interactive baselines after Holm correction; describe accuracy as comparable.


## Prompt Sampling Robustness

Prompt-sampling robustness interpretation:

- Prompt wording was sampled from scene- and prompt-type-specific pools; pool sizes and repeat rates are therefore explicitly audited.
- Prompt-feature GEE formula: `correct ~ C(method, Treatment(reference="proposed_efe")) + C(prompt_type, Treatment(reference="clear")) + C(scene_type) + candidate_set_size + prompt_length + has_my + has_color + has_spatial + has_relation + has_size + C(prompt_object_category)`.
- 73/108 scene_id x prompt_type pools were flagged as low-diversity or high-repeat (unique prompts < 3 or duplicate rate > 0.50).
- Most constrained prompt pools: scene_002:ambiguous(unique=1, dup=0.95), scene_005:ambiguous(unique=1, dup=0.95), scene_004:ambiguous(unique=1, dup=0.95), scene_006:ambiguous(unique=1, dup=0.95), scene_003:ambiguous(unique=1, dup=0.95), scene_009:ambiguous(unique=1, dup=0.93), scene_014:ambiguous(unique=1, dup=0.93), scene_015:ambiguous(unique=1, dup=0.93).
- With prompt features as covariates, 4/4 planned non-interactive accuracy contrasts remained Holm-significant.
- With prompt features as covariates, EFE accuracy remained statistically comparable to the other interactive methods.
- Sensitivity after removing flagged pools retained N=514 records across 35 scene-prompt pools.
- In the low-diversity-pool sensitivity subset, 4/4 non-interactive accuracy contrasts remained Holm-significant.
- In the same sensitivity subset, ambiguous question-count tests were not estimable because no complete paired ambiguous scenes remained.
- These analyses support treating prompt-pool imbalance as a checked robustness issue rather than a blocker for the main conclusions.


Prompt-feature adjusted GEE formula:

`correct ~ C(method, Treatment(reference="proposed_efe")) + C(prompt_type, Treatment(reference="clear")) + C(scene_type) + candidate_set_size + prompt_length + has_my + has_color + has_spatial + has_relation + has_size + C(prompt_object_category)`

Prompt-feature adjusted accuracy contrasts:

| family | method_A | method_B | log_odds_diff_A_minus_B | se | odds_ratio | or_ci95_low | or_ci95_high | p_raw | descriptive_acc_A | descriptive_acc_B | descriptive_acc_diff_A_minus_B | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | proposed_efe | top_score | 2.748 | 0.2713 | 15.62 | 9.175 | 26.58 | 4.079e-24 | 0.9213 | 0.537 | 0.3843 | 1.632e-23 |
| A | proposed_efe | random_candidate | 2.831 | 0.3092 | 16.96 | 9.249 | 31.08 | 5.492e-20 | 0.9213 | 0.5324 | 0.3889 | 1.098e-19 |
| A | proposed_efe | vlm_direct | 2.927 | 0.345 | 18.67 | 9.493 | 36.71 | 2.206e-17 | 0.9213 | 0.537 | 0.3843 | 2.206e-17 |
| A | vlm_best_question | vlm_direct | 2.888 | 0.3051 | 17.95 | 9.871 | 32.65 | 2.986e-21 | 0.9213 | 0.537 | 0.3843 | 8.959e-21 |
| B | proposed_efe | first_question | 0.1111 | 0.178 | 1.117 | 0.7884 | 1.584 | 0.5325 | 0.9213 | 0.912 | 0.009259 | 1 |
| B | proposed_efe | random_question | 0.1938 | 0.1805 | 1.214 | 0.8522 | 1.729 | 0.2829 | 0.9213 | 0.9074 | 0.01389 | 0.8488 |
| B | proposed_efe | vlm_best_question | 0.03918 | 0.1889 | 1.04 | 0.7182 | 1.506 | 0.8357 | 0.9213 | 0.9213 | 0 | 1 |

Prompt feature balance by method x prompt_type:

| method | prompt_type | N | unique_prompt_n | duplicate_rate | mean_prompt_length | has_my_rate | has_color_rate | has_spatial_rate | has_relation_rate | has_size_rate | has_modifier_rate | mean_candidate_set_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_question | ambiguous | 72 | 6 | 0.9167 | 16.07 | 1 | 0 | 0 | 0 | 0 | 1 | 4.472 |
| first_question | clear | 72 | 45 | 0.375 | 22.61 | 0 | 0.5972 | 0.3611 | 0 | 0.02778 | 0.9028 | 1.056 |
| first_question | partial | 72 | 35 | 0.5139 | 22.35 | 0 | 0.2778 | 0.7083 | 0 | 0 | 0.9722 | 2.167 |
| proposed_efe | ambiguous | 72 | 7 | 0.9028 | 16.21 | 1 | 0 | 0 | 0 | 0 | 1 | 4.472 |
| proposed_efe | clear | 71 | 35 | 0.507 | 23.03 | 0 | 0.8028 | 0.169 | 0 | 0 | 0.9296 | 1 |
| proposed_efe | partial | 73 | 33 | 0.5479 | 22.74 | 0 | 0.2466 | 0.726 | 0 | 0.0274 | 0.9863 | 2.014 |
| random_candidate | ambiguous | 73 | 6 | 0.9178 | 16.01 | 0.9863 | 0 | 0 | 0 | 0 | 0.9863 | 4.548 |
| random_candidate | clear | 71 | 44 | 0.3803 | 23.07 | 0.01408 | 0.662 | 0.2676 | 0 | 0.01408 | 0.8873 | 1.211 |
| random_candidate | partial | 72 | 40 | 0.4444 | 22.82 | 0 | 0.2639 | 0.6806 | 0.01389 | 0 | 0.9444 | 1.583 |
| random_question | ambiguous | 72 | 7 | 0.9028 | 16.11 | 1 | 0 | 0 | 0 | 0 | 1 | 4.625 |
| random_question | clear | 73 | 41 | 0.4384 | 22.71 | 0 | 0.7123 | 0.2192 | 0 | 0.0137 | 0.9041 | 1.027 |
| random_question | partial | 71 | 32 | 0.5493 | 22.75 | 0 | 0.2254 | 0.7465 | 0 | 0 | 0.9718 | 1.93 |
| top_score | ambiguous | 73 | 6 | 0.9178 | 16.07 | 0.9863 | 0 | 0 | 0 | 0 | 0.9863 | 4.452 |
| top_score | clear | 72 | 28 | 0.6111 | 22.82 | 0 | 0.8194 | 0.125 | 0 | 0 | 0.9306 | 1.264 |
| top_score | partial | 71 | 35 | 0.507 | 22.72 | 0 | 0.338 | 0.6479 | 0.01408 | 0 | 0.9859 | 1.507 |
| vlm_best_question | ambiguous | 72 | 6 | 0.9167 | 16.04 | 1 | 0 | 0 | 0 | 0 | 1 | 4.569 |
| vlm_best_question | clear | 72 | 38 | 0.4722 | 23.01 | 0 | 0.7222 | 0.1806 | 0 | 0.04167 | 0.9028 | 1.125 |
| vlm_best_question | partial | 72 | 33 | 0.5417 | 22.46 | 0 | 0.25 | 0.7778 | 0 | 0 | 1 | 1.875 |
| vlm_direct | ambiguous | 72 | 9 | 0.875 | 16.4 | 1 | 0 | 0 | 0 | 0 | 1 | 4.597 |
| vlm_direct | clear | 72 | 42 | 0.4167 | 22.78 | 0 | 0.6528 | 0.2917 | 0 | 0.05556 | 0.9444 | 1.347 |
| vlm_direct | partial | 72 | 34 | 0.5278 | 22.85 | 0 | 0.2083 | 0.7639 | 0 | 0 | 0.9722 | 1.444 |

Prompt pool diversity by scene x prompt_type:

| scene_id | prompt_type | N | unique_prompt_n | unique_method_n | mean_prompt_length | has_my_rate | has_color_rate | has_spatial_rate | has_relation_rate | has_size_rate | mean_candidate_set_size | duplicate_rate | low_diversity_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene_001 | ambiguous | 17 | 2 | 7 | 15.12 | 0.8824 | 0 | 0 | 0 | 0 | 6 | 0.8824 | True |
| scene_001 | clear | 25 | 15 | 7 | 21.28 | 0 | 0.64 | 0.36 | 0 | 0 | 1.2 | 0.4 | False |
| scene_001 | partial | 20 | 13 | 7 | 22.65 | 0 | 0.5 | 0.4 | 0 | 0 | 1.9 | 0.35 | False |
| scene_002 | ambiguous | 21 | 1 | 7 | 14.86 | 1 | 0 | 0 | 0 | 0 | 6 | 0.9524 | True |
| scene_002 | clear | 19 | 13 | 7 | 21.53 | 0 | 0.6316 | 0.2632 | 0 | 0 | 1 | 0.3158 | False |
| scene_002 | partial | 19 | 13 | 7 | 23.05 | 0 | 0.2632 | 0.6316 | 0.1053 | 0 | 3 | 0.3158 | False |
| scene_003 | ambiguous | 19 | 1 | 7 | 15 | 1 | 0 | 0 | 0 | 0 | 6 | 0.9474 | True |
| scene_003 | clear | 17 | 10 | 6 | 20.06 | 0 | 0.5882 | 0.1176 | 0 | 0.1765 | 1 | 0.4118 | False |
| scene_003 | partial | 17 | 13 | 6 | 21 | 0 | 0.2353 | 0.5294 | 0 | 0.05882 | 2.471 | 0.2353 | False |
| scene_004 | ambiguous | 20 | 1 | 7 | 15 | 1 | 0 | 0 | 0 | 0 | 6 | 0.95 | True |
| scene_004 | clear | 17 | 10 | 6 | 22.88 | 0 | 0.8235 | 0.5294 | 0 | 0 | 1.588 | 0.4118 | False |
| scene_004 | partial | 21 | 14 | 6 | 22 | 0 | 0.2381 | 0.8571 | 0 | 0 | 1.238 | 0.3333 | False |
| scene_005 | ambiguous | 21 | 1 | 7 | 15 | 1 | 0 | 0 | 0 | 0 | 6 | 0.9524 | True |
| scene_005 | clear | 16 | 13 | 6 | 22.25 | 0 | 0.3125 | 0.625 | 0 | 0 | 1.062 | 0.1875 | False |
| scene_005 | partial | 22 | 15 | 7 | 21.55 | 0 | 0.3636 | 0.5455 | 0 | 0 | 1.955 | 0.3182 | False |
| scene_006 | ambiguous | 20 | 1 | 7 | 15 | 1 | 0 | 0 | 0 | 0 | 7 | 0.95 | True |
| scene_006 | clear | 17 | 9 | 7 | 22.29 | 0 | 0.5882 | 0.2353 | 0 | 0 | 1.176 | 0.4706 | False |
| scene_006 | partial | 19 | 7 | 7 | 22.26 | 0 | 0.5263 | 0.4211 | 0 | 0 | 1.632 | 0.6316 | True |
| scene_007 | ambiguous | 13 | 1 | 7 | 16 | 1 | 0 | 0 | 0 | 0 | 7 | 0.9231 | True |
| scene_007 | clear | 13 | 6 | 7 | 22.77 | 0 | 0.9231 | 0 | 0 | 0 | 1.538 | 0.5385 | True |
| scene_007 | partial | 16 | 6 | 7 | 22.94 | 0 | 0.125 | 0.875 | 0 | 0 | 1.375 | 0.625 | True |
| scene_008 | ambiguous | 12 | 2 | 7 | 15.5 | 1 | 0 | 0 | 0 | 0 | 7 | 0.8333 | True |
| scene_008 | clear | 15 | 6 | 7 | 20.53 | 0 | 0.6667 | 0.06667 | 0 | 0 | 1 | 0.6 | True |
| scene_008 | partial | 13 | 2 | 7 | 20.85 | 0 | 0.4615 | 0.5385 | 0 | 0 | 2 | 0.8462 | True |
| scene_009 | ambiguous | 15 | 1 | 7 | 15 | 1 | 0 | 0 | 0 | 0 | 7 | 0.9333 | True |
| scene_009 | clear | 13 | 5 | 7 | 23.08 | 0 | 0.2308 | 0.1538 | 0 | 0 | 1 | 0.6154 | True |
| scene_009 | partial | 12 | 4 | 7 | 22.58 | 0 | 0.9167 | 0.08333 | 0 | 0 | 4.5 | 0.6667 | True |
| scene_010 | ambiguous | 11 | 2 | 6 | 15.73 | 1 | 0 | 0 | 0 | 0 | 8 | 0.8182 | True |
| scene_010 | clear | 14 | 6 | 7 | 26.14 | 0 | 0.8571 | 1 | 0 | 0 | 1 | 0.5714 | True |
| scene_010 | partial | 14 | 2 | 7 | 21.5 | 0 | 1 | 0 | 0 | 0 | 5 | 0.8571 | True |
| scene_011 | ambiguous | 13 | 1 | 7 | 18 | 1 | 0 | 0 | 0 | 0 | 4 | 0.9231 | True |
| scene_011 | clear | 13 | 4 | 7 | 23.54 | 0 | 0.6154 | 0.3077 | 0 | 0.07692 | 1 | 0.6923 | True |
| scene_011 | partial | 11 | 7 | 7 | 24.09 | 0 | 0 | 0.9091 | 0 | 0.09091 | 1 | 0.3636 | False |
| scene_012 | ambiguous | 13 | 1 | 7 | 18 | 1 | 0 | 0 | 0 | 0 | 4 | 0.9231 | True |
| scene_012 | clear | 12 | 4 | 6 | 24.42 | 0 | 0.9167 | 0.08333 | 0 | 0 | 1 | 0.6667 | True |
| scene_012 | partial | 11 | 3 | 6 | 24.45 | 0 | 0 | 1 | 0 | 0 | 1 | 0.7273 | True |
| scene_013 | ambiguous | 13 | 1 | 7 | 18 | 1 | 0 | 0 | 0 | 0 | 4 | 0.9231 | True |
| scene_013 | clear | 14 | 7 | 6 | 24.57 | 0 | 0.7143 | 0.2143 | 0 | 0 | 1 | 0.5 | False |
| scene_013 | partial | 14 | 5 | 6 | 24.21 | 0 | 0.1429 | 0.8571 | 0 | 0 | 1 | 0.6429 | True |
| scene_014 | ambiguous | 14 | 1 | 7 | 18 | 1 | 0 | 0 | 0 | 0 | 4 | 0.9286 | True |

Prompt-pool sensitivity summary:

| analysis | N | scenes | scene_prompt_type_pools | flagged_pools_removed |
| --- | --- | --- | --- | --- |
| main | 1512 | 36 | 108 | 0 |
| remove_low_diversity_pools | 514 | 27 | 35 | 73 |

Sensitivity accuracy contrasts:

| sensitivity | N | family | method_A | method_B | log_odds_diff_A_minus_B | se | odds_ratio | or_ci95_low | or_ci95_high | p_raw | descriptive_acc_A | descriptive_acc_B | descriptive_acc_diff_A_minus_B | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| remove_low_diversity_pools | 514 | A | proposed_efe | top_score | 1.814 | 0.5748 | 6.137 | 1.989 | 18.93 | 0.001596 | 0.9583 | 0.7857 | 0.1726 | 0.004788 |
| remove_low_diversity_pools | 514 | A | proposed_efe | random_candidate | 2.179 | 0.5639 | 8.839 | 2.927 | 26.69 | 0.0001114 | 0.9583 | 0.7778 | 0.1806 | 0.0004457 |
| remove_low_diversity_pools | 514 | A | proposed_efe | vlm_direct | 2.463 | 0.8111 | 11.74 | 2.395 | 57.55 | 0.002391 | 0.9583 | 0.7532 | 0.2051 | 0.004788 |
| remove_low_diversity_pools | 514 | A | vlm_best_question | vlm_direct | 3.529 | 1.181 | 34.09 | 3.369 | 345 | 0.002801 | 0.9867 | 0.7532 | 0.2334 | 0.004788 |
| remove_low_diversity_pools | 514 | B | proposed_efe | first_question | -0.1884 | 0.9673 | 0.8283 | 0.1244 | 5.515 | 0.8456 | 0.9583 | 0.96 | -0.001667 | 0.8456 |
| remove_low_diversity_pools | 514 | B | proposed_efe | random_question | -0.478 | 0.4625 | 0.62 | 0.2504 | 1.535 | 0.3014 | 0.9583 | 0.9726 | -0.01427 | 0.6965 |
| remove_low_diversity_pools | 514 | B | proposed_efe | vlm_best_question | -1.066 | 0.8923 | 0.3444 | 0.05991 | 1.979 | 0.2322 | 0.9583 | 0.9867 | -0.02833 | 0.6965 |

Sensitivity question-count tests:

| context | method_A | method_B | n_scenes | mean_Q_EFE | mean_Q_baseline | median_Q_EFE | median_Q_baseline | scene_level_paired_mean_diff_EFE_minus_B | scene_level_paired_median_diff_EFE_minus_B | percent_reduction_mean_Q | ci95_low | ci95_high | wilcoxon_statistic | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | proposed_efe | first_question | 0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| ambiguous | proposed_efe | random_question | 0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| ambiguous | proposed_efe | vlm_best_question | 0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| overall | proposed_efe | first_question | 25 | 0.09333 | 0.1233 | 0 | 0 | -0.03 | 0 | 24.32 | -0.14 | 0.06333 | 6 | 0.6803 | 1 |
| overall | proposed_efe | random_question | 25 | 0.09333 | 0.1073 | 0 | 0 | -0.014 | 0 | 13.04 | -0.1267 | 0.07667 | 13.5 | 0.9325 | 1 |
| overall | proposed_efe | vlm_best_question | 25 | 0.09333 | 0.12 | 0 | 0 | -0.02667 | 0 | 22.22 | -0.14 | 0.09 | 14.5 | 0.6148 | 1 |
| partial | proposed_efe | first_question | 7 | 0.4762 | 0.5238 | 0 | 0 | -0.04762 | 0 | 9.091 | -0.4048 | 0.2143 | 3 | 1 | 1 |
| partial | proposed_efe | random_question | 7 | 0.4762 | 0.3143 | 0 | 0 | 0.1619 | 0 | -51.52 | -0.1905 | 0.4857 | 3 | 0.625 | 1 |
| partial | proposed_efe | vlm_best_question | 7 | 0.4762 | 0.2143 | 0 | 0 | 0.2619 | 0 | -122.2 | 0.07143 | 0.5 | 0 | 0.25 | 0.75 |

## Question Count Results: Scene-Level Aggregation

Friedman tests:

| context | n_scenes | friedman_statistic | p_raw |
| --- | --- | --- | --- |
| ambiguous | 33 | 24.47 | 1.993e-05 |
| overall | 36 | 11.1 | 0.01119 |
| partial | 32 | 9.857 | 0.01982 |

Pairwise Wilcoxon tests:

| context | method_A | method_B | n_scenes | mean_Q_EFE | mean_Q_baseline | median_Q_EFE | median_Q_baseline | scene_level_paired_mean_diff_EFE_minus_B | scene_level_paired_median_diff_EFE_minus_B | percent_reduction_mean_Q | ci95_low | ci95_high | wilcoxon_statistic | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | proposed_efe | first_question | 33 | 1.975 | 2.611 | 2 | 2 | -0.6364 | -0.5 | 24.37 | -0.9949 | -0.2828 | 26.5 | 0.0006481 | 0.0009042 |
| ambiguous | proposed_efe | random_question | 33 | 1.975 | 2.576 | 2 | 2.5 | -0.601 | -0.5 | 23.33 | -0.8889 | -0.3182 | 25.5 | 0.0003322 | 0.0009042 |
| ambiguous | proposed_efe | vlm_best_question | 33 | 1.975 | 2.571 | 2 | 2.5 | -0.596 | -0.5 | 23.18 | -0.8939 | -0.3232 | 25 | 0.0003014 | 0.0009042 |
| overall | proposed_efe | first_question | 36 | 0.9275 | 1.11 | 0.75 | 0.5 | -0.1829 | 0 | 16.47 | -0.3156 | -0.05787 | 117 | 0.01737 | 0.05211 |
| overall | proposed_efe | random_question | 36 | 0.9275 | 1.06 | 0.75 | 0.5 | -0.1323 | 0 | 12.48 | -0.277 | 0.01142 | 117.5 | 0.0514 | 0.1028 |
| overall | proposed_efe | vlm_best_question | 36 | 0.9275 | 1.056 | 0.75 | 0.5 | -0.1289 | 0 | 12.2 | -0.2616 | 0.001562 | 121.5 | 0.06295 | 0.1028 |
| partial | proposed_efe | first_question | 32 | 0.6979 | 0.5729 | 0.5 | 0.5 | 0.125 | 0 | -21.82 | 0 | 0.2552 | 19 | 0.0591 | 0.1182 |
| partial | proposed_efe | random_question | 32 | 0.6979 | 0.6521 | 0.5 | 0.5 | 0.04583 | 0 | -7.029 | -0.09479 | 0.1854 | 39 | 0.6408 | 0.6408 |
| partial | proposed_efe | vlm_best_question | 32 | 0.6979 | 0.5365 | 0.5 | 0.5 | 0.1615 | 0 | -30.1 | 0.0625 | 0.276 | 0 | 0.01061 | 0.03184 |

Question-count interpretation:

- ambiguous: Friedman p=1.993e-05 with n_scenes=33.
  EFE reduced questions vs first_question: mean diff=-0.64, reduction=24.37%, Holm p=0.0009042.
  EFE reduced questions vs random_question: mean diff=-0.60, reduction=23.33%, Holm p=0.0009042.
  EFE reduced questions vs vlm_best_question: mean diff=-0.60, reduction=23.18%, Holm p=0.0009042.
- overall: Friedman p=0.01119 with n_scenes=36.
  No EFE pairwise question-count comparison was significant after Holm correction.
- partial: Friedman p=0.01982 with n_scenes=32.
  EFE increased questions vs vlm_best_question: mean diff=0.16, reduction=-30.10%, Holm p=0.03184.
- Main efficiency claim should emphasize ambiguous prompts if those comparisons are significant.


## Latency Results: Scene-Level Aggregation

| context | method_A | method_B | n_scenes | mean_latency_EFE | median_latency_EFE | iqr_latency_EFE | mean_latency_baseline | median_latency_baseline | iqr_latency_baseline | paired_mean_diff_EFE_minus_B | mean_diff_ci95_low | mean_diff_ci95_high | paired_median_diff_EFE_minus_B | median_diff_ci95_low | median_diff_ci95_high | wilcoxon_statistic | p_raw | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | proposed_efe | first_question | 33 | 141.7 | 119 | 37.78 | 177.1 | 141.2 | 46.71 | -35.36 | -85.75 | 10.13 | -21.49 | -53.65 | 5.273 | 216 | 0.2565 | 0.5131 |
| ambiguous | proposed_efe | random_question | 33 | 141.7 | 119 | 37.78 | 188.6 | 131 | 65.7 | -46.86 | -91.88 | -8.835 | -25.47 | -50.47 | -11.32 | 155 | 0.02412 | 0.07235 |
| ambiguous | proposed_efe | vlm_best_question | 33 | 141.7 | 119 | 37.78 | 162.3 | 130.4 | 42.75 | -20.6 | -63.27 | 23.18 | -9.419 | -40.95 | 13.96 | 232 | 0.396 | 0.5131 |
| overall | proposed_efe | first_question | 36 | 76.77 | 62.69 | 23.27 | 86.77 | 37.87 | 28.64 | -9.999 | -31.26 | 8.554 | 7.753 | -2.028 | 21.11 | 301 | 0.6247 | 1 |
| overall | proposed_efe | random_question | 36 | 76.77 | 62.69 | 23.27 | 89.2 | 58.29 | 29.3 | -12.43 | -28.76 | 3.739 | -2.824 | -17.92 | 1.779 | 239 | 0.1433 | 0.4299 |
| overall | proposed_efe | vlm_best_question | 36 | 76.77 | 62.69 | 23.27 | 77.07 | 39.15 | 25.29 | -0.3063 | -18.44 | 16.99 | 8.191 | -0.04995 | 14.18 | 315 | 0.7859 | 1 |
| partial | proposed_efe | first_question | 32 | 78.53 | 52.16 | 22.01 | 67.76 | 33.58 | 25.31 | 10.77 | -25.74 | 38.3 | 13.1 | 2.944 | 20.05 | 111 | 0.003379 | 0.01014 |
| partial | proposed_efe | random_question | 32 | 78.53 | 52.16 | 22.01 | 78.4 | 61.87 | 22.56 | 0.1336 | -21.3 | 22.75 | -1.023 | -11.99 | 2.376 | 247 | 0.7609 | 0.7609 |
| partial | proposed_efe | vlm_best_question | 32 | 78.53 | 52.16 | 22.01 | 65.39 | 36.97 | 28.15 | 13.14 | -8.763 | 31.18 | 7.258 | -0.296 | 14.97 | 138 | 0.01746 | 0.03492 |

Latency interpretation:

- Latency is long-tailed; scene-level Wilcoxon tests and median/IQR summaries are used.
- ambiguous: no Holm-corrected significant latency difference; report descriptive latency cautiously.
- overall: no Holm-corrected significant latency difference; report descriptive latency cautiously.
- partial: EFE had higher scene-level mean latency than first_question (Holm p=0.01014).
- partial: EFE had higher scene-level mean latency than vlm_best_question (Holm p=0.03492).


## Prompt Balance Checks

Prompt balance interpretation:

- Total records: 1512.
- Scenes: 36. Methods: 7.
- Method x prompt_type counts:
  - top_score: ambiguous=73, clear=72, partial=71
  - random_candidate: ambiguous=73, clear=71, partial=72
  - vlm_direct: ambiguous=72, clear=72, partial=72
  - first_question: ambiguous=72, clear=72, partial=72
  - random_question: ambiguous=72, clear=73, partial=71
  - vlm_best_question: ambiguous=72, clear=72, partial=72
  - proposed_efe: ambiguous=72, clear=71, partial=73
- Method x scene_id counts range from 1 to 10.
- Balance notes:
  - Method x scene counts are not exactly 6 for every cell (min=1, max=10); scene-clustered GEE and scene-level aggregation are used.
- GEE includes scene clustering; imbalanced available covariates are included as fixed covariates where available.


See `prompt_balance_checks.csv` and `balance_checks.csv` for full count tables.

## Prompt-Type Difficulty

Descriptives:

| prompt_type | N | accuracy | failure_rate | asked_rate | mean_Q | median_Q | mean_latency | median_latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous | 506 | 0.6047 | 0.3953 | 0.5237 | 1.441 | 1 | 109.7 | 75.49 |
| clear | 503 | 0.9602 | 0.03976 | 0.01789 | 0.02187 | 0 | 8.089 | 3.772 |
| partial | 503 | 0.6938 | 0.3062 | 0.2664 | 0.3459 | 0 | 46.6 | 18.36 |

GEE contrasts:

| contrast | log_odds_diff | se | odds_ratio | or_ci95_low | or_ci95_high | p_raw | accuracy_A | accuracy_B | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_vs_clear | -3.336 | 0.3298 | 0.03558 | 0.01864 | 0.06791 | 4.684e-24 | 0.6047 | 0.9602 | 1.405e-23 |
| partial_vs_clear | -2.775 | 0.3673 | 0.06237 | 0.03036 | 0.1281 | 4.207e-14 | 0.6938 | 0.9602 | 8.415e-14 |
| ambiguous_vs_partial | -0.5612 | 0.3103 | 0.5705 | 0.3106 | 1.048 | 0.0705 | 0.6047 | 0.6938 | 0.0705 |

Prompt-type interpretation:

- Clear prompts are high accuracy and rarely trigger questions.
- Ambiguous prompts are harder and require more clarification.
- Partial prompts are intermediate but remain challenging.
- This is supporting analysis, not the main contribution.

GEE prompt-type contrasts:
- ambiguous_vs_clear: OR=0.04, Holm p=1.405e-23.
- partial_vs_clear: OR=0.06, Holm p=8.415e-14.
- ambiguous_vs_partial: OR=0.57, Holm p=0.0705.


## Optional Supplementary Exact-Paired Subset

| status | n_exact_prompt_keys | n_records | test_type | method_A | method_B | n_pairs | mean_A | mean_B | mean_diff_A_minus_B | statistic | p_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| available | 17 | 119 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| ran | nan | nan | accuracy | proposed_efe | top_score | 17 | 0.7647 | 0.4706 | 0.2941 | 3.2 | 0.07364 |
| ran | nan | nan | accuracy | proposed_efe | random_candidate | 17 | 0.7647 | 0.3529 | 0.4118 | 5.143 | 0.02334 |
| ran | nan | nan | accuracy | proposed_efe | vlm_direct | 17 | 0.7647 | 0.4706 | 0.2941 | 3.2 | 0.07364 |
| ran | nan | nan | questions | proposed_efe | first_question | 17 | 0.9412 | 1 | -0.05882 | 6 | 0.6547 |
| ran | nan | nan | questions | proposed_efe | random_question | 17 | 0.9412 | 1.059 | -0.1176 | 2.5 | 0.3173 |
| ran | nan | nan | questions | proposed_efe | vlm_best_question | 17 | 0.9412 | 0.9412 | 0 | 10.5 | 1 |
| ran | nan | nan | latency | proposed_efe | first_question | 17 | 107 | 90.77 | 16.26 | 60 | 0.4586 |
| ran | nan | nan | latency | proposed_efe | random_question | 17 | 107 | 87.7 | 19.33 | 76 | 1 |
| ran | nan | nan | latency | proposed_efe | vlm_best_question | 17 | 107 | 91.25 | 15.79 | 67 | 0.6777 |

## Failure Analysis

Failure analysis uses the cleaned 1512-record offline dataset only. A trial is counted as failed when `correct = 0`, including both `wrong` and `unresolved` outcomes. The failure reason is taken from the audited field `audit_failure_reason`; if no reviewed reason was set, the report preserves `none` rather than inventing a reason.

Overall failure composition:

| Failure reason | Count | Share of failures | Share of all trials |
| --- | ---: | ---: | ---: |
| ambiguity_not_solved | 200 | 53.48% | 13.23% |
| gd_missing_target_candidate | 79 | 21.12% | 5.22% |
| wrong_object_selected | 65 | 17.38% | 4.30% |
| none | 26 | 6.95% | 1.72% |
| vlm_error | 3 | 0.80% | 0.20% |
| gd_label_error | 1 | 0.27% | 0.07% |

Failure reasons by method:

| Method | ambiguity_not_solved | gd_missing_target_candidate | wrong_object_selected | gd_label_error | vlm_error | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_score | 69 | 13 | 12 | 1 | 0 | 5 |
| random_candidate | 61 | 15 | 14 | 0 | 0 | 11 |
| vlm_direct | 66 | 16 | 11 | 0 | 0 | 7 |
| first_question | 1 | 11 | 6 | 0 | 0 | 1 |
| random_question | 1 | 8 | 8 | 0 | 2 | 1 |
| vlm_best_question | 0 | 9 | 7 | 0 | 1 | 0 |
| proposed_efe | 2 | 7 | 7 | 0 | 0 | 1 |

Failure reasons by prompt type:

| Prompt type | ambiguity_not_solved | gd_missing_target_candidate | wrong_object_selected | gd_label_error | vlm_error | none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clear | 7 | 0 | 10 | 1 | 0 | 2 |
| ambiguous | 156 | 28 | 3 | 0 | 1 | 12 |
| partial | 37 | 51 | 52 | 0 | 2 | 12 |

Interpretation:

- The dominant non-interactive failure mode is unresolved ambiguity: `top_score`, `random_candidate`, and `vlm_direct` together account for 176 ambiguity-not-solved failures.
- Interactive methods largely remove ambiguity failures; their remaining errors are mostly GroundingDINO candidate recall failures and wrong final object selection under partial/visually difficult prompts.
- Partial prompts expose a different bottleneck from ambiguous prompts: 51 failures come from missing the target candidate and 52 from wrong object selection, so better question selection alone cannot solve all partial failures.
- The offline data therefore supports the main system diagnosis: interaction fixes referential ambiguity, while detector recall and final candidate grounding remain the main residual failure sources.

## Paper-Ready Conclusion

- Interactive clarification substantially improves target-resolution accuracy compared with non-interactive baselines when supported by scene-clustered GEE contrasts.
- Direct VLM target selection is insufficient under ambiguous referential commands when interactive VLM questioning significantly outperforms `vlm_direct`.
- EFE should be described as maintaining comparable accuracy to other interactive clarification baselines unless GEE contrasts show significant differences.
- EFE reduces the number of clarification questions especially for ambiguous prompts when scene-level Wilcoxon tests support it.
- Because prompt wording was randomized within scene and prompt type, statistical comparisons were performed using scene-clustered or scene-level analyses rather than prompt-level paired tests.
