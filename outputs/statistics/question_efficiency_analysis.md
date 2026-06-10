# Question Efficiency Analysis Under Candidate-Splitting Assumptions

Assumptions used:
- Candidate list size is `candidate_count` stored on each trial.
- Analysis focus is asked trials: `question_count > 0`.
- Ideal lower bound uses perfect 0.5 split: `ceil(log2(candidate_count))`.
- Assumption split bound uses recursive 0.4/0.6 discriminative split, worst-case remaining side.
- Mixed VLM expectation assumes each asked/generated question is discriminative with probability 0.2; otherwise it is a direct one-vs-rest question. Target is assumed uniform over candidates.

## Overall Asked-Trial Comparison
| group | trials | asked_trials | asked_rate_pct | avg_candidate_count_asked | median_candidate_count_asked | actual_total_questions_asked_trials | actual_mean_q_asked_trials | ideal_half_total_q_asked_trials | ideal_half_mean_q_asked_trials | split04_total_q_asked_trials | split04_mean_q_asked_trials | mixed_expected_total_q_asked_trials | mixed_expected_mean_q_asked_trials | actual_minus_split04_total | actual_over_split04 | actual_minus_mixed_expected_total | actual_over_mixed_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_methods | 1512 | 408 | 26.98 | 4.17 | 4.00 | 914 | 2.24 | 842 | 2.06 | 856 | 2.10 | 884.92 | 2.17 | 58 | 1.07 | 29.08 | 1.03 |

## By Method
| method | trials | asked_trials | asked_rate_pct | avg_candidate_count_asked | actual_total_questions_asked_trials | actual_mean_q_asked_trials | ideal_half_mean_q_asked_trials | split04_mean_q_asked_trials | mixed_expected_mean_q_asked_trials | actual_over_split04 | actual_over_mixed_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_question | 216 | 101 | 46.76 | 4.22 | 239 | 2.37 | 2.10 | 2.14 | 2.21 | 1.11 | 1.07 |
| proposed_efe | 216 | 104 | 48.15 | 4.11 | 199 | 1.91 | 2.03 | 2.07 | 2.13 | 0.93 | 0.90 |
| random_candidate | 216 | 0 | 0.00 | - | 0 | - | - | - | - | - | - |
| random_question | 216 | 101 | 46.76 | 4.20 | 241 | 2.39 | 2.06 | 2.10 | 2.18 | 1.14 | 1.10 |
| top_score | 216 | 0 | 0.00 | - | 0 | - | - | - | - | - | - |
| vlm_best_question | 216 | 102 | 47.22 | 4.15 | 235 | 2.30 | 2.07 | 2.09 | 2.16 | 1.10 | 1.06 |
| vlm_direct | 216 | 0 | 0.00 | - | 0 | - | - | - | - | - | - |

## By Method x Prompt Type
| method | prompt_type | trials | asked_trials | asked_rate_pct | avg_candidate_count_asked | actual_mean_q_asked_trials | split04_mean_q_asked_trials | mixed_expected_mean_q_asked_trials | actual_over_split04 | actual_over_mixed_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| first_question | ambiguous | 72 | 67 | 93.06 | 4.73 | 2.88 | 2.43 | 2.52 | 1.18 | 1.15 |
| first_question | clear | 72 | 3 | 4.17 | 2.67 | 1.67 | 1.33 | 1.40 | 1.25 | 1.19 |
| first_question | partial | 72 | 31 | 43.06 | 3.26 | 1.32 | 1.58 | 1.62 | 0.84 | 0.81 |
| proposed_efe | ambiguous | 72 | 66 | 91.67 | 4.77 | 2.26 | 2.45 | 2.53 | 0.92 | 0.89 |
| proposed_efe | clear | 71 | 0 | 0.00 | - | - | - | - | - | - |
| proposed_efe | partial | 73 | 38 | 52.05 | 2.95 | 1.32 | 1.39 | 1.42 | 0.94 | 0.92 |
| random_candidate | ambiguous | 73 | 0 | 0.00 | - | - | - | - | - | - |
| random_candidate | clear | 71 | 0 | 0.00 | - | - | - | - | - | - |
| random_candidate | partial | 72 | 0 | 0.00 | - | - | - | - | - | - |
| random_question | ambiguous | 72 | 66 | 91.67 | 4.88 | 2.94 | 2.48 | 2.59 | 1.18 | 1.14 |
| random_question | clear | 73 | 2 | 2.74 | 2.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| random_question | partial | 71 | 33 | 46.48 | 2.97 | 1.36 | 1.39 | 1.43 | 0.98 | 0.95 |
| top_score | ambiguous | 73 | 0 | 0.00 | - | - | - | - | - | - |
| top_score | clear | 72 | 0 | 0.00 | - | - | - | - | - | - |
| top_score | partial | 71 | 0 | 0.00 | - | - | - | - | - | - |
| vlm_best_question | ambiguous | 72 | 66 | 91.67 | 4.86 | 2.92 | 2.48 | 2.59 | 1.18 | 1.13 |
| vlm_best_question | clear | 72 | 4 | 5.56 | 3.25 | 1.00 | 1.50 | 1.64 | 0.67 | 0.61 |
| vlm_best_question | partial | 72 | 32 | 44.44 | 2.78 | 1.19 | 1.34 | 1.36 | 0.88 | 0.87 |
| vlm_direct | ambiguous | 72 | 0 | 0.00 | - | - | - | - | - | - |
| vlm_direct | clear | 72 | 0 | 0.00 | - | - | - | - | - | - |
| vlm_direct | partial | 72 | 0 | 0.00 | - | - | - | - | - | - |

## By Candidate Count
| candidate_count | trials | asked_trials | asked_rate_pct | actual_mean_q_asked_trials | ideal_half_mean_q_asked_trials | split04_mean_q_asked_trials | mixed_expected_mean_q_asked_trials | actual_over_split04 | actual_over_mixed_expected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 5.00 | 0.00 | 0.00 | - | - | - | - | - | - |
| 1.00 | 810.00 | 16.00 | 1.98 | 1.00 | 0.00 | 0.00 | 0.00 | - | - |
| 2.00 | 189.00 | 100.00 | 52.91 | 1.02 | 1.00 | 1.00 | 1.00 | 1.02 | 1.02 |
| 3.00 | 92.00 | 52.00 | 56.52 | 1.87 | 2.00 | 2.00 | 1.67 | 0.93 | 1.12 |
| 4.00 | 142.00 | 82.00 | 57.75 | 2.24 | 2.00 | 2.00 | 2.20 | 1.12 | 1.02 |
| 5.00 | 28.00 | 17.00 | 60.71 | 2.59 | 3.00 | 3.00 | 2.69 | 0.86 | 0.96 |
| 6.00 | 152.00 | 86.00 | 56.58 | 3.23 | 3.00 | 3.00 | 3.15 | 1.08 | 1.03 |
| 7.00 | 72.00 | 41.00 | 56.94 | 3.63 | 3.00 | 3.00 | 3.56 | 1.21 | 1.02 |
| 8.00 | 21.00 | 14.00 | 66.67 | 3.14 | 3.00 | 4.00 | 3.95 | 0.79 | 0.80 |
| 15.00 | 1.00 | 0.00 | 0.00 | - | - | - | - | - | - |

## Asked-Trial Candidate Count Distribution
| candidate_count | asked_trials |
| --- | --- |
| 1 | 16 |
| 2 | 100 |
| 3 | 52 |
| 4 | 82 |
| 5 | 17 |
| 6 | 86 |
| 7 | 41 |
| 8 | 14 |

## Short Interpretation
- Across all asked trials, the average candidate list size was 4.17 candidates.
- Actual asked questions totaled 914, compared with 856.00 under the recursive 0.4 split bound and 884.92 under the mixed VLM direct/split expectation.
- `actual_over_split04 < 1` means the system often stopped before worst-case full identification, usually because answers/resolution ended early or the candidate set was already narrowed.
- `actual_over_mixed_expected < 1` means actual behavior used fewer user questions than the simple direct-question-heavy VLM expectation.
