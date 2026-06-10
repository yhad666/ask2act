# Question Efficiency: 80% Split-Question Assumptions

Assumptions used:
- Analysis focuses on trials where `question_count > 0`.
- Candidate list size is `candidate_count`.
- Direct fallback question means one-vs-rest, with uniform target over candidates.
- `80% 0.5-split`: each generated question has probability 0.8 of splitting candidates as evenly as possible; otherwise it is one-vs-rest.
- `80% 0.4-split`: each generated question has probability 0.8 of splitting candidates into approximately 0.4/0.6 groups; otherwise it is one-vs-rest.
- Values are expected question counts, not hard lower bounds.

## Overall
| Group | Trials | Asked | Asked % | Avg Candidates | Actual Total Q | Actual Mean Q | Exp Mean Q: 80% 0.5-split | Exp Mean Q: 80% 0.4-split | Actual / 80% 0.5 | Actual / 80% 0.4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_methods | 1512 | 408 | 26.98 | 4.17 | 914 | 2.24 | 1.96 | 1.97 | 1.14 | 1.14 |

## By Method
| Method | Trials | Asked | Asked % | Avg Candidates | Actual Total Q | Actual Mean Q | Exp Mean Q: 80% 0.5-split | Exp Mean Q: 80% 0.4-split | Actual / 80% 0.5 | Actual / 80% 0.4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_score | 216 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| random_candidate | 216 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| vlm_direct | 216 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| first_question | 216 | 101 | 46.76 | 4.22 | 239 | 2.37 | 1.99 | 2.00 | 1.19 | 1.18 |
| random_question | 216 | 101 | 46.76 | 4.20 | 241 | 2.39 | 1.96 | 1.97 | 1.22 | 1.21 |
| vlm_best_question | 216 | 102 | 47.22 | 4.15 | 235 | 2.30 | 1.96 | 1.96 | 1.18 | 1.17 |
| proposed_efe | 216 | 104 | 48.15 | 4.11 | 199 | 1.91 | 1.92 | 1.93 | 1.00 | 0.99 |

## By Method x Prompt Type
| Method | Prompt | Trials | Asked | Asked % | Avg Candidates | Actual Total Q | Actual Mean Q | Exp Mean Q: 80% 0.5-split | Exp Mean Q: 80% 0.4-split | Actual / 80% 0.5 | Actual / 80% 0.4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_score | clear | 72 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| top_score | ambiguous | 73 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| top_score | partial | 71 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| random_candidate | clear | 71 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| random_candidate | ambiguous | 73 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| random_candidate | partial | 72 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| vlm_direct | clear | 72 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| vlm_direct | ambiguous | 72 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| vlm_direct | partial | 72 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| first_question | clear | 72 | 3 | 4.17 | 2.67 | 5 | 1.67 | 1.35 | 1.35 | 1.23 | 1.23 |
| first_question | ambiguous | 72 | 67 | 93.06 | 4.73 | 193 | 2.88 | 2.26 | 2.27 | 1.27 | 1.27 |
| first_question | partial | 72 | 31 | 43.06 | 3.26 | 41 | 1.32 | 1.47 | 1.48 | 0.90 | 0.89 |
| random_question | clear | 73 | 2 | 2.74 | 2.00 | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| random_question | ambiguous | 72 | 66 | 91.67 | 4.88 | 194 | 2.94 | 2.32 | 2.33 | 1.27 | 1.26 |
| random_question | partial | 71 | 33 | 46.48 | 2.97 | 45 | 1.36 | 1.30 | 1.31 | 1.05 | 1.04 |
| vlm_best_question | clear | 72 | 4 | 5.56 | 3.25 | 4 | 1.00 | 1.49 | 1.50 | 0.67 | 0.67 |
| vlm_best_question | ambiguous | 72 | 66 | 91.67 | 4.86 | 193 | 2.92 | 2.32 | 2.33 | 1.26 | 1.25 |
| vlm_best_question | partial | 72 | 32 | 44.44 | 2.78 | 38 | 1.19 | 1.26 | 1.26 | 0.94 | 0.94 |
| proposed_efe | clear | 71 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| proposed_efe | ambiguous | 72 | 66 | 91.67 | 4.77 | 149 | 2.26 | 2.28 | 2.29 | 0.99 | 0.99 |
| proposed_efe | partial | 73 | 38 | 52.05 | 2.95 | 50 | 1.32 | 1.30 | 1.31 | 1.01 | 1.00 |

## By Candidate Count
| Candidates | Trials | Asked | Asked % | Avg Candidates | Actual Total Q | Actual Mean Q | Exp Mean Q: 80% 0.5-split | Exp Mean Q: 80% 0.4-split | Actual / 80% 0.5 | Actual / 80% 0.4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 5 | 0 | 0.00 | - | 0 | - | - | - | - | - |
| 1.0 | 810 | 16 | 1.98 | 1.00 | 16 | 1.00 | 0.00 | 0.00 | - | - |
| 2.0 | 189 | 100 | 52.91 | 2.00 | 102 | 1.02 | 1.00 | 1.00 | 1.02 | 1.02 |
| 3.0 | 92 | 52 | 56.52 | 3.00 | 97 | 1.87 | 1.67 | 1.67 | 1.12 | 1.12 |
| 4.0 | 142 | 82 | 57.75 | 4.00 | 184 | 2.24 | 2.05 | 2.05 | 1.09 | 1.09 |
| 5.0 | 28 | 17 | 60.71 | 5.00 | 44 | 2.59 | 2.45 | 2.45 | 1.06 | 1.06 |
| 6.0 | 152 | 86 | 56.58 | 6.00 | 278 | 3.23 | 2.74 | 2.77 | 1.18 | 1.17 |
| 7.0 | 72 | 41 | 56.94 | 7.00 | 149 | 3.63 | 2.98 | 2.98 | 1.22 | 1.22 |
| 8.0 | 21 | 14 | 66.67 | 8.00 | 44 | 3.14 | 3.16 | 3.25 | 0.99 | 0.97 |
| 15.0 | 1 | 0 | 0.00 | - | 0 | - | - | - | - | - |

## Short Interpretation
- Across all asked trials, average candidate count was 4.17.
- Actual questions totaled 914. The expected total was 798.51 under the 80% 0.5-split assumption and 802.18 under the 80% 0.4-split assumption.
- Overall actual/expected ratios were 1.14 for 80% 0.5-split and 1.14 for 80% 0.4-split.
- For `proposed_efe`, actual mean Q was 1.91, compared with expected 1.92 under 80% 0.5-split and 1.93 under 80% 0.4-split.
- On ambiguous asked trials for `proposed_efe`, actual mean Q was 2.26, versus expected 2.28 and 2.29.

## Saved Files
- `outputs/statistics/question_efficiency_80split_trial_level.csv`
- `outputs/statistics/question_efficiency_80split_overall.csv`
- `outputs/statistics/question_efficiency_80split_by_method.csv`
- `outputs/statistics/question_efficiency_80split_by_method_prompt_type.csv`
- `outputs/statistics/question_efficiency_80split_by_candidate_count.csv`
- `outputs/statistics/question_efficiency_80split_by_method_candidate_count.csv`
