# Final Offline Audit Metrics After Replacement

- Generated at epoch: 1779212833.87
- Filter: `balanced_method_prompt_cell_filter_seed20260519.json`
- Included records: 1512
- Excluded records: 112
- Replacement: removed `trial_20260514_212634_86dd36` and inserted `trial_20260517_171125_cd5865` with seed `20260520`; cell `random_candidate + partial`.

## Overall
| N | Reviewed | Correct | Wrong | Unresolved | Fail | Acc | Fail Rate | Asked N | Asked Rate | Avg Q Asked | Avg Latency | Median Latency |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1512 | 398 | 1138 | 292 | 82 | 374 | 75.26% | 24.74% | 408 | 26.98% | 2.24 | 54.91 | 7.67 |

## By Method
| Method | N | Reviewed | Correct | Wrong | Unresolved | Acc | Fail | Asked N | Avg Q | Avg Lat | Med Lat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_score | 216 | 104 | 116 | 83 | 17 | 53.70% | 46.30% | 0 | - | 6.77 | 5.00 |
| random_candidate | 216 | 105 | 115 | 91 | 10 | 53.24% | 46.76% | 0 | - | 7.77 | 4.23 |
| vlm_direct | 216 | 102 | 116 | 91 | 9 | 53.70% | 46.30% | 0 | - | 40.59 | 9.75 |
| first_question | 216 | 23 | 197 | 5 | 14 | 91.20% | 8.80% | 101 | 2.37 | 86.02 | 29.55 |
| random_question | 216 | 23 | 196 | 8 | 12 | 90.74% | 9.26% | 101 | 2.39 | 90.71 | 40.99 |
| vlm_best_question | 216 | 20 | 199 | 5 | 12 | 92.13% | 7.87% | 102 | 2.30 | 77.69 | 33.61 |
| proposed_efe | 216 | 21 | 199 | 9 | 8 | 92.13% | 7.87% | 104 | 1.91 | 74.84 | 42.46 |

## By Prompt Type
| Prompt | N | Reviewed | Correct | Wrong | Unresolved | Acc | Fail | Asked N | Avg Q | Avg Lat | Med Lat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clear | 503 | 34 | 483 | 19 | 1 | 96.02% | 3.98% | 9 | 1.22 | 8.09 | 3.77 |
| ambiguous | 506 | 201 | 306 | 170 | 30 | 60.47% | 39.53% | 265 | 2.75 | 109.72 | 75.49 |
| partial | 503 | 163 | 349 | 103 | 51 | 69.38% | 30.62% | 134 | 1.30 | 46.60 | 18.36 |

## Method x Prompt Type
| Method | clear N | clear Acc | clear Fail | clear Avg Lat | ambiguous N | ambiguous Acc | ambiguous Fail | ambiguous Avg Q | ambiguous Avg Lat | partial N | partial Acc | partial Fail | partial Avg Q | partial Avg Lat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_score | 72 | 93.06% | 6.94% | 6.97 | 73 | 17.81% | 82.19% | - | 6.54 | 71 | 50.70% | 49.30% | - | 6.80 |
| random_candidate | 71 | 92.96% | 7.04% | 7.37 | 73 | 17.81% | 82.19% | - | 7.31 | 72 | 50.00% | 50.00% | - | 8.63 |
| vlm_direct | 72 | 91.67% | 8.33% | 14.38 | 72 | 19.44% | 80.56% | - | 79.24 | 72 | 50.00% | 50.00% | - | 28.15 |
| first_question | 72 | 98.61% | 1.39% | 9.20 | 72 | 93.06% | 6.94% | 2.88 | 174.34 | 72 | 81.94% | 18.06% | 1.32 | 74.51 |
| random_question | 73 | 98.63% | 1.37% | 5.46 | 72 | 91.67% | 8.33% | 2.94 | 195.17 | 71 | 81.69% | 18.31% | 1.36 | 72.43 |
| vlm_best_question | 72 | 98.61% | 1.39% | 7.91 | 72 | 93.06% | 6.94% | 2.92 | 164.78 | 72 | 84.72% | 15.28% | 1.19 | 60.39 |
| proposed_efe | 71 | 98.59% | 1.41% | 5.31 | 72 | 91.67% | 8.33% | 2.26 | 143.51 | 73 | 86.30% | 13.70% | 1.32 | 74.73 |

## Requested Table
| Method | Clear Acc | Ambiguous Acc | Partial Acc | Ambiguous Avg Q | Partial Avg Q |
| --- | --- | --- | --- | --- | --- |
| top_score | 93.06% | 17.81% | 50.70% | - | - |
| random_candidate | 92.96% | 17.81% | 50.00% | - | - |
| vlm_direct | 91.67% | 19.44% | 50.00% | - | - |
| first_question | 98.61% | 93.06% | 81.94% | 2.88 | 1.32 |
| random_question | 98.63% | 91.67% | 81.69% | 2.94 | 1.36 |
| vlm_best_question | 98.61% | 93.06% | 84.72% | 2.92 | 1.19 |
| proposed_efe | 98.59% | 91.67% | 86.30% | 2.26 | 1.32 |

## By Scene Type
| Scene | N | Reviewed | Acc | Fail | Wrong | Unresolved | Avg Q | Avg Lat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bottle_only | 235 | 57 | 75.74% | 24.26% | 45 | 12 | 1.85 | 41.20 |
| cup_only | 508 | 145 | 74.02% | 25.98% | 106 | 26 | 2.97 | 74.41 |
| mixed | 381 | 74 | 80.58% | 19.42% | 58 | 16 | 1.67 | 49.58 |
| utensil_only | 388 | 122 | 71.39% | 28.61% | 83 | 28 | 1.96 | 42.93 |

## Interactive Groups
| Group | N | Reviewed | Correct | Wrong | Unresolved | Acc | Fail | Asked N | Avg Q | Avg Lat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| noninteractive | 648 | 311 | 347 | 265 | 36 | 53.55% | 46.45% | 0 | - | 18.38 |
| interactive | 864 | 87 | 791 | 27 | 46 | 91.55% | 8.45% | 408 | 2.24 | 82.32 |

## Failure Reasons Overall
| Reason | Count | Fail Share | All Share |
| --- | --- | --- | --- |
| 没有solve ambiguity | 200 | 53.48% | 13.23% |
| GroundingDINO没有包含目标candidate | 79 | 21.12% | 5.22% |
| 错误选择了物体 | 65 | 17.38% | 4.30% |
| none | 26 | 6.95% | 1.72% |
| VLM问题错误 | 3 | 0.80% | 0.20% |
| GroundingDINO label错误 | 1 | 0.27% | 0.07% |

## Failure Reason By Method
| Method | none | 没有solve ambiguity | 错误选择了物体 | GroundingDINO没有包含目标candidate | GroundingDINO label错误 | 用户失误 | VLM问题错误 | 其他 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_score | 5 | 69 | 12 | 13 | 1 | 0 | 0 | 0 |
| random_candidate | 11 | 61 | 14 | 15 | 0 | 0 | 0 | 0 |
| vlm_direct | 7 | 66 | 11 | 16 | 0 | 0 | 0 | 0 |
| first_question | 1 | 1 | 6 | 11 | 0 | 0 | 0 | 0 |
| random_question | 1 | 1 | 8 | 8 | 0 | 0 | 2 | 0 |
| vlm_best_question | 0 | 0 | 7 | 9 | 0 | 0 | 1 | 0 |
| proposed_efe | 1 | 2 | 7 | 7 | 0 | 0 | 0 | 0 |

## Failure Reason By Prompt Type
| Prompt | none | 没有solve ambiguity | 错误选择了物体 | GroundingDINO没有包含目标candidate | GroundingDINO label错误 | 用户失误 | VLM问题错误 | 其他 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clear | 2 | 7 | 10 | 0 | 1 | 0 | 0 | 0 |
| ambiguous | 12 | 156 | 3 | 28 | 0 | 0 | 1 | 0 |
| partial | 12 | 37 | 52 | 51 | 0 | 0 | 2 | 0 |

## Top Failure Groups
| Reason | Method | Prompt | Count | Top Scenes |
| --- | --- | --- | --- | --- |
| 没有solve ambiguity | top_score | ambiguous | 52 | cup_only:19, utensil_only:14, mixed:10 |
| 没有solve ambiguity | vlm_direct | ambiguous | 51 | cup_only:21, utensil_only:13, bottle_only:9 |
| 没有solve ambiguity | random_candidate | ambiguous | 50 | cup_only:18, mixed:14, utensil_only:10 |
| 没有solve ambiguity | top_score | partial | 14 | utensil_only:6, cup_only:4, bottle_only:2 |
| GroundingDINO没有包含目标candidate | vlm_direct | partial | 13 | cup_only:12, mixed:1 |
| 没有solve ambiguity | vlm_direct | partial | 12 | utensil_only:7, cup_only:4, mixed:1 |
| GroundingDINO没有包含目标candidate | random_candidate | partial | 11 | cup_only:11 |
| 错误选择了物体 | top_score | partial | 11 | bottle_only:4, cup_only:3, utensil_only:2 |
| 没有solve ambiguity | random_candidate | partial | 10 | utensil_only:5, cup_only:3, mixed:2 |
| 错误选择了物体 | random_candidate | partial | 10 | bottle_only:4, utensil_only:3, cup_only:2 |
| GroundingDINO没有包含目标candidate | top_score | partial | 9 | cup_only:8, mixed:1 |
| 错误选择了物体 | vlm_direct | partial | 8 | bottle_only:3, cup_only:2, mixed:2 |
| 错误选择了物体 | random_question | partial | 7 | bottle_only:4, cup_only:1, utensil_only:1 |
| GroundingDINO没有包含目标candidate | first_question | partial | 6 | mixed:3, cup_only:2, utensil_only:1 |
| 错误选择了物体 | proposed_efe | partial | 6 | bottle_only:4, cup_only:1, utensil_only:1 |
| none | random_candidate | ambiguous | 5 | cup_only:3, mixed:2 |
| none | random_candidate | partial | 5 | mixed:3, cup_only:1, utensil_only:1 |
| GroundingDINO没有包含目标candidate | vlm_best_question | partial | 5 | mixed:4, cup_only:1 |
| 错误选择了物体 | first_question | partial | 5 | bottle_only:3, utensil_only:1, mixed:1 |
| 错误选择了物体 | vlm_best_question | partial | 5 | bottle_only:4, utensil_only:1 |

## Audit Corrections
- Reviewed included trials: 398
- Include false among included trials: 0
- Changed counts: `{"failure_reason_set": 348, "prompt_type": 9, "outcome": 30}`
- Outcome transitions: `{"correct->wrong": 6, "wrong->correct": 15, "unresolved->correct": 8, "unresolved->wrong": 1}`
- Prompt type transitions: `{"partial->ambiguous": 2, "partial->clear": 3, "clear->partial": 4}`

