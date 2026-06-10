# Old vs Prompt-Robust Analysis Summary

## Accuracy: Original GEE vs Prompt-Feature GEE

| Contrast | Old OR | New OR + Prompt Features | OR Change | Old Holm p | New Holm p | Old conclusion | New conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| proposed_efe vs top_score | 15.52 | 15.6 | 0.09 | 1.39e-23 | 1.63e-23 | significant | significant |
| proposed_efe vs random_candidate | 15.88 | 17 | 1.08 | 1.39e-23 | 1.1e-19 | significant | significant |
| proposed_efe vs vlm_direct | 15.81 | 18.7 | 2.86 | 9.33e-19 | 2.21e-17 | significant | significant |
| vlm_best_question vs vlm_direct | 15.71 | 18 | 2.24 | 1.28e-19 | 8.96e-21 | significant | significant |
| proposed_efe vs first_question | 1.17 | 1.12 | -0.05 | 0.832 | 1 | comparable/n.s. | comparable/n.s. |
| proposed_efe vs random_question | 1.22 | 1.21 | -0.01 | 0.777 | 0.849 | comparable/n.s. | comparable/n.s. |
| proposed_efe vs vlm_best_question | 1.01 | 1.04 | 0.03 | 0.971 | 1 | comparable/n.s. | comparable/n.s. |

## Question Count: Original Scene-Level vs Asked-Only Scene-Level

| Comparison | Old scene-level all-trial EFE Q | Old scene-level all-trial baseline Q | Old reduction % | Old Holm p | Asked-only scene-level EFE Q | Asked-only scene-level baseline Q | Asked-only reduction % | Asked-only Holm p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EFE vs first_question | 1.97 | 2.61 | 24.37 | 0.000904 | 2.20 | 2.85 | 22.91 | 0.000182 |
| EFE vs random_question | 1.97 | 2.58 | 23.33 | 0.000904 | 2.16 | 2.83 | 23.88 | 0.000182 |
| EFE vs vlm_best_question | 1.97 | 2.57 | 23.18 | 0.000904 | 2.22 | 2.83 | 21.43 | 0.000555 |
