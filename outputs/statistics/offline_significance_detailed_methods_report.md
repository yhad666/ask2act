# Offline Experiment Statistical Analysis: Detailed Methods, Implementation, Results, and Interpretation

Generated for the cleaned offline target-resolution experiment.

Main analysis artifact: `outputs/statistics/significance_report.md`

Reusable analysis script: `scripts/run_significance_tests.py`

Cleaned data input used for the current statistical run:

```text
services/a6000_web/artifacts/offline_experiments/top_cups_01/analysis/balanced_method_prompt_cell_filter_seed20260519.json
```

Output directory:

```text
outputs/statistics
```

Random seed for bootstrap and deterministic sampling:

```text
20260520
```

## 1. Purpose of This Report

This document records the full statistical analysis pipeline for the cleaned offline target-resolution experiment. It is intended to be detailed enough for later paper writing, reproducibility checking, and reviewer-facing explanation.

The analysis supports the following scientific claims:

1. Interactive clarification substantially improves target-resolution accuracy compared with non-interactive baselines.
2. Direct VLM target selection is insufficient under ambiguous referential commands.
3. The proposed EFE method maintains accuracy comparable to other interactive clarification methods.
4. The proposed EFE method reduces the number of clarification questions, especially for ambiguous prompts.
5. Latency and prompt-type difficulty are analyzed as supporting results, with conservative wording.

This report also explains why prompt-level paired tests were not used as the main analysis, how prompt wording imbalance was handled, and how robustness checks were performed.

## 2. Executive Summary

The cleaned offline dataset contains 1512 included records, corresponding to 216 trials per method for 7 methods. The shared experimental unit across methods is the scene, not the exact prompt text. Prompt wording was randomized and balanced within scene and prompt type, but exact prompts were not always identical across methods. Therefore, the main analysis uses scene-clustered and scene-level methods rather than prompt-level paired tests.

Accuracy was analyzed with a scene-clustered logistic GEE model. Proposed EFE significantly outperformed all non-interactive baselines after Holm correction. VLM-best interactive clarification also significantly outperformed direct VLM selection, supporting the claim that interaction is needed under ambiguity.

EFE was not significantly better or worse than other interactive methods in accuracy. This supports the safer claim that EFE maintains comparable accuracy to interactive baselines.

Question count was analyzed at the scene level. On ambiguous prompts, EFE significantly reduced clarification questions compared with first-question, random-question, and VLM-best-question baselines. Asked-only analyses, which match the human-readable average-question tables, show the same direction and stronger practical interpretability.

Latency did not show robust statistically significant improvement for EFE after correction. The safe conclusion is that latency is long-tailed and should be reported descriptively; no strong latency-reduction claim should be made unless later data supports it.

Prompt-type analysis confirmed that clear prompts are easy, ambiguous prompts are substantially harder and trigger more questions, and partial prompts are also challenging. Ambiguous prompts were numerically harder than partial prompts, but the adjusted ambiguous-vs-partial contrast was not significant after Holm correction.

Prompt-feature adjusted GEE results were consistent with the main GEE results, indicating that prompt wording features do not explain away the method effect.

## 3. Cleaned Dataset

### 3.1 Data Source

The main analysis uses the cleaned balanced dataset generated after offline audit and replacement:

```text
services/a6000_web/artifacts/offline_experiments/top_cups_01/analysis/balanced_method_prompt_cell_filter_seed20260519.json
```

The cleaned audit summary used as the data reference is:

```text
services/a6000_web/artifacts/offline_experiments/top_cups_01/analysis/final_audit_metrics_after_replacement_seed20260520.md
```

The statistical script loads trial-level data rather than aggregate markdown tables. This is important because statistical models and scene-level tests require per-trial information.

### 3.2 Inclusion Rule

Only records marked as included in the cleaned balanced filter are used in the main analysis. Excluded records are not used. Raw data are not modified by the significance script.

The included dataset has:

| Item | Value |
| --- | ---: |
| Included records | 1512 |
| Methods | 7 |
| Records per method | 216 |
| Scenes | 36 |
| Prompt types | clear, ambiguous, partial |
| Main unit for clustered accuracy analysis | scene |
| Main unit for question/latency tests | scene-method-prompt_type aggregate |

### 3.3 Methods

The 7 evaluated methods are:

| Method | Type | Description |
| --- | --- | --- |
| `top_score` | Non-interactive | Selects the highest-scoring GroundingDINO candidate. |
| `random_candidate` | Non-interactive | Selects a random candidate from the detected candidate list. |
| `vlm_direct` | Non-interactive | Lets the VLM directly select the target without interactive clarification. |
| `first_question` | Interactive | Uses the first generated clarification question. |
| `random_question` | Interactive | Randomly chooses among generated clarification questions. |
| `vlm_best_question` | Interactive | Uses the VLM's own best clarification question. |
| `proposed_efe` | Interactive | Uses the proposed expected-free-energy question selection policy. |

### 3.4 Prompt Types

| Prompt Type | Meaning |
| --- | --- |
| `clear` | The prompt is intended to specify the target clearly. |
| `ambiguous` | The prompt intentionally leaves ambiguity, often requiring clarification. |
| `partial` | The prompt is partially specified, underspecified, or relational. |

### 3.5 Method-Level Descriptive Results

These are the cleaned descriptive results used to sanity-check the statistical analysis.

| Method | N | Correct | Wrong | Unresolved | Accuracy | Fail Rate | Asked N | Avg Q Asked | Avg Latency | Median Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `top_score` | 216 | 116 | 83 | 17 | 53.70% | 46.30% | 0 | - | 6.77 | 5.00 |
| `random_candidate` | 216 | 115 | 91 | 10 | 53.24% | 46.76% | 0 | - | 7.77 | 4.23 |
| `vlm_direct` | 216 | 116 | 91 | 9 | 53.70% | 46.30% | 0 | - | 40.59 | 9.75 |
| `first_question` | 216 | 197 | 5 | 14 | 91.20% | 8.80% | 101 | 2.37 | 86.02 | 29.55 |
| `random_question` | 216 | 196 | 8 | 12 | 90.74% | 9.26% | 101 | 2.39 | 90.71 | 40.99 |
| `vlm_best_question` | 216 | 199 | 5 | 12 | 92.13% | 7.87% | 102 | 2.30 | 77.69 | 33.61 |
| `proposed_efe` | 216 | 199 | 9 | 8 | 92.13% | 7.87% | 104 | 1.91 | 74.84 | 42.46 |

### 3.6 Prompt-Type Descriptive Results

| Prompt Type | N | Correct | Wrong | Unresolved | Accuracy | Fail Rate | Asked N | Asked Rate | Avg Q Asked | Avg Latency | Median Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `clear` | 503 | 483 | 19 | 1 | 96.02% | 3.98% | 9 | 1.79% | 1.22 | 8.09 | 3.77 |
| `ambiguous` | 506 | 306 | 170 | 30 | 60.47% | 39.53% | 265 | 52.37% | 2.75 | 109.72 | 75.49 |
| `partial` | 503 | 349 | 103 | 51 | 69.38% | 30.62% | 134 | 26.64% | 1.30 | 46.60 | 18.36 |

### 3.7 Important Note on Prompt-Type Counts

Each method has exactly 216 included trials. However, method-by-prompt-type counts are not always exactly 72 after audit correction because several records had their prompt type corrected during cleaning.

This is expected and documented. It does not break the main GEE analysis, because prompt type is included as a covariate.

| Method | Clear N | Ambiguous N | Partial N |
| --- | ---: | ---: | ---: |
| `top_score` | 72 | 73 | 71 |
| `random_candidate` | 71 | 73 | 72 |
| `vlm_direct` | 72 | 72 | 72 |
| `first_question` | 72 | 72 | 72 |
| `random_question` | 73 | 72 | 71 |
| `vlm_best_question` | 72 | 72 | 72 |
| `proposed_efe` | 71 | 72 | 73 |

## 4. Why the Main Analysis Was Changed

### 4.1 Original Assumption

The initial statistical plan assumed that every method was evaluated on the exact same task-prompt pairs. Under that assumption, prompt-level paired tests would be appropriate:

| Outcome | If exact prompt pairing existed |
| --- | --- |
| Binary accuracy | McNemar's test |
| Question count | Wilcoxon signed-rank test |
| Latency | Wilcoxon signed-rank test |

### 4.2 Actual Experimental Design

The actual cleaned experiment is balanced by scene, method, and prompt type, but exact prompt wording is not always shared across methods. About 30-40% of prompts differ in their specific attribute wording. For example, one method may receive a clear prompt based on color while another method receives a clear prompt based on location or relation in the same scene.

Therefore, exact prompt-level pairing is not valid as the main analysis.

### 4.3 Correct Shared Unit

The scene is the stable shared unit across methods. Within each scene and prompt type, prompts are sampled or randomized from a pool. This means that the correct main analysis should respect scene clustering and not assume exact prompt identity.

The updated analysis uses:

| Analysis Target | Main Statistical Method |
| --- | --- |
| Accuracy | Scene-clustered logistic GEE |
| Question count | Scene-level aggregation followed by Friedman and Wilcoxon tests |
| Latency | Scene-level aggregation followed by Wilcoxon tests |
| Prompt difficulty | Scene-clustered logistic GEE |
| Prompt wording robustness | Prompt-feature adjusted GEE |

### 4.4 Supplementary Prompt-Level Tests

Prompt-level paired tests are only used as optional supplementary analysis on a small exact-matched subset, if such a subset exists. They are not used as the main paper evidence.

In the current dataset, only 17 exact prompt keys were available, corresponding to 119 records. This subset is too small and not representative enough to replace the main scene-clustered analysis.

## 5. Main Variables and Column Mapping

The script normalizes different possible raw column names into a common internal schema.

| Internal Column | Meaning |
| --- | --- |
| `scene_id` | Scene identifier used for clustering and scene-level aggregation. |
| `method` | Evaluation method. |
| `prompt_type` | Clear, ambiguous, or partial. |
| `correct` | Binary target-resolution success, where correct is 1 and wrong or unresolved is 0. |
| `num_questions` | Number of clarification questions asked. |
| `asked` | Whether at least one clarification question was asked. |
| `latency_sec` | Trial latency in seconds. |
| `scene_type` | Scene category, if available. |
| `candidate_set_size` | Number of candidate objects, if available. |
| `prompt_text` | Natural-language prompt text, if available. |
| `target_category` | Target category, if available. |
| `failure_reason` | Human-audited failure reason, if available. |

Wrong and unresolved trials are both treated as failures for binary accuracy:

```python
correct = 1 if outcome == "correct" else 0
```

This matches the experimental goal: a system must resolve the target correctly; unresolved is not useful for execution.

## 6. Core Statistical Principles

### 6.1 No Data Modification During Testing

The significance script does not modify raw data. It reads the cleaned included dataset and writes analysis outputs under `outputs/statistics`.

### 6.2 Scene Clustering

Multiple trials from the same scene are not independent. A scene has shared visual layout, object categories, clutter, camera view, and candidate distribution. Therefore, accuracy analysis uses scene-clustered standard errors.

The main model is a logistic GEE:

```text
correct ~ method + prompt_type + scene_type + candidate_set_size
```

with:

```text
family = Binomial
groups = scene_id
covariance = robust
```

### 6.3 Scene-Level Aggregation for Counts and Latency

Question count and latency are not compared at raw prompt level, because exact prompt text is not always matched across methods. Instead, trials are aggregated by:

```text
scene_id, method, prompt_type
```

For question count:

```text
mean_num_questions
median_num_questions
n_trials
```

For latency:

```text
mean_latency
median_latency
IQR latency
n_trials
```

These scene-level aggregates are then compared across methods.

### 6.4 Multiple-Comparison Correction

Holm-Bonferroni correction is applied within each planned test family.

The accuracy contrasts are split into two families:

| Family | Purpose |
| --- | --- |
| A | EFE or interaction versus non-interactive baselines. |
| B | EFE versus other interactive baselines. |

Question-count and latency pairwise tests are corrected within each analysis context, such as ambiguous prompts or overall interactive prompts.

### 6.5 Bootstrap Confidence Intervals

Bootstrap confidence intervals are used for paired scene-level mean differences where appropriate.

Bootstrap settings:

| Item | Value |
| --- | ---: |
| Seed | 20260520 |
| Iterations | 10000 |
| Resampling unit | Scene-level paired difference |
| CI | 95% percentile interval |

### 6.6 Formula Details

This section writes the actual statistical formulas used in the analysis. It is included because the main statistical issue in this experiment is not only which test was used, but why the unit of analysis and covariates are appropriate.

#### 6.6.1 Notation

Let each trial be indexed by `i`.

| Symbol | Meaning |
| --- | --- |
| `i` | Trial index. |
| `s(i)` | Scene containing trial `i`. |
| `M_i` | Method used in trial `i`. |
| `T_i` | Prompt type for trial `i`: clear, ambiguous, or partial. |
| `C_i` | Scene type for trial `i`, if available. |
| `K_i` | Candidate set size for trial `i`, if available. |
| `X_i` | Prompt-feature vector for trial `i`, used in the robustness model. |
| `Y_i` | Binary correctness outcome for trial `i`. |
| `Q_i` | Number of clarification questions in trial `i`. |
| `L_i` | Latency in seconds for trial `i`. |

The binary outcome is:

```text
Y_i = 1 if the trial outcome is correct
Y_i = 0 if the trial outcome is wrong or unresolved
```

This matches the robotics task requirement: unresolved is a failure because the robot cannot safely execute a correct target.

#### 6.6.2 Descriptive Accuracy and Failure Rate

For method `m`, let `I(M_i = m)` be an indicator that trial `i` used method `m`.

```text
N_m = sum_i I(M_i = m)
```

```text
Correct_m = sum_i I(M_i = m) Y_i
```

```text
Accuracy_m = Correct_m / N_m
```

```text
FailureRate_m = 1 - Accuracy_m
```

The same structure is used for prompt-type accuracy by replacing `I(M_i = m)` with `I(T_i = t)`.

#### 6.6.3 Asked Rate and Asked-Only Mean Question Count

Define whether a trial asked at least one clarification question:

```text
A_i = 1 if Q_i > 0
A_i = 0 otherwise
```

For method `m`, the asked rate is:

```text
AskedRate_m = sum_i I(M_i = m) A_i / sum_i I(M_i = m)
```

The asked-only mean question count is:

```text
MeanQAsked_m =
    sum_i I(M_i = m) A_i Q_i
    /
    sum_i I(M_i = m) A_i
```

This asked-only definition is why the audit table reports values such as 1.91 for `proposed_efe`: it averages only trials where the method actually asked questions.

The all-trial mean question count is different:

```text
MeanQAll_m =
    sum_i I(M_i = m) Q_i
    /
    sum_i I(M_i = m)
```

This counts no-question trials as zero. Both are valid summaries, but they answer different questions.

#### 6.6.4 Main Logistic GEE Accuracy Model

The main accuracy model assumes:

```text
Y_i ~ Bernoulli(pi_i)
```

where:

```text
pi_i = P(Y_i = 1)
```

The logistic link is:

```text
logit(pi_i) = log(pi_i / (1 - pi_i))
```

The main model is:

```text
logit(pi_i)
  = beta_0
  + sum_m beta_m I(M_i = m)
  + sum_t gamma_t I(T_i = t)
  + sum_c delta_c I(C_i = c)
  + eta K_i
```

The implemented formula is:

```text
correct ~ C(method, Treatment(reference="proposed_efe"))
        + C(prompt_type, Treatment(reference="clear"))
        + C(scene_type)
        + candidate_set_size
```

The model is fit as a GEE with `scene_id` as the clustering group:

```text
group = scene_id
family = Binomial
working correlation = independence
covariance = robust sandwich covariance
```

The working independence assumption does not mean trials are treated as fully independent for inference. The robust sandwich covariance is clustered by scene, so standard errors account for repeated trials within the same scene.

#### 6.6.5 GEE Estimating Equation and Robust Sandwich Covariance

For scene `s`, let:

```text
Y_s = vector of outcomes in scene s
mu_s = vector of fitted probabilities in scene s
D_s = derivative of mu_s with respect to model parameters
V_s = working covariance matrix for scene s
```

GEE estimates parameters by solving:

```text
sum_s D_s^T V_s^{-1} (Y_s - mu_s) = 0
```

The robust sandwich covariance has the form:

```text
Cov(beta_hat) = A^{-1} B A^{-1}
```

where:

```text
A = sum_s D_s^T V_s^{-1} D_s
```

and:

```text
B = sum_s D_s^T V_s^{-1}
        (Y_s - mu_s)(Y_s - mu_s)^T
        V_s^{-1} D_s
```

This is why the model can use a simple working correlation while still producing scene-clustered robust standard errors.

#### 6.6.6 Odds Ratio Contrasts

Because `proposed_efe` is the reference method, its method coefficient is zero:

```text
theta_proposed_efe = 0
```

For any other method `m`, the fitted method coefficient is:

```text
theta_m = log-odds difference of method m relative to proposed_efe
```

For a planned contrast comparing method `A` against method `B`:

```text
Delta_AB = theta_A - theta_B
```

The odds ratio is:

```text
OR_AB = exp(Delta_AB)
```

Examples:

```text
log OR(proposed_efe vs top_score)
  = theta_proposed_efe - theta_top_score
  = 0 - theta_top_score
```

```text
log OR(vlm_best_question vs vlm_direct)
  = theta_vlm_best_question - theta_vlm_direct
```

The standard error of a contrast is computed from the model covariance matrix:

```text
SE(Delta_AB) = sqrt(l^T Cov(beta_hat) l)
```

where `l` is the contrast vector.

The Wald statistic is:

```text
z = Delta_AB / SE(Delta_AB)
```

The two-sided p-value is:

```text
p = 2 * (1 - Phi(abs(z)))
```

The 95% confidence interval for the log-odds contrast is:

```text
Delta_AB +/- 1.96 * SE(Delta_AB)
```

The 95% confidence interval for the odds ratio is:

```text
[exp(Delta_AB - 1.96 * SE), exp(Delta_AB + 1.96 * SE)]
```

#### 6.6.7 Prompt-Feature Adjusted GEE Model

The prompt-feature adjusted model extends the main model with prompt-level covariates:

```text
logit(pi_i)
  = beta_0
  + sum_m beta_m I(M_i = m)
  + sum_t gamma_t I(T_i = t)
  + sum_c delta_c I(C_i = c)
  + eta K_i
  + lambda_1 prompt_length_i
  + lambda_2 has_my_i
  + lambda_3 has_color_i
  + lambda_4 has_spatial_i
  + lambda_5 has_relation_i
  + lambda_6 has_size_i
  + sum_o rho_o I(object_category_i = o)
```

The implemented formula is:

```text
correct ~ C(method, Treatment(reference="proposed_efe"))
        + C(prompt_type, Treatment(reference="clear"))
        + C(scene_type)
        + candidate_set_size
        + prompt_length
        + has_my
        + has_color
        + has_spatial
        + has_relation
        + has_size
        + C(prompt_object_category)
```

This model asks whether the method effects remain after controlling for observable prompt wording difficulty.

#### 6.6.8 Scene-Level Question Aggregation

For question-count tests, raw trials are first aggregated at:

```text
scene_id, method, prompt_type
```

For scene `s`, method `m`, and prompt type `t`, let the set of matching trials be:

```text
G_smt = {i : s(i) = s, M_i = m, T_i = t}
```

The all-trial scene-level mean question count is:

```text
Qbar_smt = (1 / |G_smt|) * sum_{i in G_smt} Q_i
```

The scene-level median question count is:

```text
Qmedian_smt = median_{i in G_smt}(Q_i)
```

For asked-only analysis, the group is restricted to trials where `Q_i > 0`:

```text
Gasked_smt = {i in G_smt : Q_i > 0}
```

and:

```text
QbarAsked_smt =
    (1 / |Gasked_smt|) * sum_{i in Gasked_smt} Q_i
```

Asked-only scene-level comparisons use scenes where both EFE and the baseline have valid asked-only values.

#### 6.6.9 Friedman Test for Question Count

For a given context such as ambiguous prompts, each scene contributes one aggregated value per interactive method.

Let:

```text
n = number of complete scenes
k = number of interactive methods = 4
R_j = sum of ranks for method j across scenes
```

Within each scene, methods are ranked by their scene-level mean question count. The Friedman statistic is:

```text
Q_Friedman =
    (12 / (n k (k + 1))) * sum_j R_j^2
    - 3 n (k + 1)
```

Under the null hypothesis that all methods have the same distribution, the statistic is approximately:

```text
Q_Friedman ~ chi-square(k - 1)
```

The Friedman test is used as the omnibus test before interpreting pairwise Wilcoxon tests.

#### 6.6.10 Wilcoxon Signed-Rank Test

For a pairwise scene-level comparison between EFE and baseline `b`, define paired scene differences:

```text
d_s = Qbar_s,proposed_efe,t - Qbar_s,b,t
```

Zero differences are removed. The remaining absolute differences are ranked:

```text
rank_s = rank(abs(d_s))
```

Positive and negative signed-rank sums are:

```text
W_plus = sum_s rank_s I(d_s > 0)
```

```text
W_minus = sum_s rank_s I(d_s < 0)
```

The two-sided Wilcoxon signed-rank test evaluates whether the median paired difference is zero. In the script, this is implemented with:

```python
stats.wilcoxon(efe_values, baseline_values, zero_method="wilcox")
```

For question count, a negative paired difference means EFE asked fewer questions than the baseline.

#### 6.6.11 Percent Reduction in Mean Question Count

For each EFE-baseline comparison:

```text
PercentReduction =
    100 * (MeanQ_baseline - MeanQ_EFE) / MeanQ_baseline
```

For example, if EFE asks 1.97 questions and the baseline asks 2.61 questions:

```text
PercentReduction = 100 * (2.61 - 1.97) / 2.61 = 24.52%
```

Small differences from the reported value can occur because the actual computation uses full precision before rounding.

#### 6.6.12 Bootstrap Confidence Interval for Paired Mean Difference

For scene-level paired differences:

```text
d_s = value_s,EFE - value_s,baseline
```

the observed paired mean difference is:

```text
dbar = (1 / n) * sum_s d_s
```

Bootstrap samples are generated by resampling scenes with replacement:

```text
d_1^*, d_2^*, ..., d_n^*
```

For each bootstrap sample `b`:

```text
dbar_b^* = (1 / n) * sum_s d_s^*
```

The 95% bootstrap CI is:

```text
[percentile_2.5({dbar_b^*}), percentile_97.5({dbar_b^*})]
```

The current run used 10000 bootstrap iterations.

#### 6.6.13 Latency Formulas

Latency is aggregated with the same scene-level structure:

```text
Lbar_smt = (1 / |G_smt|) * sum_{i in G_smt} L_i
```

```text
Lmedian_smt = median_{i in G_smt}(L_i)
```

The interquartile range is:

```text
IQR_smt = percentile_75(L_i in G_smt) - percentile_25(L_i in G_smt)
```

Pairwise latency tests use:

```text
d_s = Lbar_s,proposed_efe,t - Lbar_s,baseline,t
```

and the same Wilcoxon signed-rank procedure. A negative value means EFE is faster.

#### 6.6.14 Holm-Bonferroni Correction

For a family of `m` raw p-values, sort them:

```text
p_(1) <= p_(2) <= ... <= p_(m)
```

The Holm adjusted value for the sorted p-value `p_(i)` is:

```text
p_holm_(i) =
    max_{j <= i} min((m - j + 1) * p_(j), 1)
```

A comparison is considered significant if:

```text
p_holm < 0.05
```

Holm correction is applied within planned families rather than across every number in the report, because each family corresponds to a specific scientific question.

#### 6.6.15 Prompt-Type Difficulty GEE Formula

Prompt-type difficulty is tested with:

```text
Y_i ~ Bernoulli(pi_i)
```

```text
logit(pi_i)
  = alpha_0
  + sum_t alpha_t I(T_i = t)
  + sum_m beta_m I(M_i = m)
  + sum_c delta_c I(C_i = c)
```

The implemented formula is:

```text
correct ~ C(prompt_type) + C(method) + C(scene_type)
```

The contrasts are:

```text
ambiguous vs clear
partial vs clear
ambiguous vs partial
```

The interpretation is based on odds ratios and Holm-corrected p-values.

#### 6.6.16 Supplementary McNemar Formula

McNemar's test is not the main analysis. It is only used for the small exact prompt-level matched subset.

For two methods `A` and `B`, define:

| Count | Meaning |
| --- | --- |
| `b` | A correct, B wrong. |
| `c` | A wrong, B correct. |

The continuity-corrected McNemar statistic used in the supplementary test is:

```text
chi_square = (abs(b - c) - 1)^2 / (b + c)
```

with:

```text
df = 1
```

This test is supplementary because the exact matched subset is too small to represent the full experiment.

## 7. Accuracy Analysis

### 7.1 Main Accuracy Model

The main GEE model is:

```text
correct ~ C(method, Treatment(reference="proposed_efe"))
        + C(prompt_type, Treatment(reference="clear"))
        + C(scene_type)
        + candidate_set_size
```

Rationale:

| Term | Reason |
| --- | --- |
| `method` | Main experimental factor. |
| `prompt_type` | Controls clear, ambiguous, and partial prompt difficulty. |
| `scene_type` | Controls object-group composition such as cup-only, bottle-only, utensil-only, and mixed scenes. |
| `candidate_set_size` | Controls ambiguity from the number of available candidates. |
| `scene_id` cluster | Accounts for repeated observations from the same scene. |

The reference method is `proposed_efe`, which makes planned contrasts easy to compute.

### 7.2 Planned Accuracy Contrasts

Family A tests whether EFE or interactive clarification outperforms non-interactive baselines:

| Contrast | Purpose |
| --- | --- |
| `proposed_efe` vs `top_score` | Tests whether highest detector score is enough. |
| `proposed_efe` vs `random_candidate` | Tests whether random candidate selection is enough. |
| `proposed_efe` vs `vlm_direct` | Tests whether direct VLM selection without interaction is enough. |
| `vlm_best_question` vs `vlm_direct` | Tests whether VLM needs interaction rather than direct selection. |

Family B tests whether EFE accuracy differs from other interactive methods:

| Contrast | Purpose |
| --- | --- |
| `proposed_efe` vs `first_question` | Accuracy comparison with first generated question. |
| `proposed_efe` vs `random_question` | Accuracy comparison with random generated question. |
| `proposed_efe` vs `vlm_best_question` | Accuracy comparison with VLM-selected best question. |

### 7.3 Main Accuracy Results

| Contrast | OR | 95% CI | Holm p | Conclusion |
| --- | ---: | --- | ---: | --- |
| `proposed_efe` vs `top_score` | 15.52 | [9.14, 26.37] | 1.39e-23 | Significant |
| `proposed_efe` vs `random_candidate` | 15.88 | [9.30, 27.10] | 1.39e-23 | Significant |
| `proposed_efe` vs `vlm_direct` | 15.81 | [8.57, 29.15] | 9.33e-19 | Significant |
| `vlm_best_question` vs `vlm_direct` | 15.71 | [8.70, 28.36] | 1.28e-19 | Significant |
| `proposed_efe` vs `first_question` | 1.17 | [0.80, 1.69] | 0.832 | Comparable |
| `proposed_efe` vs `random_question` | 1.22 | [0.86, 1.74] | 0.777 | Comparable |
| `proposed_efe` vs `vlm_best_question` | 1.01 | [0.71, 1.43] | 0.971 | Comparable |

### 7.4 Accuracy Interpretation

The non-interactive baselines all achieve roughly 53-54% target-resolution accuracy. The interactive methods achieve roughly 91-92% accuracy. The GEE contrasts show that EFE significantly outperforms all non-interactive baselines after Holm correction.

Direct VLM selection is not sufficient under ambiguity. This is supported by the large and significant contrast between `vlm_best_question` and `vlm_direct`.

EFE does not significantly outperform the other interactive methods in accuracy. This is not a weakness for the intended paper claim. It supports the more precise statement:

```text
EFE maintains comparable target-resolution accuracy to other interactive clarification methods while reducing clarification burden.
```

The safe paper claim should not say that EFE significantly improves accuracy over all interactive methods.

## 8. Prompt-Feature Adjusted Accuracy Analysis

### 8.1 Why This Was Added

The user identified a real design issue: prompts were created by scene and prompt type, but the number and diversity of available prompts differed across scene-prompt pools. Some pools had high repetition, and some prompts differed by attribute type, such as color, location, size, relation, or ownership.

This can affect statistical inference if one method accidentally receives easier or harder prompt wording.

To reduce this concern, a second GEE model adds prompt-level covariates.

### 8.2 Prompt Features Extracted

The script derives the following features from prompt text:

| Feature | Meaning |
| --- | --- |
| `prompt_length` | Number of words in the prompt. |
| `has_my` | Whether the prompt contains ownership wording such as "my". |
| `has_color` | Whether the prompt contains color attributes. |
| `has_spatial` | Whether the prompt contains spatial attributes such as left, right, middle, nearest, farthest. |
| `has_relation` | Whether the prompt contains relational terms such as next to, between, behind, in front of. |
| `has_size` | Whether the prompt contains size terms such as big, small, tall, short. |
| `prompt_object_category` | Coarse object category inferred from prompt text, such as cup, bottle, fork, spoon, utensil, or object. |

### 8.3 Prompt-Feature Adjusted Model

The adjusted model is:

```text
correct ~ C(method, Treatment(reference="proposed_efe"))
        + C(prompt_type, Treatment(reference="clear"))
        + C(scene_type)
        + candidate_set_size
        + prompt_length
        + has_my
        + has_color
        + has_spatial
        + has_relation
        + has_size
        + C(prompt_object_category)
```

with:

```text
family = Binomial
groups = scene_id
covariance = robust
```

### 8.4 How This Balances Prompt Difficulty

This method does not make prompts identical. Instead, it statistically adjusts for observable prompt difficulty features.

The model estimates the method effect after accounting for prompt properties. For example:

| Prompt Feature | Difficulty Issue It Helps Control |
| --- | --- |
| Color wording | Some color prompts may be visually easier than relational prompts. |
| Spatial wording | Left/right/most-left prompts rely on scene layout and detection completeness. |
| Relation wording | Relation prompts may be harder because they require comparing objects. |
| Size wording | Size can be ambiguous in perspective views. |
| Ownership wording | "my" prompts often correspond to ambiguous or user-specific reference. |
| Prompt length | Longer prompts can contain more information or more complexity. |
| Object category | Cups, bottles, and utensils may differ in detector and graspability difficulty. |

If the EFE advantage disappeared after adding these covariates, that would suggest prompt wording imbalance might explain the result. It did not disappear.

### 8.5 Prompt-Feature Adjusted Results

| Contrast | OR | 95% CI | Holm p | Conclusion |
| --- | ---: | --- | ---: | --- |
| `proposed_efe` vs `top_score` | 15.62 | [9.18, 26.58] | 1.63e-23 | Significant |
| `proposed_efe` vs `random_candidate` | 16.96 | [9.25, 31.08] | 1.10e-19 | Significant |
| `proposed_efe` vs `vlm_direct` | 18.67 | [9.49, 36.71] | 2.21e-17 | Significant |
| `vlm_best_question` vs `vlm_direct` | 17.95 | [9.87, 32.65] | 8.96e-21 | Significant |
| `proposed_efe` vs `first_question` | 1.12 | [0.79, 1.58] | 1.000 | Comparable |
| `proposed_efe` vs `random_question` | 1.21 | [0.85, 1.73] | 0.849 | Comparable |
| `proposed_efe` vs `vlm_best_question` | 1.04 | [0.72, 1.51] | 1.000 | Comparable |

### 8.6 Prompt-Feature Interpretation

The prompt-feature adjusted GEE gives essentially the same conclusion as the main GEE:

1. EFE significantly outperforms non-interactive baselines.
2. Interactive VLM clarification significantly outperforms direct VLM selection.
3. EFE accuracy remains comparable to other interactive methods.

Therefore, the main accuracy conclusion is robust to measured prompt wording features.

## 9. Question-Count Analysis

### 9.1 Why Scene-Level Analysis Is Used

Question count is not analyzed with prompt-level paired Wilcoxon tests because prompts are not exactly matched across methods.

Instead, question counts are aggregated at:

```text
scene_id, method, prompt_type
```

Then methods are compared across scenes.

### 9.2 Interactive Methods Only

Only interactive methods are included:

| Method |
| --- |
| `first_question` |
| `random_question` |
| `vlm_best_question` |
| `proposed_efe` |

Non-interactive methods ask no questions by design, so including them would not answer the EFE efficiency question.

### 9.3 Tests Used

For each context, the script runs:

1. Friedman test across all four interactive methods.
2. Pairwise Wilcoxon signed-rank tests:
   - EFE vs first-question
   - EFE vs random-question
   - EFE vs VLM-best-question
3. Holm-Bonferroni correction across the three pairwise tests.
4. Bootstrap 95% CI for paired scene-level mean difference.

Contexts:

| Context | Meaning |
| --- | --- |
| `ambiguous` | Only ambiguous prompts. Main EFE-efficiency context. |
| `overall` | All prompt types. |
| `partial` | Only partial prompts. Interpreted cautiously. |

### 9.4 All-Trial Scene-Level Question Results

The all-trial analysis treats no-question trials as zero questions. This answers:

```text
How many questions does the method ask per trial on average?
```

Ambiguous prompts:

| Comparison | Mean Q EFE | Mean Q Baseline | Reduction | Holm p | Conclusion |
| --- | ---: | ---: | ---: | ---: | --- |
| EFE vs `first_question` | 1.97 | 2.61 | 24.37% | 0.000904 | Significant |
| EFE vs `random_question` | 1.97 | 2.58 | 23.33% | 0.000904 | Significant |
| EFE vs `vlm_best_question` | 1.97 | 2.57 | 23.18% | 0.000904 | Significant |

Overall:

| Comparison | Direction | Holm p | Interpretation |
| --- | --- | ---: | --- |
| EFE vs `first_question` | EFE lower | 0.0521 | Not significant after correction. |
| EFE vs `random_question` | EFE lower | 0.1028 | Not significant after correction. |
| EFE vs `vlm_best_question` | EFE lower | 0.1028 | Not significant after correction. |

Partial prompts:

| Comparison | Direction | Holm p | Interpretation |
| --- | --- | ---: | --- |
| EFE vs `first_question` | Similar | Not significant | No clear difference. |
| EFE vs `random_question` | Similar | Not significant | No clear difference. |
| EFE vs `vlm_best_question` | EFE higher | 0.0318 | EFE asks more than VLM-best on partial prompts. |

### 9.5 Asked-Only Question Analysis

The audit tables report average question count only among trials where at least one question was asked. To match this convention, the script also produces asked-only results.

This answers:

```text
When a method decides to ask questions, how many does it ask?
```

Asked-only trial-level descriptive means:

| Context | Method | Asked N | Mean Q Asked |
| --- | --- | ---: | ---: |
| Overall | `first_question` | 101 | 2.37 |
| Overall | `random_question` | 101 | 2.39 |
| Overall | `vlm_best_question` | 102 | 2.30 |
| Overall | `proposed_efe` | 104 | 1.91 |
| Ambiguous | `first_question` | 59 | 2.88 |
| Ambiguous | `random_question` | 57 | 2.94 |
| Ambiguous | `vlm_best_question` | 59 | 2.92 |
| Ambiguous | `proposed_efe` | 62 | 2.26 |
| Partial | `first_question` | 31 | 1.32 |
| Partial | `random_question` | 33 | 1.36 |
| Partial | `vlm_best_question` | 32 | 1.19 |
| Partial | `proposed_efe` | 38 | 1.32 |

Asked-only scene-level significance:

| Context | Comparison | Mean Q EFE | Mean Q Baseline | Reduction | Holm p | Conclusion |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Overall | EFE vs `first_question` | 1.90 | 2.36 | 19.45% | 0.001108 | Significant |
| Overall | EFE vs `random_question` | 1.84 | 2.30 | 19.71% | 0.001311 | Significant |
| Overall | EFE vs `vlm_best_question` | 1.90 | 2.25 | 15.47% | 0.004218 | Significant |
| Ambiguous | EFE vs `first_question` | 2.20 | 2.85 | 22.91% | 0.000182 | Significant |
| Ambiguous | EFE vs `random_question` | 2.16 | 2.83 | 23.88% | 0.000182 | Significant |
| Ambiguous | EFE vs `vlm_best_question` | 2.22 | 2.83 | 21.43% | 0.000555 | Significant |
| Partial | EFE vs `first_question` | Similar | Similar | Small | 0.815 | Not significant |
| Partial | EFE vs `random_question` | Similar | Similar | Small | 0.815 | Not significant |
| Partial | EFE vs `vlm_best_question` | Similar | Similar | Small | 0.539 | Not significant |

### 9.6 Question-Count Interpretation

The strongest and cleanest question-count claim is:

```text
EFE significantly reduces the number of clarification questions on ambiguous prompts.
```

This is the most important subset because ambiguity resolution is the core reason for asking clarification questions.

The asked-only analysis supports an additional human-readable statement:

```text
Among trials where questions were asked, EFE required fewer clarification rounds than all interactive baselines.
```

For partial prompts, the evidence is mixed. EFE does not consistently reduce questions on partial prompts, and in the all-trial scene-level analysis it asked more than VLM-best-question in one comparison. Therefore, the paper should not claim that EFE universally reduces questions for all prompt types.

## 10. Latency Analysis

### 10.1 Why Nonparametric Scene-Level Tests Are Used

Latency is long-tailed and not normally distributed. Prompt text is not exactly paired. Therefore, the script uses scene-level aggregation and Wilcoxon signed-rank tests.

The latency aggregation is:

```text
scene_id, method, prompt_type
```

with:

```text
mean_latency
median_latency
IQR latency
```

### 10.2 Latency Results

Ambiguous prompts:

| Comparison | Mean Latency EFE | Mean Latency Baseline | Direction | Holm p | Conclusion |
| --- | ---: | ---: | --- | ---: | --- |
| EFE vs `first_question` | 141.7 | 177.1 | EFE lower | 0.513 | Not significant |
| EFE vs `random_question` | 141.7 | 188.6 | EFE lower | 0.0724 | Not significant after correction |
| EFE vs `vlm_best_question` | 141.7 | 162.3 | EFE lower | 0.513 | Not significant |

Overall:

| Comparison | Direction | Conclusion |
| --- | --- | --- |
| EFE vs `first_question` | Similar/lower depending metric | Not significant |
| EFE vs `random_question` | Similar/lower depending metric | Not significant |
| EFE vs `vlm_best_question` | Similar | Not significant |

Partial prompts:

| Comparison | Direction | Holm p | Conclusion |
| --- | --- | ---: | --- |
| EFE vs `first_question` | EFE higher latency | 0.0101 | Significant in wrong direction |
| EFE vs `random_question` | EFE higher latency | Not consistently significant | Cautious |
| EFE vs `vlm_best_question` | EFE higher latency | 0.0349 | Significant in wrong direction |

### 10.3 Latency Interpretation

Latency should be reported cautiously. The data do not support a strong claim that EFE significantly reduces latency after correction.

Safe wording:

```text
EFE reduced the number of clarification questions on ambiguous prompts. Latency was long-tailed, and although EFE showed lower mean latency in some ambiguous-prompt comparisons, the corrected scene-level latency tests did not provide robust evidence for a latency reduction.
```

Unsafe wording:

```text
EFE significantly reduces latency.
```

This should not be used unless future results support it.

## 11. Prompt-Type Difficulty Analysis

### 11.1 Purpose

Prompt-type difficulty analysis is a supporting analysis. It tests whether the three prompt types behave like different difficulty regimes.

### 11.2 Prompt-Type Descriptives

| Prompt Type | N | Accuracy | Fail Rate | Asked Rate | Mean Q All Trials | Mean Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clear` | 503 | 96.02% | 3.98% | 1.79% | 0.0219 | 8.09 |
| `ambiguous` | 506 | 60.47% | 39.53% | 52.37% | 1.441 | 109.72 |
| `partial` | 503 | 69.38% | 30.62% | 26.64% | 0.346 | 46.60 |

### 11.3 Prompt-Type GEE Model

The prompt-type GEE model is:

```text
correct ~ C(prompt_type) + C(method) + C(scene_type)
```

with:

```text
family = Binomial
groups = scene_id
covariance = robust
```

### 11.4 Prompt-Type GEE Results

| Contrast | OR | Holm p | Interpretation |
| --- | ---: | ---: | --- |
| `ambiguous` vs `clear` | 0.0356 | 1.41e-23 | Ambiguous is much harder than clear. |
| `partial` vs `clear` | 0.0624 | 8.41e-14 | Partial is much harder than clear. |
| `ambiguous` vs `partial` | 0.5705 | 0.0705 | Ambiguous is numerically harder but not significant after correction. |

### 11.5 Prompt-Type Interpretation

The descriptive and model-based results support the following:

1. Clear prompts are high accuracy and rarely require questions.
2. Ambiguous prompts are much harder and frequently require clarification.
3. Partial prompts are intermediate but still challenging.
4. Ambiguous prompts are numerically harder than partial prompts, but the adjusted ambiguous-vs-partial contrast is not significant after Holm correction.

This analysis supports experimental design validity but is not the main contribution.

## 12. Prompt Pool and Prompt Wording Robustness

### 12.1 Why This Matters

Prompt pools differed by scene and prompt type. Some pools contained many unique prompts, while others had repeated prompts. High repetition or low prompt diversity can affect interpretation because methods might not receive exactly equal prompt wording difficulty.

### 12.2 Prompt Pool Diagnostics

Prompt pools were defined by:

```text
scene_id, prompt_type
```

For each pool, the script computed:

| Quantity | Meaning |
| --- | --- |
| `prompt_pool_n` | Number of records in the pool. |
| `prompt_pool_unique_n` | Number of unique normalized prompt texts. |
| `prompt_pool_repetition_rate` | Fraction of records that repeat an already-seen prompt. |
| `low_diversity_or_high_repeat` | Flag if unique prompts are fewer than 3 or repetition rate exceeds 0.50. |

### 12.3 Pool Summary

| Prompt Type | Pools | Flagged Pools | Mean Unique Prompts | Mean Duplicate Rate |
| --- | ---: | ---: | ---: | ---: |
| `ambiguous` | 36 | 36 | 1.72 | 0.871 |
| `clear` | 36 | 10 | 8.17 | 0.417 |
| `partial` | 36 | 27 | 5.11 | 0.654 |

Ambiguous prompts are highly repeated by design. This is not automatically a flaw, but it means that prompt-level exact pairing should not be assumed and prompt pool robustness should be checked.

### 12.4 Method Balance Within Ambiguous Prompts

Even though ambiguous pools are repetitive, method-level prompt-feature balance is close.

| Method | Ambiguous N | Unique Prompts | Duplicate Rate | `has_my` Mean | Mean Candidate Set Size |
| --- | ---: | ---: | ---: | ---: | ---: |
| `top_score` | 73 | 6 | 0.9178 | 0.986 | 4.45 |
| `random_candidate` | 73 | 6 | 0.9178 | 0.986 | 4.55 |
| `vlm_direct` | 72 | 9 | 0.8750 | 1.000 | 4.60 |
| `first_question` | 72 | 6 | 0.9167 | 1.000 | 4.47 |
| `random_question` | 72 | 7 | 0.9028 | 1.000 | 4.63 |
| `vlm_best_question` | 72 | 6 | 0.9167 | 1.000 | 4.57 |
| `proposed_efe` | 72 | 7 | 0.9028 | 1.000 | 4.47 |

### 12.5 Sensitivity Analysis

A strict sensitivity analysis removed low-diversity or high-repeat pools. This removed 73 pools and left:

| Quantity | Value |
| --- | ---: |
| Remaining records | 514 |
| Remaining scenes | 27 |
| Remaining pools | 35 |
| Removed pools | 73 |

Accuracy results remained directionally and statistically robust:

| Contrast | OR | Holm p | Conclusion |
| --- | ---: | ---: | --- |
| `proposed_efe` vs `top_score` | 6.14 | 0.00479 | Significant |
| `proposed_efe` vs `random_candidate` | 8.84 | 0.000446 | Significant |
| `proposed_efe` vs `vlm_direct` | 11.74 | 0.00479 | Significant |
| `vlm_best_question` vs `vlm_direct` | 34.09 | 0.00479 | Significant |
| `proposed_efe` vs interactive baselines | Near 1 | Not significant | Comparable |

However, this strict filter removes all complete paired ambiguous scenes for question-count sensitivity, because all ambiguous pools were flagged. Therefore, ambiguous question-count sensitivity under this filter is not estimable.

### 12.6 Robustness Interpretation

The prompt-feature adjusted model and the strict pool sensitivity check both support the main accuracy conclusion. The question-count claim should rely on the full cleaned scene-level analysis, with a transparent note that ambiguous prompts are intentionally repetitive and balanced across methods.

## 13. Supplementary Exact-Paired Analysis

### 13.1 Why It Is Supplementary

Exact-paired tests are only valid when the exact same prompt key is shared across methods. The current dataset does not have enough exact prompt-level matching for this to be the main analysis.

The exact-paired subset contains:

| Quantity | Value |
| --- | ---: |
| Exact prompt keys | 17 |
| Records | 119 |

### 13.2 Supplementary Interpretation

The exact-paired subset is underpowered and not representative. It can be reported as a supplemental check, but it should not replace the scene-clustered GEE and scene-level analyses.

Safe wording:

```text
An exact prompt-level paired subset was small, so prompt-level paired tests were treated as supplementary only. Main inference used scene-clustered and scene-level analyses.
```

## 14. Offline Failure Analysis

### 14.1 Purpose

Failure analysis is separate from significance testing. The statistical tests answer whether method-level effects are reliable; failure analysis answers where the system fails and what the remaining bottlenecks are.

The failure analysis uses the same cleaned included dataset as the significance tests:

```text
outputs/statistics/cleaned_trial_level_for_statistics.csv
```

Failure counts are computed over failed trials only:

```text
failed trial = wrong or unresolved
```

This gives:

```text
374 failed trials out of 1512 included trials
```

### 14.2 Failure Reason Taxonomy

The audited failure reasons are:

| Failure Reason | Meaning |
| --- | --- |
| `ambiguity_not_solved` | The method did not resolve ambiguity and selected or returned the wrong/insufficient target. |
| `gd_missing_target_candidate` | GroundingDINO did not include the true target candidate, so downstream methods could not select it. |
| `wrong_object_selected` | A candidate was available, but the final selected object was wrong. |
| `gd_label_error` | GroundingDINO produced an incorrect label. |
| `vlm_error` | VLM question/selection behavior failed, such as invalid or misleading question/answer handling. |
| `none` | The trial failed, but no more specific audited failure reason was assigned. This is not a success category. |

### 14.3 Overall Failure Reasons

| Failure Reason | Count | Fail Share | All-Trial Share |
| --- | ---: | ---: | ---: |
| `ambiguity_not_solved` | 200 | 53.48% | 13.23% |
| `gd_missing_target_candidate` | 79 | 21.12% | 5.22% |
| `wrong_object_selected` | 65 | 17.38% | 4.30% |
| `none` | 26 | 6.95% | 1.72% |
| `vlm_error` | 3 | 0.80% | 0.20% |
| `gd_label_error` | 1 | 0.27% | 0.07% |

The largest failure mode is unsolved ambiguity. This is expected for non-interactive methods and is exactly the failure mode that interactive clarification is designed to reduce.

The second-largest failure mode is missing target candidates from GroundingDINO. This is important because no downstream VLM or EFE policy can select a target that is not in the candidate list.

### 14.4 Failure Reason by Method

| Method | ambiguity_not_solved | gd_label_error | gd_missing_target_candidate | none | vlm_error | wrong_object_selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `top_score` | 69 | 1 | 13 | 5 | 0 | 12 |
| `random_candidate` | 61 | 0 | 15 | 11 | 0 | 14 |
| `vlm_direct` | 66 | 0 | 16 | 7 | 0 | 11 |
| `first_question` | 1 | 0 | 11 | 1 | 0 | 6 |
| `random_question` | 1 | 0 | 8 | 1 | 2 | 8 |
| `vlm_best_question` | 0 | 0 | 9 | 0 | 1 | 7 |
| `proposed_efe` | 2 | 0 | 7 | 1 | 0 | 7 |

Interpretation:

1. Non-interactive methods fail mostly because ambiguity is not solved.
2. Interactive methods almost eliminate ambiguity-not-solved failures.
3. After interaction succeeds, the remaining errors are mostly detector candidate misses and final wrong-object choices.
4. EFE has no audited VLM-error failures in the cleaned data.

### 14.5 Failure Reason by Prompt Type

| Prompt Type | ambiguity_not_solved | gd_label_error | gd_missing_target_candidate | none | vlm_error | wrong_object_selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `clear` | 7 | 1 | 0 | 2 | 0 | 10 |
| `ambiguous` | 156 | 0 | 28 | 12 | 1 | 3 |
| `partial` | 37 | 0 | 51 | 12 | 2 | 52 |

Interpretation:

1. Ambiguous failures are dominated by unsolved ambiguity.
2. Partial failures are split between missing detector candidates and wrong object selection.
3. Clear prompts rarely fail, and when they do, the failure is usually not ambiguity but candidate/selection edge cases.

### 14.6 Failure Reason by Scene Type

| Scene Type | ambiguity_not_solved | gd_label_error | gd_missing_target_candidate | none | vlm_error | wrong_object_selected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bottle_only` | 28 | 0 | 0 | 1 | 0 | 28 |
| `cup_only` | 71 | 1 | 35 | 11 | 2 | 12 |
| `mixed` | 40 | 0 | 15 | 9 | 1 | 9 |
| `utensil_only` | 61 | 0 | 29 | 5 | 0 | 16 |

Interpretation:

1. Cup-only and utensil-only scenes contribute many ambiguity-not-solved failures because these scenes often contain multiple visually similar objects.
2. Cup-only and utensil-only scenes also contain many GroundingDINO missing-candidate failures, consistent with detector sensitivity for prompts such as color-modified cups or simple utensil categories.
3. Bottle-only failures are more evenly split between ambiguity and wrong object selection, rather than missing candidates.

### 14.7 Top Failure Groups

| Method | Prompt Type | Failure Reason | Count |
| --- | --- | --- | ---: |
| `top_score` | `ambiguous` | `ambiguity_not_solved` | 52 |
| `vlm_direct` | `ambiguous` | `ambiguity_not_solved` | 51 |
| `random_candidate` | `ambiguous` | `ambiguity_not_solved` | 50 |
| `top_score` | `partial` | `ambiguity_not_solved` | 14 |
| `vlm_direct` | `partial` | `gd_missing_target_candidate` | 13 |
| `vlm_direct` | `partial` | `ambiguity_not_solved` | 12 |
| `top_score` | `partial` | `wrong_object_selected` | 11 |
| `random_candidate` | `partial` | `gd_missing_target_candidate` | 11 |
| `random_candidate` | `partial` | `wrong_object_selected` | 10 |
| `random_candidate` | `partial` | `ambiguity_not_solved` | 10 |

This table explains why interactive clarification has such a large accuracy effect: the dominant non-interactive failure cluster is ambiguous prompts where no clarification is available.

### 14.8 Offline Failure Takeaway

The offline failure analysis supports the same conclusion as the statistical tests:

```text
Interactive clarification primarily helps by converting ambiguity failures into solvable target-resolution decisions.
```

The next bottleneck after interactive clarification is not question selection itself. It is the perception candidate set:

```text
If GroundingDINO does not include the true target candidate, EFE and VLM selection cannot recover it.
```

This motivates the later online-policy work on better detector prompting, category fallback, and segmentation-assisted point-cloud cropping.

## 15. Important Code Implementation Details

### 15.1 Main Command

The current analysis can be reproduced with:

```bash
python scripts/run_significance_tests.py \
  --input services/a6000_web/artifacts/offline_experiments/top_cups_01/analysis/balanced_method_prompt_cell_filter_seed20260519.json \
  --out outputs/statistics \
  --bootstrap 10000
```

### 15.2 Script Responsibilities

The script `scripts/run_significance_tests.py` performs the following:

| Step | Functionality |
| --- | --- |
| Load data | Reads CSV, JSON, JSONL, or Parquet trial-level data. |
| Normalize schema | Maps project-specific columns to analysis columns. |
| Validate data | Checks total N, method counts, scene counts, prompt-type balance, and duplicates. |
| Add prompt features | Extracts prompt wording covariates. |
| Descriptive tables | Writes audit, method, prompt, and cross-tab tables. |
| Accuracy tests | Runs main and prompt-feature adjusted GEE contrasts. |
| Question tests | Runs scene-level Friedman and Wilcoxon tests. |
| Asked-only question tests | Recomputes question statistics only among asked trials. |
| Latency tests | Runs scene-level latency Wilcoxon tests. |
| Prompt difficulty | Runs prompt-type descriptives and GEE contrasts. |
| Prompt balance | Writes prompt-feature and prompt-pool balance checks. |
| Sensitivity | Re-runs selected tests after removing low-diversity/high-repeat pools. |
| Supplementary exact-paired | Runs optional McNemar/Wilcoxon tests on exact prompt matches only. |
| Figures | Saves publication-resolution matplotlib figures. |
| Report | Writes `significance_report.md`. |

### 15.3 Column Normalization Logic

The script searches for equivalent column names and maps them to internal names. Conceptually:

```python
mapping = {
    "scene_id": scene_col,
    "method": method_col,
    "prompt_type": prompt_type_col,
    "correct": correct_col,
    "num_questions": question_col,
    "latency_sec": latency_col,
    "scene_type": scene_type_col,
    "candidate_set_size": candidate_col,
    "prompt_text": prompt_text_col,
}
```

If a required column cannot be found, the script stops and reports the missing field.

### 15.4 Correctness Encoding

The binary accuracy variable treats unresolved as failure:

```python
df["correct"] = df["outcome"].map(lambda x: 1 if x == "correct" else 0)
```

This is appropriate because a real robot pipeline needs a resolved target. Wrong and unresolved outcomes both mean the target-resolution module did not provide a successful target.

### 15.5 Prompt Feature Extraction

Prompt features are generated from normalized lowercase prompt text.

Representative implementation pattern:

```python
def add_prompt_features(df):
    text = df["prompt_text"].fillna("").astype(str).str.lower()
    df["prompt_length"] = text.str.split().map(len)
    df["has_my"] = text.str.contains(r"\bmy\b").astype(int)
    df["has_color"] = text.str.contains(color_pattern).astype(int)
    df["has_spatial"] = text.str.contains(spatial_pattern).astype(int)
    df["has_relation"] = text.str.contains(relation_pattern).astype(int)
    df["has_size"] = text.str.contains(size_pattern).astype(int)
    df["prompt_object_category"] = text.map(infer_object_category)
    return df
```

These features are not used to change labels. They are only used as covariates or balance diagnostics.

### 15.6 Main GEE Implementation

The main accuracy model is implemented with statsmodels GEE:

```python
model = smf.gee(
    formula=accuracy_formula,
    groups=df["scene_id"],
    data=df,
    family=sm.families.Binomial(),
)
result = model.fit()
```

Method contrasts are then computed from the fitted coefficient vector and covariance matrix. Odds ratios are computed by exponentiating log-odds differences:

```python
odds_ratio = np.exp(log_odds_difference)
```

### 15.7 Holm-Bonferroni Correction

Holm correction is applied inside each planned family:

```python
def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted
```

This controls family-wise error more powerfully than Bonferroni while maintaining conservative inference.

### 15.8 Scene-Level Question Aggregation

Question-count tests aggregate before testing:

```python
agg = (
    df[df["method"].isin(interactive_methods)]
    .groupby(["scene_id", "method", "prompt_type"])
    .agg(
        mean_num_questions=("num_questions", "mean"),
        median_num_questions=("num_questions", "median"),
        n_trials=("num_questions", "size"),
    )
    .reset_index()
)
```

The paired comparison is then performed across scenes, not raw prompts.

### 15.9 Wilcoxon Signed-Rank Tests

Pairwise scene-level tests use:

```python
stat, p = scipy.stats.wilcoxon(
    efe_values,
    baseline_values,
    zero_method="wilcox",
    alternative="two-sided",
)
```

The two-sided test is conservative. If the paper later pre-registers a one-sided efficiency hypothesis, one-sided tests could be reported separately, but the current report uses two-sided tests.

### 15.10 Bootstrap Confidence Intervals

Bootstrap CIs are computed by resampling scene-level paired differences:

```python
diffs = efe_values - baseline_values
samples = rng.choice(diffs, size=(n_bootstrap, len(diffs)), replace=True)
boot_means = samples.mean(axis=1)
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
```

This respects the scene-level comparison unit.

## 16. Output Files

### 16.1 Main Reports

| File | Purpose |
| --- | --- |
| `outputs/statistics/significance_report.md` | Main concise statistical report. |
| `outputs/statistics/offline_significance_detailed_methods_report.md` | This detailed methods and results report. |
| `outputs/statistics/robustness_old_vs_new_summary.md` | Comparison between old and updated analysis variants. |

### 16.2 Core CSV Outputs

| File | Purpose |
| --- | --- |
| `outputs/statistics/dataset_audit.csv` | Dataset audit and validation. |
| `outputs/statistics/balance_checks.csv` | Balance checks by method, prompt type, scene, and available covariates. |
| `outputs/statistics/accuracy_gee_contrasts.csv` | Main GEE accuracy contrasts. |
| `outputs/statistics/accuracy_prompt_feature_gee_contrasts.csv` | Prompt-feature adjusted GEE contrasts. |
| `outputs/statistics/question_count_scene_level_tests.csv` | Main scene-level question-count tests. |
| `outputs/statistics/question_count_asked_only_descriptives.csv` | Asked-only question descriptive statistics. |
| `outputs/statistics/question_count_asked_only_scene_level_tests.csv` | Asked-only scene-level question tests. |
| `outputs/statistics/latency_scene_level_tests.csv` | Scene-level latency tests. |
| `outputs/statistics/prompt_type_descriptives.csv` | Prompt-type descriptive statistics. |
| `outputs/statistics/prompt_type_gee_tests.csv` | Prompt-type GEE contrasts. |
| `outputs/statistics/prompt_feature_balance_by_method_prompt_type.csv` | Prompt-feature balance diagnostics. |
| `outputs/statistics/prompt_pool_diversity_by_scene_prompt_type.csv` | Prompt-pool diversity diagnostics. |
| `outputs/statistics/prompt_pool_sensitivity_summary.csv` | Summary of strict prompt-pool sensitivity filtering. |
| `outputs/statistics/sensitivity_accuracy_prompt_feature_gee_contrasts.csv` | Accuracy sensitivity after removing flagged prompt pools. |
| `outputs/statistics/supplementary_exact_paired_tests.csv` | Supplementary exact-paired tests, not main analysis. |

### 16.3 Figures

| File | Purpose |
| --- | --- |
| `outputs/statistics/fig_accuracy_by_method.png` | Accuracy by method. |
| `outputs/statistics/fig_accuracy_method_prompt_type.png` | Accuracy by method and prompt type. |
| `outputs/statistics/fig_scene_level_questions_ambiguous.png` | Scene-level ambiguous question counts. |
| `outputs/statistics/fig_accuracy_vs_questions.png` | Accuracy vs mean question count. |
| `outputs/statistics/fig_latency_interactive.png` | Interactive-method latency distribution. |

## 17. Paper-Ready Interpretation

The following wording is statistically safe:

```text
Because prompt wording was randomized within scene and prompt type rather than exactly matched across methods, statistical comparisons used scene-clustered and scene-level analyses. Accuracy was analyzed using a scene-clustered logistic GEE model, while clarification count and latency were analyzed using scene-level aggregates.
```

```text
Interactive clarification substantially improved target-resolution accuracy compared with non-interactive baselines. Proposed EFE significantly outperformed top-score detector selection, random candidate selection, and direct VLM target selection after Holm correction.
```

```text
Direct VLM target selection was insufficient under ambiguous referential commands. A VLM interactive clarification baseline significantly outperformed direct VLM selection, indicating that interaction rather than direct one-shot VLM selection is important for this task.
```

```text
EFE maintained comparable target-resolution accuracy to other interactive clarification baselines. EFE was not significantly better or worse than first-question, random-question, or VLM-best-question baselines in the adjusted accuracy model.
```

```text
EFE significantly reduced the number of clarification questions on ambiguous prompts. In asked-only scene-level analyses, EFE also required fewer clarification rounds overall and on ambiguous prompts.
```

```text
Latency was long-tailed. Although EFE showed lower observed mean latency in some ambiguous-prompt comparisons, corrected scene-level latency tests did not provide robust support for a latency-reduction claim.
```

```text
Prompt-type analyses confirmed that clear prompts were easy and rarely triggered questions, whereas ambiguous and partial prompts were substantially more difficult.
```

## 18. Claims to Avoid

The following claims are not supported by the current analysis and should be avoided:

| Unsafe Claim | Why It Should Be Avoided |
| --- | --- |
| EFE significantly improves accuracy over all interactive methods. | EFE vs interactive baselines was not significant in GEE. |
| EFE significantly reduces latency. | Corrected latency tests did not robustly support this. |
| Prompt-level paired tests are the main analysis. | Exact prompts are not shared across all methods. |
| Ambiguous prompts are significantly harder than partial prompts. | Ambiguous is numerically harder, but adjusted Holm p is not significant. |
| Prompt wording imbalance is irrelevant. | It is addressed by covariates and sensitivity checks, not ignored. |

## 19. Reviewer-Facing Justifications

### 18.1 Why GEE Instead of McNemar?

McNemar's test requires exact paired binary outcomes for the same task-prompt pair. In this experiment, methods are balanced by scene and prompt type, but exact prompt wording differs across methods. Therefore, McNemar's test would incorrectly assume stronger pairing than the design provides.

GEE is appropriate because it models binary correctness while accounting for repeated observations clustered within scenes.

### 18.2 Why Aggregate Questions by Scene?

Question count depends on prompt wording. Since prompt wording is not exactly matched, raw prompt-level Wilcoxon tests would compare non-identical prompts. Aggregating by scene, method, and prompt type respects the experimental balance and compares methods at the shared scene level.

### 18.3 Why Use Prompt Features?

Prompt features help control observed difficulty differences from wording. They do not solve all possible unobserved prompt differences, but they reduce the concern that method effects are caused by obvious prompt attributes such as color, spatial terms, relation terms, size terms, object category, or prompt length.

### 18.4 Why Still Trust the Main Result?

The main accuracy conclusion is robust across:

1. Descriptive method-level accuracy.
2. Scene-clustered GEE.
3. Prompt-feature adjusted GEE.
4. Strict prompt-pool sensitivity analysis.

The same qualitative pattern appears in all analyses: interactive methods are much better than non-interactive methods, and EFE is comparable to other interactive methods in accuracy.

### 18.5 What Is the Main Contribution?

The strongest contribution is not that EFE is more accurate than every interactive method. The strongest supported contribution is:

```text
EFE preserves the high accuracy of interactive clarification while reducing the number of clarification questions, especially under ambiguity.
```

## 20. Known Warnings and Caveats

### 19.1 GEE Numerical Warnings

During fitting, statsmodels may emit warnings such as:

```text
RuntimeWarning: invalid value encountered in sqrt
```

This can occur when robust covariance estimates for some nuisance covariates are unstable. The method contrasts used in the report are finite and interpretable. The warning should be noted if exact reproducibility logs are inspected.

### 19.2 Prompt Pool Sensitivity Limitation

The strict low-diversity/high-repeat pool filter removes all complete ambiguous prompt pools, so ambiguous question-count sensitivity cannot be estimated under that strict filter. This is expected because ambiguous prompts are deliberately repetitive within many scenes.

### 19.3 Asked-Only Versus All-Trial Question Count

There are two valid question-count summaries:

| Summary | Meaning |
| --- | --- |
| All-trial mean Q | Average questions per trial, with no-question trials counted as zero. |
| Asked-only mean Q | Average questions among trials where at least one question was asked. |

The audit tables use asked-only means. The scene-level statistical tests report both all-trial and asked-only variants. For paper text, clearly state which one is being reported.

### 19.4 Prompt Wording Is Not Fully Randomized Uniformly

Prompt pools differ in size and repetition rate. The analysis addresses this with prompt-feature covariates and sensitivity checks, but future experiments could improve design by enforcing equal prompt pool sizes or exact prompt matching across methods.

## 21. Recommendations for Paper Reporting

### 20.1 Main Results Table

For the main paper, include:

1. Accuracy by method.
2. Accuracy by method and prompt type.
3. Asked-only mean question count for interactive methods.
4. Ambiguous-prompt question-count reductions.
5. GEE contrast table for accuracy.
6. Scene-level Wilcoxon table for question count.

### 20.2 Statistical Methods Paragraph

Suggested methods paragraph:

```text
Because prompt wording was randomized within scene and prompt type and was not exactly identical across methods, we did not use prompt-level paired tests as the primary analysis. Instead, binary target-resolution accuracy was analyzed using logistic generalized estimating equations with scene-level clustering and robust covariance. The model included method, prompt type, scene type, and candidate set size as covariates. A second robustness model additionally controlled for prompt-level features including prompt length, ownership terms, color terms, spatial terms, relation terms, size terms, and object category. Clarification question count and latency were analyzed after aggregation at the scene-method-prompt-type level using Friedman tests and Wilcoxon signed-rank tests with Holm-Bonferroni correction. Bootstrap confidence intervals used scene-level resampling with seed 20260520.
```

### 20.3 Results Paragraph

Suggested results paragraph:

```text
Interactive clarification methods achieved approximately 91-92% target-resolution accuracy, compared with approximately 53-54% for non-interactive baselines. In the scene-clustered logistic GEE, proposed EFE significantly outperformed top-score selection, random candidate selection, and direct VLM selection after Holm correction. VLM-best interactive clarification also significantly outperformed direct VLM selection, indicating that direct VLM target selection is insufficient under ambiguous referential commands. EFE did not significantly differ in accuracy from the other interactive baselines, supporting comparable accuracy. On ambiguous prompts, EFE significantly reduced the number of clarification questions compared with first-question, random-question, and VLM-best-question baselines.
```

### 20.4 Limitations Paragraph

Suggested limitations paragraph:

```text
The offline experiment was balanced by scene, method, and prompt type, but exact prompt wording was not always identical across methods. To address this, the primary inference used scene-clustered and scene-level analyses rather than prompt-level paired tests. We further included prompt-feature covariates and prompt-pool sensitivity checks. Future experiments could enforce exact prompt matching or equal prompt-pool sizes across methods to enable stronger paired analyses.
```

## 22. Final Conclusions

The cleaned offline experiment provides strong evidence that interactive clarification improves target-resolution accuracy over non-interactive baselines. Direct VLM target selection is insufficient under ambiguity. Proposed EFE maintains the high accuracy of other interactive methods and significantly reduces clarification questions on ambiguous prompts.

The most defensible contribution statement is:

```text
EFE improves clarification efficiency while preserving target-resolution accuracy.
```

The current data do not justify claiming that EFE is significantly more accurate than other interactive methods or that EFE significantly reduces latency.
