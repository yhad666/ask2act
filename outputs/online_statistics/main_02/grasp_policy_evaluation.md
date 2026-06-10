# Online Grasping Policy Evaluation

This report evaluates the current real-robot grasping policy using the cleaned `main_02` online experiment and the current code path.

Primary data file:

```text
outputs/online_statistics/main_02/cleaned_online_main02_for_statistics.csv
```

Relevant implementation files:

```text
services/a6000_web/grasp_runtime.py
simulation/ask2act_grasp/planning/motion_planner.py
real/stretch_transport/scripts/dispatch_grasp.py
```

## 1. Main Conclusion

The online experiment shows that the target-resolution layer is the main driver of method-level end-to-end performance, but the grasping policy remains the main residual bottleneck after a target is correctly selected.

The safety policy is working correctly:

```text
If the operator marks the selected target as wrong, the robot does not grasp.
```

This makes the online wrong-object grasp rate:

```text
0 / 228 = 0.00%
```

However, skipped wrong-target trials still count as task failures, which is correct for the paper because the full system did not complete the command.

## 2. Online Evidence

### 2.1 By Method

| Method | Target Correct | Target Acc | Grasp Attempted | Physical Success | Physical Success Given Attempt | Correct-Object Success | Task Success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `proposed_efe` | 53/57 | 92.98% | 53 | 50 | 94.34% | 50 | 87.72% |
| `vlm_best_question` | 52/57 | 91.23% | 52 | 46 | 88.46% | 46 | 80.70% |
| `top_score` | 34/57 | 59.65% | 34 | 30 | 88.24% | 30 | 52.63% |
| `random_candidate` | 35/57 | 61.40% | 35 | 29 | 82.86% | 29 | 50.88% |

Interpretation:

1. EFE creates more correct grasp attempts because target selection is much better.
2. EFE also has the highest physical success rate conditional on attempted grasps, but this should be interpreted cautiously because the attempted-object distribution differs by method.
3. Non-interactive methods are not mainly failing because the gripper cannot grasp; they fail because the wrong target is selected and execution is skipped.

### 2.2 By Object Category for Attempted Grasps

| Object Category | Attempted | Physical Success | Physical Fail | Physical Success Rate |
| --- | ---: | ---: | ---: | ---: |
| cup | 59 | 57 | 2 | 96.61% |
| bottle | 61 | 52 | 9 | 85.25% |
| fork | 26 | 21 | 5 | 80.77% |
| spoon | 28 | 25 | 3 | 89.29% |

Interpretation:

1. Cup grasping is already strong.
2. Bottles are weaker, mainly because tall objects stress top-down height selection and collision clearance.
3. Forks are the weakest utensil category. This matches the observed problem that narrow objects need reliable wrist yaw alignment and sufficient gripper closure.
4. Spoons are better than forks but still not as stable as cups.

### 2.3 By Scene Type for Attempted Grasps

| Scene Type | Attempted | Physical Success | Physical Success Rate |
| --- | ---: | ---: | ---: |
| cup_only | 42 | 40 | 95.24% |
| mixed | 37 | 35 | 94.59% |
| utensil_only | 41 | 35 | 85.37% |
| bottle_only | 54 | 45 | 83.33% |

Interpretation:

1. Cup-only and mixed scenes are physically robust once the correct object is selected.
2. Bottle-only and utensil-only scenes are the main physical grasping weaknesses.
3. This is consistent with the pilot observations: bottles expose vertical/top-down clearance issues; forks and spoons expose narrow-object orientation and closure issues.

## 3. Current Policy Mechanics

### 3.1 Perception and Point-Cloud Crop

The real pipeline starts from the selected detection candidate and produces a target point cloud.

Current behavior:

1. The selected candidate bbox is used as the initial target crop.
2. If SAM is enabled, Segment Anything predicts a mask from that bbox.
3. The point cloud is generated from the mask if the mask has enough valid depth points.
4. If the mask produces too few points, the system falls back to bbox-based point-cloud cropping.
5. The chosen target point cloud is saved for diagnostics.

Important code path:

```text
services/a6000_web/grasp_runtime.py
```

Relevant behavior:

```text
_predict_sam_mask_for_bbox(...)
PointCloudGenerator.generate(... target_mask_2d=target_mask_2d)
fallback_to_bbox if mask_point_cloud_too_small
```

Policy evaluation:

1. This is the right direction because bbox-only cropping can include nearby background objects, batteries, handles, or adjacent utensils.
2. The fallback is necessary for robustness, but it can reintroduce contamination when SAM fails.
3. For grasping, VLM should continue to receive the original image; segmentation should only affect point-cloud crop and grasp pose estimation.

### 3.2 Geometric Grasp Estimation

The geometric grasp is computed from target points:

```text
geometric_grasp = _compute_geometric_grasp(point_cloud.world_points_xyz, ...)
```

Fallback behavior estimates:

1. object top and bottom using z percentiles.
2. grasp z from center or capped distance below the top.
3. xy center from median point location.
4. minor axis from PCA for narrow-object orientation.
5. open width from cross-section width plus clearance.

Important current formula for top-down height:

```text
grasp_z = max(center_z, object_top_z - max_top_grasp_delta_m)
```

when `max_top_grasp_delta_m > 0`.

Current interpretation:

1. This fixes tall-bottle collisions better than pure center-height grasping.
2. For very tall bottles, the gripper contact point is prevented from being too far below the top.
3. The policy still depends heavily on the point cloud being a clean target-only cloud.

### 3.3 Top-Down Motion Planning

The selected grasp is executed as a top-down plan:

```text
preferred_approach = "top_down"
approach_type = "top_down"
```

The planner computes:

1. base rotation.
2. lift height.
3. arm extension.
4. wrist yaw.
5. gripper open and close commands.
6. postgrasp lift.
7. arm retraction and base return.

Current waypoint structure:

```text
retract_and_tuck
move_lift_to_pregrasp
orient_wrist
open_gripper
move_to_pregrasp
open_gripper_at_pregrasp
descend_to_grasp
close_gripper
secure_grasp
postgrasp_lift
retract_arm_after_grasp
return_base_rotate_after_grasp
```

Policy evaluation:

1. The safety and sequencing are reasonable.
2. The policy is conservative enough to avoid wrong-object grasps.
3. It still has too many object-specific geometric edge cases for utensils and tall bottles.

## 4. Current Compensation Parameters

The current policy includes tunable corrections:

| Parameter Type | Purpose |
| --- | --- |
| Rubber local X/Y/Z correction | Compensates the offset between planned contact and actual rubber finger contact. |
| Approx X/Y correction | Used by approximate top-down fallback when SimpleIK is unavailable. |
| Side X bias | Pushes grasp target outward for left/right objects to compensate side approach error. |
| Right extra X bias | Adds extra correction for right-side objects. |
| Left/center Y and right Y bias | Compensates front/back error depending on side region. |
| Slender-specific corrections | Separate correction set for forks/spoons or high-aspect-ratio objects. |
| Max top delta | Prevents the grasp point from being too far below a tall object's top. |
| Open/close gripper commands | Controls full-open and stronger close behavior. |

This is practical for fast robot experimentation, but it also means the policy is still calibration-heavy.

## 5. What Is Working

### 5.1 Safety Gate

The operator-confirmed target gate is working well:

```text
wrong_object_grasp = 0
wrong_target_prevented = 54
```

This is exactly what we want for online experiments with fragile hardware.

### 5.2 Cup Grasps

Cup grasping is strong:

```text
57 / 59 = 96.61% physical success
```

This means the basic top-down gripper policy, head-camera calibration, and table-height estimate are good enough for cup-like objects.

### 5.3 EFE Transfer to Physical Pipeline

EFE improves full task success mainly by creating more correct target selections:

```text
proposed_efe task success = 50 / 57 = 87.72%
top_score task success = 30 / 57 = 52.63%
random_candidate task success = 29 / 57 = 50.88%
```

So the offline target-resolution advantage does transfer into online correct-object grasp success.

## 6. What Is Still Weak

### 6.1 Bottles

Bottle physical success is:

```text
52 / 61 = 85.25%
```

Likely causes:

1. Tall bottles make pure center-height top-down grasping risky.
2. If the point cloud includes background or the bottle shoulder/rim unevenly, the estimated center can shift.
3. Top-down approach can collide with the top or side if the contact point is too low or too far back.

Current mitigation already implemented:

```text
grasp_z >= object_top_z - max_top_grasp_delta_m
```

Remaining recommendation:

1. Keep max-top-delta logic.
2. Prefer SAM mask point cloud when valid.
3. Consider bottle-specific side or upper-body grasp modes if bottle failures persist.

### 6.2 Forks

Fork physical success is:

```text
21 / 26 = 80.77%
```

Likely causes:

1. Forks are thin, so small xy error matters.
2. If wrist yaw does not align with the narrow grasp axis, the gripper closes along the wrong direction.
3. Forks can slip or fly out if the gripper closes with lateral pressure instead of pinching across the narrow dimension.
4. The effective gripper fingertip position changes with opening/closing, making table-near grasps sensitive.

Current code attempts to use geometric yaw when:

```text
requested_open_width < wrist_yaw_open_width_threshold
```

or when:

```text
xy_aspect_ratio >= slender_aspect_ratio
```

Remaining concern:

If the underlying geometric grasp does not mark the object as slender, or if the estimated `grip_angle_rad` is noisy, the wrist yaw may still not rotate in the physically desired way.

Recommendation:

1. Log and inspect `use_geometric_wrist_yaw`, `wrist_yaw_reason`, `grip_angle_rad`, and `total_gripper_yaw_rad` for failed utensil trials.
2. For fork/spoon categories, force wrist yaw from the mask PCA axis even when open width is not below threshold.
3. Add utensil-specific grasp point bias toward the handle center rather than the fork head/tines.
4. Keep strong close command but do not verify final full closure, because an object can block closure.

### 6.3 Spoons

Spoon physical success is:

```text
25 / 28 = 89.29%
```

Spoons are better than forks but still sensitive to orientation. The bowl/head can bias the point-cloud center, so grasping near the center of the entire mask may not always be best.

Recommendation:

1. For spoons, prefer handle-region grasp if the mask/PCA can separate long axis endpoints.
2. Use wrist yaw aligned across the narrow axis, not cup-like default yaw.

## 7. Policy-Level Recommendations

### 7.1 Highest Priority

The highest priority is to make segmentation-based point-cloud cropping the normal path:

```text
GroundingDINO bbox -> SAM mask -> target-only point cloud -> geometric grasp
```

This directly addresses the earlier observed root cause: bbox crops can include nearby batteries, handles, or adjacent clutter, shifting the grasp center backward or sideways.

### 7.2 Utensil-Specific Grasping

Fork/spoon should not simply be treated as small cups.

Recommended policy:

1. Use SAM mask PCA to estimate long axis.
2. Choose wrist yaw so the gripper closes across the short axis.
3. Choose the contact point near the handle/body stable region, not necessarily the full-mask centroid.
4. Use slender-specific rubber corrections and side biases.
5. Use stronger close command, but keep close verification disabled.

### 7.3 Bottle-Specific Grasping

Recommended policy:

1. Continue using `max_top_grasp_delta_m`.
2. Avoid grasp points too far below the top for tall bottles.
3. If top-down remains unstable, add a bottle-specific side grasp or upper-body grasp option.

### 7.4 Calibration Discipline

The UI tuning controls are useful, but the policy should not become only a collection of manual offsets.

Recommended next step:

1. Keep the current explicit corrections for online experimentation.
2. Record the final chosen values in each trial.
3. Use the logs to determine whether errors are systematic by object category, side region, or head pose.
4. Convert stable offsets into category-aware defaults.

## 8. Paper-Ready Summary

Safe wording:

```text
In the online experiment, wrong target selections were not executed for hardware safety. As a result, wrong-object grasp rate was zero, while wrong target selections were counted as task failures. EFE achieved the highest target-selection accuracy and correct-object grasp success. Conditional physical grasp success was high for cups but lower for bottles and utensils, indicating that target resolution is the main method-level driver while object-specific grasp execution remains a residual bottleneck.
```

Do not overclaim:

```text
The current grasping policy is solved for all object categories.
```

Better claim:

```text
The current grasping policy is sufficient to evaluate the target-resolution contribution online, but remaining physical failures show that bottle and utensil grasping can be further improved with segmentation-aware and object-category-specific grasp planning.
```

