# Stretch MuJoCo 机器人运动仿真验证

更新时间：2026-04-05

## 脚本与日志

脚本：

- `/home/yhad/robot/sim_grasping/motion_validation.py`
- `/home/yhad/robot/scripts/run_sim_motion_validation.sh`

日志：

- `/home/yhad/robot/logs/sim_validation/motion/latest/motion_summary.json`

## 本次验证的动作

已执行：

1. `home`
2. `stow`
3. `base` 前进
4. `base` 后退
5. `base` 原地转向
6. `lift` 上下
7. `arm` 伸缩
8. `head_pan / head_tilt`
9. `wrist_yaw / wrist_pitch / wrist_roll`
10. `gripper` 开合
11. 一个简单 `pre-grasp` 风格模板动作

## 动作结果

### 1. `home / stow`

- `已验证`：接口可用
- `home`、`stow` 都能执行，且状态会更新到对应附近位姿

### 2. base 前进 / 后退 / 转向

记录值：

- 前进 `v_linear=0.15, duration=1.5s`，实际 `delta_x ~= +0.038 m`
- 后退 `v_linear=-0.15, duration=1.5s`，实际 `delta_x ~= -0.038 m`
- 转向 `omega=0.45, duration=1.5s`，实际 `delta_theta ~= +0.093 rad`

结论：

- `已验证`：base 速度控制接口可用
- `已验证`：base 姿态会按命令变化
- `推断`：后续不要只靠“给一个 wall-clock 时长”来走位，应该闭环看 `base_pose`

### 3. lift / arm / head / wrist / gripper

在收紧动作容差后，本次记录为：

- `lift`：误差约 `0.03 m`
- `arm`：误差约 `0.02 m`
- `head_pan / head_tilt`：误差约 `0.02 rad`
- `wrist_yaw / wrist_pitch / wrist_roll`：误差约 `0.02 ~ 0.03 rad`
- `gripper`：误差约 `0.0095`

结论：

- `已验证`：这些接口都可用
- `已验证`：gripper 在更严格容差下仍能稳定开合
- `已验证`：没有出现 NaN、爆炸、明显抖动失控

## pre-grasp 风格模板动作

本次模板顺序：

1. 调到一个安全的 pre-grasp 关节组合
2. 记录“目标上方”位姿
3. `lift` 下压一点
4. `lift` 再抬起
5. `arm` 回撤

`motion_summary.json` 中，这个模板动作包含 12 个子步骤。

## 哪些动作接口最适合作为后续抓取 primitive

优先推荐直接复用：

1. `home`
2. `stow`
3. `set_base_velocity(...)` + `base_pose` 闭环
4. `move_to(lift, ...)`
5. `move_to(arm, ...)`
6. `move_to(wrist_pitch / wrist_yaw / wrist_roll, ...)`
7. `move_to(gripper, ...)`

建议的 primitive 组合：

- `approach_base`
- `raise_to_pregrasp_height`
- `extend_arm`
- `set_wrist_for_grasp`
- `open_gripper`
- `descend_small_delta`
- `close_gripper`
- `lift_after_grasp`
- `retract_arm`
- `return_home_or_stow`

## 是否存在明显不稳定 / 抖动 / 不连续

本次观察结果：

- `已验证`：没有出现明显仿真爆炸
- `已验证`：轨迹整体连续
- `已验证`：关节能稳定到达目标附近
- `推断`：base 速度控制更像“可用但需闭环”的 primitive，不建议依赖纯时间积分

## 对后续抓取 primitive 的建议

最小可行抓取 primitive，建议先做成 joint-space 版本：

1. `home`
2. `base coarse align`
3. `lift to height`
4. `arm extend`
5. `wrist pitch/yaw align`
6. `gripper open`
7. `lift descend`
8. `gripper close`
9. `lift ascend`
10. `arm retract`

原因：

- 这些接口已经验证过
- 不依赖额外 IK 或复杂任务规划
- 最适合先和 RGB/depth 感知结果做最小闭环

## 结论

当前 Stretch MuJoCo 的运动仿真能力已经足够支撑“primitive 级”的后续抓取开发。

如果下一步要接 grasp pipeline，建议先不要追求复杂轨迹，而是：

- 用 `base + lift + arm + wrist + gripper` 拼出一个保守的 pre-grasp / grasp / retreat 模板
- 让感知模块只负责给出一个桌面局部目标
- 先把“能到、能闭环、能抬起”跑通
