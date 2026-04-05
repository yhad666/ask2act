# Stretch MuJoCo 面向抓取的下一步总结

更新时间：2026-04-05

## 1. 当前 Stretch MuJoCo 仿真是否已经足够支撑后续抓取算法开发

结论：已经足够支撑“第一阶段抓取开发”。

这里的“第一阶段”指：

- 感知输入接入
- RGB / depth / K / 状态数据整理
- 简单桌面场景
- 基础抓取 primitive
- pre-grasp / descend / close / lift / retract 这类最小闭环

本次已经验证到的能力：

- GUI 可启动
- headless 可运行
- RGB / depth 可保存
- `K_saved_image` 已可直接用于保存图像
- joint / base / ee / sensor 状态可保存
- 机器人基础动作接口可用
- 简单 tabletop 场景可复现

## 2. 现有最适合的场景是什么

现阶段最适合的是：

1. `/home/yhad/robot/sim_grasping/tabletop_minimal_scene.xml`
2. `/home/yhad/robot/stretch_mujoco/stretch_mujoco/models/scene.xml`

二者分工建议：

- `tabletop_minimal_scene.xml`
  - 作为抓取前置联调主场景
  - 更适合 wrist depth / 点云 / primitive
- 官方 `scene.xml`
  - 作为最稳定的回退基线
  - 用来验证“是不是算法改坏了”，而不是场景改坏了

## 3. 是否建议先用简单 tabletop 场景，而不是复杂厨房场景

结论：强烈建议先用简单 tabletop。

原因：

- 复杂厨房场景引入的变量太多
- 当前 `robocasa` 相关依赖在你的现有环境里并没有装好
- 你当前优先目标是“仿真验收与场景准备”，不是任务级 benchmark

推荐节奏：

1. 先在简单 tabletop 跑通感知输入
2. 再跑通最小抓取 primitive
3. 最后再迁移到 Robocasa 厨房

## 4. 后续接 grasp pipeline 前还差哪些验证

还差但范围不大的验证：

1. 图像与状态的严格同步
   - 当前已经能分别拿到，但还可以再做更明确的同帧绑定
2. wrist 相机坐标系到 grasp target 的最小几何链路
   - `depth + K_saved_image + ee_pose_4x4`
3. 一个真正以“桌面目标点”为输入的 primitive
   - 不是纯脚本预设动作，而是基于目标点的 approach / descend
4. 简单接触成功判据
   - 例如 gripper 闭合量、lift 后物体是否随动
5. tabletop 场景下的 GUI 人工检查
   - 当前已验证可启动，但还值得你在 viewer 中亲眼看一次物体布局是否正好符合你的预期

## 5. 我建议的下一个最小可行步骤

建议你下一步做这个最小闭环：

1. 固定使用 `/home/yhad/robot/sim_grasping/tabletop_minimal_scene.xml`
2. 读取 `wrist_d405_rgb + wrist_d405_depth + cam_d405_K_saved_image`
3. 在图像里先手工指定一个桌面目标点
4. 把这个目标点变成 wrist 相机系下的 3D 点
5. 写一个非常保守的 joint-space pre-grasp primitive
6. 验证“能否稳定靠近该点并做一次开合”

为什么这是下一步最优：

- 不依赖大模型感知栈
- 不依赖复杂场景
- 最快暴露“感知坐标系”和“动作 primitive”之间真正的接口问题

## 最终一句话结论

Stretch MuJoCo 在你当前环境里已经从“能跑”进入到“可作为抓取开发底座”的阶段了。

现在最该做的不是继续扩环境，而是：

- 锁定 simple tabletop
- 锁定 wrist depth 主输入
- 锁定一组保守 primitive
- 尽快跑通一个最小抓取闭环
