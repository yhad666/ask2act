# Stretch MuJoCo 基础仿真验收报告

更新时间：2026-04-05

## 验收范围

本次只做“仿真验收与场景准备”，不接 ROS 主工作区，不接 Contact-GraspNet / Grounding DINO 等抓取算法主栈。

默认约束：

- 仓库：`/home/yhad/robot/stretch_mujoco`
- headless 渲染：继续使用 `MUJOCO_GL=egl`
- 复用已有 `uv` 环境

## 实际运行过的命令

基础 smoke：

- `/home/yhad/robot/scripts/smoke_test_sim.sh headless`
- `/home/yhad/robot/scripts/smoke_test_sim.sh perception`
- `/home/yhad/robot/scripts/smoke_test_sim.sh gui`

完整验收入口：

- `/home/yhad/robot/scripts/run_stretch_mujoco_acceptance.sh`

数据链路：

- `/home/yhad/robot/scripts/run_sim_data_capture.sh /home/yhad/robot/stretch_mujoco/stretch_mujoco/models/scene.xml default_scene`
- `/home/yhad/robot/scripts/run_sim_data_capture.sh /home/yhad/robot/sim_grasping/tabletop_minimal_scene.xml tabletop_scene`

动作验收：

- `/home/yhad/robot/scripts/run_sim_motion_validation.sh /home/yhad/robot/stretch_mujoco/stretch_mujoco/models/scene.xml`

截图导出：

- `/home/yhad/robot/scripts/run_sim_scene_snapshots.sh`

tabletop GUI 烟测：

- `cd /home/yhad/robot/stretch_mujoco && timeout 20s uv run launch_sim --scene-xml-path /home/yhad/robot/logs/sim_validation/tabletop_scene_resolved.xml`

## 结果结论

| 项目 | 结果 | 备注 |
| --- | --- | --- |
| 默认 GUI 场景 | 成功 | passive viewer 可启动 |
| 默认 headless 场景 | 成功 | 物理循环、控制、状态读取正常 |
| headless 相机与深度 | 成功 | 已验证 d405 / d435i RGB+depth |
| 状态读取 | 成功 | joint / base / ee / 传感器可拉取 |
| lidar / IMU | 成功 | `gyro / accel / lidar` 均返回数据 |
| 自定义 tabletop 场景 headless | 成功 | 通过 resolved XML 运行 |
| 自定义 tabletop 场景 GUI | 基本成功 | viewer 启动成功，超时强停时有连接重置噪声 |

## GUI 是否正常

- `已验证`：默认场景 GUI 正常启动。
- `已验证`：tabletop resolved 场景 GUI 可启动。
- `已验证`：当前 GUI 采用 passive viewer。
- `已验证`：在 WSLg 下会看到图形层日志噪声，但不影响 viewer 启动。

典型 WSLg 日志：

- `libigdgmm*_w.so`
- `D3D12: Removing Device.`

这些更像图形后端日志，不是本次功能失败的直接证据。

## headless 是否正常

- `已验证`：默认 headless 物理与控制正常。
- `已验证`：在 `MUJOCO_GL=egl` 下，RGB / depth 获取正常。
- `已验证`：headless + 单次少量相机组合更稳定。

## 是否必须依赖 `MUJOCO_GL=egl`

- `已验证`：本次所有 headless 感知验证都在 `MUJOCO_GL=egl` 下成功。
- `推断`：当前 WSL2 环境下也许存在其他 backend 组合，但本次未验证，因此后续建议继续固定使用 `egl`。

## WSL2 / 当前执行环境特有问题

### 1. 代理沙箱内 `multiprocessing.Manager()` 无法起本地 socket

失败命令：

- 我第一次直接在当前 agent 沙箱里运行数据采样和动作脚本。

关键报错：

- `PermissionError: [Errno 1] Operation not permitted`
- 位置：`multiprocessing.managers.Listener(...)`

处理：

- 改为在沙箱外运行同样的脚本。

结论：

- 这是当前执行沙箱限制，不是 Stretch MuJoCo 本身的 bug。

### 2. 早期版本的数据采样脚本把多路相机和动作混在同一次会话里，导致 server 提前退出

失败现象：

- 相机接通后，在同一会话里再驱动 lift / arm 到“观察位姿”，server 会中途退出。

关键报错：

- `ConnectionError: The Stretch Mujoco Simulator is not running.`

已尝试修复：

- 第一版：一次会话同时打开 d435i + d405 + nav，再移动机器人
- 第二版：分相机会话采样，但仍在采样阶段移动机器人
- 最终做法：状态采样与相机采样拆分；相机按头 / 腕 / nav 三个会话分别采；采样阶段保持 `home`

结论：

- 当前环境里，“少量相机 + 不额外动作”的采样方式最稳。

### 3. 自定义 tabletop XML 的相对资源路径会失效

失败命令：

- 直接让 simulator 读取 `/home/yhad/robot/sim_grasping/tabletop_minimal_scene.xml`

关键报错：

- `Error opening file ... base_link_0.obj: No such file or directory`

原因：

- `stretch.xml` 内部的 `assetdir="assets"` 对顶层 XML 路径很敏感。

修复：

- 新增 `/home/yhad/robot/sim_grasping/tabletop_scene_builder.py`
- 运行时生成 `/home/yhad/robot/logs/sim_validation/tabletop_scene_resolved.xml`
- 把 Stretch 机器人 include 改成绝对路径展开版本

### 4. offscreen 截图默认 framebuffer 太小

关键报错：

- `Image width 1280 > framebuffer width 640`

修复：

- 在 `render_scene_snapshot.py` 里，创建 `Renderer` 前主动设置 `offwidth / offheight`

## 失败项与修复建议

已发生并记录的失败：

1. 沙箱内 `Manager()` 起不来：已通过沙箱外执行规避。
2. 采样阶段同时做多相机+运动导致 simulator 退出：已通过分会话采样规避。
3. tabletop 场景 XML 相对资源路径失效：已通过 resolved XML 规避。
4. timeout 强停 GUI 时，可能出现 `ConnectionResetError / BrokenPipeError` 收尾噪声：建议手动关闭 viewer，而不是用粗暴超时。

后续建议：

- headless 感知采样继续固定 `MUJOCO_GL=egl`
- 一次会话尽量只开必须的相机
- GUI 验证适合人工短时验证，不适合用 `timeout` 长时间粗暴杀进程
- 自定义场景统一先生成 resolved XML，再交给 `launch_sim`

## 总结

当前 Stretch MuJoCo 在你的 WSL2 环境里已经达到“稳定可用”的程度：

- GUI 能起
- headless 能跑
- 控制接口能用
- 状态 / RGB / depth / 传感器能拿
- 自定义简化 tabletop 场景可扩展

剩下的重点不是“能不能跑”，而是“如何把现有能力更干净地对齐到后续 grasp pipeline 的输入与 primitive 上”。
