# Stretch MuJoCo 场景与示例盘点

更新时间：2026-04-05

说明：

- `已验证`：本次在当前 WSL2 + `~/robot/stretch_mujoco` 环境中实际跑过。
- `代码支持`：仓库里已有入口或生成逻辑，但本次没有在当前环境里完整跑通。
- `推断`：根据代码结构和已有结果做出的工程判断。

## 默认入口

- GUI 默认入口：`cd /home/yhad/robot/stretch_mujoco && uv run launch_sim`
- headless 默认入口：`cd /home/yhad/robot/stretch_mujoco && uv run launch_sim --headless`
- 带相机场景：`cd /home/yhad/robot/stretch_mujoco && uv run launch_sim --headless --imagery`
- 自定义场景 XML：`cd /home/yhad/robot/stretch_mujoco && uv run launch_sim --scene-xml-path <scene.xml>`

其中 `launch_sim` 来自 `pyproject.toml` 的 `project.gui-scripts`，实际入口是 `stretch_mujoco.launch_sim:main`。

## 现有场景与示例

| 项目 | 启动方式 | 适合验证什么 | 当前判断 |
| --- | --- | --- | --- |
| 默认场景 `stretch_mujoco/models/scene.xml` | `uv run launch_sim` | GUI、基础物理、默认桌子+简单物体、headless | 已验证 |
| 默认场景 headless | `uv run launch_sim --headless` | 无 GUI 物理循环、控制接口 | 已验证 |
| 默认场景 + imagery | `uv run launch_sim --headless --imagery` | 相机渲染链路 | 代码支持，等价能力已通过自定义采样脚本验证 |
| `examples/move_joints.py` | `uv run examples/move_joints.py` | home/stow、lift、head、base 运动 | 代码支持；等价能力已验证 |
| `examples/camera_feeds.py` | `uv run examples/camera_feeds.py` | RGB/depth 拉取与 OpenCV 显示 | 代码支持；等价能力已验证 |
| `examples/laser_scan.py` | `uv run examples/laser_scan.py` | lidar / gyro / accel | 代码支持；lidar 数据已验证 |
| `examples/world_frames.py` | `uv run examples/world_frames.py` | 可视化坐标轴、世界标记 | 代码支持 |
| `examples/start_pose.py` | `uv run examples/start_pose.py` | 改起始底盘位姿 | 代码支持 |
| `examples/keyboard_teleop.py` | `uv run examples/keyboard_teleop.py` | 人工遥操作 | 代码支持，本次未交互验证 |
| `examples/gamepad_teleop.py` | `uv run examples/gamepad_teleop.py` | 手柄遥操作 | 代码支持，本次未交互验证 |
| `examples/robocasa_environment.py` | `uv run examples/robocasa_environment.py` | 厨房任务场景生成、复杂抓取前置环境 | 代码支持，但当前环境未装 `robocasa` / `robosuite`，本次未运行 |

## 场景生成与模型生成相关代码

- 默认机器人模型：`stretch_mujoco/models/stretch.xml`
- 默认完整场景：`stretch_mujoco/models/scene.xml`
- 默认场景内容：
  - 平面地面
  - 一张桌子
  - 一个 box 物体
  - 一个 cylinder 物体
- Robocasa 生成入口：`stretch_mujoco/robocasa_gen.py`
  - 支持按 `task / layout / style` 生成厨房 XML
  - 支持把 Stretch 机器人插入厨房环境
  - 支持写出绝对路径 XML

## 相机与传感器相关能力

仓库里直接暴露的相机：

- `cam_d405_rgb`
- `cam_d405_depth`
- `cam_d435i_rgb`
- `cam_d435i_depth`
- `cam_nav_rgb`

传感器：

- `base_gyro`
- `base_accel`
- `base_lidar`

本次已确认：

- 相机 RGB / depth 可拉取
- `K` 可由 simulator 返回
- joint / base / ee 状态可拉取
- lidar / gyro / accel 可拉取

## 哪些场景更适合后续抓取验证

优先级建议：

1. `/home/yhad/robot/sim_grasping/tabletop_minimal_scene.xml`
   - 已验证可生成 resolved XML、可 headless 采样、可导出截图
   - 物体简单、可复现、便于后续感知与 primitive 验证
2. `/home/yhad/robot/stretch_mujoco/stretch_mujoco/models/scene.xml`
   - 已验证最稳定
   - 自带桌子和两个简单物体，适合最小闭环
3. Robocasa 厨房场景
   - 更贴近真实家居任务
   - 但依赖额外安装，且环境复杂度明显更高
   - 更适合作为后续第二阶段，而不是当前抓取主验证场景

## 是否有现成 tabletop 或近似场景

- `已验证`：官方 `scene.xml` 已经是一个“近似 tabletop”场景，含桌子和两个简单物体。
- `已验证`：本次新增 `/home/yhad/robot/sim_grasping/tabletop_minimal_scene.xml`，专门面向桌面抓取前置验证。
- `推断`：Robocasa 厨房中的台面也可用于抓取，但更适合在简单桌面场景跑通后再接入。

## 对后续抓取的直接建议

- 第一阶段优先复用默认场景和本次新增 tabletop 场景，不要一上来上 Robocasa。
- RGB / depth / 状态链路已经够支撑后续 grasp pipeline 的输入准备。
- 真正接抓取算法前，先继续补“相机-位姿同步”和“primitive 到桌面目标点的闭环控制”。
