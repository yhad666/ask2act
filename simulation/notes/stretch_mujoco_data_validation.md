# Stretch MuJoCo 数据链路验证

更新时间：2026-04-05

## 产物位置

默认场景样例：

- `/home/yhad/robot/logs/sim_validation/data_samples/default_scene/`

tabletop 场景样例：

- `/home/yhad/robot/logs/sim_validation/data_samples/tabletop_scene/`

截图：

- `/home/yhad/robot/logs/sim_validation/screenshots/`

## 采样脚本

主脚本：

- `/home/yhad/robot/sim_grasping/capture_validation_data.py`

包装脚本：

- `/home/yhad/robot/scripts/run_sim_data_capture.sh`

本次最终采样策略：

- `status_only` 会话：无相机、快速 headless，用于得到稳定的 joint / base / ee / sensor 状态
- `head` 会话：只采 `d435i RGB + depth`
- `wrist` 会话：只采 `d405 RGB + depth`
- `nav` 会话：只采 `nav RGB`

这样做的原因是：当前环境下，少量相机会话明显比“全相机 + 动作”更稳定。

## 已保存的数据类型

| 数据 | 位置示例 | 格式 |
| --- | --- | --- |
| 头相机 RGB | `head_d435i_rgb.png` | PNG |
| 头相机 depth | `head_d435i_depth.npy` | `float32`，单位米 |
| 头相机 depth 彩图 | `head_d435i_depth_color.png` | PNG |
| 腕相机 RGB | `wrist_d405_rgb.png` | PNG |
| 腕相机 depth | `wrist_d405_depth.npy` | `float32`，单位米 |
| 腕相机 depth 彩图 | `wrist_d405_depth_color.png` | PNG |
| nav RGB | `nav_rgb.png` | PNG |
| 相机内参 | `camera_intrinsics.json` | JSON |
| 稳态状态 | `status_snapshot.json` | JSON |
| 每次采样说明 | `sample_manifest.json` | JSON |

## 各相机当前能拿到什么

| 相机 | RGB | depth | K | 当前建议用途 |
| --- | --- | --- | --- | --- |
| head `d435i` | 有 | 有 | 有 | 桌面粗检测、全局目标定位 |
| wrist `d405` | 有 | 有 | 有 | 近距离抓取前处理、局部点云、抓取 refinement |
| nav | 有 | 无 | 无 | 环境观察 / 导航辅助，不是抓取主相机 |

## 图像与 depth 形状

本次保存结果：

- `d435i RGB`：`424 x 240 x 3`
- `d435i depth`：`424 x 240`
- `d405 RGB`：`270 x 480 x 3`
- `d405 depth`：`270 x 480`
- `nav RGB`：`800 x 600 x 3`

注意：

- `d435i` 在官方 helper 里会做旋转校正，因此保存图像是“旋转后”的朝向。
- `d405` 保存方向与输出分辨率更直观。

## K 的格式是什么

本次我在 `camera_intrinsics.json` 里保留了两套 K：

1. `*_K_raw`
   - 来自 simulator 直接返回
   - 更接近传感器原始分辨率坐标系
2. `*_K_saved_image`
   - 已对齐到本次保存下来的图像尺寸
   - 后续点云 / 投影 / 抓取前处理应优先使用这一套

当前文件示例：

- `cam_d405_K_saved_image`
  - `[[242.56, 0, 240], [0, 242.34, 135], [0, 0, 1]]`
- `cam_d435i_K_saved_image`
  - `[[304.07, 0, 119], [0, 304.24, 212], [0, 0, 1]]`

说明：

- `已验证`：`*_K_raw` 与保存图像尺寸并不直接一致。
- `已修复`：采样脚本现已额外输出 `*_K_saved_image`，供后续算法直接使用。

## depth 是否可直接用于后续点云 / 抓取前处理

结论：可以，但要按下面方式使用。

已验证：

- depth 以 `float32` 的米制保存
- 可直接作为点云重建或抓取前处理输入

使用建议：

- 只使用 `depth > 0` 的像素
- 和 `*_K_saved_image` 配对使用
- 对 `d435i`，使用保存后的旋转图像时，也要使用对应旋转后的 `K_saved_image`

## 状态数据验证结果

状态文件：

- `/home/yhad/robot/logs/sim_validation/data_samples/default_scene/status_snapshot.json`
- `/home/yhad/robot/logs/sim_validation/data_samples/tabletop_scene/status_snapshot.json`

当前包含：

- `sim_time`
- `status_fps`
- `base_pose`
- `ee_pose_4x4`
- 全 joint 状态
- `base_gyro / base_imu / lidar`

默认场景稳态示例：

- `sim_time ~= 2.84`
- `lift ~= 0.589`
- `arm ~= 0.100`
- `head_tilt ~= -0.0045`

tabletop 场景稳态示例：

- `sim_time ~= 3.01`
- `lift ~= 0.589`
- `arm ~= 0.100`

这说明：

- `已验证`：状态链路本身是可用的
- `已验证`：在把桌面场景整体后移之后，tabletop 场景也能保持稳定初始位姿

## 哪些数据最适合后续 grasp pipeline

优先推荐：

1. `wrist_d405_rgb.png`
2. `wrist_d405_depth.npy`
3. `cam_d405_K_saved_image`
4. `status_snapshot.json` 里的 `ee_pose_4x4` 和 `base_pose`

第二优先级：

1. `head_d435i_rgb.png`
2. `head_d435i_depth.npy`
3. `cam_d435i_K_saved_image`

不建议作为抓取主输入：

- `nav_rgb.png`

## 默认场景与 tabletop 场景对比

tabletop wrist depth 统计：

- `min ~= 0.075 m`
- `mean ~= 0.352 m`
- `valid_pixel_count ~= 113718`

这比默认场景更像“有明确近距离桌面结构”的输入，说明 wrist 视角下的 tabletop 场景更适合作为抓取前处理的主测试集。

## 结论

当前数据链路已经满足后续抓取算法前置接入的最小要求：

- RGB 有
- depth 有
- K 有，而且已补齐对齐到保存图像的版本
- base / joint / ee 状态有
- 数据保存格式已固定

真正进入 grasp pipeline 前，建议继续做两件小事：

1. 把 `status_snapshot.json` 和每次图像采样做更严格的“同一时刻”同步
2. 先用 `d405 + K_saved_image + ee_pose_4x4` 打通一个最小点云/裁剪前处理脚本
