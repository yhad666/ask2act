# Stretch MuJoCo 简单桌面场景扩展说明

更新时间：2026-04-05

## 最终采用的做法

最终采用的是：

1. 在 `/home/yhad/robot/sim_grasping/tabletop_minimal_scene.xml` 中维护一个可读、可修改的桌面场景模板
2. 通过 `/home/yhad/robot/sim_grasping/tabletop_scene_builder.py` 在运行时生成绝对路径展开后的
   `/home/yhad/robot/logs/sim_validation/tabletop_scene_resolved.xml`
3. 让 headless / GUI / 截图脚本统一使用这个 resolved XML

这样做的原因：

- 模板文件可读性好
- 运行时不再受 `stretch.xml` 相对 `assetdir` 的影响
- 更适合后续重复试验

## 场景内容

包含：

- 一张桌子
- 一个杯子
- 一个小盘子
- 一个叉子
- 一个勺子

## 哪些物体是真实 mesh，哪些是简化替代

桌子与餐具物体：

- `已验证`：全部使用简化 primitive geom

具体做法：

- 桌子：`box` + 4 条桌腿
- 杯子：`cylinder`
- 小盘子：薄 `cylinder`
- 叉子：`capsule + box`
- 勺子：`capsule + ellipsoid`

Stretch 机器人：

- `已验证`：继续复用官方 `stretch.xml` 模型

结论：

- 当前桌面物体不是高保真真实 mesh
- 但它们对 RGB / depth / 场景可视化 / primitive 验证已经足够

## 如何启动

### 1. 生成并采样 tabletop 场景

- `/home/yhad/robot/scripts/run_sim_data_capture.sh /home/yhad/robot/sim_grasping/tabletop_minimal_scene.xml tabletop_scene`

这个命令会自动生成：

- `/home/yhad/robot/logs/sim_validation/tabletop_scene_resolved.xml`

### 2. GUI 启动 resolved 场景

- `cd /home/yhad/robot/stretch_mujoco && uv run launch_sim --scene-xml-path /home/yhad/robot/logs/sim_validation/tabletop_scene_resolved.xml`

### 3. 导出 overview 截图

- `/home/yhad/robot/scripts/run_sim_scene_snapshots.sh`

## 在 GUI 中是否可见

- `已验证`：`tabletop_scene_resolved.xml` 可以启动 passive viewer
- `已验证`：我用 `timeout 20s` 做了 GUI smoke，viewer 在超时前已经连接成功
- `已知现象`：如果用超时强杀 GUI，会出现 `ConnectionResetError / BrokenPipeError` 这类退出噪声

因此：

- “场景能起 GUI”这件事是已验证的
- “优雅退出 GUI”建议人工关闭，不建议靠强制超时

## 在 RGB / depth 中是否可见

已保存截图：

- `/home/yhad/robot/logs/sim_validation/screenshots/tabletop_scene_overview.png`
- `/home/yhad/robot/logs/sim_validation/screenshots/tabletop_scene_head_rgb.png`
- `/home/yhad/robot/logs/sim_validation/screenshots/tabletop_scene_head_depth.png`
- `/home/yhad/robot/logs/sim_validation/screenshots/tabletop_scene_wrist_rgb.png`
- `/home/yhad/robot/logs/sim_validation/screenshots/tabletop_scene_wrist_depth.png`

结论：

- `已验证`：offscreen overview 图已成功导出
- `已验证`：wrist RGB / depth 图已成功导出
- `已验证`：head RGB / depth 图已成功导出
- `推断`：在当前几何与相机位置下，wrist 视角比 head 视角更适合直接看桌面餐具

支持这个判断的数据：

- tabletop wrist depth：`min ~= 0.075 m`，`mean ~= 0.352 m`，`valid_pixel_count ~= 113718`
- 说明 wrist 视角确实拿到了较密集的近距离桌面结构

## 这个场景是否适合后续 grasp pipeline 接入

结论：适合，且比复杂厨房场景更适合做第一阶段接入。

原因：

- 场景简单
- 物体数量受控
- RGB / depth 可用
- wrist 视角有足够近距离桌面结构
- 可以稳定生成 resolved XML 并重复运行

当前限制：

- 物体是简化几何体，不是高保真餐具 mesh
- 还没有做真实抓取接触成功率验证

因此更准确的定位是：

- 非常适合作为“感知 + primitive + 控制”联调场景
- 适合作为 grasp pipeline 第一阶段接入场景
- 不适合作为最终真实餐具泛化能力评估场景

## 本次为稳定性做过的修正

中途遇到过两个关键问题，已经修正：

1. 相对资源路径导致 Stretch mesh 找不到
   - 通过 resolved XML 修复
2. 桌子初始位置太近，影响 `home` 稳态
   - 已把整张桌子整体后移
   - 现在 tabletop 场景稳态状态与默认场景接近

## 结论

当前 tabletop 场景已经满足你的当前目标：

- 可重复生成
- 可 GUI / headless 使用
- 可 RGB / depth 观察
- 可作为后续抓取算法前的场景准备基线

如果下一步继续迭代，我建议先不要换复杂 mesh，而是先在这个场景里完成：

1. 目标点选择
2. wrist 点云裁剪
3. pre-grasp primitive 闭环
4. 简单抓取闭环
