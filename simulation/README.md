# Ask2Act Simulation

`simulation/` 现在有一条可直接运行的单杯抓取主线，入口在 [pipeline.py](/home/yhad/robot/ask2act/simulation/pipeline.py)，核心代码在 [ask2act_grasp/](/home/yhad/robot/ask2act/simulation/ask2act_grasp)。

当前这条主线的目标不是“完整 MoveIt2 + CGN 终版”，而是先把 Stretch 在 MuJoCo 里完成一套稳定的单红杯抓取流程：

1. `head` 先转到固定观察位
2. `arm/wrist` 回到默认准备位
3. 已知单杯目标点侧抓
4. 闭爪
5. 抬起

## 当前哪些文件是主线

这次单杯抓取主要依赖这些文件：

- [pipeline.py](/home/yhad/robot/ask2act/simulation/pipeline.py)
- [ask2act_grasp/pipeline.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/pipeline.py)
- [ask2act_grasp/config/scene_config.yaml](/home/yhad/robot/ask2act/simulation/ask2act_grasp/config/scene_config.yaml)
- [ask2act_grasp/config/grasp_config.yaml](/home/yhad/robot/ask2act/simulation/ask2act_grasp/config/grasp_config.yaml)
- [ask2act_grasp/perception/head_alignment.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/perception/head_alignment.py)
- [ask2act_grasp/planning/motion_planner.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/planning/motion_planner.py)
- [ask2act_grasp/execution/grasp_executor.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/execution/grasp_executor.py)
- [ask2act_grasp/scene/scene_setup.py](/home/yhad/robot/ask2act/simulation/ask2act_grasp/scene/scene_setup.py)
- [scripts/source_sim.sh](/home/yhad/robot/ask2act/simulation/scripts/source_sim.sh)
- [CURRENT_LOCKED_PARAMS.md](/home/yhad/robot/ask2act/simulation/CURRENT_LOCKED_PARAMS.md)

如果只想把当前仿真主线跑起来，上面这些文件已经够了。

## 目录结构

```text
simulation/
├── ask2act_grasp/
│   ├── config/
│   ├── execution/
│   ├── grasp/
│   ├── perception/
│   ├── planning/
│   ├── scene/
│   ├── tests/
│   └── utils/
├── scripts/
├── stretch_mujoco/
└── pipeline.py
```

## 首次准备

```bash
cd simulation
./scripts/bootstrap_stretch_mujoco.sh
```

这一步会：

- fresh clone 上游 `stretch_mujoco`
- 创建本地 `.venv`
- 安装当前主线需要的 Python 依赖

## 怎么跑

### 1. GUI 仿真

```bash
cd /home/yhad/robot/ask2act/simulation
source ./scripts/source_sim.sh >/dev/null
MUJOCO_GL=glfw \
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
__NV_PRIME_RENDER_OFFLOAD=1 \
__GLX_VENDOR_LIBRARY_NAME=nvidia \
python3 pipeline.py --show-viewer-ui --run-dir /tmp/ask2act_pipeline_gui
```

适合人工观察：

- `head` 有没有先转到位
- `arm/wrist` 有没有先回准备位
- 夹爪是不是侧抓
- 抓住后有没有继续莫名往前顶

### 2. Headless

```bash
cd /home/yhad/robot/ask2act/simulation
source ./scripts/source_sim.sh >/dev/null
python3 pipeline.py --headless
```

### 3. 只检查 head

```bash
cd /home/yhad/robot/ask2act/simulation
source ./scripts/source_sim.sh >/dev/null
python3 scripts/diag_head_motion.py --camera-profile head --timeout-s 20
```

## 当前默认行为

当前单杯抓取不是完全依赖视觉估抓取位姿，而是“已知单杯位置 + 固定侧抓策略”：

- head 固定看桌面
- 杯子默认是场景里唯一红色圆柱
- planner 用 oracle 单杯抓取点
- 夹爪提前张开，再平着伸过去
- 闭爪目标不是 `-0.2`，而是更接近真实抓住物体时的受限闭合

## 现在最常调的参数

### 场景参数

文件：
[ask2act_grasp/config/scene_config.yaml](/home/yhad/robot/ask2act/simulation/ask2act_grasp/config/scene_config.yaml)

最重要的是：

- `table_position_m`
- `table_size_m`
- `cup_position_m`
- `cup_radius_m`
- `cup_height_m`
- `head_pan_rad`
- `head_tilt_rad`

当前锁定的 head 参数：

- `head_pan_rad = -1.57`
- `head_tilt_rad = -0.55`

这两个现在是基线，不建议随便改。

### 抓取参数

文件：
[ask2act_grasp/config/grasp_config.yaml](/home/yhad/robot/ask2act/simulation/ask2act_grasp/config/grasp_config.yaml)

最常调的是：

- `oracle_grasp_height_ratio`
  作用：抓杯身的高度比例
- `oracle_side_grasp_wrist_pitch_rad`
  作用：侧抓时 wrist pitch
- `oracle_tucked_wrist_yaw_rad`
  作用：启动和避桌沿时 wrist yaw tuck 姿态
- `oracle_side_open_width_cmd`
  作用：侧抓阶段的夹爪张开命令
- `oracle_lateral_offset_m`
  作用：左右偏置
- `oracle_forward_offset_m`
  作用：前后偏置
- `oracle_arm_backoff_m`
  作用：最终抓取前的保守后撤
- `oracle_final_arm_delta_m`
  作用：最后一小段前送

## 这几个参数怎么理解

### 1. 抓得太高/太低

调：

- `oracle_grasp_height_ratio`

经验：

- 变小：抓得更低
- 变大：抓得更高

### 2. 图片视角里偏左/偏右

调：

- `oracle_lateral_offset_m`

经验：

- 变大：目标往图片右侧修
- 变小：目标往图片左侧修

### 3. 不够前 / 伸过头

调：

- `oracle_forward_offset_m`
- `oracle_arm_backoff_m`
- `oracle_final_arm_delta_m`

经验：

- `oracle_forward_offset_m` 更负：整体更往前
- `oracle_arm_backoff_m` 更小：前一阶段更靠近物体
- `oracle_final_arm_delta_m` 更大：最后一小段更敢往前送

### 4. 夹爪开得太晚 / 太窄

调：

- `oracle_side_open_width_cmd`

说明：

- 这个值是 Stretch gripper 的控制命令，不是肉眼看到的“净开口宽度”
- 当前主线里已经把开夹时机提前到真正接近目标之前
- 如果视觉上还是觉得不够开，可以继续往上调

### 5. 容易碰桌子下沿

优先检查：

- `oracle_tucked_wrist_yaw_rad`
- startup pose
- `arm` 有没有先收回

不要第一反应就改 head 或乱改抓取高度。

## 已知稳定基线

当前建议把这些值当成默认基线：

- `head_pan_rad = -1.57`
- `head_tilt_rad = -0.55`
- `oracle_grasp_height_ratio = 0.30`
- `oracle_tucked_wrist_yaw_rad = 2.50`

更完整的锁定说明看：
[CURRENT_LOCKED_PARAMS.md](/home/yhad/robot/ask2act/simulation/CURRENT_LOCKED_PARAMS.md)

## 验证

```bash
cd /home/yhad/robot/ask2act/simulation
python3 -m pytest ask2act_grasp/tests -q
```

当前基础回归应为：

```text
4 passed
```

## 当前状态

当前这条主线已经能做：

- 启动 Stretch MuJoCo 单杯场景
- head 先看桌面
- 已知单杯目标做侧抓
- 闭爪并抬起
- 用 GUI 反复调抓取点、夹爪开度、前送距离

当前还没有完全做完的事情：

- 真正的 MoveIt2 规划还没接上
- Contact-GraspNet 还不是主闭环
- 自动“是否真的抓住”判定现在仍然很弱

所以这条线目前更适合：

- 先把仿真动作本身调顺
- 再逐步替换成真实感知和规划模块
