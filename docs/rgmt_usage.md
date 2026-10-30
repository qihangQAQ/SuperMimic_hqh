# RGMT 融合版本：训练与评估

实现包含可选的因果历史编码与 cross-attention 策略、站立恢复课程、带外力输入的世界模型，以及与想象 rollout 一致的恢复终止逻辑。训练仍使用 SuperMimic 的可微策略更新。

2026-10-30 更新：baseline 与完整 RGMT 已接入 `dataset=lafan1` / `dataset=boneseed`，默认学习率 `4e-4`、策略 EMA `0.995`，并支持评估环境复用。新训练的服务器命令见 [BONES-SEED 训练指南](boneseed_training.md)。本文末尾的 2026-10-29 验证记录仅对应之前版本。

## 1. 实验配置

在仓库根目录执行；准备资产的方法沿用 README。默认后端为 mjwarp，下面用 `training.log_dir` 区分输出目录。

| 实验 | Hydra 参数 |
| --- | --- |
| 原始 SuperMimic | 不指定 `experiment` |
| 只加编码器 | `experiment=rgmt` |
| 只加恢复课程 | `experiment=recovery` |
| 编码器 + 恢复课程 | `experiment=rgmt_recovery` |
| 与 RGMT 使用相同参考窗口的 MLP 对照 | `experiment=mlp_window` |

```bash
uv run unisupermimic-train experiment=rgmt_recovery \
  training.log_dir=runs/rgmt_recovery_seed0 algo.seed=0
```

CPU/MuJoCo 可添加 `task=g1_super_mimic/mujoco`。网络、环境数量和训练时长仍由原有 `algo.*` 参数控制；训练规模较大时应使用可用的 GPU 后端。辅助力通过 UniLab 的 pelvis COM 外力接口施加，不支持该能力的后端会明确报错。无辅助实验使用：

```bash
uv run unisupermimic-train experiment=rgmt_recovery \
  env.commands.motion.recovery.assistance_max=0 \
  training.log_dir=runs/rgmt_recovery_no_assistance
```

RGMT 默认使用 10 步本体历史、21 帧局部参考、128 维 token、4 个注意力头。窗口由 `algo.reference_past` 和 `algo.goal_frames` 共同定义；例如前后各 2 帧使用 `algo.reference_past=2 algo.goal_frames=5`。动作接口、73 维参考特征和基础奖励沿用 SuperMimic。注意力部分使用 FP32，动作解码器和世界模型保留原有混合精度设置。

## 2. 恢复任务的具体语义

- 默认 15% 的训练 reset 被分配为恢复 episode；普通 motion 的自适应采样统计独立维护。
- 追加一个专用的静止站立参考，姿态固定，参考速度为零。该参考不进入普通 motion 采样池和全库跟踪指标。
- 静止参考的时间索引编码恢复计时；默认 3 秒后结束 episode。想象轨迹也在此处停止累计未来收益和 bootstrap。
- 训练初期随机姿态倾斜较小，之后逐渐扩展；辅助力在 reset 时按课程进度采样，整个 episode 内保持恒定，到下一 episode 才使用新的退火幅值。
- 世界模型读取每条真实 transition 记录的世界坐标外力，并在内部转为本体坐标。critic 额外读取恢复模式、剩余时间和外力；actor 不读取辅助力。
- 恢复成功需在窗口末尾满足高度、姿态、关节和速度阈值，并连续稳定指定时长。默认稳定时长为 0.3 秒。
- 参考初始位置、朝向使用站立参考的坐标；初始化根据所有身体节点的最低高度留出 0.12 m 余量，随后由物理仿真处理接触。这不是完整碰撞几何穿透检查。

主要参数位于 `env.commands.motion.recovery`，包括 `probability`、`seconds`、`stand_height`、`assistance_max`、`curriculum_iterations`、`initial_angle`、`final_angle`、成功阈值和 `stable_seconds`。默认沿用跟踪奖励；可选 `shaping_weight` 对恢复阶段增加姿态/高度引导，默认值为 0。

目前独立恢复评估检验“随机失稳后恢复到站立”；推扰跟踪评估保持参考动作连续推进及原失败阈值。未实现跌倒检测后切换站立参考、暂停原 motion、恢复后再接续原相位的运行时状态机，不能把独立起身成功率解释成这种能力。

## 3. 数据划分

正式泛化实验建议使用 JSON manifest。文件路径相对于 manifest 所在目录；`source_group` 必须标识同一条原始录制序列，其裁剪、增广或重定向版本使用同一个值。

```json
{
  "train": [
    {
      "path": "src/unisupermimic/assets/motions/g1/lafan1_supermimic/walk1_subject1.csv",
      "source_group": "lafan1/walk1_subject1"
    }
  ],
  "test": [
    {
      "path": "src/unisupermimic/assets/motions/g1/lafan1_supermimic/jumps1_subject2.csv",
      "source_group": "lafan1/jumps1_subject2"
    }
  ]
}
```

这是格式示例，正式实验应制定有代表性的动作清单。加载时拒绝跨集合的同源组、相同内容，以及集合内重复路径；motion 文件名需互不重复。

```bash
uv run unisupermimic-train experiment=rgmt_recovery \
  training.motion_manifest=motion_splits.json training.motion_split=train \
  training.log_dir=runs/rgmt_split_seed0 algo.seed=0
```

每个新训练 run 写入 `motion_dataset.json`，保存数据内容 SHA-256 和分组身份。没有 manifest 时仍记录内容身份，但不能认证原始序列级的未见泛化。程序无法自行识别被错误标成不同组的相邻片段，组标签应按数据来源填写。

## 4. 跟踪、扰动与恢复评估

评估、视频和交互播放会从 `run_config.json` 恢复策略与参考窗口配置。旧 MLP checkpoint 使用原策略构造路径。

```bash
# 普通全 clip 跟踪
uv run unisupermimic-eval --backend mujoco \
  --ckpt runs/rgmt_recovery_seed0/last.pt

# 固定种子的参考噪声与推扰；外力单位为 N，方向使用世界坐标
uv run unisupermimic-eval --backend mujoco \
  --ckpt runs/rgmt_recovery_seed0/last.pt \
  --reference-noise 1.0 --push-force 100 0 0 \
  --push-at 1.0 --push-duration 0.1 --seed 0 \
  --output runs/rgmt_recovery_seed0/perturb_seed0.json

# 随机失稳初始化；始终关闭辅助力
uv run unisupermimic-eval --backend mujoco \
  --ckpt runs/rgmt_recovery_seed0/last.pt \
  --suite recovery --trials 32 --seed 0

# 显式更换 motion 集合，继续严格检查机器人、网络和动作接口
uv run unisupermimic-eval --backend mujoco \
  --ckpt runs/rgmt_split_seed0/last.pt \
  --motions motion_splits.json --split test --require-unseen
```

`--require-unseen` 要求训练 run 有完整分组记录，且测试文件内容和来源组均不与训练集重叠。它包括用于世界模型采集的数据，而不只检查 policy 的参考。

跟踪评估增加以下指标：

- 完成率、归一化回报和失败动作列表。
- 按 reward 的朝向和 root XY 对齐后的 14 个跟踪身体点平均位置误差（mm）。
- 世界坐标身体点速度误差（m/s）、关节角平均绝对误差（rad）、全局 root 位置误差（m）。
- 按 clip 平均和按有效帧加权的统计，以及评估耗时。

误差统计包含结束 transition，只计算首次 episode 结束前的有效帧；应与完成率一起解读。恢复评估另外输出窗口末尾的成功率，以及成功 trial 首次达到连续稳定条件的时间。

`--reference-noise` 只扰动 policy 的参考窗口，基础均匀噪声幅度按特征分别为：位置 0.02 m、6D 旋转表示 0.05、线速度/角速度各 0.2、关节角 0.1 rad、缩放后的关节速度 0.01。该参数对所有幅度统一缩放；奖励参考保持原值。该协议属于本项目实验设置。

默认写入 checkpoint 同目录的 `*_tracking_eval.json` 或 `*_recovery_eval.json`；使用 `--output` 为不同扰动和种子保留独立结果。恢复试验可用于基线策略，评估入口会在结构校验后构造独立恢复任务。

## 5. 续训、日志与验证

```bash
# 使用与源 run 相同的实验、数据和结构参数
uv run unisupermimic-train experiment=rgmt_recovery \
  algo.resume=true algo.load_run=runs/rgmt_recovery_seed0 \
  algo.checkpoint=last.pt algo.iterations=16000
```

续训检查策略类型、参考窗口、网络结构、恢复配置和数据身份；课程进度由 checkpoint 的 iteration 恢复。恢复开启时会重新初始化环境以使用恢复后的课程进度，replay 仍按原实现重新预热。恢复配置不能通过普通 resume 静默改变。

原有 `train_log.csv` 和 TensorBoard 日志继续有效。RGMT/恢复实验另写 `rgmt_diagnostics.jsonl`，记录 GPU 峰值显存；开启恢复时，还记录跟踪/恢复的单步与多步世界模型误差和恢复样本实际占比。模型诊断只计算日志迭代的最后一个采样 batch，避免每次更新都同步诊断数据；这些值不是整个日志区间的平均值。恢复误差额外按真实 root 高度低于/不低于 0.45 m 分组；它只是便于排查的高度分组，不等同于接触状态分类。未采样到的组记录为 `null`。

```bash
make check
make test

# 下载机器人资产后，运行 MuJoCo 的完整融合短程测试
uv run pytest -m slow \
  tests/envs/test_supermimic_env.py::test_rgmt_recovery_mujoco_training_and_checkpoint
```

如果系统中的 ROS `launch_testing` 插件与项目 pytest 版本冲突，可在测试命令前添加 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，禁用系统自动注入的测试插件。

当前验证覆盖数值测试、MuJoCo 短程训练、真实外力效果、checkpoint、无辅助恢复与扰动评估。正式的四组消融、多随机种子训练、高动态未见动作结论，以及 GPU/其他后端的性能，需由完整实验确认。

### 2026-10-29 验证记录

- Ruff、格式检查、mypy、Pyright 全部通过。
- 默认测试集：98 passed、2 skipped、7 deselected；慢速后端测试不在默认集合内。
- 单独的 MuJoCo 融合集成测试通过，覆盖世界模型/策略更新、课程续训、真实外力、外力清除和固定种子的无辅助恢复评估。
- CPU 上使用小网络和短合成 motion，完成 `compile=true` 的训练迭代，再从 `last.pt` 续训，随后通过独立入口输出恢复及参考噪声/推扰评估 JSON。
- 视频 rollout 回归覆盖恢复参考追加后的 clip 边界。

本地未使用 GPU 执行训练验证，也未运行完整训练。短程测试仅确认工程链路和接口；恢复成功率、泛化提升及训练效率需在服务器正式实验中测量。
