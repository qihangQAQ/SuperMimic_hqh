# baseline / 完整 RGMT：LAFAN1 与 BONES-SEED 服务器训练

2026-10-30：在 `supermimic-rgmt` 上整合上游 `05a9772` 之前的数据管线、计算优化、策略 EMA 和评估环境复用。本页面向从头开始的新训练。

## 选择方法与数据集

| 方法 | LAFAN1（默认 40 段） | BONES-SEED（准备后的紧凑动作库） |
| --- | --- | --- |
| baseline：MLP，无恢复课程 | `dataset=lafan1` | `dataset=boneseed` |
| 完整 RGMT：时序编码 + cross-attention + 恢复课程 | `experiment=rgmt_recovery dataset=lafan1` | `experiment=rgmt_recovery dataset=boneseed` |

不指定 `dataset` 就是 LAFAN1，因此旧训练命令仍使用原来的 40 段动作。数据集选择与方法选择互相独立。

## 服务器准备

```bash
cd /root/gpufree-data/Uni-SuperMimic
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

使用更新后的代码执行以下命令。已有 G1 机器人资产时可以跳过机器人准备：

```bash
# 仅准备机器人，不下载 LAFAN1
uv run tools/prepare_supermimic_assets.py --robot-only

# BONES-SEED：下载 G1 数据 -> 转换 -> 过滤 -> 打包
uv run unisupermimic-boneseed all --workers 16
```

数据目录：

- 原始 CSV：`data/raw/bones_seed`
- 转换后的 NPZ 和过滤清单：`data/npz/bones_seed`
- 训练读取的文件：`data/library/bones_seed_compact.pt`

上游参考规模是过滤后 129,487 段、约 4,730 万帧；实际数量以处理输出为准。71 维 fp16 参考库约 6.72 GB，常驻训练设备；仿真、网络、replay、梯度还会占用额外显存。原始数据和中间文件也需要磁盘空间。

已有原始数据时，可以分别执行：

```bash
uv run unisupermimic-boneseed convert --workers 16
uv run unisupermimic-boneseed filter --workers 16
uv run unisupermimic-boneseed pack
```

转换默认跳过已有 NPZ。需要重建时使用 `convert --overwrite`。选择 LAFAN1 时，沿用 `uv run tools/prepare_supermimic_assets.py`。

## 从头训练

两组命令在同一 GPU 上顺序执行，或自行通过 `CUDA_VISIBLE_DEVICES` 分配 GPU。使用独立输出目录保存结果。

```bash
# baseline + BONES-SEED
uv run unisupermimic-train dataset=boneseed \
  training.log_dir=runs/baseline_boneseed_seed0 algo.seed=0

# 完整 RGMT + BONES-SEED
uv run unisupermimic-train experiment=rgmt_recovery dataset=boneseed \
  training.log_dir=runs/rgmt_boneseed_seed0 algo.seed=0
```

LAFAN1 使用相同入口：

```bash
uv run unisupermimic-train dataset=lafan1 \
  training.log_dir=runs/baseline_lafan1_seed0 algo.seed=0

uv run unisupermimic-train experiment=rgmt_recovery dataset=lafan1 \
  training.log_dir=runs/rgmt_lafan1_seed0 algo.seed=0
```

如果紧凑库放在其他目录，覆盖文件路径：

```bash
uv run unisupermimic-train experiment=rgmt_recovery dataset=boneseed \
  'env.commands.motion.motion_files=[/root/gpufree-data/datasets/bones_seed_compact.pt]' \
  training.log_dir=runs/rgmt_boneseed_seed0 algo.seed=0
```

默认 mjwarp、4096 个训练环境、12000 轮。两种方法共享 `lr=4e-4`、`no_sync=true`、`pi_ema=0.995`，CUDA 使用 fused Adam，旋转与运动学采用上游优化实现。可直接覆盖 `algo.lr`、`algo.no_sync`、`algo.pi_ema`。

`no_sync` 减少归一化与采样循环的同步，仍保留非有限梯度跳步及空有效样本检查。EMA 只对浮点状态求平均，RGMT 的布尔掩码直接复制。策略开始更新后，checkpoint 的 `pi` 用于 EMA 评估；`pi_live` 保留实时权重，`pi_ema_updates` 记录更新次数。世界模型预热阶段尚无 EMA 更新时保存实时策略。无需为本次新训练处理旧 checkpoint。

恢复课程仍使用原来的 15% reset 概率、21 帧局部参考和带辅助力输入的世界模型。新增的站立参考不参与普通动作采样，也不计入跟踪全库完成率。

## 评估和视频

评估自动读取 run 的数据集、策略和恢复配置，无需再次传 `dataset`。`--num-envs` 限制同时运行的跟踪环境数，默认上限 4096；长动作优先，结束的环境分配给下一段动作。

```bash
uv run unisupermimic-eval --backend mjwarp \
  --ckpt runs/baseline_boneseed_seed0/last.pt --num-envs 4096

uv run unisupermimic-eval --backend mjwarp \
  --ckpt runs/rgmt_boneseed_seed0/last.pt --num-envs 4096

# 完整 RGMT 的独立、无辅助恢复评估
uv run unisupermimic-eval --backend mjwarp \
  --ckpt runs/rgmt_boneseed_seed0/last.pt --suite recovery --trials 128 --seed 0

# 推扰相对每段动作开始计时；复用环境后也从该动作第 1 秒开始施力
uv run unisupermimic-eval --backend mjwarp \
  --ckpt runs/rgmt_boneseed_seed0/last.pt --num-envs 4096 \
  --push-force 100 0 0 --push-at 1 --push-duration 0.1 --seed 0 \
  --output runs/rgmt_boneseed_seed0/push_x_eval.json
```

跟踪输出默认是 `last_tracking_eval.json`，包括逐段回报、完成率、失败列表、误差和评估耗时；恢复输出为 `last_recovery_eval.json`。`policy_weights` 标明 EMA 或实时策略。参考噪声使用 `clip-step-hash-v1`，与环境复用顺序无关，与历史版本噪声评估使用的随机序列不同。

视频只创建一个仿真环境，`--clip` 填评估 JSON 中实际的 BONES 动作名称：

```bash
uv run unisupermimic-video --backend mjwarp \
  --ckpt runs/rgmt_boneseed_seed0/last.pt \
  --clip YOUR_BONES_CLIP_NAME --seconds 8 --output runs/rgmt_boneseed_seed0/demo.mp4
```

## 记录与展示

每个训练目录保存 `run_config.json`、`motion_dataset.json`、`train_log.csv`、TensorBoard、checkpoint 和结束时的 `training_summary.json`。后者包含训练耗时、仿真步数及 PyTorch 峰值 allocated memory；RGMT 还保留 `rgmt_diagnostics.jsonl`。

通过 run 目录评估多个 checkpoint 时会写入 `eval.csv`：

```bash
uv run unisupermimic-eval --backend mjwarp --run_dir runs/baseline_boneseed_seed0
uv run unisupermimic-eval --backend mjwarp --run_dir runs/rgmt_boneseed_seed0

uv run tools/plot_boneseed_optimized_curve.py \
  runs/baseline_boneseed_seed0/train_log.csv --label baseline \
  --eval-csv runs/baseline_boneseed_seed0/eval.csv \
  --compare-log runs/rgmt_boneseed_seed0/train_log.csv --compare-label RGMT \
  --compare-eval-csv runs/rgmt_boneseed_seed0/eval.csv \
  --out runs/comparison_boneseed.png
```

图表只读取传入的实际日志，不内置上游回报。RGMT 训练奖励包含恢复任务，因此训练平均奖励应结合全库跟踪、恢复成功率与耗时解读。

README 中 BONES-SEED 曲线与指标来自上游；`docs/experiments/rgmt_vs_baseline_2026-10-29` 是之前 LAFAN1 的历史实验，两者均不是本次整合后的新结果。默认全库训练再全库评估衡量已见动作跟踪，不能当作未见动作泛化。紧凑库目前记录文件身份，不提供逐段来源隔离认证。

本次本地仅进行代码静态检查和四种配置组合解析；没有下载全量 BONES-SEED，也没有执行本地训练、仿真评估或 GPU 性能验证。运行效果以服务器实验为准。
