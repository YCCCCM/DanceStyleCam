# DanceStyleCam-Official 迁移验证

验证日期：2026-08-03。

## 全量 DCM-style++

- 从公开 DCM 结构生成 108 个完整序列、344,015 帧、540 个 NPY 文件。
- 数据目录总大小约 330 MB；旧 `DanceStyleCam/DCM++` 为 30 GB。
- `validate_dcm_style_pp.py` 检查 schema、shape、dtype、checksum、manifest 和 split 引用，
  最终 `issues: []`。
- `train_pre.json`/`test_pre.json` 只在运行时解析，manifest 不包含 split 或 style 分配。

## 与旧 DCM++ 的逐值比较

使用 `tools/data/compare_legacy_dcmpp.py` 比较旧 DCM++ 中 154 个物理片段
（Train 105、Test 49）。在相机有效帧范围内结果如下：

| 字段 | 通过片段 | 最大绝对误差 |
| --- | ---: | ---: |
| camera20 | 154/154 | 0 |
| keyframe_mask | 154/154 | 0 |
| bone_mask60 | 154/154 | 0 |
| motion180 | 154/154 | 0 |
| music35_clip | 154/154 | 0 |

旧文件中的 motion/music 在部分片段尾部比相机多 1-2 个不可用帧；新 Dataset 继续以相机
长度作为有效范围，范围内逐值一致。

旧 `train.json`/`test.json` 协议下的窗口数量也完全一致：训练 CKD 21,005、CS 28,546、
插入关键帧 1,153；测试 CKD 654、CS 1,942、插入关键帧 185。

旧训练缓存与新 Dataset 现场拟合的 MinMax 统计也逐值一致：CKD pose、CS pose、camera
distance/position/rotation/FOV/eye 的 minimum 和 maximum 最大绝对误差均为 0。

## 默认分割

当前 `train_pre.json`/`test_pre.json` 实际构建结果：

| split | 虚拟片段 | CKD 窗口 | CS 窗口 | 插入关键帧 |
| --- | ---: | ---: | ---: | ---: |
| train | 145 | 18,437 | 24,587 | 997 |
| test | 43 | 533 | 2,000 | 164 |

## 模型链路

- 公开 `DSC_CKD.pt` 和 `DSC_CS.pt` 均严格加载成功。
- 推理默认读取 `model_state_dict`，与原测试脚本 `EMA_tag=False` 一致；EMA 只能通过配置
  显式选择。
- 使用同一公开权重和同一随机输入比较原模型类与迁移模型类，CKD/CS 前向输出均
  `torch.equal == true`，最大绝对误差为 0。
- `test_pre` 样本 `66_6` 完成 CKD -> CS 推理，输出 camera20 `(999, 20)` 和关键帧
  `(999,)`；相机为 float32 且全部有限。CKD 输出不改写首尾，CS Dataset 按原协议在
  内存中补首尾关键帧。
- 论文 benchmark 入口成功读取同一 NPY generation run 并生成指标报告。
- 原始 49 条 `test.json` 上分别运行原版与迁移推理，再统一使用原始
  `DanceStyleCam/scripts/evaluate.py`：DMR 完全相等，FID-S/Div-S/LCD 差异均小于
  `0.001`，FID-K 差异为 `0.0142`。完整数值见 `docs/FULL_DATASET_EVALUATION.md`。
- 同一 generation run 成功独立导出 VMD，并生成 5 帧 H.264、1280x720 的可视化冒烟视频。
- CKD、CS-GAN、CS-noGAN 均完成一次训练反向传播并保存 checkpoint。
- noGAN checkpoint 以 `use_gan: true` 续训时成功初始化新判别器，并保存包含判别器状态的
  GAN checkpoint。

## 软件检查

- Pytest：23 passed。
- Ruff：All checks passed。
- `compileall`：通过。
- 数据准备、数据校验、CKD/CS 训练、推理、指标、可视化和 VMD 的 Python CLI 均通过
  `--help` 入口检查。

108 个完整序列同时作为 train/test 的重组验证，以及公开 CKD/CS 对原始 49 条
`test.json` 的原版/迁移对照指标见 `docs/FULL_DATASET_EVALUATION.md`。
