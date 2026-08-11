# 全量分割与公开模型评估

验证日期：2026-08-03。

## 全量 train=test 重组

`train_pre.json + test_pre.json` 的并集只有 106 个序列，缺少 29 和 83，因此本次使用
`C_0` 到 `C_107` 共 108 个完整序列，并让临时 train/test 同时指向这份列表。

| Dataset | train clips | test clips | train windows | test windows |
| --- | ---: | ---: | ---: | ---: |
| CKD | 108 | 108 | 22,980 | 5,785 |
| CS | 108 | 108 | 30,041 | 30,041 |

CS train/test 均插入 1,296 个间隔关键帧。CKD 与 CS 的首、中、末窗口均完成实际读取，
camera、motion、music、mask 和 style 数组 shape 正确且全部为有限值。

未向 Dataset 传入 normalizer。代码自动扫描临时 train split，并为 train/test 分别创建
pose、camera distance/position/rotation/FOV/eye 统计。两个 split 的所有 minimum/maximum
数组逐值相等，CKD 与 CS 的 pose normalizer 也逐值相等。这证明 split JSON 可以自由重组；
测试集始终自动使用配置中训练集所定义的统计范围，不需要 pickle 缓存或重建 NPY。

## 旧窗口协议复核

为排除指标差异来自重构，额外将旧 DCM++ 保存的窗口与新 Dataset 逐项比较：

- CKD：654/654 个窗口，keyframe、padding、music 完全相等；motion 最大误差
  `2.98e-7`。
- CS：1,942/1,942 个窗口，inference mask、bone mask、music、style、padding 完全相等；
  camera/motion 最大误差分别为 `2.38e-7`/`2.98e-7`。
- 所有连续字段误差均小于 `1e-6`，没有 shape mismatch。
- 原模型类与迁移模型类在相同公开权重和输入下，CKD/CS 前向最大误差均为 0。

## 公开权重原版协议复现

公开模型验证必须使用旧 `DCM_data/split/test.json` 对应的 49 个 `DCM++/Test` 片段，而
不是默认开源分割 `test_pre.json` 的 43 个片段。本次使用：

- `DSC_CKD.pt`：SHA-256 `7a4080475e47c45374e66ef12574b94da8a0f22111fdb4b043220b20bba30985`
- `DSC_CS.pt`：SHA-256 `a90dd6cfcc083f4e5fe0a5d945729b8ba5041d4647245e743fc944113c520cb3`
- checkpoint 权重：`model_state_dict`，与原始测试脚本的 `EMA_tag=False` 一致。

先在 `DanceStyleCam` 中运行原始 CKD、原始 CS，再运行原始 `scripts/evaluate.py`。随后在
`DanceStyleCam-Official` 中读取 `DCM-style++` NPY 完成同样的 49 条推理，将结果临时导出
为原 JSON 格式，并再次使用同一个原始 `scripts/evaluate.py` 评价。

## 论文生成指标对照

| 指标 | 原版推理 | 迁移推理 | 迁移 - 原版 |
| --- | ---: | ---: | ---: |
| FID-K (result) | 1.096687 | 1.110885 | +0.014197 |
| Div-K (result) | 3.260620 | 3.253655 | -0.006965 |
| Div-K (test) | 3.331566 | 3.331566 | 0 |
| FID-S (result) | 0.104586 | 0.104218 | -0.000368 |
| Div-S (result) | 1.455577 | 1.456462 | +0.000885 |
| Div-S (test) | 1.734189 | 1.734189 | 0 |
| DMR (result) | 0.00393223 | 0.00393223 | 0 |
| DMR (test) | 0.00931595 | 0.00931595 | 0 |
| LCD | 0.121292 | 0.121255 | -0.000037 |

两次评价的 kinetic result/test 均为 480 个特征，shot result/test 均为 37,843 个特征。
DMR 完全一致；FID-S、Div-S 和 LCD 的差异均小于 `0.001`。

原始 JSON 窗口和新 NPY 窗口的连续输入最大误差小于 `3e-7`。公开 CKD 的少数 logits
位于分类边界，float32 NPY 量化使本次生成结果中 52/37,892 个关键帧标签与原版不同；
其中 2 个是迁移代码曾提前强制首尾关键帧造成，现已改为与原版一样只在 CS Dataset
内部补首尾。剩余边界翻转解释了 FID-K 的小幅差异，不改变整体评价结论。

此前在 43 条 `test_pre` 上得到的高 DMR / 高 FID-S 使用了错误的 EMA 权重，同时与原版
49 条评价协议混用，不能作为公开模型结果，已从本报告撤销。

## 风格一致性对照

两组结果均使用原始 `scripts/evaluate_style_consistency.py` 和 49 条原型集。原脚本会递归
同时发现根目录与评价阶段复制到 `CameraCentric` 的同一批文件，因此日志显示 98 条；
每条结果被等量重复，不改变以下比例和均值。

| 指标 | 原版推理 | 迁移推理 |
| --- | ---: | ---: |
| Top-1 accuracy | 0.693878 | 0.693878 |
| Top-2 accuracy | 0.775510 | 0.775510 |
| Top-3 accuracy | 0.836735 | 0.836735 |
| Top-5 accuracy | 1.000000 | 1.000000 |
| Average target distance | 1.902795 | 1.894360 |
| Average target similarity | 0.726640 | 0.734175 |
| Similarity std | 0.239630 | 0.231987 |
| Similarity max | 0.989629 | 0.989585 |
| Similarity min | 0.095875 | 0.095876 |
| Similarity median | 0.827490 | 0.827490 |
| Quality score | 0.778599 | 0.780482 |
