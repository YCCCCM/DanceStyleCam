# DanceStyleCam-Official 项目结构

项目采用 AnchorDance 风格的顶层模块结构，不使用额外的 `dancestylecam/` 包装目录。
架构约束以根目录 `MIGRATION_SCOPE.md` 为准。

## 顶层目录

| 路径 | 作用 |
| --- | --- |
| `DCM_data/` | 网络下载并解压后的公开 DCM 原始数据，只读。 |
| `DCM-style++/` | 从原始数据生成的完整序列 NPY，不包含物理 Train/Test 副本。 |
| `common/` | 配置读取、路径解析以及跨模块公共 IO。 |
| `configs/` | 数据、训练、推理、可视化和评估 YAML。 |
| `data/` | 数据预处理、数据协议和虚拟 Dataset 代码，只放源码。 |
| `models/` | CKD、CS、Transformer 主干和可选判别器。 |
| `train/` | CKD 与 CS 训练入口及训练公共实现。 |
| `infer/` | CKD -> CS 推理和生成结果协议。 |
| `metric/` | 评估指标实现。 |
| `tools/` | 数据准备、可视化、VMD 和评估命令入口。 |
| `generation/` | 推理运行产物。 |
| `runs/` | 训练日志和周期 checkpoint。 |
| `checkpoints/` | 下载的公开模型文件。 |
| `tests/` | 软件自动化测试，不是模型测试输出。 |

## `common/`

| 文件 | 作用 |
| --- | --- |
| `config.py` | 读取 YAML，支持 `base_config` 和深度合并，并验证必需配置段。 |
| `paths.py` | 定义项目根路径，统一解析相对/绝对数据路径。 |

## `data/`

| 文件 | 作用 |
| --- | --- |
| `schema.py` | 固定 NPY schema、30 FPS、camera20 字段和规范风格列表。 |
| `raw_dcm.py` | 根据 sequence ID 定位 WAV、相机 JSON 和动作 JSON。 |
| `audio_features.py` | 提取 35 维 AIST 音乐特征；支持完整序列存储和旧 DCM++ 的片段级精确兼容模式。 |
| `camera_geometry.py` | 相机插值、坐标变换、camera20 和 bone mask 计算。 |
| `prepare.py` | `DCM_data -> DCM-style++` 转换实现，支持原子写入、校验和与断点续做。 |
| `validate.py` | 校验原始文件、NPY shape/dtype/checksum、manifest 和 split 范围。 |
| `store.py` | 通过 `np.load(..., mmap_mode="r")` 按需读取完整序列 NPY。 |
| `splits.py` | 读取 `train_pre/test_pre/long2short`，构建不复制数据的虚拟 clip。 |
| `style_labels.py` | 读取 16 类标注，并显式管理规范顺序与公开 checkpoint 历史顺序。 |
| `normalization.py` | 按训练 split 计算和保存小型 min-max 统计量。 |
| `dataset_common.py` | Dataset 上下文、零填充窗口和公共切片逻辑。 |
| `ckd_dataset.py` | 动态 CKD 滑窗，不写 pickle 或窗口级 NPY。 |
| `cs_dataset.py` | 动态关键帧中心 CS 窗口，支持 CKD 生成 mask 和插入关键帧。 |

## `models/`

| 文件 | 作用 |
| --- | --- |
| `backbone.py` | 与公开 CKD/CS checkpoint 严格匹配的 Transformer 主干。 |
| `rotary_embedding.py` | Rotary position embedding。 |
| `transformer_utils.py` | Transformer 公共层与位置编码。 |
| `ckd.py` | 构建公开 60+60 CKD 模型。 |
| `cs.py` | 构建公开 8 维 polar、16 类风格 CS 生成器。 |
| `discriminator.py` | 仅 `training.use_gan: true` 时创建的风格判别器。 |
| `checkpoint.py` | 加载公开旧 checkpoint、自定义 normalizer 兼容和严格权重加载。 |

不迁移 long-sequence、LSC 或 AWTS 模型文件。

## `train/`

| 文件 | 作用 |
| --- | --- |
| `train_ckd.py` | CKD 1500-epoch 配置训练入口。 |
| `train_cs.py` | GAN/no-GAN 共用的 CS 训练入口。 |
| `losses.py` | 重建、速度、加速度和 body-attention 损失。 |
| `train_utils.py` | Accelerator、EMA、运行目录、JSONL 指标和 portable checkpoint。 |
| `adan.py` | 原项目使用的 Adan optimizer。 |

`train_cs.py` 的 checkpoint 恢复规则：

- `use_gan: true` 且 checkpoint 有 D：恢复 D 和 D optimizer。
- `use_gan: true` 且 checkpoint 无 D：初始化新的 D 和 D optimizer。
- `use_gan: false`：不创建 D，忽略 checkpoint 中可能存在的 D。

## `infer/`

| 文件 | 作用 |
| --- | --- |
| `generate.py` | `python infer/generate.py --config ...` 命令入口。 |
| `generate_test_controlled.py` | test split 可控推理；支持覆盖关键帧时序、关键帧相机参数和风格。 |
| `generate_custom.py` | 自定义 VMD/PMX 或 motion180 舞蹈与音乐推理，并现场提取 music35。 |
| `pipeline.py` | CKD 自回归 mask 拼接、CS 自回归相机拼接和 camera20 重建。 |
| `result_io.py` | generation run 的 manifest、camera NPY 和 keyframe NPY 协议。 |

推理只加载 generator/EMA 和 normalizer，不加载判别器。

## `metric/`

| 文件 | 作用 |
| --- | --- |
| `evaluate.py` | 读取已有 generation run，计算轨迹指标、论文 kinetic/shot FID 与 diversity、dancer missing rate、limbs capture difference，以及可选的 30-D style consistency。 |
| `features.py` | 在 NPY 上按旧论文定义提取 kinetic、shot 和 style 特征；不创建 JSON 或特征缓存副本。 |

## `tools/`

| 文件 | 作用 |
| --- | --- |
| `data/prepare_dcm_style_pp.py` | 数据转换命令入口。 |
| `data/validate_dcm_style_pp.py` | 数据校验命令入口。 |
| `data/compare_legacy_dcmpp.py` | 将 NPY/虚拟片段逐字段与已有旧 DCM++ 比较，输出 JSON 报告。 |
| `visualization/visualize.py` | 读取 camera20 和动作 NPY，在同一 run 中创建 `vis/`。 |
| `visualization/export_vmd.py` | 将 camera20 转成 VMD，在同一 run 中创建 `vmd/`。 |
| `visualization/vmd.py` | 无额外项目依赖的 VMD camera writer。 |
| `eval/run_benchmark.py` | 调用 `metric/evaluate.py` 的评估入口。 |

## `configs/`

| 文件 | 作用 |
| --- | --- |
| `data/dcm_style_pp.yaml` | 原始目录、NPY 目录、标注、split 和动态窗口设置。 |
| `train/ckd.yaml` | 公开 CKD 1500-epoch 配置。 |
| `train/cs.yaml` | 公开 GAN CS 1200-epoch 配置。 |
| `train/cs_nogan.yaml` | 继承 CS 配置并关闭 D 的 no-GAN 配置。 |
| `infer/default.yaml` | 公开 CKD/CS 权重和 test split 推理配置。 |
| `infer/controlled_test.yaml` | test split 三类可控推理配置。 |
| `infer/custom.yaml` | 自定义舞蹈、音乐和可选控制推理配置。 |
| `visualization/default.yaml` | 已有 generation run 的可视化配置。 |
| `eval/benchmark.yaml` | 已有 generation run 的评估配置。 |

## 数据与运行产物

`DCM-style++/manifest.json` 只记录稳定数值数据：schema、帧数、dtype、相对路径和
checksum。它不记录 style 或 split。更改 `music_style_16cat.json`、`train_pre.json` 或
`test_pre.json` 后无需重新预处理。

`music35/` 中的 NPY 是完整序列表示。默认 `dataset.music_feature_mode: legacy_clip` 会在
内存中按虚拟片段从 `DCM_data` 音频提取音乐特征，以复现旧 DCM++ “先裁 WAV、再提特征”
的边界和 beat 行为；结果使用进程内缓存，不写 split 专属文件。若只需要更快的完整序列
切片，可显式使用 `sequence_npy`，但该模式不保证片段边界处与旧版逐值一致。

```text
generation/<run-name>/
├── manifest.json
├── config.yaml
├── camera/
├── keyframes/
├── inputs/    # 自定义输入模式按需保存 motion180/music35
├── vis/       # 按需创建
├── metrics/   # 按需创建
└── vmd/       # 按需创建
```

顶层不设置含义模糊的 `outputs/`：训练产物进入 `runs/`，推理和后处理产物进入
`generation/`。
