<div align="center">
<h2><font color="red"> 🎥🎥🎥 DanceStyleCam 🎥🎥🎥 </font></center> <br> Style-Based 3D Multi-Style Dance Camera Movement Synthesis</h2>

</div>

<p float="left">
	<img src="assets/1.gif" width="200" /> <img src="assets/2.gif" width="200" /> <img width="200" src="assets/3.gif"/> <img src="assets/4.gif" width="200" />
	</p>

## Abstract
Fully automatic camera movement directly affects the art quality of dance expressiveness, especially in terms of visual expression, as well as choreography and music. 
Current studies mainly focus on synthesizing camera movements conditioned on dance and music, but they overlook the camera movement style, which is essential factor for artistic and visual coherence. In this paper, we introduce DanceStyleCam, a unified framework that incorporates the style-consistent characteristic into dance camera movement synthesis with diverse stylistic characteristics. 
Specifically, a style-aware feature learning module is proposed to map dance style information into compact embeddings, facilitating stable and discriminative style learning. To further guarantee that the generated camera movements remain faithful to the target style, we propose a style-consistent adversarial training scheme, leading and optimizing the model to learn better style-consistent representations. In addition, we also enrich the DCM dataset with diverse camera movement style annotations. 
Extensive experiments demonstrate that DanceStyleCam outperforms state-of-the-art methods in both generation quality and style consistency. Qualitative results further show that our method produces stylistically consistent camera movements while preserving smooth trajectories and natural shot transitions.

# Setup Environment
Our method is trained using cuda 12.1 toolkit on 5 Nvidia Geforce RTX3090 GPUs.
``` 
pip install -r requirements.txt     # RTX3090 is OK
```
* We recommend Linux for performance and compatibility reasons. Windows is OK, please see `dsc_win.yaml`.
* 64-bit Python 3.10
* PyTorch 2.3.1 or PyTorch 2.9.0
* At least 24 GB RAM per GPU
* CUDA 12.1 toolkit or CUDA 12.8 toolkit.
* Recommended training configuration: 5 NVIDIA GPUs with at least 24 GB of GPU memory.
* !!! Because it includes adversarial training, do not easily modify the recommended configuration !!! 

The train and inference example build this repo was validated on:
* Ubuntu 24.04 LTS or Windows11 25H2
* 64-bit Python 3.10
* Train: Ubuntu 24.04 LTS, 5 x NVIDIA Geforce RTX3090, CUDA 12.1 toolkit, PyTorch 2.3.1, 256 GB RAM
* Inference, Test, Render and Visualization: Windows11 25H2, 1 x NVIDIA Geforce RTX5090, CUDA 12.8 toolkit, PyTorch 2.9.0, 96 GB RAM


# Getting started

## Data preparation

### Download DCM Dataset and prepare for DanceStyleCam
* Download and Check the DCM dataset.
* Put ```music_style_16cat.json``` in ```DCM_data``` and ```DCM_data/split```,
* Preprocess the data by running code.
```bash
python tools/data/prepare_dcm_style_pp.py --config configs/data/dcm_style_pp.yaml
```

* Make DCM++ dataset which is keyframe-aware by running
```bash
python tools/data/prepare_dcm_style_pp.py --config configs/data/dcm_style_pp.yaml
```

The dataset file structure is as follows:
```bash
DanceStyleCam
├── DCM_data
│   ├── amc_aligned_data
│   ├── amc_aligned_data_split
│   │── amc_camera_json
│   ├── amc_data_split_by_categories
│   ├── amc_raw_data
│   ├── Simplified_MotionGlobalTransform
│   ├── split
│   │   ├──music_style_16cat.json
│   │   ├──...
│   └── music_style_16cat.json
└── DCM++
```

## Model Test
* Contact the first author to obtain a link to download our trained checkpoints.
* Put the downloaded checkpoints under `checkpoints` folder and rename them as `DSC_CKD.pt` and `DSC_CS.pt`.
* synthesis keyframe information with CKD model by running
```.bash
python infer/generate.py --config configs/infer/default.yaml
```

* synthesis camera movement with CS model (given keyframe information from CKD model)
```.bash
python infer/generate.py --config configs/infer/default.yaml
```

* controllable inference on the test split or custom MMD dance and music
```bash
python infer/generate_test_controlled.py --config configs/infer/controlled_test.yaml
python infer/generate_custom.py --config configs/infer/custom.yaml --dance input/dance.vmd --pmx input/model.pmx --music input/music.wav
```
Control file formats are documented in [docs/CONTROLLABLE_INFERENCE.md](docs/CONTROLLABLE_INFERENCE.md).

## Evaluate
* Evaluation of quantitative results of synthesis and quantitative results of style consistency
```.bash
python tools/eval/run_benchmark.py --config configs/eval/benchmark.yaml --input generation/<run-name>
```

## Model Training
Training the Camera Keyframe Detection model
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 train/train_ckd.py --config configs/train/ckd.yaml
```

Training the Camera Synthesis model
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes 2 train/train_cs.py --config configs/train/cs.yaml
```

## Visualization and Render
### Visualization I
Once the training is done, run inference and render:
```bash
python tools/visualization/visualize.py --config configs/visualization/default.yaml --input generation/<run-name>
```

### Visualization II
* If you want to experience better visualization, convert the results to `.vmd` format that can be viewed in [Saba_Viewer] by running

```.bash
python tools/visualization/export_vmd.py --input generation/<run-name>
```

# Citation 
If you think this project is helpful, please cite our paper:
```bibtex
@inproceedings{huang2026dsc,
  title={DanceStyleCam: Style-Based 3D Multi-Style Dance Camera Movement Synthesis},
  author={Xiaoying Huang},
  booktitle={2026 IJCAI},
  year={2026},
}
``` 

# Acknowledgements
Thank you for reviewing my manuscript.
