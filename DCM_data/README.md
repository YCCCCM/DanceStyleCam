## About DCM Dataset

### Brief Structure

```
.DCM_data
+-- amc_camera_json # our preprocessed camera data with keyframe information
|   +-- c0.json
|   +-- c1.json
|   +-- ···
|   +-- c107.json
+-- amc_raw_data # Incomplete raw data including music audio, camera and motion data
|   +-- amc0
|   |   +-- a0.wav # for raw music audio; we have provided
|   |   +-- c0.vmd # for raw camera data; you may need to download from original links
|   |   +-- m0.vmd # for raw motion data; you may need to download from original links
|   +-- ···
|   +-- amc107
|   |   +-- a107.wav
|   |   +-- c107.vmd
|   |   +-- m107.vmd
+-- Simplified_MotionGlobalTransform # our preprocessed motion data with interpolated frames
|   +-- m0_GlobalTransform.json
|   +-- ···
|   +-- m107_GlobalTransform.json
+-- LinkOfRawData.xlsx # list of original links for raw camera and motion data
+-- README.md
```

### Usage

* **Try the DanceCamera3D model**: if you just want to train or test our model, you don't need to download the raw data. Or you can download the motion data of test set to use [Saba_Viewer](https://github.com/benikabocha/saba) to render results.
* **More Investigation to DCM dataset**: you can download, rename and move the raw data following `LinkOfRawData.xlsx` to get complete DCM dataset.
