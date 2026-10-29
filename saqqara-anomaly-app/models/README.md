# Model files

The app needs these five files in this folder. File names must match `config/pipeline_config.json`.

| File | Made by | Included |
|---|---|---|
| `stage1_bucket_ref.pkl` | Stage_01_Pipeline_FINAL_v2 (cell 7) | yes |
| `segformer_b0_epoch30.pt` | Stage_02_Pipeline_FINAL (training cell) | yes |
| `track_a_model.pkl` | Stage_03_TrackA_v4_with_images_v3 (save cell) | **add it** |
| `cetacean_detector_v2.onnx` | `tools/export_detector.py` from `cetacean_detector_v2.pt` | **add it** |
| `cetacean_detector_v2.json` | Stage3_TrackB_Cetacean_Detector_v2 (cell 9) | **add it** |

Use the same `track_a_model.pkl` the integrated end-to-end test used (the one saved 2026-09-30).
The app checks that its settings match the ones stored inside that file, and refuses to start if they differ.

The `.json` file holds the detector's validated cut-off (0.10) and image size (352). If it is missing, the values
in `config/pipeline_config.json` are used.
