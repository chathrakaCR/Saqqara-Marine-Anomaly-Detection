# Marine Anomaly Detection App

**Project P000285DS · RMIT University × Saqqara Aerospace**

A Docker app that watches drone footage of the open ocean and reports where **prawn swarms**, **tuna** and
**cetaceans** (whales and dolphins) appear. Give it a video file, a live stream or a folder of photos. It splits the
video into frames, runs each sampled frame through the project's final detection pipeline, and gives you a list of
sightings with times, annotated images and a per-frame report.

![How a video becomes answers](docs/images/architecture.png)

---

## Contents

1. [What the app does](#1-what-the-app-does)
2. [Project summary: the pipeline and its results](#2-project-summary-the-pipeline-and-its-results)
3. [How a frame is analysed](#3-how-a-frame-is-analysed)
4. [Final configuration](#4-final-configuration)
5. [Requirements](#5-requirements)
6. [Setup and first run (Windows)](#6-setup-and-first-run-windows)
7. [Using the web app](#7-using-the-web-app)
8. [Using the command line](#8-using-the-command-line)
9. [Testing without a drone video](#9-testing-without-a-drone-video)
10. [Results and output files](#10-results-and-output-files)
11. [Web API](#11-web-api)
12. [Project structure](#12-project-structure)
13. [Troubleshooting](#13-troubleshooting)
14. [Limitations and next steps](#14-limitations-and-next-steps)
15. [Full documentation](#15-full-documentation)

---

## 1. What the app does

- **Input:** a video file (mp4, mov, avi…), an `rtsp://` or `http://` live stream, a camera, or a folder of photos.
- **Frames:** analyses one frame per second by default, which you can change.
- **Answer per frame:** exactly one of **prawn · tuna · cetacean · none**.
- **Sightings:** consecutive frames with the same answer are merged into one sighting with a start and end time.
- **Output:** a CSV row for every frame (the answer and *why*), a sightings list, annotated images of every
  detection, an optional annotated video, and a summary of every setting used.
- **Two ways to use it:** a **web page** (upload, watch progress, browse results) and a **command line** for batch runs.
- **Self-contained:** runs offline on a normal laptop CPU. The models and settings are built into the Docker image.

---

## 2. Project summary: the pipeline and its results

The system is a chain of four parts. Each part was built, tuned and tested on its own, then the whole chain was
tested end to end. The table summarises the final results.

| Part | What it does | Final result |
|---|---|---|
| **Stage 1** Anomaly detector | Flags any frame where something stands out from normal sea, using texture **or** local brightness. No species knowledge. | **Recall 97.5%** (470 / 482 target photos flagged: prawn 82/82, tuna 189/200, cetacean 199/200). Flags 46% of empty sea on purpose, because later stages remove those. Precision 69.9%, F1 0.815. |
| **Stage 2** SegFormer-B0 mask | Outlines the flagged object on the whole frame. | Mask produced for **205 / 257** unseen prawn and tuna frames and only **2 / 1,000** empty-sea frames (precision 99.0%, recall 79.8%, F1 0.884, specificity 99.8%). Against hand-drawn masks: **tuna IoU 76.2%**, prawn IoU 47.4%. |
| **Stage 3 Track A** Prawn / tuna / other | Takes 12 measurements of the mask and its surroundings and uses a logistic regression. "Other" means a false alarm. | **62 / 65** held-out masks correct (**95.4%**, macro-F1 0.955). Prawn 10/10, tuna 38/40, other 14/15. |
| **Stage 3 Track B** Cetacean detector | A trained object detector that finds and boxes cetaceans on the whole frame. Cut-off 0.10, chosen on validation. | **Recall 95.9%** (424/442), **precision 99.1%**, **F1 0.975**, mAP@0.5 0.873. False alarms: prawn 2/32, tuna 1/36, sea 1/60. |
| **Combined pipeline** | All of the above in order, one answer per frame. | **238 / 256 unseen frames correct (93.0%)**, macro-F1 0.886. Prawn 26/32, tuna 135/143, cetacean 58/61, clean sea 19/20. |

**Combined pipeline: confusion matrix on 256 unseen frames.** Rows are the true class, columns are the app's answer.

|                | prawn | tuna | cetacean | none | total |
|----------------|------:|-----:|---------:|-----:|------:|
| **prawn**      | **26** | 2 | 2 | 2 | 32 |
| **tuna**       | 0 | **135** | 0 | 8 | 143 |
| **cetacean**   | 0 | 0 | **58** | 3 | 61 |
| **clean sea**  | 0 | 0 | 1 | **19** | 20 |

| Answer | Precision | Recall | F1 |
|---|---:|---:|---:|
| Prawn | 1.000 | 0.813 | 0.897 |
| Tuna | 0.985 | 0.944 | 0.964 |
| Cetacean | 0.951 | 0.951 | 0.951 |
| None | 0.594 | 0.950 | 0.731 |
| **Macro average** | 0.882 | 0.914 | **0.886** |

When the system names a target it is almost always right. Most errors are **missed** targets reported as "none",
for example 8 tuna frames that Stage 1 never flagged. Very few are wrong labels.

**What mattered most during development**

- **Stage 1:** adding the local-brightness rule raised tuna detection from 41.5% to 94.5%. The trade-off is more
  false alarms, which the later stages remove.
- **Stage 2:** running SegFormer on the whole frame instead of Stage 1's cut-out raised tuna IoU from 41.1% to 77.8%.
- **Track A:** measuring texture at the photo's own resolution made it robust to how the crop is resized.
- **Track B:** Stage 1 and SegFormer cannot locate small animals, so cetaceans need their own detector.
  Fine-tuning with prawn frames as negatives cut prawn false alarms from 12 to 2. Without a cetacean track the
  combined score drops to 70.7%.

---

## 3. How a frame is analysed

1. **Stage 1 (anomaly check).** Not flagged → **none**.
2. **Track B (cetacean detector).** Any box with confidence ≥ 0.10 → **cetacean**. This runs before Stage 2
   because Track A has no cetacean class.
3. **Stage 2 (SegFormer mask on the whole frame).** No mask of at least 80 px → **none**.
4. **Track A (12 measurements, then logistic regression).** prawn → **prawn**, tuna → **tuna**, other → **none**.

The detector only runs on frames Stage 1 has flagged. The end-to-end test ran it on every frame but only counted
its answer after the Stage 1 check, so the final answers are identical and the app does less work.

---

## 4. Final configuration

All settings are in [`config/pipeline_config.json`](config/pipeline_config.json). They are the settings validated
in the end-to-end test.

| Part | Setting | Value |
|---|---|---|
| **Stage 1** | image · superpixels · compactness · GLCM levels · glint | 512 × 512 · SLIC 250 · 20 · 32 · L > 220 |
| | sea-state buckets | 3, edges 98.49 / 132.11 (from 3,283 normal-sea photos) |
| | texture rule | z > 2.5 on ≥ 2 of 5 features, region ≥ 2,000 px |
| | local-brightness rule | z > 2.5, region ≥ 1,000 px |
| | region growing | dilation × 2 |
| **Track B** | detector · image size · cut-off | detector v2 · 352 px · **0.10** |
| | boxes kept · overlap removal | confidence ≥ 0.05 · IoU 0.7 |
| **Stage 2** | model · input | SegFormer-B0, epoch-30 checkpoint · whole frame |
| | probability · blobs · minimum | > 0.35 · largest only · ≥ 80 px |
| | crop for Track A | mask box + 20 px, joined with Stage 1's box if they overlap |
| **Track A** | features · model | 12 (shape 3, disturbance 3, scene colour 6) · logistic regression |
| | ring · native-size limits | 25 px · 24 px / 30 px |
| **Video** | sampling · sighting gap | 1 frame per second · gap > 2 s starts a new sighting |

> **Stage 2 note.** The Stage 2 page's best stand-alone setting is epoch 20 / probability 0.50. The app uses
> epoch 30 / 0.35 because that is the output Track A was trained on, and the end-to-end 93.0% was measured with it.
> Switching to epoch 20 / 0.50 would need Track A to be retrained first.

**Safety check.** On start-up the app compares these settings with the ones stored inside `track_a_model.pkl`. If
they differ, it refuses to run, so Track A always gets the input it was trained on.

### Model files (`models/`)

| File | What it is | Made by | In the zip |
|---|---|---|---|
| `stage1_bucket_ref.pkl` | normal-sea texture statistics per bucket | Stage_01_Pipeline_FINAL_v2 | ✅ |
| `segformer_b0_epoch30.pt` | SegFormer weights | Stage_02_Pipeline_FINAL | ✅ |
| `track_a_model.pkl` | Track A classifier, feature names, training settings | Stage_03_TrackA_v4_with_images_v3 | add |
| `cetacean_detector_v2.onnx` | cetacean detector in ONNX format | `tools/export_detector.py` | add |
| `cetacean_detector_v2.json` | detector cut-off (0.10) and image size (352) | Stage3_TrackB_Cetacean_Detector_v2 | add |

The detector runs through ONNX Runtime, so the container does not need the library it was trained with. The export
tool checks the converted detector against the original. On test images the confidences match to about 1e-7 and
the boxes to under a pixel, with no answer changing at the 0.10 cut-off.

---

## 5. Requirements

- **Docker Desktop** (Windows or Mac, with WSL 2 on Windows) or Docker Engine on Linux.
- About **4 GB** of free disk space for the image.
- For the one-time detector conversion: the Python environment the cetacean detector was trained in.
- No GPU needed.

---

## 6. Setup and first run (Windows)

Run these in **PowerShell**. Open Docker Desktop first and wait for **"Engine running"**.

**1. Folders and unzip**
```powershell
$PROJ = "C:\Users\bcura\AI PostGrad Project\Saqqara-Marine-Anomaly-Detection"
$APP  = "$PROJ\saqqara-anomaly-app"
Expand-Archive "$HOME\Downloads\saqqara-anomaly-app.zip" -DestinationPath $PROJ -Force
cd $APP
```

**2. The Python your notebooks use.** In any notebook cell run `import sys; print(sys.executable)`, then paste the
path it prints:
```powershell
$NBPY = "C:\path\to\python.exe"
```

**3. Add the model files**
```powershell
Copy-Item "$PROJ\Combined Pipeline\track_a_model.pkl" models\
Copy-Item "$PROJ\Stage_03\stage3_output\cetacean_detector_v2.json" models\
& $NBPY -m pip install onnx onnxslim onnxruntime opencv-python
& $NBPY tools\export_detector.py --weights "$PROJ\Stage_03\stage3_output\cetacean_detector_v2.pt" --out models --check-images "$PROJ\Stage_03\Test_data\SEA"
dir models          # five files: .json, .onnx, segformer .pt, bucket .pkl, track_a .pkl
```
The export should end with **"0 images would get a different cetacean answer"**.

**4. Match the scikit-learn version** that saved `track_a_model.pkl`:
```powershell
$SK = (& $NBPY -c "import sklearn; print(sklearn.__version__)").Trim()
(Get-Content docker-compose.yml) -replace 'SKLEARN_VERSION: "[^"]*"', "SKLEARN_VERSION: `"$SK`"" | Set-Content docker-compose.yml
```

**5. Build and start.** The first build takes about 5–10 minutes.
```powershell
docker compose build
docker compose up -d
Start-Process "http://localhost:8000"
```
Wait until the page says **"Models ready · cpu · detector cut-off 0.1"**.

**Every time after that:**
```powershell
cd "C:\Users\bcura\AI PostGrad Project\Saqqara-Marine-Anomaly-Detection\saqqara-anomaly-app"
docker compose up -d        # start
docker compose down         # stop
```
Rebuild (`docker compose build`) only after changing a model file, the config or the code.

<details>
<summary>Linux / macOS equivalent</summary>

```bash
cd saqqara-anomaly-app
cp /path/to/track_a_model.pkl /path/to/cetacean_detector_v2.json models/
python tools/export_detector.py --weights /path/to/cetacean_detector_v2.pt --out models --check-images /path/to/images
docker compose build --build-arg SKLEARN_VERSION=$(python -c "import sklearn; print(sklearn.__version__)")
docker compose up -d
```
</details>

---

## 7. Using the web app

Open **http://localhost:8000**.

![A finished run](docs/images/web_run_overview.png)

- **New analysis (left).** Choose a video file, or paste a stream URL. Set how often to analyse a frame and when to
  stop (0 = end of video). Tick *Annotated video* and/or *Keep every extracted frame*, then click **Start**.
- **Runs.** Every run in this session. Runs are processed one at a time.
- **Run panel (right).** Status, progress bar, frame counts per answer, download links (`frames.csv`,
  `events.json`, `summary.json`, `annotated.mp4`), how many frames Stage 1 flagged, and the time per frame.

![Sightings table](docs/images/web_sightings.png)

**Sightings** list each run of consecutive frames with the same answer: the answer, from / to time, number of
frames and, for cetaceans, the best detector confidence.

![Detection gallery](docs/images/web_detection_gallery.png)

**Gallery.** Shows one annotated frame per sighting. Click an image to open it full size. On each image:

- **Banner:** final answer and time.
- **Yellow box:** Stage 1's region.
- **Green boxes:** cetaceans found by the detector, with confidence.
- **Coloured outline:** the SegFormer mask that Track A labelled.

---

## 8. Using the command line

Put videos in the app's `data\` folder. Inside the container that folder is `/data`.

```powershell
docker compose run --rm marine-anomaly python -m app.cli -i /data/flight1.mp4 -o /data/results/flight1 --every 1 --save-video
```

| Option | Meaning |
|---|---|
| `-i` | video file, `rtsp://` / `http://` URL, camera number, or folder of photos |
| `-o` | results folder (default `/data/results/run`) |
| `--every 1` | analyse one frame every N seconds (default from the config: 1) |
| `--every-n-frames 30` | analyse every 30th frame instead (files only) |
| `--max-seconds 600` | stop after 10 minutes, useful for streams |
| `--save-video` | write `annotated.mp4` |
| `--save-frames` | keep every extracted frame as a JPEG |
| `--device cuda` | use a GPU if the image was built with one |

**Live stream on the same PC:** use `rtsp://host.docker.internal:8554/...`, because inside Docker `localhost` means
the container.

---

## 9. Testing without a drone video

You can test with the labelled photos in `Stage_03\Test_data`. First copy them into `data\`:

```powershell
New-Item -ItemType Directory -Force data | Out-Null
if (-not (Test-Path data\Test_data)) { Copy-Item -Recurse "$PROJ\Stage_03\Test_data" data\Test_data }
python tools\copy_cetacean_test.py      # copies the detector's 61 test cetacean photos to data\cetacean_test
```

**Option A: run one folder per class and score it.** Each photo is analysed once at its own size, the same way
the end-to-end test read them.
```powershell
docker compose run --rm marine-anomaly python -m app.cli -i /data/Test_data/TUNA -o /data/results/tuna
docker compose run --rm marine-anomaly python tools/score_run.py --results /data/results/tuna --label tuna
```
Repeat with `PRAWNS` (`--label prawn`), `SEA` (`--label none`) and `/data/cetacean_test` (`--label cetacean`).

**Option B: build a labelled test video, run it, and score it.**
```powershell
docker compose run --rm marine-anomaly python tools/make_test_video.py --folder prawn=/data/Test_data/PRAWNS --folder tuna=/data/Test_data/TUNA --folder none=/data/Test_data/SEA --folder cetacean=/data/cetacean_test --per-class 15 --out /data/test_video.mp4
docker compose run --rm marine-anomaly python -m app.cli -i /data/test_video.mp4 -o /data/results/test_video --every 1 --save-video
docker compose run --rm marine-anomaly python tools/score_run.py --results /data/results/test_video --truth /data/test_video_truth.csv
```
You can also upload `data\test_video.mp4` on the web page. To score that run, use `--results /data/results/<run id>`.

`score_run.py` prints the confusion matrix, accuracy, precision, recall and F1, and lists every wrong answer. It
also saves `score.json`.

**Example test run** (60-second video, 15 photos each of prawn, tuna, clean sea and cetacean, one frame per second):

| Measure | Result |
|---|---|
| Frames analysed | 60 |
| Flagged by Stage 1 | 57 of 60 |
| Answers | prawn 13 · tuna 14 · cetacean 15 · none 18 |
| Sightings | 35 (each photo is a new scene, so most sightings are one frame long) |
| Speed | 0.91 s per frame · 60.6 s in total |

`Test_data` still contains photos used during development, and the video is cropped to 16:9 and compressed. Treat
this as a functional check of the app, not a new accuracy figure. The independent figure is the 93.0% in
[section 2](#2-project-summary-the-pipeline-and-its-results).

---

## 10. Results and output files

Each run writes `data/results/<run>/`:

| File | Contents |
|---|---|
| `frames.csv` | One row per analysed frame: frame number, time, source file (folders), **answer**, which step decided it and **why**, Stage 1 rule and box, detector best confidence and boxes, mask size, SegFormer max probability, Track A label and probabilities, time taken, annotated image |
| `events.json` | Sightings: answer, start and end time, frames, best detector confidence, example image |
| `summary.json` | Video details, frames analysed, counts per answer, frames flagged by Stage 1, timing, and **every setting used** |
| `detections/` | Annotated JPEG of every prawn, tuna or cetacean frame |
| `annotated.mp4` | (optional) every analysed frame with the answer drawn on it |
| `frames/` | (optional) every analysed frame as extracted |
| `score.json` | (after `score_run.py`) confusion matrix, accuracy, per-class precision / recall / F1 |

Results stay in the `data` folder after the app stops.

---

## 11. Web API

| Method | Path | Use |
|---|---|---|
| GET | `/api/health` | models loaded, device, settings |
| POST | `/api/jobs` | start a run. Form fields: `file` or `url`, `every`, `max_seconds`, `save_video`, `save_frames` |
| GET | `/api/jobs` · `/api/jobs/{id}` | status, progress, counts, sightings, summary |
| POST | `/api/jobs/{id}/stop` | stop a run |
| GET | `/api/jobs/{id}/files/{path}` | download a result file or image |

Example:
```bash
curl -F file=@flight1.mp4 -F every=1 http://localhost:8000/api/jobs
```

---

## 12. Project structure

```
saqqara-anomaly-app/
├── app/
│   ├── stage1.py          Stage 1 anomaly check (ported unchanged from the notebook)
│   ├── track_b.py         cetacean detector with ONNX Runtime
│   ├── stage2.py          SegFormer mask on the whole frame + crop for Track A
│   ├── track_a.py         12 measurements, classifier, settings check
│   ├── pipeline.py        decision order for one frame
│   ├── video.py           reading videos / streams / folders, frame sampling
│   ├── runner.py          runs a source, writes results and sightings
│   ├── annotate.py        draws boxes, masks and the answer
│   ├── cli.py             command line
│   ├── server.py          web API
│   └── static/index.html  web page
├── config/pipeline_config.json   all final settings
├── models/                        model files (see section 4)
├── tools/
│   ├── export_detector.py     convert the trained detector to ONNX and check it
│   ├── make_test_video.py     build a labelled test video from photos
│   ├── score_run.py           confusion matrix / precision / recall / F1 for a run
│   └── copy_cetacean_test.py  copy the detector's test photos
├── tests/smoke_test.py    quick end-to-end check on a synthetic video
├── docs/images/           pictures used in this README
├── Dockerfile             CPU image (Python 3.11, PyTorch CPU, ONNX Runtime)
├── docker-compose.yml     port 8000, ./data mounted at /data, scikit-learn version
└── requirements.txt
```

---

## 13. Troubleshooting

Check the logs first: `docker compose logs`

| Problem | Fix |
|---|---|
| "Model files not found" | A file in `models\` is missing or has a different name (see section 4). |
| "Settings differ from what Track A was trained with" | The message lists what differs. Fix `config/pipeline_config.json`, or set `ALLOW_SETTINGS_MISMATCH=1` (not recommended). |
| `InconsistentVersionWarning` or the Track A file fails to load | Wrong scikit-learn version. Repeat step 4, then `docker compose build`. |
| `can't open file '/app/tools/…'` | The image is older than the code. Run `docker compose build`. |
| "No module named …" when exporting | You are using a different Python. Use the notebook's `python.exe` (step 2). |
| Port 8000 already in use | In `docker-compose.yml`, change `"8000:8000"` to `"8080:8000"`. |
| A stream on the same PC won't open | Use `host.docker.internal` instead of `localhost`. |
| USB webcam on Windows | Docker Desktop cannot pass webcams through. Stream from the camera instead. |
| Very slow | Analyse fewer frames (`--every 2`). Close other heavy programs. |

---

## 14. Limitations and next steps

- **Not yet tested on real drone video.** The 93.0% comes from photographs. Motion blur, compression and changing
  altitude may change results. Next step: run one or two raw flight recordings from Saqqara and review the
  sightings by eye.
- **Small test groups.** The prawn (32) and clean-sea (20) test groups are small, so their figures are uncertain.
  A fresh hold-out set from new footage, scored once, would give an unbiased number.
- **Stage 2 setting.** Epoch 30 / 0.35 is used to match Track A. See the [Stage 2 note](#4-final-configuration).
- **One answer per frame.** A frame showing two kinds of target gets a single label.
- **Speed.** About 0.9 s per frame on CPU, mostly Stage 1.
- **Local use only.** No login, and the run list is kept in memory (result files are kept on disk). Run it on a
  laptop or a private network, not the open internet.

---

## 15. Full documentation

Confluence space *RMIT - Saqqara - Documentation*:

- [Stage 1 – Anomaly detector](https://saqqara-aerospace-s2-2026.atlassian.net/wiki/spaces/~712020f83fcc8ee2794144a9b9b17f0661e021/pages/13697026)
- [Stage 2 – SegFormer mask](https://saqqara-aerospace-s2-2026.atlassian.net/wiki/spaces/~712020f83fcc8ee2794144a9b9b17f0661e021/pages/13697065)
- [Stage 3 Track A – Prawn / tuna / other classifier](https://saqqara-aerospace-s2-2026.atlassian.net/wiki/spaces/~712020f83fcc8ee2794144a9b9b17f0661e021/pages/13762608)
- [Stage 3 Track B – Cetacean detector](https://saqqara-aerospace-s2-2026.atlassian.net/wiki/spaces/~712020f83fcc8ee2794144a9b9b17f0661e021/pages/13762638)
- [Combined pipeline – End-to-end results](https://saqqara-aerospace-s2-2026.atlassian.net/wiki/spaces/~712020f83fcc8ee2794144a9b9b17f0661e021/pages/13729877)
- [Marine Anomaly Detection App (Docker)](https://saqqara-aerospace-s2-2026.atlassian.net/wiki/spaces/~712020f83fcc8ee2794144a9b9b17f0661e021/pages/14090243)

---

*Author: Chathraka Ranaweera · Master of Data Science, RMIT University · Capstone P000285DS with Saqqara Aerospace*
