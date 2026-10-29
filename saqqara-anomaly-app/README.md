# Marine anomaly detection app (P000285DS)

A Docker app that takes a drone video (a file, a live stream or a camera), splits it into frames and runs each
sampled frame through the final pipeline:

1. **Stage 1 – anomaly detector.** Texture rule OR local-brightness rule. Not flagged → **none**.
2. **Stage 3 Track B – cetacean detector (v2).** Any box with confidence ≥ 0.10 → **cetacean**.
3. **Stage 2 – SegFormer-B0** on the whole frame (probability > 0.35, largest blob, ≥ 80 px). No mask → **none**.
4. **Stage 3 Track A – logistic regression** on 12 measurements of the mask → **prawn**, **tuna**, or other → **none**.

Every frame ends as exactly one of prawn · tuna · cetacean · none. Consecutive frames with the same answer are
merged into **sightings** with start and end times.

All final settings are in [`config/pipeline_config.json`](config/pipeline_config.json).

## 1. Add the model files

Copy into `models/` (see [`models/README.md`](models/README.md)):

- `track_a_model.pkl`
- `cetacean_detector_v2.json`
- `cetacean_detector_v2.onnx`. Make it once from the trained `.pt` file on the machine where you trained it:

  ```bash
  python tools/export_detector.py --weights path/to/cetacean_detector_v2.pt --out models/ --check-images path/to/some/test/images
  ```

  The check compares the original and the converted detector on those images and prints the largest difference
  (expect about 0.000001 and 0 changed answers).

## 2. Build

Find the scikit-learn version that saved `track_a_model.pkl` (in that environment run
`python -c "import sklearn; print(sklearn.__version__)"`) and build with it:

```bash
docker build --build-arg SKLEARN_VERSION=1.6.1 -t marine-anomaly .
```

or edit `SKLEARN_VERSION` in `docker-compose.yml` and run `docker compose build`.

## 3. Run

**Web app** — upload a video or give a stream URL, follow progress, look through the sightings, and download the results:

```bash
docker compose up -d          # then open http://localhost:8000
```

**Command line** — one video, results written to `./data/results/flight1`:

```bash
docker run --rm -v "$PWD/data:/data" marine-anomaly \
  python -m app.cli -i /data/flight1.mp4 -o /data/results/flight1 --every 1 --save-video
```

Live stream: `-i rtsp://<drone-or-server>:8554/stream --max-seconds 600`.
Camera: `--device /dev/video0` on `docker run` and `-i 0`.

| Option | Meaning |
|---|---|
| `--every 1` | analyse one frame per second of video (default from the config) |
| `--every-n-frames 30` | analyse every 30th frame instead |
| `--max-seconds 600` | stop after 10 minutes |
| `--save-frames` | keep every extracted frame as a JPEG |
| `--save-video` | write `annotated.mp4` |

## No video? Test with your labelled photos

**A. Run on a folder of photos.** Every photo is analysed once at its own size, the same way the end-to-end test
read them. Put the folder under `data/` first.

```bash
docker compose run --rm marine-anomaly python -m app.cli -i /data/Test_data/TUNA -o /data/results/tuna
docker compose run --rm marine-anomaly python tools/score_run.py --results /data/results/tuna --label tuna
```

**B. Make a test video whose content you know**, then run it like a real flight (web page or command line):

```bash
python tools/copy_cetacean_test.py          # on your machine: copies the detector's 61 test photos to data/cetacean_test
docker compose run --rm marine-anomaly python tools/make_test_video.py \
    --folder prawn=/data/Test_data/PRAWNS --folder tuna=/data/Test_data/TUNA \
    --folder none=/data/Test_data/SEA --folder cetacean=/data/cetacean_test \
    --per-class 15 --out /data/test_video.mp4
docker compose run --rm marine-anomaly python -m app.cli -i /data/test_video.mp4 -o /data/results/test_video --every 1 --save-video
docker compose run --rm marine-anomaly python tools/score_run.py --results /data/results/test_video --truth /data/test_video_truth.csv
```

`score_run.py` prints the confusion matrix, accuracy, precision / recall / F1 and the wrong answers, and saves
`score.json`. `Test_data` holds photos that were also used during development (the end-to-end test set them aside),
so expect a somewhat higher score than the 93.0% reported on the 256 unseen frames.

## Results

Each run writes a folder:

| File | Contents |
|---|---|
| `frames.csv` | one row per analysed frame: time, answer, which step decided it and why, Stage 1 box, detector confidence and boxes, mask size, Track A probabilities, time taken |
| `events.json` | sightings: answer, start and end time, number of frames, best detector confidence, example image |
| `summary.json` | counts per answer, frames flagged by Stage 1, timing, and every setting used |
| `detections/` | annotated JPEG of every prawn, tuna or cetacean frame |
| `annotated.mp4` | (optional) every analysed frame with the answer drawn on it |
| `frames/` | (optional) every analysed frame as extracted |

On the images: yellow box = Stage 1's region, green boxes = cetaceans found by the detector (with confidence),
coloured outline = SegFormer's mask, banner = final answer.

## Speed

About 1–1.5 s per frame on 2 CPU cores, most of it Stage 1. Analysing one frame per second, a 10-minute flight takes
about 10–15 minutes on a laptop. For a faster run, analyse fewer frames (`--every 2`).

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | models loaded, settings |
| POST | `/api/jobs` | form fields: `file` or `url`, `every`, `max_seconds`, `save_video`, `save_frames` |
| GET | `/api/jobs`, `/api/jobs/{id}` | status, progress, counts, sightings |
| POST | `/api/jobs/{id}/stop` | stop a run |
| GET | `/api/jobs/{id}/files/{path}` | `frames.csv`, `events.json`, `summary.json`, `annotated.mp4`, `detections/...` |

## Notes

- The settings match the integrated end-to-end test (93.0% of 256 unseen test frames correct). That test used
  photographs; video frames from new footage may behave differently (motion blur, compression), so check a few
  flights by eye before relying on the counts.
- The detector is run only on frames Stage 1 flags. The end-to-end test ran it on every frame but only counted it
  after the Stage 1 check, so the answers are the same.
- Change `config/pipeline_config.json` only after re-validating. The app refuses to start if the Stage 1/Stage 2
  settings differ from those stored in `track_a_model.pkl` (override with `ALLOW_SETTINGS_MISMATCH=1`).
- GPU: build from a CUDA base image with the matching PyTorch build and `onnxruntime-gpu`; the app uses CUDA when it is available.
