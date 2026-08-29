# ALPR Pipeline Prototype

A working end-to-end ALPR pipeline: video/RTSP input → plate detection & tracking
(YOLOv8 + ByteTrack) → OCR (PaddleOCR) → multi-frame voting → CSV logging.

This is the ingestion layer for the "High-Accuracy ANPR/OCR Engine" and
"Single Plate Trajectory Tracking" components of the hackathon problem
statement. Each CSV row (camera, GPS, timestamp, plate) is exactly the input
your later Trajectory Reconstruction Engine and Analytics Dashboard will
consume — load them into PostgreSQL/PostGIS as your next step.

## Setup

```bash
cd alpr_project
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Before running

1. **Plate detector model**: `src/config.py` points at
   `models/plate_detector.pt`. You don't have a fine-tuned plate detector
   yet, so for a first test:
   - Quick hack: download a general YOLOv8 model (`yolov8n.pt`, auto-downloads
     on first use) and point `PLATE_DETECTOR_MODEL_PATH` at it temporarily —
     it will detect vehicles, not plates, but proves the pipeline runs.
   - Real version: fine-tune YOLOv8 on a labeled Indian license plate
     dataset (Roboflow has several public ones) and save the resulting
     `.pt` file into `models/`.

2. **Video source**: edit `VIDEO_SOURCE` in `src/config.py`:
   - Video file: `"sample_videos/traffic.mp4"` — drop a sample clip in that folder
   - Live camera: `"rtsp://user:pass@camera_ip:554/stream1"`
   - Webcam (quick test): `0`

3. **Camera metadata**: set `CAMERA_ID`, `CAMERA_LAT`, `CAMERA_LON` in
   `src/config.py` per physical camera — this is what lets you plot results
   on a GIS map later.

## Run

```bash
python -m src.main
```

Output is written to `output/plate_log.csv` and printed to console as plates
are finalized. Press `q` in the video window to stop early.

## How it works

```
StreamReader   -> threaded frame grabbing, handles live RTSP reconnects
                  and keeps only the latest frame so processing never lags
PlateDetector  -> YOLOv8 detection + ByteTrack tracking (stable track_id
                  per physical vehicle across frames)
OCREngine      -> blur check -> CLAHE/denoise preprocessing -> PaddleOCR
                  -> regex validation & common-confusion character correction
VoteTracker    -> collects multiple OCR reads per track_id, reports the
                  majority-vote (or highest-confidence) result once enough
                  votes are in or the vehicle leaves frame
PlateLogger    -> writes finalized reads to CSV with camera + GPS + timestamp
```

## Tuning for accuracy

All the knobs that matter are in `src/config.py`:
- `DETECTION_CONFIDENCE_THRESHOLD` — raise if you get false-positive plate boxes
- `BLUR_VARIANCE_THRESHOLD` — raise to skip more blurry frames (fewer but cleaner reads)
- `MIN_VOTES_BEFORE_REPORT` — raise for higher confidence, at the cost of latency
- `CHAR_CORRECTIONS` — extend based on the actual OCR error patterns you observe

## Next steps for the full hackathon solution

- Fine-tune the plate detector and PaddleOCR recognition model on real
  Indian plate crops (aim for a few hundred labeled images minimum)
- Push CSV rows into PostgreSQL + PostGIS instead of a flat file
- Build the Trajectory Reconstruction Engine: query by plate_text across
  all camera_id rows, order by timestamp, plot on Leaflet/Mapbox
- Build the Analytics Dashboard: aggregate rows for heatmaps, density,
  origin-destination pairs
- Add the Alert System: check each finalized plate against a blacklist
  table, push a websocket/notification event on match
