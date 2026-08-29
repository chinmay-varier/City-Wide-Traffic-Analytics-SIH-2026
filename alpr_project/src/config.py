"""
Central configuration for the ALPR pipeline.
Tweak these values instead of hunting through the code.
"""

# ---- Model paths ----
# If you haven't fine-tuned a plate detector yet, use a pretrained YOLOv8
# general model temporarily (e.g. "yolov8n.pt") and detect vehicles instead
# of plates, then crop the top/bottom region as an approximation.
# Once you have a fine-tuned plate model, point this at that .pt file.
PLATE_DETECTOR_MODEL_PATH = "models/plate_detector.pt"

# ---- Video source ----
# For a video file:      "sample_videos/traffic.mp4"
# For a live RTSP camera: "rtsp://username:password@camera_ip:554/stream1"
# For a webcam (testing): 0
VIDEO_SOURCE = "sample_videos/traffic1.mp4"

# Force RTSP over TCP (more reliable, slightly higher latency than UDP)
FORCE_RTSP_TCP = True

# ---- Detection ----
DETECTION_CONFIDENCE_THRESHOLD = 0.4
PROCESS_EVERY_N_FRAMES = 2          # skip frames to save compute
TRACKER_CONFIG = "bytetrack.yaml"    # built into ultralytics

# ---- OCR ----
OCR_LANG = "en"
OCR_USE_ANGLE_CLS = True
MIN_OCR_CONFIDENCE = 0.5

# Minimum number of OCR reads collected for a track before we
# finalize/report a plate (higher = more accurate, slower to report)
MIN_VOTES_BEFORE_REPORT = 3

# A track is considered "finished" (vehicle left frame) after this many
# consecutive frames with no matching detection
TRACK_TIMEOUT_FRAMES = 30

# ---- Blur filtering ----
# Frames with a Laplacian variance below this are considered too blurry
# to bother running OCR on. Tune this against your own footage.
BLUR_VARIANCE_THRESHOLD = 60.0

# ---- Indian plate format validation ----
# Standard format: 2 letters (state) + 1-2 digits (RTO code) +
#                   1-3 letters (series) + 4 digits (number)
# e.g. KA01AB1234, MH12DE1433, DL3CAB1234
INDIAN_PLATE_REGEX = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$"

# Common OCR character confusions used for auto-correction attempts
CHAR_CORRECTIONS = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
}

# ---- Output ----
OUTPUT_CSV_PATH = "output/plate_log.csv"
CAMERA_ID = "CAM_01"          # identifies which camera this instance is running on
CAMERA_LAT = 12.9716          # for GIS mapping later — set per-camera
CAMERA_LON = 77.5946

# ---- Display ----
SHOW_LIVE_WINDOW = True        # set False when running headless on a server
