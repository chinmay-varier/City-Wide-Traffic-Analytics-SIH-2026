"""
License plate detection + multi-object tracking.

Uses YOLOv8 for detection and Ultralytics' built-in ByteTrack integration
for tracking, so each physical plate gets a stable track_id across frames
instead of being treated as a brand-new detection every frame.
"""

from ultralytics import YOLO
from . import config


class PlateDetector:
    def __init__(self, model_path=None):
        model_path = model_path or config.PLATE_DETECTOR_MODEL_PATH
        self.model = YOLO(model_path)

    def detect_and_track(self, frame):
        """
        Runs detection + tracking on a single frame.

        Returns a list of dicts:
            [{"track_id": int, "bbox": (x1, y1, x2, y2), "conf": float}, ...]
        """
        results = self.model.track(
            frame,
            persist=True,                # remember tracks between calls
            conf=config.DETECTION_CONFIDENCE_THRESHOLD,
            tracker=config.TRACKER_CONFIG,
            verbose=False,
        )

        detections = []
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, conf, tid in zip(boxes, confs, track_ids):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "track_id": int(tid),
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(conf),
                })

        return detections

    @staticmethod
    def crop(frame, bbox, padding=4):
        """Crop a bounding box out of a frame with a small padding margin."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        return frame[y1:y2, x1:x2]
