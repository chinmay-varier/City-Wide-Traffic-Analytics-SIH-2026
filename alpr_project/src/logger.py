"""
Logs finalized plate reads to CSV, tagged with camera ID, GPS location, and
timestamp. This is the output your Trajectory Reconstruction Engine and
Analytics Dashboard would later consume (e.g. load into PostgreSQL/PostGIS).
"""

import csv
import os
from datetime import datetime, timezone

from . import config


class PlateLogger:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path or config.OUTPUT_CSV_PATH
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        file_exists = os.path.isfile(self.csv_path)
        self.file = open(self.csv_path, "a", newline="")
        self.writer = csv.writer(self.file)

        if not file_exists:
            self.writer.writerow([
                "timestamp_utc", "camera_id", "latitude", "longitude",
                "track_id", "plate_text", "confidence", "num_votes", "format_valid",
            ])
            self.file.flush()

    def log(self, track_id, plate_text, confidence, num_votes, format_valid):
        self.writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            config.CAMERA_ID,
            config.CAMERA_LAT,
            config.CAMERA_LON,
            track_id,
            plate_text,
            round(confidence, 3),
            num_votes,
            format_valid,
        ])
        self.file.flush()

    def close(self):
        self.file.close()
