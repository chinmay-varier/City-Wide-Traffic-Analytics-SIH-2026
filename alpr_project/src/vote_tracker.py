"""
Multi-frame voting per vehicle track.

A single vehicle is seen across many frames as it crosses the camera's view.
Rather than trusting any one frame's OCR read, we collect several reads per
track_id and pick the best consensus answer once we have enough data or the
vehicle leaves frame. This is what pushes effective accuracy above the
single-frame OCR baseline.
"""

import time
from collections import Counter, defaultdict

from . import config


class TrackVoteRecord:
    def __init__(self, track_id):
        self.track_id = track_id
        self.reads = []              # list of (text, confidence)
        self.last_seen_frame = 0
        self.first_seen_time = time.time()
        self.reported = False

    def add_read(self, text, confidence):
        if text:
            self.reads.append((text, confidence))

    def best_guess(self):
        """
        Majority vote weighted lightly by confidence. Falls back to the
        single highest-confidence read if there's no clear majority.
        """
        if not self.reads:
            return None, 0.0

        counts = Counter(text for text, _ in self.reads)
        most_common_text, freq = counts.most_common(1)[0]

        # If there's a genuine majority (appeared more than once), use it
        if freq > 1:
            confs = [c for t, c in self.reads if t == most_common_text]
            return most_common_text, sum(confs) / len(confs)

        # No repeats — fall back to the single highest-confidence read
        best_text, best_conf = max(self.reads, key=lambda r: r[1])
        return best_text, best_conf

    def vote_count(self):
        return len(self.reads)


class VoteTracker:
    """Manages TrackVoteRecords for all currently-active vehicle tracks."""

    def __init__(self):
        self.records = {}          # track_id -> TrackVoteRecord
        self.current_frame = 0

    def update(self, track_id, ocr_text, ocr_conf):
        if track_id not in self.records:
            self.records[track_id] = TrackVoteRecord(track_id)
        record = self.records[track_id]
        record.add_read(ocr_text, ocr_conf)
        record.last_seen_frame = self.current_frame
        return record

    def advance_frame(self):
        self.current_frame += 1

    def get_ready_to_report(self):
        """
        Returns tracks that have either:
          (a) collected enough votes to confidently report, or
          (b) gone stale (vehicle left frame) and have at least one read.
        Marks them as reported so they aren't returned again.
        """
        ready = []
        for track_id, record in self.records.items():
            if record.reported:
                continue

            frames_since_seen = self.current_frame - record.last_seen_frame
            enough_votes = record.vote_count() >= config.MIN_VOTES_BEFORE_REPORT
            went_stale = frames_since_seen >= config.TRACK_TIMEOUT_FRAMES and record.vote_count() > 0

            if enough_votes or went_stale:
                text, conf = record.best_guess()
                if text:
                    ready.append({
                        "track_id": track_id,
                        "plate_text": text,
                        "confidence": conf,
                        "num_votes": record.vote_count(),
                    })
                record.reported = True

        return ready

    def cleanup_stale(self):
        """Remove tracks that have been idle way past timeout to free memory."""
        stale_ids = [
            tid for tid, r in self.records.items()
            if (self.current_frame - r.last_seen_frame) > (config.TRACK_TIMEOUT_FRAMES * 3)
        ]
        for tid in stale_ids:
            del self.records[tid]
