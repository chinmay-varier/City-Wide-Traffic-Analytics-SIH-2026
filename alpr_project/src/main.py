"""
Main ALPR pipeline entry point.

Flow:
    Video/RTSP stream
        -> StreamReader (threaded, handles live feeds + file playback)
        -> PlateDetector (YOLOv8 + ByteTrack — detects & tracks plates)
        -> OCREngine (PaddleOCR on each cropped plate, per frame)
        -> VoteTracker (combines multiple frame-reads per vehicle track)
        -> PlateLogger (writes finalized reads to CSV)

Run with:
    python -m src.main
(from the alpr_project/ directory, after editing src/config.py)
"""

import cv2
import time

from . import config
from .stream_reader import StreamReader
from .detector import PlateDetector
from .ocr_engine import OCREngine
from .vote_tracker import VoteTracker
from .logger import PlateLogger


def draw_overlay(frame, bbox, track_id, text=None):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"ID:{track_id}"
    if text:
        label += f" {text}"
    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def run():
    print(f"[main] Starting ALPR pipeline on source: {config.VIDEO_SOURCE}")

    stream = StreamReader(config.VIDEO_SOURCE, force_tcp=config.FORCE_RTSP_TCP)
    detector = PlateDetector()
    ocr_engine = OCREngine()
    vote_tracker = VoteTracker()
    plate_logger = PlateLogger()

    # Cache the latest OCR text per track_id, just for on-screen display
    display_cache = {}

    frame_index = 0

    try:
        while True:
            if stream.is_finished():
                print("[main] Video source exhausted, stopping.")
                break

            frame = stream.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            frame_index += 1
            vote_tracker.advance_frame()

            run_detection = (frame_index % config.PROCESS_EVERY_N_FRAMES == 0)

            if run_detection:
                detections = detector.detect_and_track(frame)

                for det in detections:
                    track_id = det["track_id"]
                    bbox = det["bbox"]

                    plate_crop = detector.crop(frame, bbox)
                    raw_text, conf = ocr_engine.read_plate(plate_crop)

                    if raw_text:
                        corrected_text, is_valid = ocr_engine.validate_and_correct(raw_text)
                        vote_tracker.update(track_id, corrected_text, conf)
                        display_cache[track_id] = corrected_text

                    if config.SHOW_LIVE_WINDOW:
                        draw_overlay(frame, bbox, track_id, display_cache.get(track_id))

            # Report any tracks that have enough votes or went stale
            for finalized in vote_tracker.get_ready_to_report():
                text, is_valid = ocr_engine.validate_and_correct(finalized["plate_text"])
                print(
                    f"[PLATE] track={finalized['track_id']} "
                    f"text={text} valid_format={is_valid} "
                    f"conf={finalized['confidence']:.2f} votes={finalized['num_votes']}"
                )
                plate_logger.log(
                    track_id=finalized["track_id"],
                    plate_text=text,
                    confidence=finalized["confidence"],
                    num_votes=finalized["num_votes"],
                    format_valid=is_valid,
                )

            vote_tracker.cleanup_stale()

            if config.SHOW_LIVE_WINDOW:
                cv2.imshow("ALPR Pipeline", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[main] Quit key pressed, stopping.")
                    break

    finally:
        stream.stop()
        plate_logger.close()
        if config.SHOW_LIVE_WINDOW:
            cv2.destroyAllWindows()
        print("[main] Shutdown complete.")


if __name__ == "__main__":
    run()
