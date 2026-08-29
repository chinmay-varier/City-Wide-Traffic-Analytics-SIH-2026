"""
Threaded video reader that works for both video files and live RTSP feeds.

For live feeds: reads frames continuously in a background thread and always
keeps only the LATEST frame, so the main pipeline never processes a backlog
of stale frames if it's running slower than the incoming stream.

For file input: behaves like a normal sequential reader (get_frame() returns
None once the file is exhausted and the buffer is drained).
"""

import os
import time
import threading
import cv2

from . import config


class StreamReader:
    def __init__(self, source, force_tcp=True):
        self.source = source
        self.is_live = isinstance(source, str) and source.startswith("rtsp://")

        if self.is_live and force_tcp:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

        self.cap = cv2.VideoCapture(source)

        # Low buffer size = fewer stale frames queued internally (live feeds only)
        if self.is_live:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.latest_frame = None
        self.frame_available = False
        self.file_exhausted = False
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                if self.is_live:
                    # Live feed dropped — attempt reconnect
                    print(f"[StreamReader] Lost connection to {self.source}, reconnecting...")
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = cv2.VideoCapture(self.source)
                    continue
                else:
                    # Video file ended
                    self.file_exhausted = True
                    break

            with self.lock:
                self.latest_frame = frame
                self.frame_available = True

            # For file playback, pace reads roughly to source FPS so we don't
            # blow through the file instantly (optional, comment out for max speed)
            if not self.is_live:
                time.sleep(1.0 / 30.0)

    def get_frame(self):
        """Returns the latest available frame, or None if nothing new yet / stream ended."""
        with self.lock:
            if not self.frame_available:
                return None
            frame = self.latest_frame.copy()
            # For live feeds we allow re-reading same frame if processing is slower;
            # for files we mark as consumed so we don't reprocess it.
            if not self.is_live:
                self.frame_available = False
            return frame

    def is_finished(self):
        """Only meaningful for file input — True once file is fully read and drained."""
        return (not self.is_live) and self.file_exhausted and not self.frame_available

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()
