"""
OCR recognition + Indian license plate format validation/correction.

PaddleOCR only operates on single images (plate crops), never on video —
video handling lives in stream_reader.py. This module just takes a cropped
plate image and returns the best-guess text + confidence.
"""

import re
import cv2
import numpy as np
from paddleocr import PaddleOCR

from . import config


class OCREngine:
    def __init__(self):
        # PaddleOCR 3.x renamed/removed several init params from 2.x:
        #   use_angle_cls -> use_textline_orientation
        #   show_log      -> removed entirely (no longer accepted)
        # We also disable document-specific preprocessing (orientation
        # classification / unwarping) since plate crops are not documents
        # and those stages just add latency for no benefit here.
        self.ocr = PaddleOCR(
            lang=config.OCR_LANG,
            use_textline_orientation=config.OCR_USE_ANGLE_CLS,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            enable_mkldnn=False,
        )
        self.plate_pattern = re.compile(config.INDIAN_PLATE_REGEX)

    # ---------- Preprocessing ----------

    @staticmethod
    def is_too_blurry(image):
        """Laplacian-variance blur check. Skip OCR on frames that are too blurry to read."""
        if image is None or image.size == 0:
            return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < config.BLUR_VARIANCE_THRESHOLD

    @staticmethod
    def preprocess(image):
        """Light cleanup before OCR: CLAHE contrast boost + mild denoising."""
        if image is None or image.size == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Adaptive histogram equalization — helps with glare / poor lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)

        # Mild denoise for dirty/weathered plates
        denoised = cv2.fastNlMeansDenoising(equalized, h=10)

        # Back to BGR since PaddleOCR expects 3-channel input
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    # ---------- OCR ----------

    def read_plate(self, plate_crop):
        """
        Runs OCR on a single cropped plate image.
        Returns (text, confidence) or (None, 0.0) if nothing usable was read.
        """
        if self.is_too_blurry(plate_crop):
            return None, 0.0

        processed = self.preprocess(plate_crop)

        # PaddleOCR 3.x: .ocr() is deprecated in favor of .predict(), which
        # returns an iterable of result objects (one per input image). Each
        # result behaves like a dict with 'rec_texts' and 'rec_scores' lists
        # — one entry per detected text region in the image.
        results = self.ocr.predict(processed)

        if not results:
            return None, 0.0

        res = results[0]
        text_parts = res.get("rec_texts", [])
        confidences = res.get("rec_scores", [])

        if not text_parts:
            return None, 0.0

        # A plate may be read as multiple text fragments (e.g. "KA01" and "AB1234"
        # on separate lines for some plate layouts) — concatenate them in order.
        raw_text = "".join(text_parts).upper()
        raw_text = re.sub(r"[^A-Z0-9]", "", raw_text)  # strip spaces/punctuation
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        if avg_conf < config.MIN_OCR_CONFIDENCE:
            return None, avg_conf

        return raw_text, avg_conf

    # ---------- Validation & correction ----------

    def validate_and_correct(self, text):
        """
        Checks text against the Indian plate regex. If it doesn't match,
        attempts common OCR-confusion character swaps (O<->0, I<->1, etc.)
        and re-checks. Returns (corrected_text, is_valid).
        """
        if text is None:
            return None, False

        if self.plate_pattern.match(text):
            return text, True

        # Try correcting letters that are commonly confused with digits
        # in the numeric segments of the plate (very naive positional fix —
        # good enough as a first pass, refine per your real OCR error patterns)
        corrected = list(text)
        for i, ch in enumerate(corrected):
            if ch in config.CHAR_CORRECTIONS:
                corrected[i] = config.CHAR_CORRECTIONS[ch]
        corrected_text = "".join(corrected)

        if self.plate_pattern.match(corrected_text):
            return corrected_text, True

        # Still invalid — return original text but flag as unvalidated.
        # Don't silently discard; a human reviewer or the voting mechanism
        # downstream may still want it.
        return text, False
