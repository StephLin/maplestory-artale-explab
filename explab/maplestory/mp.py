# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
import re
from dataclasses import dataclass, field

import numpy as np
import structlog

from explab.ocr import ocr
from explab.ocr.base import TextRecognitionResult
from explab.ocr.ocr import recognize_text_from_images_batch  # Added import
from explab.preprocessing import cropper

MP_REGEX = re.compile(r"^\[(?P<current>\d+)/(?P<total>\d+)\].*$")

logger = structlog.get_logger(__name__)


@dataclass
class MpCheckpoint:
    """
    Represents an MP checkpoint in MapleStory.

    Attributes:
        current_mp (int): The current MP.
        total_mp (int): The total MP.
        ts (datetime.datetime): The timestamp of the checkpoint.
    """

    current_mp: int
    total_mp: int
    ts: datetime.datetime = field(default_factory=datetime.datetime.now)

    @staticmethod
    def from_app_capture(
        capture: np.ndarray,
        ts: datetime.datetime | None = None,
    ) -> "MpCheckpoint | None":
        """
        Creates an MpCheckpoint instance from screen capture results.

        Args:
            capture (np.ndarray): The screen capture.
            ts (datetime.datetime | None): Optional timestamp.

        Returns:
            MpCheckpoint | None: An instance of MpCheckpoint or None if parsing fails.
        """
        mp_results = ocr.recognize_text_from_image(
            cropper.get_mp_crop(capture, ocr_friendly=True),
            allowlist="0123456789/[]",
            width_ths=10.0,
        )

        return MpCheckpoint.from_ocr_results(
            mp_results=mp_results,
            ts=ts,
        )

    @staticmethod
    def from_app_captures(
        captures: list[np.ndarray],
        ts_list: list[datetime.datetime] | None = None,
        ocr_batch_size: int | None = None,
    ) -> list["MpCheckpoint | None"]:
        """
        Creates a list of MpCheckpoint instances from a batch of screen captures.

        Args:
            captures (list[np.ndarray]): A list of screen captures.
            ts_list (list[datetime.datetime | None] | None): Optional list of timestamps,
                                                             one for each capture. If None,
                                                             current time is used for each.
                                                             If provided, its length must match captures.
            ocr_batch_size (int | None): Optional batch size for OCR processing.

        Returns:
            list[MpCheckpoint | None]: A list of MpCheckpoint instances or None
                                       for captures where parsing fails.
        """
        if ts_list is not None and len(ts_list) != len(captures):
            raise ValueError(
                "Length of ts_list must match the number of captures if provided."
            )

        cropped_images = [
            cropper.get_mp_crop(capture, ocr_friendly=True) for capture in captures
        ]

        batch_ocr_results = recognize_text_from_images_batch(
            cropped_images,
            allowlist="0123456789/[]",
            width_ths=10.0,
            batch_size=ocr_batch_size,
        )

        checkpoints: list[MpCheckpoint | None] = []
        for i, single_image_ocr_results in enumerate(batch_ocr_results):
            current_ts = ts_list[i] if ts_list else None
            checkpoint = MpCheckpoint.from_ocr_results(
                mp_results=single_image_ocr_results,
                ts=current_ts,
            )
            checkpoints.append(checkpoint)

        return checkpoints

    @staticmethod
    def from_ocr_results(
        mp_results: list[TextRecognitionResult],
        ts: datetime.datetime | None = None,
    ) -> "MpCheckpoint | None":
        """
        Creates an MpCheckpoint instance from OCR results.

        Args:
            mp_results (list[TextRecognitionResult]): OCR results for MP.
            ts (datetime.datetime | None): Optional timestamp.

        Returns:
            MpCheckpoint | None: An instance of MpCheckpoint or None if parsing fails.
        """
        parsed_mp = MpCheckpoint.parse_mp_results(mp_results)

        if parsed_mp is None:
            return None

        current_mp, total_mp = parsed_mp

        if ts is None:
            ts = datetime.datetime.now()

        return MpCheckpoint(current_mp=current_mp, total_mp=total_mp, ts=ts)

    @staticmethod
    def parse_mp_results(
        mp_results: list[TextRecognitionResult],
    ) -> tuple[int, int] | None:
        """
        Parses the MP from OCR results.

        Args:
            mp_results (list[TextRecognitionResult]): OCR results for MP.

        Returns:
            tuple[int, int] | None: A tuple containing (current_mp, total_mp) or None if parsing fails.
        """
        for result in mp_results:
            text = result.text.replace(" ", "")
            match = MP_REGEX.match(text)
            if match:
                current_mp = int(match.group("current"))
                total_mp = int(match.group("total"))
                return current_mp, total_mp

        logger.warning(
            "Failed to parse MP from OCR results: {}".format(
                ", ".join(result.text for result in mp_results)
            )
        )
        return None
