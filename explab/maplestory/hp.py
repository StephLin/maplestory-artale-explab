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

HP_REGEX = re.compile(r"^\[(?P<current>\d+)/(?P<total>\d+)\].*$")

logger = structlog.get_logger(__name__)


@dataclass
class HpCheckpoint:
    """
    Represents an HP checkpoint in MapleStory.

    Attributes:
        current_hp (int): The current HP.
        total_hp (int): The total HP.
        ts (datetime.datetime): The timestamp of the checkpoint.
    """

    current_hp: int
    total_hp: int
    ts: datetime.datetime = field(default_factory=datetime.datetime.now)

    @staticmethod
    def from_app_capture(
        capture: np.ndarray,
        ts: datetime.datetime | None = None,
    ) -> "HpCheckpoint | None":
        """
        Creates an HpCheckpoint instance from screen capture results.

        Args:
            capture (np.ndarray): The screen capture.
            ts (datetime.datetime | None): Optional timestamp.

        Returns:
            HpCheckpoint | None: An instance of HpCheckpoint or None if parsing fails.
        """
        hp_results = ocr.recognize_text_from_image(
            cropper.get_hp_crop(capture, ocr_friendly=True),
            allowlist="0123456789/[]",
            width_ths=10.0,
        )

        return HpCheckpoint.from_ocr_results(
            hp_results=hp_results,
            ts=ts,
        )

    @staticmethod
    def from_app_captures(
        captures: list[np.ndarray],
        ts_list: list[datetime.datetime] | None = None,
        ocr_batch_size: int | None = None,
    ) -> list["HpCheckpoint | None"]:
        """
        Creates a list of HpCheckpoint instances from a batch of screen captures.

        Args:
            captures (list[np.ndarray]): A list of screen captures.
            ts_list (list[datetime.datetime | None] | None): Optional list of timestamps,
                                                             one for each capture. If None,
                                                             current time is used for each.
                                                             If provided, its length must match captures.
            ocr_batch_size (int | None): Optional batch size for OCR processing.

        Returns:
            list[HpCheckpoint | None]: A list of HpCheckpoint instances or None
                                       for captures where parsing fails.
        """
        if ts_list is not None and len(ts_list) != len(captures):
            raise ValueError(
                "Length of ts_list must match the number of captures if provided."
            )

        if len(captures) == 0:
            logger.warning("No captures provided for HP checkpoint extraction.")
            return []

        cropped_images = [
            cropper.get_hp_crop(capture, ocr_friendly=True) for capture in captures
        ]

        batch_ocr_results = recognize_text_from_images_batch(
            cropped_images,
            allowlist="0123456789/[]",
            width_ths=10.0,
            batch_size=ocr_batch_size,
        )

        checkpoints: list[HpCheckpoint | None] = []
        for i, single_image_ocr_results in enumerate(batch_ocr_results):
            current_ts = ts_list[i] if ts_list else None
            checkpoint = HpCheckpoint.from_ocr_results(
                hp_results=single_image_ocr_results,
                ts=current_ts,
            )
            checkpoints.append(checkpoint)

        return checkpoints

    @staticmethod
    def from_ocr_results(
        hp_results: list[TextRecognitionResult],
        ts: datetime.datetime | None = None,
    ) -> "HpCheckpoint | None":
        """
        Creates an HpCheckpoint instance from OCR results.

        Args:
            hp_results (list[TextRecognitionResult]): OCR results for HP.
            ts (datetime.datetime | None): Optional timestamp.

        Returns:
            HpCheckpoint | None: An instance of HpCheckpoint or None if parsing fails.
        """
        parsed_hp = HpCheckpoint.parse_hp_results(hp_results)

        if parsed_hp is None:
            return None

        current_hp, total_hp = parsed_hp

        if ts is None:
            ts = datetime.datetime.now()

        return HpCheckpoint(current_hp=current_hp, total_hp=total_hp, ts=ts)

    @staticmethod
    def parse_hp_results(
        hp_results: list[TextRecognitionResult],
    ) -> tuple[int, int] | None:
        """
        Parses the HP from OCR results.

        Args:
            hp_results (list[TextRecognitionResult]): OCR results for HP.

        Returns:
            tuple[int, int] | None: A tuple containing (current_hp, total_hp) or None if parsing fails.
        """
        for result in hp_results:
            text = result.text.replace(" ", "")
            match = HP_REGEX.match(text)
            if match:
                current_hp = int(match.group("current"))
                total_hp = int(match.group("total"))
                return current_hp, total_hp

        logger.warning(
            "Failed to parse HP from OCR results: {}".format(
                ", ".join(result.text for result in hp_results)
            )
        )
        return None
