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
        )

        return HpCheckpoint.from_ocr_results(
            hp_results=hp_results,
            ts=ts,
        )

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
