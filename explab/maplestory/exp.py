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

LEVEL_REGEX = re.compile(r"^\d{1,3}$")
EXP_REGEX = re.compile(r"^(?P<value>\d+)\[(?P<ratio>\d{1,2}\.\d{1,2})%\]?$")

logger = structlog.get_logger(__name__)


@dataclass
class ExpCheckpoint:
    """
    Represents an experience checkpoint in MapleStory.

    Attributes:
        level (int): The level at which the experience is recorded.
        exp (int): The total experience points required to reach this level.
    """

    level: int
    exp: int
    exp_ratio: float

    ts: datetime.datetime = field(default_factory=datetime.datetime.now)

    @staticmethod
    def from_app_capture(
        capture: np.ndarray,
        ts: datetime.datetime | None = None,
    ) -> "ExpCheckpoint | None":
        """
        Creates a Checkpoint instance from screen capture results.

        Args:
            level_crop (TextRecognitionResult): OCR result for the level.
            exp_crop (TextRecognitionResult): OCR result for the experience points.

        Returns:
            Checkpoint: An instance of Checkpoint with level and exp set.
        """
        level_results = ocr.recognize_text_from_image(
            cropper.get_level_crop(capture, ocr_friendly=True), allowlist="0123456789LV"
        )
        exp_results = ocr.recognize_text_from_image(
            cropper.get_exp_crop(capture, ocr_friendly=True), allowlist="0123456789[]%."
        )

        return ExpCheckpoint.from_ocr_results(
            level_results=level_results,
            exp_results=exp_results,
            ts=ts,
        )

    @staticmethod
    def from_ocr_results(
        level_results: list[TextRecognitionResult],
        exp_results: list[TextRecognitionResult],
        ts: datetime.datetime | None = None,
    ) -> "ExpCheckpoint | None":
        """
        Creates a Checkpoint instance from OCR results.

        Args:
            level_results (list[TextRecognitionResult]): OCR results for the level.
            exp_results (list[TextRecognitionResult]): OCR results for the experience points.

        Returns:
            Checkpoint: An instance of Checkpoint with level and exp set.
        """
        level = ExpCheckpoint.parse_level_results(level_results)

        if level is None:
            return None

        exp = ExpCheckpoint.parse_exp_results(exp_results)

        if exp is None:
            return None

        if ts is None:
            ts = datetime.datetime.now()

        return ExpCheckpoint(level=level, exp=exp[0], exp_ratio=exp[1], ts=ts)

    @staticmethod
    def parse_level_results(
        level_results: list[TextRecognitionResult],
    ) -> int | None:
        """
        Parses the level from OCR results.

        Args:
            level_results (list[TextRecognitionResult]): OCR results for the level.

        Returns:
            int: The parsed level.
        """

        for result in level_results:
            text = result.text.replace(" ", "")
            if LEVEL_REGEX.match(text):
                return int(text)

        logger.warning(
            "Failed to parse level from OCR results: {}".format(
                ", ".join(result.text for result in level_results)
            )
        )

        return None

    @staticmethod
    def parse_exp_results(
        exp_results: list[TextRecognitionResult],
    ) -> tuple[int, float] | None:
        """
        Parses the experience points from OCR results.

        Args:
            exp_results (list[TextRecognitionResult]): OCR results for the experience points.

        Returns:
            int: The parsed experience points.
        """
        for result in exp_results:
            text = result.text.replace(" ", "")
            match = EXP_REGEX.match(text)
            if match:
                return int(match.group("value")), float(match.group("ratio")) / 100

        logger.warning(
            "Failed to parse experience from OCR results: {}".format(
                ", ".join(result.text for result in exp_results)
            )
        )

        return None
