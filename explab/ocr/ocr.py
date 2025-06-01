# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os  # Add os import

import easyocr
import numpy as np
import structlog

from .base import BoundingBox, TextRecognitionResult

logger = structlog.get_logger()

reader: easyocr.Reader | None = None


def initialize(lang_list: list[str] | None = None) -> None:
    """
    Initializes the EasyOCR reader with the specified languages.

    Args:
        lang_list (list[str], optional): List of languages to initialize the reader with.
                                         Defaults to ["en"].

    Returns:
        easyocr.Reader: The initialized EasyOCR reader.
    """
    global reader
    if lang_list is None:
        lang_list = ["en"]

    if reader is None:
        logger.info("Initializing EasyOCR Reader...")
        # Determine GPU usage from environment variable
        use_gpu_str = os.getenv("USE_GPU", "False").lower()
        use_gpu = use_gpu_str in ("true", "1", "t")
        logger.info(
            f"USE_GPU environment variable set to: {use_gpu_str}, parsed as: {use_gpu}"
        )

        # Initialize the EasyOCR reader with the specified languages and GPU support
        reader = easyocr.Reader(lang_list=lang_list, gpu=use_gpu)
    else:
        logger.info("EasyOCR Reader already initialized.")


def recognize_text_from_image(
    image_array: np.ndarray, allowlist: str | None = None, width_ths: float = 0.5
) -> list[TextRecognitionResult]:
    if reader is None:
        initialize()

    logger.debug("Feeding image array to EasyOCR for text extraction...")
    # results is a list of (bbox, text, confidence)
    # bbox is [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
    raw_results = reader.readtext(image_array, allowlist=allowlist, width_ths=width_ths)  # type: ignore
    logger.debug("Text extraction results:")

    processed_results: list[TextRecognitionResult] = []
    for raw_result in raw_results:
        bbox_points, text, confidence = raw_result
        # Convert easyocr bbox to our BoundingBox dataclass
        # easyocr bbox: [[ul_x, ul_y], [ur_x, ur_y], [lr_x, lr_y], [ll_x, ll_y]]
        # We need: x_min, y_min, x_max, y_max
        # x_min = top_left_x, y_min = top_left_y
        # x_max = bottom_right_x, y_max = bottom_right_y
        x_min = int(bbox_points[0][0])
        y_min = int(bbox_points[0][1])
        x_max = int(bbox_points[2][0])
        y_max = int(bbox_points[2][1])

        bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        processed_results.append(
            TextRecognitionResult(
                text=str(text), confidence=float(confidence), bounding_box=bbox
            )
        )
        logger.debug(
            f"Detected text: {text} with confidence {float(confidence):.2f} at {bbox}"
        )

    return processed_results
