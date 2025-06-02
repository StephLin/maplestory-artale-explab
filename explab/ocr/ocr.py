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


def recognize_text_from_images_batch(
    image_arrays: list[np.ndarray],
    allowlist: str | None = None,
    width_ths: float = 0.5,
    batch_size: int | None = None,  # Add batch_size parameter for EasyOCR
) -> list[list[TextRecognitionResult]]:
    """
    Recognizes text from a batch of images using EasyOCR.

    Args:
        image_arrays (list[np.ndarray]): A list of image arrays (NumPy arrays).
        allowlist (str | None, optional): A string of characters to allow. Defaults to None.
        width_ths (float, optional): Width threshold for text detection. Defaults to 0.5.
        batch_size (int | None, optional): Batch size for OCR processing. Defaults to None (EasyOCR default).

    Returns:
        list[list[TextRecognitionResult]]: A list of lists, where each inner list
                                           contains TextRecognitionResult objects for an image.
    """
    if reader is None:
        initialize()

    logger.debug(
        f"Feeding {len(image_arrays)} image arrays to EasyOCR for batch text extraction..."
    )

    # EasyOCR's readtext can handle a list of images.
    # It returns a list, where each element is the result for the corresponding image.
    # Each result itself is a list of (bbox, text, confidence) tuples.
    readtext_kwargs = {"allowlist": allowlist, "width_ths": width_ths}
    if batch_size is not None:
        readtext_kwargs["batch_size"] = batch_size

    all_raw_results = reader.readtext_batched(image_arrays, **readtext_kwargs)  # type: ignore

    batch_processed_results: list[list[TextRecognitionResult]] = []

    for i, raw_results_for_image in enumerate(all_raw_results):
        processed_results_for_image: list[TextRecognitionResult] = []
        logger.debug(f"Text extraction results for image {i}:")
        for raw_result in raw_results_for_image:
            bbox_points, text, confidence = raw_result
            x_min = int(bbox_points[0][0])
            y_min = int(bbox_points[0][1])
            x_max = int(bbox_points[2][0])
            y_max = int(bbox_points[2][1])

            bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
            recognition_result = TextRecognitionResult(
                text=str(text), confidence=float(confidence), bounding_box=bbox
            )
            processed_results_for_image.append(recognition_result)
            logger.debug(
                f"  Detected text: {text} with confidence {float(confidence):.2f} at {bbox}"
            )
        batch_processed_results.append(processed_results_for_image)

    return batch_processed_results
