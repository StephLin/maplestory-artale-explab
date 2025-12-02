# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os

import numpy as np
import opencc
import paddleocr
import structlog

from .base import BoundingBox, TextRecognitionResult

logger = structlog.get_logger()

reader: paddleocr.PaddleOCR | None = None
converter: opencc.OpenCC | None = None


def initialize() -> None:
    """
    Initializes the PaddleOCR reader.

    Args:
        lang_list (list[str], optional): List of languages to initialize the reader with.
                                         Not used in PaddleOCR, kept for API compatibility.
                                         Defaults to ["en"].

    Returns:
        None
    """
    global reader, converter

    if reader is None:
        logger.info("Initializing PaddleOCR Reader...")
        # Determine GPU usage from environment variable
        use_gpu_str = os.getenv("USE_GPU", "False").lower()
        use_gpu = use_gpu_str in ("true", "1", "t")
        logger.info(
            f"USE_GPU environment variable set to: {use_gpu_str}, parsed as: {use_gpu}"
        )

        # Initialize the PaddleOCR reader with Chinese language support
        reader = paddleocr.PaddleOCR(
            lang="en",
            # use_gpu=use_gpu,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
        )

        # Initialize OpenCC converter for Simplified to Traditional Chinese
        converter = opencc.OpenCC("s2t.json")

        logger.info("PaddleOCR Reader initialized.")
    else:
        logger.info("PaddleOCR Reader already initialized.")


def recognize_text_from_image(
    image_array: np.ndarray,
    allowlist: str | None = None,
    width_ths: float = 0.5,
    recognize_only: bool = False,
) -> list[TextRecognitionResult]:
    """
    Recognizes text from an image using PaddleOCR.

    Args:
        image_array: Input image as numpy array
        allowlist: Not used in PaddleOCR, kept for API compatibility
        width_ths: Not used in PaddleOCR, kept for API compatibility
        recognize_only: Not used in PaddleOCR, kept for API compatibility

    Returns:
        List of TextRecognitionResult objects
    """
    if reader is None:
        initialize()

    if reader is None:
        raise RuntimeError("Failed to initialize OCR engine.")

    logger.debug("Feeding image array to PaddleOCR for text extraction...")

    # Handle RGBA images by removing alpha channel
    if len(image_array.shape) == 3 and image_array.shape[2] == 4:
        image_array = image_array[:, :, :3].copy()
    else:
        image_array = image_array.copy()

    # Get OCR results from PaddleOCR
    paddleocr_results = reader.predict(image_array)  # type: ignore

    logger.debug("Text extraction results:")

    processed_results: list[TextRecognitionResult] = []
    for res in paddleocr_results:
        for text, score, poly in zip(
            res["rec_texts"],
            res["rec_scores"],
            res["dt_polys"],
        ):
            # Convert PaddleOCR bbox to our BoundingBox dataclass
            # PaddleOCR poly: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            # We use top-left and bottom-right corners
            x1, y1 = poly[0]
            x2, y2 = poly[2]
            x_min = int(x1)
            y_min = int(y1)
            x_max = int(x2)
            y_max = int(y2)

            bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

            # Convert text from Simplified to Traditional Chinese if converter is available
            converted_text = converter.convert(text) if converter else text

            processed_results.append(
                TextRecognitionResult(
                    text=str(converted_text), confidence=float(score), bounding_box=bbox
                )
            )
            logger.debug(
                f"Detected text: {converted_text} with confidence {float(score):.2f} at {bbox}"
            )

    return processed_results


def recognize_text_from_images_batch(
    image_arrays: list[np.ndarray],
    allowlist: str | None = None,
    width_ths: float = 0.5,
    batch_size: int | None = None,
) -> list[list[TextRecognitionResult]]:
    """
    Recognizes text from a batch of images using PaddleOCR.

    Args:
        image_arrays (list[np.ndarray]): A list of image arrays (NumPy arrays).
        allowlist (str | None, optional): Not used in PaddleOCR, kept for API compatibility.
        width_ths (float, optional): Not used in PaddleOCR, kept for API compatibility.
        batch_size (int | None, optional): Not used in PaddleOCR, kept for API compatibility.

    Returns:
        list[list[TextRecognitionResult]]: A list of lists, where each inner list
                                           contains TextRecognitionResult objects for an image.
    """
    if reader is None:
        initialize()

    if reader is None:
        raise RuntimeError("Failed to initialize OCR engine.")

    logger.debug(
        f"Feeding {len(image_arrays)} image arrays to PaddleOCR for batch text extraction..."
    )

    batch_processed_results: list[list[TextRecognitionResult]] = []

    for i, image_array in enumerate(image_arrays):
        processed_results_for_image: list[TextRecognitionResult] = []
        logger.debug(f"Text extraction results for image {i}:")

        # Handle RGBA images by removing alpha channel
        if len(image_array.shape) == 3 and image_array.shape[2] == 4:
            image_array = image_array[:, :, :3].copy()
        else:
            image_array = image_array.copy()

        # Get OCR results from PaddleOCR
        paddleocr_results = reader.predict(image_array)  # type: ignore

        for res in paddleocr_results:
            for text, score, poly in zip(
                res["rec_texts"],
                res["rec_scores"],
                res["dt_polys"],
            ):
                # Convert PaddleOCR bbox to our BoundingBox dataclass
                x1, y1 = poly[0]
                x2, y2 = poly[2]
                x_min = int(x1)
                y_min = int(y1)
                x_max = int(x2)
                y_max = int(y2)

                bbox = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)

                # Convert text from Simplified to Traditional Chinese if converter is available
                converted_text = converter.convert(text) if converter else text

                recognition_result = TextRecognitionResult(
                    text=str(converted_text), confidence=float(score), bounding_box=bbox
                )
                processed_results_for_image.append(recognition_result)
                logger.debug(
                    f"  Detected text: {converted_text} with confidence {float(score):.2f} at {bbox}"
                )

        batch_processed_results.append(processed_results_for_image)

    return batch_processed_results
