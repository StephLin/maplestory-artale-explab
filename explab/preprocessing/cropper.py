# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import structlog
from skimage import morphology

from explab.utils.imgproc import binarize_image

logger = structlog.get_logger()


@dataclass
class CropAxisBounds:
    start: int
    end: int


@dataclass
class CropBouonds:
    x: CropAxisBounds
    y: CropAxisBounds


def remove_large_objects(boolean_image: np.ndarray, min_size: int) -> np.ndarray:
    """Removes large connected components from a boolean image.

    Args:
        boolean_image (np.ndarray): The input boolean image.
        min_size (int): The minimum size of objects to keep. Objects larger than this will be removed.

    Returns:
        np.ndarray: The image with large objects removed.
    """
    # Remove small holes (which were originally large objects)
    cleaned_inverted_image = morphology.remove_small_objects(
        boolean_image, min_size=min_size
    )
    # Invert back to get the original image with large objects removed
    result_image = np.where(boolean_image & (~cleaned_inverted_image), 1, 0).astype(
        bool
    )
    return result_image


def extract_tall_thin_text_patterns(binarized_image: np.ndarray) -> np.ndarray:
    """
    Extracts tall and thin text patterns from a binary image using morphological operations.

    Args:
        binary_image (np.ndarray): The input binary image.

    Returns:
        np.ndarray: The processed image with tall and thin text patterns extracted.
    """
    input_for_morph = binarized_image
    was_3d = False
    if binarized_image.ndim == 3:
        was_3d = True
        # Assuming all channels are the same after binarization, take the first one.
        input_for_morph = binarized_image[:, :, 0]

    # Define a kernel for morphological operations
    kernel_size = max(5, int(5 / 50 * input_for_morph.shape[0]))
    kernel = np.ones((1, kernel_size), dtype=np.uint8)

    # Perform opening operation on the 2D image
    opened_image_2d = morphology.opening(
        input_for_morph,
        footprint=kernel,
    )

    # Identify eroded pixels (pixels in binary but not in opened_image_2d)
    eroded_pixels_2d = input_for_morph - opened_image_2d

    processed_image_2d = morphology.reconstruction(eroded_pixels_2d, input_for_morph)
    processed_image_2d = (
        remove_large_objects(
            processed_image_2d.astype(bool),
            min_size=binarized_image.shape[0] * (kernel_size // 2),
        ).astype(np.uint8)
        * 255
    )
    # Ensure no overflow and correct dtype
    processed_image_2d = np.clip(processed_image_2d, 0, 255).astype(
        input_for_morph.dtype
    )

    if was_3d:
        # Convert back to 3D if the original binarized image was 3D
        final_image = np.stack([processed_image_2d] * 3, axis=-1)
    else:
        final_image = processed_image_2d

    return final_image


@lru_cache(maxsize=128)
def infer_standard_resolution(
    raw_resolution: tuple[int, int], standard_ratio: tuple[int, int] = (1540, 2640)
) -> tuple[int, int]:
    """
    Infers the standard resolution of the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        tuple[int, int]: The inferred width and height of the image.
    """
    x_length = raw_resolution[0]
    y_length = raw_resolution[1]

    if x_length * standard_ratio[1] < y_length * standard_ratio[0]:
        y_length = int(x_length * standard_ratio[1] / standard_ratio[0])
    elif x_length * standard_ratio[1] > y_length * standard_ratio[0]:
        x_length = int(y_length * standard_ratio[0] / standard_ratio[1])

    logger.debug(
        f"Inferred standard resolution: {x_length}x{y_length} (from shape {raw_resolution})"
    )

    return x_length, y_length


def get_status_bar_x_bounds(image_array: np.ndarray) -> CropAxisBounds:
    """
    Calculates the height of the status bar in the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        CropAxisBounds: The bounds of the status bar.
    """
    std_x, _ = infer_standard_resolution(image_array.shape[:2])

    # Calculate the height of the status bar
    status_bar_height = int(std_x * 0.075)  # 5% from the top

    return CropAxisBounds(
        start=image_array.shape[0] - status_bar_height,
        end=image_array.shape[0],
    )


def get_level_area_bounds(image_array: np.ndarray) -> CropBouonds:
    """
    Gets the bounds of the level area in the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        CropBouonds: The bounds of the level area.
    """
    _, std_y = infer_standard_resolution(image_array.shape[:2])

    return CropBouonds(
        x=get_status_bar_x_bounds(image_array),
        y=CropAxisBounds(
            start=0,
            end=int(std_y * 0.1),
        ),
    )


def get_level_crop(image_array: np.ndarray, ocr_friendly: bool = False) -> np.ndarray:
    """
    Crops the level area from the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        np.ndarray: The cropped image array containing only the level area.
    """
    bounds = get_level_area_bounds(image_array)

    cropped_image = image_array[
        bounds.x.start : bounds.x.end, bounds.y.start : bounds.y.end, :
    ]

    if ocr_friendly:
        cropped_image = binarize_image(cropped_image, threshold=180)

    return cropped_image


def get_exp_area_bounds(image_array: np.ndarray) -> CropBouonds:
    """
    Gets the bounds of the experience area in the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        CropBouonds: The bounds of the experience area.
    """
    _, std_y = infer_standard_resolution(image_array.shape[:2])

    x = get_status_bar_x_bounds(image_array)

    return CropBouonds(
        x=CropAxisBounds(start=x.start, end=int((x.start + x.end) / 2)),
        y=CropAxisBounds(start=int(std_y * 0.53), end=int(std_y * 0.68)),
    )


def get_exp_crop(capture: np.ndarray, ocr_friendly: bool = False) -> np.ndarray:
    """
    Crops the experience area from the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        np.ndarray: The cropped image array containing only the experience area.
    """
    bounds = get_exp_area_bounds(capture)

    # Crop the image array
    cropped_image = capture[
        bounds.x.start : bounds.x.end, bounds.y.start : bounds.y.end, :
    ]

    if ocr_friendly:
        binarized_image = binarize_image(cropped_image, threshold=190)
        cropped_image = extract_tall_thin_text_patterns(binarized_image)

    return cropped_image


def get_hp_area_bounds(image_array: np.ndarray) -> CropBouonds:
    """
    Gets the bounds of the experience area in the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        CropBouonds: The bounds of the experience area.
    """
    _, std_y = infer_standard_resolution(image_array.shape[:2])

    x = get_status_bar_x_bounds(image_array)

    return CropBouonds(
        x=CropAxisBounds(start=x.start, end=int((x.start + x.end) / 2)),
        y=CropAxisBounds(start=int(std_y * 0.26), end=int(std_y * 0.393)),
    )


def get_hp_crop(capture: np.ndarray, ocr_friendly: bool = False) -> np.ndarray:
    """
    Crops the HP area from the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        np.ndarray: The cropped image array containing only the HP area.
    """
    bounds = get_hp_area_bounds(capture)

    # Crop the image array
    cropped_image = capture[
        bounds.x.start : bounds.x.end, bounds.y.start : bounds.y.end, :
    ]

    if ocr_friendly:
        binarized_image = binarize_image(cropped_image, threshold=190)
        cropped_image = extract_tall_thin_text_patterns(binarized_image)

    return cropped_image


def get_mp_area_bounds(image_array: np.ndarray) -> CropBouonds:
    """
    Gets the bounds of the MP area in the given image array.

    Args:
        image_array (np.ndarray): The input image array.

    Returns:
        CropBouonds: The bounds of the MP area.
    """
    _, std_y = infer_standard_resolution(image_array.shape[:2])

    x = get_status_bar_x_bounds(image_array)

    return CropBouonds(
        x=CropAxisBounds(start=x.start, end=int((x.start + x.end) / 2)),
        y=CropAxisBounds(start=int(std_y * 0.39), end=int(std_y * 0.535)),
    )


def get_mp_crop(capture: np.ndarray, ocr_friendly: bool = False) -> np.ndarray:
    """
    Crops the MP area from the given image array.
    Args:
        image_array (np.ndarray): The input image array.
    Returns:
        np.ndarray: The cropped image array containing only the MP area.
    """
    bounds = get_mp_area_bounds(capture)

    # Crop the image array
    cropped_image = capture[
        bounds.x.start : bounds.x.end, bounds.y.start : bounds.y.end, :
    ]

    if ocr_friendly:
        binarized_image = binarize_image(cropped_image, threshold=190)
        cropped_image = extract_tall_thin_text_patterns(binarized_image)

    return cropped_image
