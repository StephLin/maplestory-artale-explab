# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np


def binarize_image(
    image_array: np.ndarray, threshold: tuple[int, int, int] | int = 180
) -> np.ndarray:
    """
    Converts a grayscale image to a binary image using a specified threshold.

    Args:
        image_array (np.ndarray): The input grayscale image as a NumPy array.
        threshold (int): The threshold value for binarization. Default is 128.

    Returns:
        np.ndarray: The binarized image as a NumPy array.
    """
    # Apply the threshold to create a binary image
    binary_image = image_array[:]

    if isinstance(threshold, int):
        threshold = (threshold, threshold, threshold)

    binary_ = np.where(
        (image_array[..., 0] > threshold[0])
        & (image_array[..., 1] > threshold[1])
        & (image_array[..., 2] > threshold[2]),
        np.ones_like(image_array[..., 1]) * 255,
        np.zeros_like(image_array[..., 0]),
    ).astype(np.uint8)

    for i in range(3):
        binary_image[..., i] = binary_[...]

    return binary_image
