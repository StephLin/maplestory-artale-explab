# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Represents a bounding box with min and max coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class TextRecognitionResult:
    """
    Represents the result of text recognition from an image.

    Attributes:
        text (str): The recognized text.
        confidence (float): The confidence score of the recognition.
        bounding_box (BoundingBox): The coordinates of the bounding box for the recognized text.
    """

    text: str
    confidence: float
    bounding_box: BoundingBox
