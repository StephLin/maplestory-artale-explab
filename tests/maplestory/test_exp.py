# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from explab.maplestory.exp import ExpCheckpoint
from explab.ocr.base import BoundingBox, TextRecognitionResult


@patch("explab.maplestory.exp.ocr.recognize_text_from_image")
@patch("explab.maplestory.exp.cropper.get_exp_crop")
@patch("explab.maplestory.exp.cropper.get_level_crop")
def test_exp_checkpoint_from_app_capture(
    mock_get_level_crop: MagicMock,
    mock_get_exp_crop: MagicMock,
    mock_recognize_text: MagicMock,
):
    """
    Tests that Checkpoint.from_app_capture correctly processes OCR results
    from a given capture.
    """
    dummy_capture = np.array([])
    dummy_level_crop_img = np.array([1])
    dummy_exp_crop_img = np.array([2])

    mock_get_level_crop.return_value = dummy_level_crop_img
    mock_get_exp_crop.return_value = dummy_exp_crop_img

    # Simulate OCR results
    # First call for level, second for exp
    mock_recognize_text.side_effect = [
        [
            TextRecognitionResult(
                text="LV", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
            ),
            TextRecognitionResult(
                text="123", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
            ),
        ],  # Level OCR result
        [
            TextRecognitionResult(
                text="456789[12.34%",
                confidence=0.9,
                bounding_box=BoundingBox(0, 0, 10, 10),
            )
        ],  # Exp OCR result
    ]

    expected_level = 123
    expected_exp = 456789
    expected_exp_ratio = 0.1234
    now = datetime.datetime.now()

    checkpoint = ExpCheckpoint.from_app_capture(capture=dummy_capture, ts=now)

    assert checkpoint is not None
    assert checkpoint.level == expected_level
    assert checkpoint.exp == expected_exp
    assert checkpoint.exp_ratio == pytest.approx(expected_exp_ratio)
    assert checkpoint.ts == now

    mock_get_level_crop.assert_called_once()
    mock_get_exp_crop.assert_called_once()
    assert mock_recognize_text.call_count == 2


@patch("explab.maplestory.exp.ocr.recognize_text_from_image")
@patch("explab.maplestory.exp.cropper.get_exp_crop")
@patch("explab.maplestory.exp.cropper.get_level_crop")
def test_checkpoint_from_app_capture_no_valid_level(
    mock_get_level_crop: MagicMock,
    mock_get_exp_crop: MagicMock,
    mock_recognize_text: MagicMock,
):
    """
    Tests that Checkpoint.from_app_capture returns None if level is not found.
    """
    dummy_capture = np.array([])
    mock_get_level_crop.return_value = np.array([1])
    mock_get_exp_crop.return_value = np.array([2])

    mock_recognize_text.side_effect = [
        [
            TextRecognitionResult(
                text="LV", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
            ),
            TextRecognitionResult(
                text="ABC", confidence=0.1, bounding_box=BoundingBox(0, 0, 10, 10)
            ),
        ],  # Invalid Level OCR
        [
            TextRecognitionResult(
                text="456789[12.34%]",
                confidence=0.9,
                bounding_box=BoundingBox(0, 0, 10, 10),
            )
        ],
    ]

    checkpoint = ExpCheckpoint.from_app_capture(capture=dummy_capture)
    assert checkpoint is None


@patch("explab.maplestory.exp.ocr.recognize_text_from_image")
@patch("explab.maplestory.exp.cropper.get_exp_crop")
@patch("explab.maplestory.exp.cropper.get_level_crop")
def test_checkpoint_from_app_capture_no_valid_exp(
    mock_get_level_crop: MagicMock,
    mock_get_exp_crop: MagicMock,
    mock_recognize_text: MagicMock,
):
    """
    Tests that Checkpoint.from_app_capture returns None if exp is not found.
    """
    dummy_capture = np.array([])
    mock_get_level_crop.return_value = np.array([1])
    mock_get_exp_crop.return_value = np.array([2])

    mock_recognize_text.side_effect = [
        [
            TextRecognitionResult(
                text="123", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
            )
        ],
        [
            TextRecognitionResult(
                text="EXP GONE", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
            )
        ],  # Invalid Exp OCR
    ]

    checkpoint = ExpCheckpoint.from_app_capture(capture=dummy_capture)
    assert checkpoint is None
