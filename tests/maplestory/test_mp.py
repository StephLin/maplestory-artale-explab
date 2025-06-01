# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from explab.maplestory.mp import MpCheckpoint
from explab.ocr.base import BoundingBox, TextRecognitionResult


@patch("explab.maplestory.mp.ocr.recognize_text_from_image")
@patch("explab.maplestory.mp.cropper.get_mp_crop")
def test_mp_checkpoint_from_app_capture(
    mock_get_mp_crop: MagicMock,
    mock_recognize_text: MagicMock,
):
    """
    Tests that MpCheckpoint.from_app_capture correctly processes OCR results
    from a given capture.
    """
    dummy_capture = np.array([])
    dummy_mp_crop_img = np.array([2])

    mock_get_mp_crop.return_value = dummy_mp_crop_img

    # Simulate OCR results
    mock_recognize_text.return_value = [
        TextRecognitionResult(
            text="[12345/20000]", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
        )
    ]  # MP OCR result

    expected_current_mp = 12345
    expected_total_mp = 20000
    now = datetime.datetime.now()

    checkpoint = MpCheckpoint.from_app_capture(capture=dummy_capture, ts=now)

    assert checkpoint is not None
    assert checkpoint.current_mp == expected_current_mp
    assert checkpoint.total_mp == expected_total_mp
    assert checkpoint.ts == now

    mock_get_mp_crop.assert_called_once_with(dummy_capture, ocr_friendly=True)
    mock_recognize_text.assert_called_once_with(
        dummy_mp_crop_img, allowlist="0123456789/[]", width_ths=10.0
    )


@patch("explab.maplestory.mp.ocr.recognize_text_from_image")
@patch("explab.maplestory.mp.cropper.get_mp_crop")
def test_mp_checkpoint_from_app_capture_no_valid_mp(
    mock_get_mp_crop: MagicMock,
    mock_recognize_text: MagicMock,
):
    """
    Tests that MpCheckpoint.from_app_capture returns None if MP is not found.
    """
    dummy_capture = np.array([])
    dummy_mp_crop_img = np.array([2])
    mock_get_mp_crop.return_value = dummy_mp_crop_img

    # Simulate OCR results - invalid MP
    mock_recognize_text.return_value = [
        TextRecognitionResult(
            text="MP GONE", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
        )
    ]

    checkpoint = MpCheckpoint.from_app_capture(capture=dummy_capture)
    assert checkpoint is None
