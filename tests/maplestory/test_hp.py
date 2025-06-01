# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
from unittest.mock import MagicMock, patch

import numpy as np

from explab.maplestory.hp import HpCheckpoint
from explab.ocr.base import BoundingBox, TextRecognitionResult


@patch("explab.maplestory.hp.ocr.recognize_text_from_image")
@patch("explab.maplestory.hp.cropper.get_hp_crop")
def test_hp_checkpoint_from_app_capture(
    mock_get_hp_crop: MagicMock,
    mock_recognize_text: MagicMock,
):
    """
    Tests that HpCheckpoint.from_app_capture correctly processes OCR results
    from a given capture.
    """
    dummy_capture = np.array([])
    dummy_hp_crop_img = np.array([2])

    mock_get_hp_crop.return_value = dummy_hp_crop_img

    # Simulate OCR results
    mock_recognize_text.return_value = [
        TextRecognitionResult(
            text="[12345/20000]", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
        )
    ]  # HP OCR result

    expected_current_hp = 12345
    expected_total_hp = 20000
    now = datetime.datetime.now()

    checkpoint = HpCheckpoint.from_app_capture(capture=dummy_capture, ts=now)

    assert checkpoint is not None
    assert checkpoint.current_hp == expected_current_hp
    assert checkpoint.total_hp == expected_total_hp
    assert checkpoint.ts == now

    mock_get_hp_crop.assert_called_once_with(dummy_capture, ocr_friendly=True)
    mock_recognize_text.assert_called_once_with(
        dummy_hp_crop_img, allowlist="0123456789/[]", width_ths=10.0
    )


@patch("explab.maplestory.hp.ocr.recognize_text_from_image")
@patch("explab.maplestory.hp.cropper.get_hp_crop")
def test_hp_checkpoint_from_app_capture_no_valid_hp(
    mock_get_hp_crop: MagicMock,
    mock_recognize_text: MagicMock,
):
    """
    Tests that HpCheckpoint.from_app_capture returns None if HP is not found.
    """
    dummy_capture = np.array([])
    dummy_hp_crop_img = np.array([2])
    mock_get_hp_crop.return_value = dummy_hp_crop_img

    # Simulate OCR results - invalid HP
    mock_recognize_text.return_value = [
        TextRecognitionResult(
            text="HP GONE", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)
        )
    ]

    checkpoint = HpCheckpoint.from_app_capture(capture=dummy_capture)
    assert checkpoint is None
