# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


@patch("explab.maplestory.hp.recognize_text_from_images_batch")
@patch("explab.maplestory.hp.cropper.get_hp_crop")
def test_hp_checkpoint_from_app_captures_success(
    mock_get_hp_crop: MagicMock,
    mock_recognize_batch: MagicMock,
):
    """
    Tests that HpCheckpoint.from_app_captures correctly processes a batch of captures
    when all OCR results are successful.
    """
    num_captures = 3
    dummy_captures = [np.array([i]) for i in range(num_captures)]
    dummy_hp_crop_imgs = [np.array([i + 10]) for i in range(num_captures)]

    mock_get_hp_crop.side_effect = dummy_hp_crop_imgs

    # Simulate OCR results for batch
    mock_ocr_results_batch = [
        [
            TextRecognitionResult(
                text=f"[{10000 + i}/{20000 + i}]",
                confidence=0.9,
                bounding_box=BoundingBox(0, 0, 10, 10),
            )
        ]
        for i in range(num_captures)
    ]
    mock_recognize_batch.return_value = mock_ocr_results_batch

    expected_hps = [
        {"current": 10000 + i, "total": 20000 + i} for i in range(num_captures)
    ]
    now_timestamps = [datetime.datetime.now() for _ in range(num_captures)]

    checkpoints = HpCheckpoint.from_app_captures(
        captures=dummy_captures, ts_list=now_timestamps
    )

    assert len(checkpoints) == num_captures
    for i, checkpoint in enumerate(checkpoints):
        assert checkpoint is not None
        assert checkpoint.current_hp == expected_hps[i]["current"]
        assert checkpoint.total_hp == expected_hps[i]["total"]
        assert checkpoint.ts == now_timestamps[i]

    assert mock_get_hp_crop.call_count == num_captures
    for i in range(num_captures):
        mock_get_hp_crop.assert_any_call(dummy_captures[i], ocr_friendly=True)

    mock_recognize_batch.assert_called_once_with(
        dummy_hp_crop_imgs,
        allowlist="0123456789/[]",
        width_ths=10.0,
        batch_size=None,
    )


@patch("explab.maplestory.hp.recognize_text_from_images_batch")
@patch("explab.maplestory.hp.cropper.get_hp_crop")
def test_hp_checkpoint_from_app_captures_mixed_results(
    mock_get_hp_crop: MagicMock,
    mock_recognize_batch: MagicMock,
):
    """
    Tests that HpCheckpoint.from_app_captures handles mixed valid and invalid OCR results.
    """
    captures = [np.array([1]), np.array([2]), np.array([3])]
    cropped_images = [np.array([11]), np.array([12]), np.array([13])]
    mock_get_hp_crop.side_effect = cropped_images

    ocr_results_batch = [
        [
            TextRecognitionResult(
                text="[100/200]", confidence=0.9, bounding_box=BoundingBox(0, 0, 1, 1)
            )
        ],  # Valid
        [
            TextRecognitionResult(
                text="Invalid HP", confidence=0.9, bounding_box=BoundingBox(0, 0, 1, 1)
            )
        ],  # Invalid
        [
            TextRecognitionResult(
                text="[300/400]", confidence=0.9, bounding_box=BoundingBox(0, 0, 1, 1)
            )
        ],  # Valid
    ]
    mock_recognize_batch.return_value = ocr_results_batch

    ts_list = [datetime.datetime.now() for _ in captures]
    checkpoints = HpCheckpoint.from_app_captures(captures=captures, ts_list=ts_list)

    assert len(checkpoints) == 3
    assert checkpoints[0] is not None
    assert checkpoints[0].current_hp == 100
    assert checkpoints[0].total_hp == 200
    assert checkpoints[1] is None
    assert checkpoints[2] is not None
    assert checkpoints[2].current_hp == 300
    assert checkpoints[2].total_hp == 400


@patch("explab.maplestory.hp.recognize_text_from_images_batch")
@patch("explab.maplestory.hp.cropper.get_hp_crop")
def test_hp_checkpoint_from_app_captures_all_invalid(
    mock_get_hp_crop: MagicMock,
    mock_recognize_batch: MagicMock,
):
    """
    Tests that HpCheckpoint.from_app_captures returns None for all checkpoints
    if all OCR results are invalid.
    """
    captures = [np.array([1]), np.array([2])]
    cropped_images = [np.array([11]), np.array([12])]
    mock_get_hp_crop.side_effect = cropped_images

    ocr_results_batch = [
        [
            TextRecognitionResult(
                text="Invalid1", confidence=0.9, bounding_box=BoundingBox(0, 0, 1, 1)
            )
        ],
        [
            TextRecognitionResult(
                text="Invalid2", confidence=0.9, bounding_box=BoundingBox(0, 0, 1, 1)
            )
        ],
    ]
    mock_recognize_batch.return_value = ocr_results_batch

    checkpoints = HpCheckpoint.from_app_captures(captures=captures)

    assert len(checkpoints) == 2
    assert checkpoints[0] is None
    assert checkpoints[1] is None


def test_hp_checkpoint_from_app_captures_empty_input():
    """
    Tests that HpCheckpoint.from_app_captures handles an empty list of captures.
    """
    checkpoints = HpCheckpoint.from_app_captures(captures=[])
    assert checkpoints == []


@patch("explab.maplestory.hp.recognize_text_from_images_batch")
@patch("explab.maplestory.hp.cropper.get_hp_crop")
def test_hp_checkpoint_from_app_captures_no_ts_list(
    mock_get_hp_crop: MagicMock,
    mock_recognize_batch: MagicMock,
):
    """
    Tests that HpCheckpoint.from_app_captures uses datetime.now() if ts_list is not provided.
    """
    captures = [np.array([1])]
    cropped_images = [np.array([11])]
    mock_get_hp_crop.side_effect = cropped_images

    ocr_results_batch = [
        [
            TextRecognitionResult(
                text="[100/200]", confidence=0.9, bounding_box=BoundingBox(0, 0, 1, 1)
            )
        ]
    ]
    mock_recognize_batch.return_value = ocr_results_batch

    before_call = datetime.datetime.now()
    checkpoints = HpCheckpoint.from_app_captures(captures=captures)
    after_call = datetime.datetime.now()

    assert len(checkpoints) == 1
    assert checkpoints[0] is not None
    assert before_call <= checkpoints[0].ts <= after_call


def test_hp_checkpoint_from_app_captures_ts_list_mismatch():
    """
    Tests that HpCheckpoint.from_app_captures raises ValueError if ts_list length
    does not match captures length.
    """
    captures = [np.array([1]), np.array([2])]
    ts_list = [datetime.datetime.now()]  # Mismatched length

    with pytest.raises(ValueError) as excinfo:
        HpCheckpoint.from_app_captures(captures=captures, ts_list=ts_list)
    assert "Length of ts_list must match the number of captures" in str(excinfo.value)
