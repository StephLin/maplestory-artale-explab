# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from skimage.io import imread

from explab.maplestory.exp import ExpCheckpoint
from explab.maplestory.hp import HpCheckpoint
from explab.maplestory.mp import MpCheckpoint


@dataclass
class TestScreenshot:
    screenshot_path: Path
    expected_exp_checkpoint: ExpCheckpoint
    expected_hp_checkpoint: HpCheckpoint
    expected_mp_checkpoint: MpCheckpoint


ASSETS_DIR = Path(__file__).parents[1].absolute() / "assets"
# Test data using the TestScreenshot dataclass
TEST_DATA: list[TestScreenshot] = [
    TestScreenshot(
        screenshot_path=ASSETS_DIR / "screenshot1.png",
        expected_exp_checkpoint=ExpCheckpoint(level=42, exp=15432, exp_ratio=0.0432),
        expected_hp_checkpoint=HpCheckpoint(current_hp=1323, total_hp=1419),
        expected_mp_checkpoint=MpCheckpoint(current_mp=743, total_mp=962),
    ),
    TestScreenshot(
        screenshot_path=ASSETS_DIR / "screenshot2.png",
        expected_exp_checkpoint=ExpCheckpoint(level=43, exp=113291, exp_ratio=0.2896),
        expected_hp_checkpoint=HpCheckpoint(current_hp=1440, total_hp=1440),
        expected_mp_checkpoint=MpCheckpoint(current_mp=976, total_mp=976),
    ),
    TestScreenshot(
        screenshot_path=ASSETS_DIR / "screenshot3.png",
        expected_exp_checkpoint=ExpCheckpoint(level=43, exp=113291, exp_ratio=0.2896),
        expected_hp_checkpoint=HpCheckpoint(current_hp=1440, total_hp=1440),
        expected_mp_checkpoint=MpCheckpoint(current_mp=976, total_mp=976),
    ),
]


class TestCheckpointFromCapture:
    @pytest.mark.parametrize("test_case", TEST_DATA)
    def test_exp_checkpoint_from_capture(self, test_case: TestScreenshot):
        image_path = test_case.screenshot_path
        assert os.path.exists(image_path), f"Screenshot file not found: {image_path}"
        capture = imread(image_path)
        assert capture is not None, f"Failed to load image: {image_path}"

        checkpoint = ExpCheckpoint.from_app_capture(capture)

        assert checkpoint is not None, (
            f"Failed to create ExpCheckpoint from {test_case.screenshot_path.name}"
        )

        expected = test_case.expected_exp_checkpoint
        assert checkpoint.level == expected.level, (
            f"Level mismatch for {test_case.screenshot_path.name}"
        )
        assert checkpoint.exp == expected.exp, (
            f"EXP mismatch for {test_case.screenshot_path.name}"
        )
        assert checkpoint.exp_ratio == pytest.approx(expected.exp_ratio), (
            f"EXP Ratio mismatch for {test_case.screenshot_path.name}"
        )

    @pytest.mark.parametrize("test_case", TEST_DATA)
    def test_hp_checkpoint_from_capture(self, test_case: TestScreenshot):
        image_path = test_case.screenshot_path
        assert os.path.exists(image_path), f"Screenshot file not found: {image_path}"
        capture = imread(image_path)
        assert capture is not None, f"Failed to load image: {image_path}"

        checkpoint = HpCheckpoint.from_app_capture(capture)

        assert checkpoint is not None, (
            f"Failed to create HpCheckpoint from {test_case.screenshot_path.name}"
        )

        expected = test_case.expected_hp_checkpoint
        assert checkpoint.current_hp == expected.current_hp, (
            f"Current HP mismatch for {test_case.screenshot_path.name}"
        )
        assert checkpoint.total_hp == expected.total_hp, (
            f"Total HP mismatch for {test_case.screenshot_path.name}"
        )

    @pytest.mark.parametrize("test_case", TEST_DATA)
    def test_mp_checkpoint_from_capture(self, test_case: TestScreenshot):
        image_path = test_case.screenshot_path
        assert os.path.exists(image_path), f"Screenshot file not found: {image_path}"
        capture = imread(image_path)
        assert capture is not None, f"Failed to load image: {image_path}"

        checkpoint = MpCheckpoint.from_app_capture(capture)

        assert checkpoint is not None, (
            f"Failed to create MpCheckpoint from {test_case.screenshot_path.name}"
        )

        # The TODO for filling values is now handled by the TestScreenshot data
        expected = test_case.expected_mp_checkpoint
        assert checkpoint.current_mp == expected.current_mp, (
            f"Current MP mismatch for {test_case.screenshot_path.name}"
        )
        assert checkpoint.total_mp == expected.total_mp, (
            f"Total MP mismatch for {test_case.screenshot_path.name}"
        )
