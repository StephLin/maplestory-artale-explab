# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from unittest import mock

import numpy as np
import pytest

from explab.screen_capture import capture_app_window


@mock.patch("sys.platform", "darwin")
@mock.patch("explab.screen_capture.mac_capture.capture_app_window")
def test_capture_app_window_macos(mock_mac_capture):
    """Test capture_app_window on macOS."""
    expected_image = np.array([[[0, 0, 0]]], dtype=np.uint8)
    mock_mac_capture.return_value = expected_image
    app_name = "TestApp"

    image = capture_app_window(app_name)

    mock_mac_capture.assert_called_once_with(app_name)
    assert isinstance(image, np.ndarray)
    np.testing.assert_array_equal(image, expected_image)


@mock.patch("sys.platform", "linux")  # Any non-darwin platform
def test_capture_app_window_non_macos():
    """Test capture_app_window on a non-macOS platform."""
    app_name = "TestApp"

    with pytest.raises(
        NotImplementedError,
        match="Screen capture is not implemented for the current platform: linux.",
    ):
        capture_app_window(app_name)


@mock.patch("sys.platform", "darwin")
@mock.patch("explab.screen_capture.mac_capture.capture_app_window", return_value=None)
def test_capture_app_window_macos_capture_failed(mock_mac_capture):
    """Test capture_app_window on macOS when mac_capture returns None."""
    app_name = "TestApp"

    image = capture_app_window(app_name)

    mock_mac_capture.assert_called_once_with(app_name)
    assert image is None
