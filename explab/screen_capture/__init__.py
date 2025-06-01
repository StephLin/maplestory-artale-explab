# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import sys

import numpy as np


def capture_app_window(app_name: str) -> np.ndarray | None:
    """
    Captures the window of the specified application and returns it as a NumPy array.

    Args:
        app_name: The name of the application to capture.

    Returns:
        numpy.ndarray: The captured window image as a NumPy array,
                       or None if capturing failed.

    Raises:
        NotImplementedError: If the current platform is not macOS.
    """
    if sys.platform == "darwin":
        from .mac_capture import capture_app_window as mac_capture_app_window

        return mac_capture_app_window(app_name)
    else:
        raise NotImplementedError(
            "Screen capture is only implemented for macOS at the moment."
        )


__all__ = ["capture_app_window"]
