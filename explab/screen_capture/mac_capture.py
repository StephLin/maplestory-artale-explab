# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# type: ignore
import ctypes
import sys

import numpy as np
import Quartz
import structlog
from AppKit import NSWorkspace
from Cocoa import NSBitmapImageRep

logger = structlog.get_logger()


def is_app_running(app_name) -> bool:
    """
    Checks if a given application is running on macOS.
    """
    if "NSWorkspace" in globals():
        for app in NSWorkspace.sharedWorkspace().runningApplications():
            if app.localizedName() == app_name:
                return True
        return False
    else:
        logger.warning(
            "NSWorkspace not loaded, cannot check if the application is running."
        )
        return False  # Or True, depending on desired behavior when NSWorkspace is not available


def capture_app_window(app_name) -> np.ndarray | None:
    """
    Screenshots the window of the specified application, even if it's not on top, and returns a NumPy array.

    Args:
        app_name (str): The name of the application to screenshot (e.g., "Safari", "Google Chrome", "Terminal").

    Returns:
        numpy.ndarray: The captured window image as a NumPy array (Height, Width, Channels),
                       or None on failure.
    """
    # Check if the app is running first
    if sys.platform == "darwin":
        if not is_app_running(app_name):
            logger.error(
                f"Application '{app_name}' is not running on macOS. Please start it first."
            )
            # Consider if sys.exit(1) is appropriate here or just return
            return  # Exit the function if app not running

    """
    Screenshots the window of the specified application, even if it's not on top.

    Args:
        app_name (str): The name of the application to screenshot (e.g., "Safari", "Google Chrome", "Terminal").
        output_filename (str): The filename to save the screenshot to.
    """
    options = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements
    )
    window_list = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID)

    target_window_id = None

    logger.info(f"Looking for window of application: {app_name}...")

    for window in window_list:
        owner_name = window.get(Quartz.kCGWindowOwnerName)
        window_title = window.get(Quartz.kCGWindowName)
        window_layer = window.get(Quartz.kCGWindowLayer)
        window_id = window.get(Quartz.kCGWindowNumber)

        if owner_name == app_name:
            if window_layer == 0:
                logger.debug(
                    f"Found window for application '{app_name}': ID={window_id}, Title='{window_title or 'Untitled'}'"
                )
                target_window_id = window_id
                break

    if target_window_id is None:
        logger.error(
            f"Could not find any window for application '{app_name}'. Please ensure the application is running and has a visible window."
        )
        return

    try:
        image_ref = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            target_window_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming,
        )

        if not image_ref:
            logger.error(f"Could not capture window ID {target_window_id}.")
            return

        bitmap_rep = NSBitmapImageRep.alloc().initWithCGImage_(image_ref)
        if not bitmap_rep:
            logger.error("Could not create NSBitmapImageRep from CGImageRef.")
            return

        # Convert NSBitmapImageRep to NumPy array
        width = int(bitmap_rep.pixelsWide())
        height = int(bitmap_rep.pixelsHigh())
        spp = int(bitmap_rep.samplesPerPixel())  # Samples per pixel
        bps = int(bitmap_rep.bitsPerSample())  # Bits per sample

        if bps != 8:
            logger.error(f"Unsupported bits per sample: {bps}. Expected 8.")
            return None

        # Common samples per pixel: 1 (Grayscale), 3 (RGB), 4 (RGBA)
        if spp not in [1, 3, 4]:
            logger.error(f"Unsupported samples per pixel: {spp}. Expected 1, 3, or 4.")
            return None

        data_ptr = bitmap_rep.bitmapData()
        if not data_ptr:
            logger.error("Failed to get bitmap data from NSBitmapImageRep.")
            return None

        # data_ptr is a memoryview.
        # Shape definition is crucial and depends on spp, height, width.
        if spp == 1:  # Grayscale
            shape = (height, width)
        else:  # RGB or RGBA (ARGB)
            shape = (height, width, spp)

        logger.debug(
            f"Bitmap properties - Width: {width}, Height: {height}, SPP: {spp}, BPS: {bps}, Shape: {shape}"
        )

        try:
            # Calculate the total number of ubyte elements needed for the given shape.
            # This is also the number of bytes since elements are c_ubyte.
            if spp == 1:
                num_elements = height * width
            else:
                num_elements = height * width * spp

            # Ensure the memoryview (data_ptr) provides enough data.
            if data_ptr.nbytes < num_elements:
                logger.error(
                    f"Memoryview size mismatch. Expected at least {num_elements} bytes, "
                    f"but memoryview (data_ptr) has {data_ptr.nbytes} bytes. "
                    f"Details - Target Shape: {shape}, Height: {height}, Width: {width}, SPP: {spp}"
                )
                return None

            # Create a ctypes array type for the exact number of elements required by the shape.
            c_ubyte_array_type = ctypes.c_ubyte * num_elements

            # Create a ctypes array from the memoryview's buffer.
            # This operation shares memory; no data is copied here.
            # It will use the first 'num_elements' bytes of 'data_ptr'.
            # Use from_buffer_copy to ensure the ctypes array is writable.
            c_array = c_ubyte_array_type.from_buffer_copy(data_ptr)

            # Use np.ctypeslib.as_array to get a NumPy view into the ctypes array.
            # This also shares memory (creates a view).
            np_array_view = np.ctypeslib.as_array(c_array).reshape(shape)

            # Finally, copy the data from the view to a new NumPy array.
            # This makes image_numpy independent of the original C buffer's lifecycle.
            image_numpy = np_array_view.copy()
        except Exception as e_np:
            logger.error(f"Error converting bitmap data to NumPy array: {e_np}")
            return None

        logger.debug(
            f"Successfully captured window for '{app_name}' with shape {image_numpy.shape}"
        )

        if spp == 4:
            # If the image is RGBA (ARGB), convert it to RGBA by removing the alpha channel
            image_numpy = image_numpy[..., [1, 2, 3, 0]]

        return image_numpy

    except Exception as e:
        logger.exception(f"Error during screenshot process: {e}")
