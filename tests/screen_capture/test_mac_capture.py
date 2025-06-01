# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import ctypes  # Added import
import sys
from unittest.mock import MagicMock  # 'patch' is not used directly

import numpy as np
import pytest

MOCK_MACOS_LIBS = True
# Mock the modules at the system level before mac_capture tries to import them
_appkit_mock = MagicMock()
_appkit_mock.NSWorkspace = MagicMock()  # Define NSWorkspace on the mock
sys.modules["AppKit"] = _appkit_mock

_quartz_mock = MagicMock()
sys.modules["Quartz"] = _quartz_mock

_cocoa_mock = MagicMock()
_cocoa_mock.NSBitmapImageRep = MagicMock()  # Define NSBitmapImageRep on the mock
sys.modules["Cocoa"] = _cocoa_mock


# Now import the module under test.
# If on non-macOS, the mocks above will be used by mac_capture.py
from explab.screen_capture import mac_capture


# Fixture to reset mocks for NSWorkspace if needed between tests,
# though pytest typically isolates tests well.
@pytest.fixture(autouse=True)
def reset_macos_mocks(mocker):
    # This ensures that if we modify a mock (e.g. mac_capture.NSWorkspace),
    # it's reset for the next test.
    if MOCK_MACOS_LIBS:
        # If on a non-macOS system, ensure the mocks in sys.modules are consistently MagicMock
        # and have the necessary top-level attributes if they were to be accessed via sys.modules.
        _appkit_mock_fixture = MagicMock()
        _appkit_mock_fixture.NSWorkspace = MagicMock()
        _cocoa_mock_fixture = MagicMock()
        _cocoa_mock_fixture.NSBitmapImageRep = MagicMock()
        _quartz_mock_fixture = MagicMock()  # Basic mock for Quartz at sys.modules level

        mocker.patch.dict(
            sys.modules,
            {
                "AppKit": _appkit_mock_fixture,
                "Cocoa": _cocoa_mock_fixture,
                "Quartz": _quartz_mock_fixture,
            },
        )

    # Patch NSWorkspace within the mac_capture module's scope for tests
    # This is more direct for testing the functions in mac_capture.py
    # Ensure these mocks have the attributes that will be accessed.
    mock_ns_workspace = MagicMock()
    mocker.patch.object(mac_capture, "NSWorkspace", mock_ns_workspace)

    mock_quartz = MagicMock()
    # Define attributes on mock_quartz that are used in mac_capture.py
    mock_quartz.kCGWindowListOptionOnScreenOnly = 0
    mock_quartz.kCGWindowListExcludeDesktopElements = 0
    mock_quartz.kCGNullWindowID = "kCGNullWindowID_val"
    mock_quartz.kCGWindowOwnerName = "kCGWindowOwnerName_val"
    mock_quartz.kCGWindowName = "kCGWindowName_val"
    mock_quartz.kCGWindowLayer = "kCGWindowLayer_val"
    mock_quartz.kCGWindowNumber = "kCGWindowNumber_val"
    mock_quartz.CGRectNull = "CGRectNull_val"
    mock_quartz.kCGWindowListOptionIncludingWindow = (
        "kCGWindowListOptionIncludingWindow_val"
    )
    mock_quartz.kCGWindowImageBoundsIgnoreFraming = (
        "kCGWindowImageBoundsIgnoreFraming_val"
    )
    mock_quartz.CGWindowListCopyWindowInfo = MagicMock()
    mock_quartz.CGWindowListCreateImage = MagicMock()
    mocker.patch.object(mac_capture, "Quartz", mock_quartz)

    mock_ns_bitmap_image_rep_class = MagicMock(name="NSBitmapImageRepClassMock")
    mock_ns_bitmap_image_rep_instance = MagicMock(name="NSBitmapImageRepInstanceMock")

    # Define attributes that will be accessed on the instance returned by initWithCGImage_
    # These are called by the mac_capture.py code.
    mock_ns_bitmap_image_rep_instance.pixelsWide = MagicMock(return_value=0)
    mock_ns_bitmap_image_rep_instance.pixelsHigh = MagicMock(return_value=0)
    mock_ns_bitmap_image_rep_instance.samplesPerPixel = MagicMock(return_value=0)
    mock_ns_bitmap_image_rep_instance.bitsPerSample = MagicMock(return_value=0)
    mock_ns_bitmap_image_rep_instance.bitmapData = MagicMock(
        return_value=None
    )  # This will be a memoryview

    mock_ns_bitmap_image_rep_class.alloc().initWithCGImage_.return_value = (
        mock_ns_bitmap_image_rep_instance
    )
    mocker.patch.object(mac_capture, "NSBitmapImageRep", mock_ns_bitmap_image_rep_class)
    # Note: Pylance might still issue warnings about unknown attributes on mac_capture.Quartz
    # within test functions. This is often a limitation of static analysis with dynamic mocks.
    # The mocks are configured here with specific attributes, and tests should execute correctly.


def test_is_app_running_when_app_is_running(mocker):
    """Test is_app_running when the target application is running."""
    mock_app = MagicMock()
    mock_app.localizedName.return_value = "TestApp"

    # Ensure mac_capture.NSWorkspace is the one being patched and used
    mac_capture.NSWorkspace.sharedWorkspace().runningApplications.return_value = [
        mock_app
    ]

    assert mac_capture.is_app_running("TestApp") is True


def test_is_app_running_when_app_is_not_running(mocker):
    """Test is_app_running when the target application is not running."""
    mock_app = MagicMock()
    mock_app.localizedName.return_value = "OtherApp"
    mac_capture.NSWorkspace.sharedWorkspace().runningApplications.return_value = [
        mock_app
    ]

    assert mac_capture.is_app_running("TestApp") is False


def test_is_app_running_when_no_apps_are_running(mocker):
    """Test is_app_running when no applications are running."""
    mac_capture.NSWorkspace.sharedWorkspace().runningApplications.return_value = []
    assert mac_capture.is_app_running("TestApp") is False


def test_is_app_running_when_nsworkspace_not_in_globals_effectively(mocker):
    """
    Test is_app_running when NSWorkspace is effectively not available.
    The original code checks `if "NSWorkspace" in globals():`.
    We can simulate this by temporarily making mac_capture.NSWorkspace None or unpatching it.
    """
    # To simulate NSWorkspace not being available in the context of mac_capture.py,
    # we can patch it to raise an AttributeError or set it to None if the code handles that.
    # The original code checks `if "NSWorkspace" in globals():` which refers to its own module's globals.

    # We will mock the `globals()` call within `is_app_running` to not include 'NSWorkspace'
    # This is a bit more involved as `globals()` is a built-in.
    # A simpler way is to ensure `mac_capture.NSWorkspace` itself is not available
    # or to patch the `globals()` call if it were `mac_capture.globals()`.

    # Given the code `if "NSWorkspace" in globals():`, this check happens within mac_capture.py.
    # The most robust way to test this branch is to ensure that `mac_capture.NSWorkspace`
    # is not resolvable or is None, and the code handles it.
    # However, the code as written `from AppKit import NSWorkspace` means NSWorkspace
    # *will* be in its globals if AppKit is importable.
    # The check `if "NSWorkspace" in globals():` is a bit misleading if NSWorkspace is imported directly.
    # It would make more sense if NSWorkspace was conditionally imported or accessed.

    # Let's assume the intent is to check if NSWorkspace was successfully imported and is usable.
    # We can achieve a similar effect by making NSWorkspace raise an error or be None.

    mock_logger = mocker.patch.object(mac_capture, "logger")

    # Simulate NSWorkspace not being loaded by removing it from the module's context for this test
    # This is tricky because of how Python imports work.
    # A more direct test of the `else` branch of `if "NSWorkspace" in globals():`
    # would require modifying the source or a more complex patching of `globals` itself.

    # Let's try to simulate the condition where the check `if "NSWorkspace" in globals()` fails.
    # This means `NSWorkspace` is not in the `mac_capture` module's global scope.
    # We can achieve this by temporarily deleting it from the module if it was imported.

    original_nsworkspace = None
    if hasattr(mac_capture, "NSWorkspace"):
        original_nsworkspace = mac_capture.NSWorkspace
        del mac_capture.NSWorkspace  # Temporarily remove it

    # To ensure the `else` branch in `is_app_running` is hit, we need `NSWorkspace` to not be in `mac_capture.globals()`.
    # The `from AppKit import NSWorkspace` line in `mac_capture.py` makes this tricky without altering the file
    # or using advanced import manipulation.
    # The provided code's check `if "NSWorkspace" in globals()` will almost always be true if the import succeeds.
    # If the import `from AppKit import NSWorkspace` fails, an ImportError would occur before `is_app_running` is called.

    # For the purpose of this test, let's assume the `globals()` check is meant to be more robust,
    # perhaps against `NSWorkspace` being `None`.
    # A direct way to test the logger warning is to control the `NSWorkspace` object.
    # If we set `mac_capture.NSWorkspace = None`, the `sharedWorkspace()` call would fail.
    # The current code structure makes the `else` branch of `if "NSWorkspace" in globals()` hard to reach
    # if `AppKit` is successfully imported.

    # Let's assume the most likely scenario for the warning is if `AppKit` itself couldn't be imported,
    # and `NSWorkspace` was thus never added to `mac_capture`'s globals.
    # Our initial MOCK_MACOS_LIBS setup handles this by replacing `AppKit` with a mock.
    # If `AppKit` is a mock, `NSWorkspace` would be an attribute of that mock.

    # Re-evaluating the original code: `from AppKit import NSWorkspace`
    # If this line executes, NSWorkspace IS in globals().
    # The only way it wouldn't be is if AppKit itself is not found, leading to ImportError.
    # So, the `else` branch of `if "NSWorkspace" in globals()` in `is_app_running` seems unreachable
    # if the module loads without an ImportError on `from AppKit import NSWorkspace`.

    # Let's assume the check is more about `NSWorkspace` being usable.
    # We will patch `mac_capture.NSWorkspace` to be something that causes the intended behavior (log warning).
    # The current code would try `NSWorkspace.sharedWorkspace()`. If `NSWorkspace` is not what's expected,
    # this would fail. The `if "NSWorkspace" in globals()` is the primary gate.

    # To truly test the `else` branch of `if "NSWorkspace" in globals():`, we'd need to ensure
    # that `mac_capture.py` is loaded in an environment where `NSWorkspace` is not in its global scope.
    # This can be done by carefully managing `sys.modules` or by directly patching `mac_capture.globals`.
    # Patching `mac_capture.globals` is not straightforward.

    # Given the structure, the most practical way to test the warning is to ensure
    # `NSWorkspace` is not the expected object.
    # However, the test is specifically for the `else` branch of `if "NSWorkspace" in globals():`.
    # This implies `NSWorkspace` is not defined in that scope.

    # Let's assume the `MOCK_MACOS_LIBS` path: `sys.modules['AppKit'] = MagicMock()`.
    # Then `from AppKit import NSWorkspace` would make `mac_capture.NSWorkspace` a `MagicMock` (or an attribute of it).
    # The check `if "NSWorkspace" in globals()` would still be true.

    # The most direct way to test the warning path is to assume the `globals()` check fails.
    # This is hard to simulate perfectly without changing the source or very deep magic.
    # Let's assume the spirit of the test is "what if NSWorkspace is unusable".
    # The current `is_app_running` will call `NSWorkspace.sharedWorkspace()`.
    # If `NSWorkspace` is, say, `None`, this will error.
    # The `if "NSWorkspace" in globals()` is the key.

    # If we want to test the *exact* `else` branch, we need `NSWorkspace` to not be in `mac_capture.globals`.
    # This means the `from AppKit import NSWorkspace` line must effectively fail to add it,
    # or it's removed.
    # This test aims to simulate the scenario where 'NSWorkspace' is not in the globals
    # of the mac_capture module when is_app_running is called.
    mock_logger_obj = mocker.patch.object(mac_capture, "logger")

    # Store original NSWorkspace if it exists in mac_capture's __dict__ (globals for the module)
    # The autouse fixture patches `mac_capture.NSWorkspace` (as an attribute),
    # but the code checks `if "NSWorkspace" in globals():` which refers to the module's __dict__.
    original_ns_workspace_in_dict = mac_capture.__dict__.pop(
        "NSWorkspace", "SENTINEL_NOT_FOUND"
    )

    try:
        assert mac_capture.is_app_running("TestApp") is False
        mock_logger_obj.warning.assert_called_with(
            "NSWorkspace not loaded, cannot check if the application is running."
        )
    finally:
        # Restore NSWorkspace in module's __dict__ if it was removed
        if original_ns_workspace_in_dict != "SENTINEL_NOT_FOUND":
            mac_capture.__dict__["NSWorkspace"] = original_ns_workspace_in_dict
        # The autouse fixture `reset_macos_mocks` will re-patch `mac_capture.NSWorkspace`
        # for subsequent tests, ensuring it's a MagicMock attribute.


# Tests for capture_app_window will be more involved
# and require mocking Quartz, NSBitmapImageRep, and ctypes interactions.


def test_capture_app_window_app_not_running_on_macos(mocker):
    """Test capture_app_window when the app is not running on macOS."""
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=False)
    mock_logger = mocker.patch.object(mac_capture, "logger")

    assert mac_capture.capture_app_window("TestApp") is None
    mock_logger.error.assert_called_with(
        "Application 'TestApp' is not running on macOS. Please start it first."
    )


def test_capture_app_window_not_macos(mocker):
    """Test capture_app_window behavior when not on macOS (should not try to check if app is running)."""
    mocker.patch.object(sys, "platform", "linux")  # Simulate non-macOS
    # is_app_running should not be called if not darwin
    mock_is_app_running = mocker.patch.object(mac_capture, "is_app_running")

    # The function will proceed to try and use Quartz, which will be mocked.
    # We expect it to fail finding the window if Quartz is just a basic mock.
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = []  # type: ignore[attr-defined]
    mock_logger = mocker.patch.object(mac_capture, "logger")

    assert mac_capture.capture_app_window("TestApp") is None
    mock_is_app_running.assert_not_called()
    mock_logger.error.assert_called_with(
        "Could not find any window for application 'TestApp'. Please ensure the application is running and has a visible window."
    )


def test_capture_app_window_no_window_found(mocker):
    """Test capture_app_window when no window is found for the app."""
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)

    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = []  # type: ignore[attr-defined]
    mock_logger = mocker.patch.object(mac_capture, "logger")

    assert mac_capture.capture_app_window("TestApp") is None
    mock_logger.error.assert_called_with(
        "Could not find any window for application 'TestApp'. Please ensure the application is running and has a visible window."
    )


# More detailed tests for capture_app_window success path would go here,
# mocking Quartz.CGWindowListCreateImage, NSBitmapImageRep, ctypes, etc.
# This would involve creating mock CGImageRef, mock NSBitmapImageRep, and mock bitmapData.


# Example of a more involved test for capture_app_window (simplified)
def test_capture_app_window_successful_capture_rgb(mocker):
    """Test a successful window capture for an RGB image."""
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)

    mock_window_info = {
        mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowName: "Test Window",  # type: ignore[attr-defined]
    }
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [mock_window_info]  # type: ignore[attr-defined]

    mock_cg_image_ref = MagicMock(name="CGImageRef")
    mac_capture.Quartz.CGWindowListCreateImage.return_value = mock_cg_image_ref  # type: ignore[attr-defined]

    # The fixture mocks NSBitmapImageRep.alloc().initWithCGImage_ to return an instance
    # (mock_ns_bitmap_image_rep_instance) that has mock attributes.
    mock_bitmap_rep_instance = mac_capture.NSBitmapImageRep.alloc().initWithCGImage_(
        mock_cg_image_ref
    )

    width, height, spp, bps = 100, 50, 3, 8  # RGB
    mock_bitmap_rep_instance.pixelsWide.return_value = width  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.pixelsHigh.return_value = height  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.samplesPerPixel.return_value = spp  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.bitsPerSample.return_value = bps  # type: ignore[attr-defined]

    # Create fake bitmap data (RGB, 8-bit per sample)
    # This needs to be a memoryview-like object or something np.ctypeslib.as_array can handle
    fake_image_data = np.random.randint(
        0, 256, size=(height, width, spp), dtype=np.uint8
    )

    # The `bitmapData()` method in the original code returns a memoryview.
    # We need to mock this to return something that `ctypes.c_ubyte.from_buffer` can use.
    # A bytes object or a ctypes array itself.
    fake_data_bytes = fake_image_data.tobytes()

    # Mocking memoryview behavior:
    mock_data_ptr = memoryview(fake_data_bytes)
    mock_bitmap_rep_instance.bitmapData.return_value = mock_data_ptr  # type: ignore[attr-defined]

    # Mock np.ctypeslib.as_array if direct control is needed, or ensure data flows through
    # For simplicity, we assume the data conversion works if inputs are correct.
    # The code uses np.ctypeslib.as_array(c_array).reshape(shape)
    # and then .copy(). We need to ensure c_array is formed correctly.

    # Mock ctypes part
    mock_ctypes_array = (ctypes.c_ubyte * (width * height * spp)).from_buffer_copy(
        fake_data_bytes
    )
    mocker.patch("ctypes.c_ubyte", new=ctypes.c_ubyte)  # Ensure we use the real c_ubyte

    # The internal `c_ubyte_array_type.from_buffer(data_ptr)` needs `data_ptr` to be a buffer.
    # `memoryview(bytes_obj)` is a buffer.

    result_image = mac_capture.capture_app_window("TestApp")

    assert result_image is not None
    assert isinstance(result_image, np.ndarray)
    assert result_image.shape == (height, width, spp)
    assert result_image.dtype == np.uint8
    # Potentially compare content if fake_image_data was more deterministic
    np.testing.assert_array_equal(result_image, fake_image_data)

    mac_capture.Quartz.CGWindowListCreateImage.assert_called_with(  # type: ignore[attr-defined]
        mac_capture.Quartz.CGRectNull,  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowListOptionIncludingWindow,  # type: ignore[attr-defined]
        123,  # target_window_id
        mac_capture.Quartz.kCGWindowImageBoundsIgnoreFraming,  # type: ignore[attr-defined]
    )


def test_capture_app_window_successful_capture_rgba_to_rgb(mocker):
    """Test a successful window capture for an RGBA image, checking conversion."""
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)

    mock_window_info = {
        mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowName: "Test Window RGBA",  # type: ignore[attr-defined]
    }
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [mock_window_info]  # type: ignore[attr-defined]

    mock_cg_image_ref = MagicMock(name="CGImageRef_RGBA")
    mac_capture.Quartz.CGWindowListCreateImage.return_value = mock_cg_image_ref  # type: ignore[attr-defined]

    mock_bitmap_rep_instance = mac_capture.NSBitmapImageRep.alloc().initWithCGImage_(
        mock_cg_image_ref
    )

    width, height, spp_original, bps = 10, 5, 4, 8  # RGBA (or ARGB depending on system)
    mock_bitmap_rep_instance.pixelsWide.return_value = width  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.pixelsHigh.return_value = height  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.samplesPerPixel.return_value = (  # type: ignore[attr-defined]
        spp_original  # e.g., 4 for RGBA/ARGB
    )
    mock_bitmap_rep_instance.bitsPerSample.return_value = bps  # type: ignore[attr-defined]

    # Create fake ARGB data (as macOS might provide)
    # A B G R (order might vary, common is BGRA or ARGB for underlying buffers)
    # The code converts from ARGB to RGBA if spp is 4: image_numpy[..., [1, 2, 3, 0]]
    # This means input is assumed to be ARGB.
    fake_argb_data = np.random.randint(
        0, 256, size=(height, width, spp_original), dtype=np.uint8
    )

    fake_data_bytes = fake_argb_data.tobytes()
    mock_data_ptr = memoryview(fake_data_bytes)
    mock_bitmap_rep_instance.bitmapData.return_value = mock_data_ptr  # type: ignore[attr-defined]

    result_image = mac_capture.capture_app_window("TestApp")

    assert result_image is not None
    assert isinstance(result_image, np.ndarray)
    # The output shape should be RGBA (H, W, 4) after conversion
    assert result_image.shape == (height, width, 4)
    assert result_image.dtype == np.uint8

    # Construct the expected RGBA from the ARGB source
    expected_rgba_data = fake_argb_data[..., [1, 2, 3, 0]]  # A,R,G,B -> R,G,B,A
    np.testing.assert_array_equal(result_image, expected_rgba_data)


def test_capture_app_window_unsupported_bps(mocker):
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)
    # ... setup mocks for window list and image ref ...
    # Use the mocked Quartz constants as keys
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [  # type: ignore[attr-defined]
        {
            mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
        }
    ]
    mac_capture.Quartz.CGWindowListCreateImage.return_value = MagicMock()  # type: ignore[attr-defined]

    mock_bitmap_rep_instance = mac_capture.NSBitmapImageRep.alloc().initWithCGImage_(
        MagicMock()
    )
    mock_bitmap_rep_instance.pixelsWide.return_value = 10  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.pixelsHigh.return_value = 10  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.samplesPerPixel.return_value = 3  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.bitsPerSample.return_value = 16  # type: ignore[attr-defined]

    mock_logger = mocker.patch.object(mac_capture, "logger")
    assert mac_capture.capture_app_window("TestApp") is None
    mock_logger.error.assert_called_with("Unsupported bits per sample: 16. Expected 8.")


def test_capture_app_window_unsupported_spp(mocker):
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [  # type: ignore[attr-defined]
        {
            mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
        }
    ]
    mac_capture.Quartz.CGWindowListCreateImage.return_value = MagicMock()  # type: ignore[attr-defined]

    mock_bitmap_rep_instance = mac_capture.NSBitmapImageRep.alloc().initWithCGImage_(
        MagicMock()
    )
    mock_bitmap_rep_instance.pixelsWide.return_value = 10  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.pixelsHigh.return_value = 10  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.samplesPerPixel.return_value = 2  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.bitsPerSample.return_value = 8  # type: ignore[attr-defined]

    mock_logger = mocker.patch.object(mac_capture, "logger")
    assert mac_capture.capture_app_window("TestApp") is None
    mock_logger.error.assert_called_with(
        "Unsupported samples per pixel: 2. Expected 1, 3, or 4."
    )


def test_capture_app_window_failed_to_get_bitmap_data(mocker):
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [  # type: ignore[attr-defined]
        {
            mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
        }
    ]
    mac_capture.Quartz.CGWindowListCreateImage.return_value = MagicMock()  # type: ignore[attr-defined]

    mock_bitmap_rep_instance = mac_capture.NSBitmapImageRep.alloc().initWithCGImage_(
        MagicMock()
    )
    mock_bitmap_rep_instance.pixelsWide.return_value = 10  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.pixelsHigh.return_value = 10  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.samplesPerPixel.return_value = 3  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.bitsPerSample.return_value = 8  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.bitmapData.return_value = None  # type: ignore[attr-defined]

    mock_logger = mocker.patch.object(mac_capture, "logger")
    assert mac_capture.capture_app_window("TestApp") is None
    mock_logger.error.assert_called_with(
        "Failed to get bitmap data from NSBitmapImageRep."
    )


def test_capture_app_window_cg_image_creation_fails(mocker):
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [  # type: ignore[attr-defined]
        {
            mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowName: "Test Window",  # type: ignore[attr-defined]
        }
    ]
    mac_capture.Quartz.CGWindowListCreateImage.return_value = (  # type: ignore[attr-defined]
        None  # CGImage creation fails
    )

    mock_logger = mocker.patch.object(mac_capture, "logger")
    assert mac_capture.capture_app_window("TestApp") is None
    mock_logger.error.assert_called_with("Could not capture window ID 123.")


def test_capture_app_window_nsbitmapimagerep_creation_fails(mocker):
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [  # type: ignore[attr-defined]
        {
            mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowName: "Test Window",  # type: ignore[attr-defined]
        }
    ]
    mac_capture.Quartz.CGWindowListCreateImage.return_value = (  # type: ignore[attr-defined]
        MagicMock()
    )  # CGImage creation succeeds
    mac_capture.NSBitmapImageRep.alloc().initWithCGImage_.return_value = (
        None  # NSBitmapImageRep creation fails
    )

    mock_logger = mocker.patch.object(mac_capture, "logger")
    assert mac_capture.capture_app_window("TestApp") is None
    mock_logger.error.assert_called_with(
        "Could not create NSBitmapImageRep from CGImageRef."
    )


def test_capture_app_window_memory_view_size_mismatch(mocker):
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)

    mock_window_info = {
        mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
        mac_capture.Quartz.kCGWindowName: "Test Window",  # type: ignore[attr-defined]
    }
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [mock_window_info]  # type: ignore[attr-defined]
    mac_capture.Quartz.CGWindowListCreateImage.return_value = MagicMock()  # type: ignore[attr-defined]

    mock_bitmap_rep_instance = mac_capture.NSBitmapImageRep.alloc().initWithCGImage_(
        MagicMock()
    )

    width, height, spp, bps = 100, 50, 3, 8
    mock_bitmap_rep_instance.pixelsWide.return_value = width  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.pixelsHigh.return_value = height  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.samplesPerPixel.return_value = spp  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.bitsPerSample.return_value = bps  # type: ignore[attr-defined]

    # Data too small
    small_fake_data_bytes = np.random.randint(
        0, 255, size=(height // 2, width, spp), dtype=np.uint8
    ).tobytes()
    mock_data_ptr = memoryview(small_fake_data_bytes)
    mock_bitmap_rep_instance.bitmapData.return_value = mock_data_ptr  # type: ignore[attr-defined]

    mock_logger = mocker.patch.object(mac_capture, "logger")
    result = mac_capture.capture_app_window("TestApp")
    assert result is None
    expected_elements = height * width * spp
    mock_logger.error.assert_called_with(
        (
            f"Memoryview size mismatch. Expected at least {expected_elements} bytes, "
            f"but memoryview (data_ptr) has {mock_data_ptr.nbytes} bytes. "
            f"Details - Target Shape: {(height, width, spp)}, Height: {height}, "
            f"Width: {width}, SPP: {spp}"
        )
    )  # Ruff E501 fix


def test_capture_app_window_numpy_conversion_exception(mocker):
    mocker.patch.object(sys, "platform", "darwin")
    mocker.patch.object(mac_capture, "is_app_running", return_value=True)
    # ... setup mocks up to bitmapData ...
    mac_capture.Quartz.CGWindowListCopyWindowInfo.return_value = [  # type: ignore[attr-defined]
        {
            mac_capture.Quartz.kCGWindowOwnerName: "TestApp",  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowLayer: 0,  # type: ignore[attr-defined]
            mac_capture.Quartz.kCGWindowNumber: 123,  # type: ignore[attr-defined]
        }
    ]
    mac_capture.Quartz.CGWindowListCreateImage.return_value = MagicMock()  # type: ignore[attr-defined]
    mock_bitmap_rep_instance = mac_capture.NSBitmapImageRep.alloc().initWithCGImage_(
        MagicMock()
    )
    mock_bitmap_rep_instance.pixelsWide.return_value = 10  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.pixelsHigh.return_value = 10  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.samplesPerPixel.return_value = 3  # type: ignore[attr-defined]
    mock_bitmap_rep_instance.bitsPerSample.return_value = 8  # type: ignore[attr-defined]
    # Valid data for shape, but we'll make np.ctypeslib.as_array fail
    fake_data = np.zeros((10 * 10 * 3), dtype=np.uint8).tobytes()
    mock_bitmap_rep_instance.bitmapData.return_value = memoryview(fake_data)  # type: ignore[attr-defined]

    mocker.patch(
        "numpy.ctypeslib.as_array",
        side_effect=ValueError("Test NumPy conversion error"),
    )

    mock_logger = mocker.patch.object(mac_capture, "logger")
    assert mac_capture.capture_app_window("TestApp") is None
    mock_logger.error.assert_called_with(
        "Error converting bitmap data to NumPy array: Test NumPy conversion error"
    )
