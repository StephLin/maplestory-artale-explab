# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


import ctypes.wintypes  # Import wintypes

import numpy as np
import structlog

logger = structlog.get_logger()

# Windows API functions and constants
user32 = ctypes.windll.user32
psapi = ctypes.windll.psapi
kernel32 = ctypes.windll.kernel32
gdi32 = ctypes.windll.gdi32

# Constants for GetWindowThreadProcessId, OpenProcess, EnumProcessModules, GetModuleBaseNameW
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

# For GetWindowText or GetModuleFileNameExW
MAX_PATH = 260


def is_app_running(app_name_target: str) -> bool:
    """
    Checks if a given application is running on Windows by checking process names.
    Compares against the executable name (e.g., "notepad.exe").
    """
    if not isinstance(app_name_target, str):
        logger.error("app_name_target must be a string.")
        return False

    # Ensure ctypes types are correctly referenced
    DWORD = ctypes.wintypes.DWORD
    WCHAR_ARRAY = ctypes.c_wchar * MAX_PATH

    # Get all process IDs.
    array_size = 1024  # Max number of processes to retrieve
    pids_array = (DWORD * array_size)()
    bytes_returned = DWORD()

    if not psapi.EnumProcesses(
        ctypes.byref(pids_array),
        ctypes.sizeof(pids_array),
        ctypes.byref(bytes_returned),
    ):
        logger.error(
            f"Failed to enumerate processes. Error code: {kernel32.GetLastError()}"
        )
        return False

    num_pids = bytes_returned.value // ctypes.sizeof(DWORD)

    for i in range(num_pids):
        pid = pids_array[i]
        if pid == 0:  # Skip idle process or system process
            continue

        h_process = kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
        )
        if not h_process:
            # logger.debug(f"Could not open process {pid}. Error: {kernel32.GetLastError()}. Skipping.")
            continue

        try:
            exe_name_buffer = WCHAR_ARRAY()
            if psapi.GetModuleFileNameExW(h_process, None, exe_name_buffer, MAX_PATH):
                process_exe_name = exe_name_buffer.value.split("\\")[-1]

                # Case-insensitive comparison
                if process_exe_name.lower() == app_name_target.lower():
                    logger.debug(
                        f"Application '{app_name_target}' found running with PID {pid} (exact match)."
                    )
                    kernel32.CloseHandle(h_process)
                    return True
                # If app_name_target doesn't have .exe, try matching without it (less precise)
                # Also handle if app_name_target has .exe but process_exe_name doesn't (unlikely for GetModuleFileNameExW)
                elif app_name_target.lower().endswith(
                    ".exe"
                ) and process_exe_name.lower() == app_name_target.lower().replace(
                    ".exe", ""
                ):
                    logger.debug(
                        f"Application '{app_name_target}' found running as '{process_exe_name}' with PID {pid} (name match without .exe)."
                    )
                    kernel32.CloseHandle(h_process)
                    return True
                elif (
                    not app_name_target.lower().endswith(".exe")
                    and process_exe_name.lower() == app_name_target.lower() + ".exe"
                ):
                    logger.debug(
                        f"Application '{app_name_target}' found running as '{process_exe_name}' with PID {pid} (name match with .exe)."
                    )
                    kernel32.CloseHandle(h_process)
                    return True

        except Exception as e:
            logger.debug(
                f"Error processing PID {pid} ('{process_exe_name if 'process_exe_name' in locals() else 'N/A'}'): {e}"
            )
        finally:
            kernel32.CloseHandle(h_process)

    logger.info(
        f"Application '{app_name_target}' not found running after checking {num_pids} processes."
    )
    return False


# For EnumWindows callback
WNDENUMPROC = ctypes.WINFUNCTYPE(
    ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
)

# For GetClientRect
RECT = ctypes.wintypes.RECT

# For BitBlt
SRCCOPY = 0x00CC0020

# For DIBs
BI_RGB = 0
DIB_RGB_COLORS = 0

# For PrintWindow
PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002  # Windows 8.1+

# For RedrawWindow
RDW_INVALIDATE = 0x0001
RDW_UPDATENOW = 0x0100
RDW_ERASE = 0x0004


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.wintypes.DWORD),
        ("biWidth", ctypes.wintypes.LONG),
        ("biHeight", ctypes.wintypes.LONG),
        ("biPlanes", ctypes.wintypes.WORD),
        ("biBitCount", ctypes.wintypes.WORD),
        ("biCompression", ctypes.wintypes.DWORD),
        ("biSizeImage", ctypes.wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.wintypes.LONG),
        ("biYPelsPerMeter", ctypes.wintypes.LONG),
        ("biClrUsed", ctypes.wintypes.DWORD),
        ("biClrImportant", ctypes.wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", ctypes.wintypes.DWORD * 1),
    ]  # RGBQUAD array, only one for BI_RGB


# Global variable to store HWND found by EnumWindows
found_hwnd = None


def _enum_windows_callback_find_title(
    hwnd: ctypes.wintypes.HWND, lparam_app_title_substr_ptr: ctypes.wintypes.LPARAM
) -> bool:
    """Callback function for EnumWindows to find a window by title substring."""
    global found_hwnd
    if not user32.IsWindowVisible(hwnd) or not user32.IsWindowEnabled(hwnd):
        return True  # Continue enumeration

    length = user32.GetWindowTextLengthW(hwnd) + 1
    if length == 1:  # No title
        return True

    buffer = (ctypes.c_wchar * length)()
    user32.GetWindowTextW(hwnd, buffer, length)
    window_title = buffer.value
    logger.debug(
        f"EnumWindows Callback: Checking HWND={hwnd}, Title='{window_title}', Visible={user32.IsWindowVisible(hwnd)}, Enabled={user32.IsWindowEnabled(hwnd)}"
    )

    # Simplified: Assume target_title_substr is accessible via a global set before calling EnumWindows
    if (
        target_title_substr_for_callback
        and target_title_substr_for_callback.lower() in window_title.lower()
    ):
        logger.debug(
            f"EnumWindows Callback: HWND={hwnd}, Title='{window_title}' contains target substring '{target_title_substr_for_callback}'."
        )

        # Check if it's a main application window (not a child, popup, etc.)
        is_main_window_candidate = (
            user32.GetParent(hwnd) == 0
            and user32.GetWindow(hwnd, ctypes.wintypes.DWORD(4)) == 0
        )  # GW_OWNER (4)
        logger.debug(
            f"EnumWindows Callback: HWND={hwnd}, Is main window candidate (GetParent=0, GetOwner=0): {is_main_window_candidate}"
        )

        if is_main_window_candidate:
            # Check if it has a valid process ID
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            logger.debug(f"EnumWindows Callback: HWND={hwnd}, PID={pid.value}")

            if pid.value != 0:
                # Additional check: ensure the window has some dimensions, as sometimes invisible/stub windows might pass other checks
                rect = RECT()
                if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                    width = rect.right - rect.left
                    height = rect.bottom - rect.top
                    logger.debug(
                        f"EnumWindows Callback: HWND={hwnd}, ClientRect Width={width}, Height={height}"
                    )
                    if width > 0 and height > 0:
                        found_hwnd = hwnd
                        logger.info(  # Changed to INFO for the selected window
                            f"EnumWindows Callback: Selected HWND={hwnd}, Title='{window_title}' matching '{target_title_substr_for_callback}', PID={pid.value}, Size=({width}x{height})"
                        )
                        return False  # Stop enumeration
                    else:
                        logger.debug(
                            f"EnumWindows Callback: HWND={hwnd}, Title='{window_title}' has zero dimensions, skipping."
                        )
                else:
                    logger.debug(
                        f"EnumWindows Callback: HWND={hwnd}, Title='{window_title}', GetClientRect failed, skipping."
                    )
            else:
                logger.debug(
                    f"EnumWindows Callback: HWND={hwnd}, Title='{window_title}' has PID 0, skipping."
                )
        else:
            logger.debug(
                f"EnumWindows Callback: HWND={hwnd}, Title='{window_title}' is not a main window candidate, skipping."
            )
    return True


target_title_substr_for_callback = None  # Global to pass target title to callback


def find_window_by_title_substring(title_substring: str) -> ctypes.wintypes.HWND | None:
    """Finds a top-level window HWND whose title contains title_substring."""
    global found_hwnd, target_title_substr_for_callback
    found_hwnd = None
    target_title_substr_for_callback = title_substring

    # Create a WNDENUMPROC from the callback function
    callback_ptr = WNDENUMPROC(_enum_windows_callback_find_title)

    user32.EnumWindows(callback_ptr, 0)  # lparam is 0, using global for target title

    target_title_substr_for_callback = None  # Clear global
    if found_hwnd:
        return found_hwnd
    else:
        logger.warning(f"No window found with title containing '{title_substring}'.")
        return None


def get_app_title_by_hwnd(hwnd: ctypes.wintypes.HWND) -> str:
    """
    Retrieves the title of the application window given its HWND.
    Returns an empty string if the title cannot be retrieved.
    """
    length = user32.GetWindowTextLengthW(hwnd) + 1
    if length <= 1:  # No title
        return ""

    buffer = (ctypes.c_wchar * length)()
    user32.GetWindowTextW(hwnd, buffer, length)
    return buffer.value


def find_main_window_by_pid(target_pid: int) -> ctypes.wintypes.HWND | None:
    """Finds the main window HWND for a given process ID (PID)."""
    result_hwnd = None

    def callback(hwnd, lparam):
        nonlocal result_hwnd
        if not user32.IsWindowVisible(hwnd) or not user32.IsWindowEnabled(hwnd):
            return True  # Continue
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == target_pid:
            # Main window: no parent, no owner
            if user32.GetParent(hwnd) == 0 and user32.GetWindow(hwnd, 4) == 0:
                # Check window size
                rect = RECT()
                if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                    width = rect.right - rect.left
                    height = rect.bottom - rect.top
                    if width > 0 and height > 0:
                        result_hwnd = hwnd
                        return False  # Stop
        return True  # Continue

    enum_proc = WNDENUMPROC(callback)
    user32.EnumWindows(enum_proc, 0)
    return result_hwnd


def capture_app_window(app_name_or_title_substr: str) -> np.ndarray | None:
    """
    Screenshots the window of the specified application on Windows.
    Tries to find the window by a substring of its title or by exe name.
    """
    if not isinstance(app_name_or_title_substr, str):
        logger.error("app_name_or_title_substr must be a string.")
        return None

    hwnd = None
    app_title = ""

    if app_name_or_title_substr.lower().endswith(".exe"):
        # Check if exe is running
        if not is_app_running(app_name_or_title_substr):
            logger.error(f"Application '{app_name_or_title_substr}' is not running.")
            return None
        # Find PID(s) for the exe
        DWORD = ctypes.wintypes.DWORD
        array_size = 1024
        pids_array = (DWORD * array_size)()
        bytes_returned = DWORD()
        psapi.EnumProcesses(
            ctypes.byref(pids_array),
            ctypes.sizeof(pids_array),
            ctypes.byref(bytes_returned),
        )
        num_pids = bytes_returned.value // ctypes.sizeof(DWORD)
        found = False
        for i in range(num_pids):
            pid = pids_array[i]
            if pid == 0:
                continue
            h_process = kernel32.OpenProcess(
                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
            )
            if not h_process:
                continue
            try:
                exe_name_buffer = (ctypes.c_wchar * MAX_PATH)()
                if psapi.GetModuleFileNameExW(
                    h_process, None, exe_name_buffer, MAX_PATH
                ):
                    process_exe_name = exe_name_buffer.value.split("\\")[-1]
                    if process_exe_name.lower() == app_name_or_title_substr.lower():
                        hwnd = find_main_window_by_pid(pid)
                        if hwnd:
                            app_title = get_app_title_by_hwnd(hwnd)
                            found = True
                            break
            finally:
                kernel32.CloseHandle(h_process)
        if not found or not hwnd:
            logger.error(
                f"Could not find main window for exe '{app_name_or_title_substr}'."
            )
            return None
    else:
        hwnd = find_window_by_title_substring(app_name_or_title_substr)
        if hwnd:
            app_title = get_app_title_by_hwnd(hwnd)

    if not hwnd:
        logger.error(f"Could not find window for '{app_name_or_title_substr}'.")
        return None

    logger.info(
        f"Found window: '{app_title}' (HWND: {hwnd}) for capture. Attempting to capture..."
    )

    # Get window client rectangle
    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        logger.error(
            f"GetClientRect failed for HWND {hwnd}. Error: {kernel32.GetLastError()}"
        )
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top

    if width <= 0 or height <= 0:
        logger.error(f"Window HWND {hwnd} has invalid dimensions: {width}x{height}.")
        return None

    logger.debug(f"Window dimensions: {width}x{height}")

    # Create device contexts
    h_window_dc = None
    h_mem_dc = None
    h_bitmap = None

    try:
        h_window_dc = user32.GetWindowDC(hwnd)
        if not h_window_dc:
            logger.error(f"GetWindowDC failed. Error: {kernel32.GetLastError()}")
            return None
        logger.debug(f"h_window_dc: {h_window_dc}")

        h_mem_dc = gdi32.CreateCompatibleDC(h_window_dc)
        if not h_mem_dc:
            logger.error(f"CreateCompatibleDC failed. Error: {kernel32.GetLastError()}")
            return None
        logger.debug(f"h_mem_dc: {h_mem_dc}")

        h_bitmap = gdi32.CreateCompatibleBitmap(h_window_dc, width, height)
        if not h_bitmap:
            logger.error(
                f"CreateCompatibleBitmap failed. Error: {kernel32.GetLastError()}"
            )
            return None
        logger.debug(f"h_bitmap: {h_bitmap}")

        # Select the bitmap into the memory DC
        old_bitmap = gdi32.SelectObject(h_mem_dc, h_bitmap)
        if not old_bitmap:
            logger.error(f"SelectObject failed. Error: {kernel32.GetLastError()}")
            return None
        logger.debug(f"Old bitmap selected in h_mem_dc: {old_bitmap}")

        # Attempt to force the window to redraw its content before capturing
        logger.debug(f"Calling RedrawWindow for HWND {hwnd} before PrintWindow.")
        redraw_flags = RDW_INVALIDATE | RDW_UPDATENOW | RDW_ERASE
        if not user32.RedrawWindow(hwnd, None, None, redraw_flags):
            logger.warning(
                f"RedrawWindow failed for HWND {hwnd}. Error: {kernel32.GetLastError()}. Proceeding with PrintWindow anyway."
            )
        else:
            logger.debug(
                f"RedrawWindow for HWND {hwnd} seems to have been called successfully."
            )

        # Call PrintWindow to capture the window content
        # Try with PW_CLIENTONLY | PW_RENDERFULLCONTENT first
        print_window_flags_enhanced = PW_CLIENTONLY | PW_RENDERFULLCONTENT
        logger.debug(
            f"Attempting PrintWindow with flags: {hex(print_window_flags_enhanced)} (PW_CLIENTONLY | PW_RENDERFULLCONTENT)"
        )
        print_window_result = user32.PrintWindow(
            hwnd, h_mem_dc, print_window_flags_enhanced
        )
        logger.debug(f"PrintWindow (enhanced flags) result: {print_window_result}")

        if not print_window_result:
            logger.warning(
                f"PrintWindow with enhanced flags (PW_CLIENTONLY | PW_RENDERFULLCONTENT) failed for HWND {hwnd}. "
                f"Error: {kernel32.GetLastError()}. Falling back to PW_CLIENTONLY."
            )
            print_window_flags_basic = PW_CLIENTONLY
            logger.debug(
                f"Attempting PrintWindow with flags: {hex(print_window_flags_basic)} (PW_CLIENTONLY)"
            )
            print_window_result = user32.PrintWindow(
                hwnd, h_mem_dc, print_window_flags_basic
            )
            logger.debug(f"PrintWindow (PW_CLIENTONLY) result: {print_window_result}")

            if not print_window_result:
                logger.error(
                    f"PrintWindow also failed for HWND {hwnd} with PW_CLIENTONLY. Error: {kernel32.GetLastError()}."
                )
                return None

        # Prepare BITMAPINFO structure
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height  # Negative for top-down DIB (easier for numpy)
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32  # Create a 32-bit bitmap (BGRA)
        bmi.bmiHeader.biCompression = BI_RGB
        bmi.bmiHeader.biSizeImage = width * height * 4  # 4 bytes per pixel (BGRA)

        # Create buffer for pixel data
        buffer_size = width * height * 4
        image_data = ctypes.create_string_buffer(buffer_size)

        # Get bitmap bits
        bits_copied = gdi32.GetDIBits(
            h_mem_dc,
            h_bitmap,
            0,  # Start scan line
            height,  # Number of scan lines
            image_data,  # Pointer to buffer
            ctypes.byref(bmi),  # Pointer to BITMAPINFO
            DIB_RGB_COLORS,  # Usage
        )
        logger.debug(f"GetDIBits copied {bits_copied} scan lines.")

        if bits_copied == 0:
            logger.error(f"GetDIBits failed. Error: {kernel32.GetLastError()}")
            return None

        # Convert buffer to NumPy array
        # The data is BGRA if biBitCount is 32.
        # Explicitly convert to bytes to ensure compatibility and address Pylance warning.
        np_array = np.frombuffer(bytes(image_data), dtype=np.uint8)
        np_array = np_array.reshape((height, width, 4))  # BGRA format

        # BGRA to RGBA conversion
        np_array = np_array[:, :, [2, 1, 0, 3]]  # Rearrange channels to RGBA

        # If biHeight was positive, the image would be bottom-up and need np.flipud(np_array, axis=0)
        # Since we used negative biHeight, it's top-down, so no flip needed.

        logger.info(
            f"Successfully captured window '{app_title}' (HWND: {hwnd}). Image shape: {np_array.shape}"
        )
        return np_array

    except Exception as e:
        logger.error(f"Exception during window capture: {e}", exc_info=True)
        return None
    finally:
        # Clean up GDI objects
        if (
            old_bitmap and h_mem_dc
        ):  # old_bitmap is only valid if SelectObject succeeded
            gdi32.SelectObject(h_mem_dc, old_bitmap)  # Deselect h_bitmap
        if h_bitmap:
            gdi32.DeleteObject(h_bitmap)
            logger.debug(f"Deleted h_bitmap: {h_bitmap}")
        if h_mem_dc:
            gdi32.DeleteDC(h_mem_dc)
            logger.debug(f"Deleted h_mem_dc: {h_mem_dc}")
        if h_window_dc:  # GetWindowDC needs ReleaseDC, not DeleteDC
            user32.ReleaseDC(hwnd, h_window_dc)
            logger.debug(f"Released h_window_dc for HWND: {hwnd}")
