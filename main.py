# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import logging
import os
import shutil
import subprocess as sp
import sys
from pathlib import Path

import click
import dotenv
import structlog
from click_default_group import DefaultGroup

PROJECT_ROOT = Path(__file__).parent.absolute()

logger = structlog.get_logger()

# Load environment variables from .env file
# If .env file does not exist, copy from .env.example
if not (PROJECT_ROOT / ".env").exists():
    logger.info("Creating .env file from .env.example ...")
    shutil.copyfile(PROJECT_ROOT / ".env.example", PROJECT_ROOT / ".env")
dotenv.load_dotenv()

# Configure structlog
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
if log_level not in logging._nameToLevel:
    logger.warning(f"Invalid LOG_LEVEL '{log_level}' in .env, defaulting to INFO.")
    log_level = "INFO"

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(log_level),
)


@click.group(cls=DefaultGroup, default="ui", default_if_no_args=True)
def main():
    """
    MapleStory Worlds Screen Capture Utility
    """
    pass


@main.command()
@click.option(
    "--use-real-capture",
    is_flag=True,
    default=False,
    help="Use real screen capture instead of a static image.",
)
@click.option(
    "--no-save-screenshot",
    is_flag=True,
    default=False,
    help="Do not save the screenshot to disk.",
)
@click.option(
    "--disable-ocr",
    is_flag=True,
    default=False,
    help="Disable OCR processing on the captured screenshot.",
)
def single_shot(use_real_capture: bool, no_save_screenshot: bool, disable_ocr: bool):
    from skimage.io import imread, imsave

    from explab.maplestory.exp import ExpCheckpoint
    from explab.maplestory.hp import HpCheckpoint
    from explab.maplestory.mp import MpCheckpoint
    from explab.ocr import ocr
    from explab.preprocessing import cropper
    from explab.screen_capture import capture_app_window

    save_screenshot = not no_save_screenshot
    enable_ocr = not disable_ocr

    if enable_ocr:
        ocr.initialize()

    app_to_screenshot = os.getenv("MAPLESTORY_APP_NAME", "MapleStory Worlds")

    try:
        logger.info(
            f"Attempting to screenshot '{app_to_screenshot}' and get NumPy array..."
        )
        if use_real_capture:
            capture = capture_app_window(app_to_screenshot)
        else:
            capture = imread(PROJECT_ROOT / "./tests/assets/screenshot1.png")

        if capture is not None:
            logger.info(
                f"Successfully obtained screenshot NumPy array, shape: {capture.shape}"
            )

            if save_screenshot:
                tmp_dir = PROJECT_ROOT / "temp"
                tmp_dir.mkdir(exist_ok=True)

                imsave(tmp_dir / "screenshot.png", capture)
                imsave(tmp_dir / "exp_crop.png", cropper.get_exp_crop(capture))
                imsave(
                    tmp_dir / "exp_crop.ocr.png",
                    cropper.get_exp_crop(capture, ocr_friendly=True),
                )

                imsave(tmp_dir / "hp_crop.png", cropper.get_hp_crop(capture))
                imsave(
                    tmp_dir / "hp_crop.ocr.png",
                    cropper.get_hp_crop(capture, ocr_friendly=True),
                )

                imsave(tmp_dir / "mp_crop.png", cropper.get_mp_crop(capture))
                imsave(
                    tmp_dir / "mp_crop.ocr.png",
                    cropper.get_mp_crop(capture, ocr_friendly=True),
                )

            if enable_ocr:
                exp_ckpt = ExpCheckpoint.from_app_capture(capture)
                logger.info(f"Exp Checkpoint created: {exp_ckpt}")

                hp_checkpoint = HpCheckpoint.from_app_capture(capture)
                logger.info(f"HP Checkpoint created: {hp_checkpoint}")

                mp_checkpoint = MpCheckpoint.from_app_capture(capture)
                logger.info(f"MP Checkpoint created: {mp_checkpoint}")

        else:
            logger.error(f"Failed to get screenshot of '{app_to_screenshot}'.")
    except NotImplementedError as e:
        logger.error(f"NotImplementedError: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


@main.command()
def ui():
    sp.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "explab" / "ui" / "main.py"),
        ]
    )


if __name__ == "__main__":
    main()
