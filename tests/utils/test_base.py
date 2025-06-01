# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from pathlib import Path

from explab.utils.base import PROJECT_ROOT


def test_project_root_is_correct_path():
    """
    Tests if PROJECT_ROOT is a Path object and points to the correct directory.
    """
    # Check if PROJECT_ROOT is a Path object
    assert isinstance(PROJECT_ROOT, Path), "PROJECT_ROOT should be a Path object"

    # Check if PROJECT_ROOT is an existing directory
    assert PROJECT_ROOT.exists(), (
        f"PROJECT_ROOT directory does not exist: {PROJECT_ROOT}"
    )
    assert PROJECT_ROOT.is_dir(), f"PROJECT_ROOT is not a directory: {PROJECT_ROOT}"

    # Check if a known file or directory exists at the project root
    # This helps confirm it's the correct project root
    expected_file_at_root = "pyproject.toml"
    assert (PROJECT_ROOT / expected_file_at_root).exists(), (
        f"Expected file '{expected_file_at_root}' not found in PROJECT_ROOT: {PROJECT_ROOT}"
    )

    # Check the name of the project root directory
    # This assumes the project directory is named 'maplestory-artale-explab'
    # Adjust if your project directory has a different name.
    assert PROJECT_ROOT.name == "maplestory-artale-explab", (
        f"PROJECT_ROOT name is not 'maplestory-artale-explab': {PROJECT_ROOT.name}"
    )
