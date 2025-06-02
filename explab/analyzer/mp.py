# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
from dataclasses import dataclass

from explab.maplestory.mp import MpCheckpoint


@dataclass
class MpAnalyzerConfig:
    interval: int = 1  # seconds between checkpoints
    batch_size: int = 5  # Number of checkpoints to process in a batch
    max_checkpoints: int = 3600  # Keep a similar max_checkpoints
    min_checkpoints: int = 10  # Keep a similar min_checkpoints


@dataclass
class MpAnalyzerResult:
    """
    Represents the result of an MP analysis.
    Contains the current MP and MP lost per minute.
    """

    current_mp: int
    total_mp: int
    mp_lost_per_minute: float
    ts: datetime.datetime


class MpAnalyzer:
    def __init__(self):
        self.config: MpAnalyzerConfig = MpAnalyzerConfig()
        self.checkpoints: list[MpCheckpoint] = []

    def reset(self):
        """
        Resets the analyzer by clearing the checkpoints.
        """
        self.checkpoints.clear()

    def add_checkpoint(self, checkpoint: MpCheckpoint):
        """
        Adds a new checkpoint to the analyzer.

        Args:
            checkpoint (MpCheckpoint): The checkpoint to add.
        """
        if len(self.checkpoints) >= self.config.max_checkpoints:
            self.checkpoints.pop(0)
        self.checkpoints.append(checkpoint)

    def _compute_mp_lost_per_minute(self) -> float | None:
        """
        Computes the MP lost per minute.
        Only considers instances where MP has decreased.
        """
        mp_lost = 0
        time_diff_seconds = 0

        if len(self.checkpoints) < 2:
            return None

        for i in range(1, len(self.checkpoints)):
            prev_checkpoint = self.checkpoints[i - 1]
            curr_checkpoint = self.checkpoints[i]

            # Only consider MP loss
            if curr_checkpoint.current_mp < prev_checkpoint.current_mp:
                mp_lost += prev_checkpoint.current_mp - curr_checkpoint.current_mp

            time_diff_seconds += (
                curr_checkpoint.ts - prev_checkpoint.ts
            ).total_seconds()

        if time_diff_seconds > 0:
            # Convert total time difference to minutes for per-minute calculation
            time_diff_minutes = time_diff_seconds / 60
            if time_diff_minutes > 0:
                return mp_lost / time_diff_minutes
            return 0.0  # No time elapsed in minutes, or no loss
        return None

    def get_result(self) -> MpAnalyzerResult | None:
        """
        Analyzes the checkpoints and returns the result.

        Returns:
            MpAnalyzerResult: The result of the analysis, or None if not enough data.
        """
        if len(self.checkpoints) < self.config.min_checkpoints:
            return None

        last_checkpoint = self.checkpoints[-1]
        current_mp = last_checkpoint.current_mp
        total_mp = last_checkpoint.total_mp

        mp_lost_per_minute = self._compute_mp_lost_per_minute()

        if mp_lost_per_minute is None:
            # If not enough data for rate, we might still return current MP
            # but the task implies we need the rate.
            # For now, let's return None if rate can't be computed.
            # Or, we could decide to return 0.0 if not enough data for rate.
            # Based on exp_analyzer, it returns None if exp_per_minute is None.
            return None

        return MpAnalyzerResult(
            current_mp=current_mp,
            total_mp=total_mp,
            mp_lost_per_minute=mp_lost_per_minute,
            ts=datetime.datetime.now(),
        )
