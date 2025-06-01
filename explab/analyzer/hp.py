# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
from dataclasses import dataclass

from explab.maplestory.hp import HpCheckpoint


@dataclass
class HpAnalyzerConfig:
    interval: int = 1  # seconds between checkpoints
    max_checkpoints: int = 3600  # Keep a similar max_checkpoints
    min_checkpoints: int = 10  # Keep a similar min_checkpoints


@dataclass
class HpAnalyzerResult:
    """
    Represents the result of an HP analysis.
    Contains the current HP and HP lost per minute.
    """

    current_hp: int
    total_hp: int
    hp_lost_per_minute: float
    ts: datetime.datetime


class HpAnalyzer:
    def __init__(self):
        self.config: HpAnalyzerConfig = HpAnalyzerConfig()
        self.checkpoints: list[HpCheckpoint] = []

    def reset(self):
        """
        Resets the analyzer by clearing the checkpoints.
        """
        self.checkpoints.clear()

    def add_checkpoint(self, checkpoint: HpCheckpoint):
        """
        Adds a new checkpoint to the analyzer.

        Args:
            checkpoint (HpCheckpoint): The checkpoint to add.
        """
        if len(self.checkpoints) >= self.config.max_checkpoints:
            self.checkpoints.pop(0)
        self.checkpoints.append(checkpoint)

    def _compute_hp_lost_per_minute(self) -> float | None:
        """
        Computes the HP lost per minute.
        Only considers instances where HP has decreased.
        """
        hp_lost = 0
        time_diff_seconds = 0

        if len(self.checkpoints) < 2:
            return None

        for i in range(1, len(self.checkpoints)):
            prev_checkpoint = self.checkpoints[i - 1]
            curr_checkpoint = self.checkpoints[i]

            # Only consider HP loss
            if curr_checkpoint.current_hp < prev_checkpoint.current_hp:
                hp_lost += prev_checkpoint.current_hp - curr_checkpoint.current_hp

            time_diff_seconds += (
                curr_checkpoint.ts - prev_checkpoint.ts
            ).total_seconds()

        if time_diff_seconds > 0:
            # Convert total time difference to minutes for per-minute calculation
            time_diff_minutes = time_diff_seconds / 60
            if time_diff_minutes > 0:
                return hp_lost / time_diff_minutes
            return 0.0  # No time elapsed in minutes, or no loss
        return None

    def get_result(self) -> HpAnalyzerResult | None:
        """
        Analyzes the checkpoints and returns the result.

        Returns:
            HpAnalyzerResult: The result of the analysis, or None if not enough data.
        """
        if len(self.checkpoints) < self.config.min_checkpoints:
            return None

        last_checkpoint = self.checkpoints[-1]
        current_hp = last_checkpoint.current_hp
        total_hp = last_checkpoint.total_hp

        hp_lost_per_minute = self._compute_hp_lost_per_minute()

        if hp_lost_per_minute is None:
            # If not enough data for rate, we might still return current HP
            # but the task implies we need the rate.
            # For now, let's return None if rate can't be computed.
            # Or, we could decide to return 0.0 if not enough data for rate.
            # Based on exp_analyzer, it returns None if exp_per_minute is None.
            return None

        return HpAnalyzerResult(
            current_hp=current_hp,
            total_hp=total_hp,
            hp_lost_per_minute=hp_lost_per_minute,
            ts=datetime.datetime.now(),
        )
