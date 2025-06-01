# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
from dataclasses import dataclass

from explab.maplestory.exp import ExpCheckpoint


@dataclass
class ExpAnalyzerConfig:
    interval: int = 5  # seconds between checkpoints

    max_checkpoints: int = 720
    min_checkpoints: int = 2


@dataclass
class ExpAnalyzerResult:
    """
    Represents the result of an experience analysis.
    Contains the predicted time to reach the next level based on the current experience rate.
    """

    current_exp: int
    current_level: int
    exp_per_minute: float
    exp_ratio_per_minute: float

    minutes_to_next_level: float

    ts: datetime.datetime


class ExpAnalyzer:
    def __init__(self):
        self.config: ExpAnalyzerConfig = ExpAnalyzerConfig()

        self.checkpoints: list[ExpCheckpoint] = []

    def reset(self):
        """
        Resets the analyzer by clearing the checkpoints.
        """
        self.checkpoints.clear()

    def add_checkpoint(self, checkpoint: ExpCheckpoint):
        """
        Adds a new checkpoint to the analyzer.

        Args:
            checkpoint (Checkpoint): The checkpoint to add.
        """
        if len(self.checkpoints) >= self.config.max_checkpoints:
            self.checkpoints.pop(0)
        self.checkpoints.append(checkpoint)

    def _compute_exp_per_minute(self) -> float | None:
        """
        Computes the experience per minute based on the given experience and time in minutes.

        Args:
            exp (int): The total experience points.
            minutes (int): The time in minutes.

        Returns:
            float: The experience per minute.
        """
        exp_diff = 0
        time_diff = 0

        for i in range(1, len(self.checkpoints)):
            if self.checkpoints[i].level != self.checkpoints[i - 1].level:
                continue

            exp_diff += self.checkpoints[i].exp - self.checkpoints[i - 1].exp
            time_diff += (
                self.checkpoints[i].ts - self.checkpoints[i - 1].ts
            ).total_seconds() / 60  # convert to minutes

        if time_diff > 0:
            return exp_diff / time_diff

        return None

    def _compute_total_exp(self) -> float | None:
        """
        Computes the total experience points based on the last checkpoint.

        Returns:
            float: The total experience points.
        """
        if len(self.checkpoints) == 0:
            return None

        last_checkpoint = self.checkpoints[-1]

        if last_checkpoint.exp_ratio <= 0:
            return None

        return last_checkpoint.exp / last_checkpoint.exp_ratio

    def _compute_minutes_to_next_level(self, exp_per_minute: float) -> float | None:
        if len(self.checkpoints) == 0:
            return None

        last_checkpoint = self.checkpoints[-1]
        total_exp = self._compute_total_exp()

        if total_exp is None:
            return None

        remaining_exp = total_exp - last_checkpoint.exp

        if remaining_exp <= 0:
            return 0

        if exp_per_minute <= 0:
            return float("inf")

        return remaining_exp / exp_per_minute

    def get_result(self) -> ExpAnalyzerResult | None:
        """
        Analyzes the checkpoints and returns the result.

        Returns:
            ExpAnalyzerResult: The result of the analysis.
        """
        if not self.checkpoints:
            raise ValueError("No checkpoints available for analysis.")

        if len(self.checkpoints) < self.config.min_checkpoints:
            return None

        last_checkpoint = self.checkpoints[-1]
        current_exp = last_checkpoint.exp
        current_level = last_checkpoint.level

        exp_per_minute = self._compute_exp_per_minute()

        if exp_per_minute is None:
            return None

        total_exp = self._compute_total_exp()

        exp_ratio_per_minute = exp_per_minute / total_exp if total_exp else float("nan")
        if exp_per_minute == 0:
            exp_ratio_per_minute = 0

        minutes_to_next_level = self._compute_minutes_to_next_level(
            exp_per_minute=exp_per_minute
        )

        if minutes_to_next_level is None:
            minutes_to_next_level = float("nan")

        return ExpAnalyzerResult(
            current_exp=current_exp,
            current_level=current_level,
            exp_per_minute=exp_per_minute,
            exp_ratio_per_minute=exp_ratio_per_minute,
            minutes_to_next_level=minutes_to_next_level,
            ts=datetime.datetime.now(),
        )
