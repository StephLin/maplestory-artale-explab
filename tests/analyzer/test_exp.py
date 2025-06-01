# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime
import math

import pytest

from explab.analyzer.exp import ExpAnalyzer, ExpAnalyzerConfig, ExpAnalyzerResult
from explab.maplestory.exp import ExpCheckpoint


@pytest.fixture
def now_time():
    return datetime.datetime(2023, 1, 1, 12, 0, 0)


@pytest.fixture
def analyzer():
    return ExpAnalyzer()


def create_checkpoint(
    exp: int,
    exp_ratio: float,
    level: int,
    now_time_fixture: datetime.datetime,
    time_offset_seconds: int = 0,
) -> ExpCheckpoint:
    return ExpCheckpoint(
        exp=exp,
        exp_ratio=exp_ratio,
        level=level,
        ts=now_time_fixture + datetime.timedelta(seconds=time_offset_seconds),
    )


def test_initialization(analyzer: ExpAnalyzer):
    assert len(analyzer.checkpoints) == 0
    assert isinstance(analyzer.config, ExpAnalyzerConfig)


def test_reset(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.1, 10, now_time)
    analyzer.add_checkpoint(cp1)
    assert len(analyzer.checkpoints) == 1
    analyzer.reset()
    assert len(analyzer.checkpoints) == 0


def test_add_checkpoint(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.1, 10, now_time)
    analyzer.add_checkpoint(cp1)
    assert len(analyzer.checkpoints) == 1
    assert analyzer.checkpoints[0] == cp1

    cp2 = create_checkpoint(200, 0.2, 10, now_time, time_offset_seconds=60)
    analyzer.add_checkpoint(cp2)
    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[1] == cp2


def test_add_checkpoint_max_limit(analyzer: ExpAnalyzer, now_time):
    analyzer.config.max_checkpoints = 2
    cp1 = create_checkpoint(100, 0.1, 10, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(200, 0.2, 10, now_time, time_offset_seconds=60)
    cp3 = create_checkpoint(300, 0.3, 10, now_time, time_offset_seconds=120)

    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[0] == cp1
    assert analyzer.checkpoints[1] == cp2

    analyzer.add_checkpoint(cp3)
    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[0] == cp2  # cp1 should be popped
    assert analyzer.checkpoints[1] == cp3


def test_get_result_no_checkpoints(analyzer: ExpAnalyzer):
    with pytest.raises(ValueError, match="No checkpoints available for analysis."):
        analyzer.get_result()


def test_get_result_one_checkpoint(analyzer: ExpAnalyzer, now_time):
    # _compute_exp_per_minute will return None with only one checkpoint
    cp1 = create_checkpoint(100, 0.1, 10, now_time)
    analyzer.add_checkpoint(cp1)
    result = analyzer.get_result()
    assert result is None


def test_compute_exp_per_minute_no_checkpoints(analyzer: ExpAnalyzer):
    assert analyzer._compute_exp_per_minute() is None


def test_compute_exp_per_minute_one_checkpoint(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.1, 10, now_time)
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_exp_per_minute() is None


def test_compute_exp_per_minute_valid(analyzer: ExpAnalyzer, now_time):
    # 100 exp in 1 minute (60 seconds)
    cp1 = create_checkpoint(100, 0.1, 10, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(200, 0.2, 10, now_time, time_offset_seconds=60)
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    assert analyzer._compute_exp_per_minute() == pytest.approx(100.0)

    # 150 exp in 1.5 minutes (90 seconds) -> 100 exp/min
    cp3 = create_checkpoint(
        350, 0.35, 10, now_time, time_offset_seconds=60 + 90
    )  # 90s after cp2
    analyzer.add_checkpoint(cp3)
    assert analyzer._compute_exp_per_minute() == pytest.approx(100.0)


def test_compute_exp_per_minute_level_up(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(900, 0.9, 10, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(50, 0.05, 11, now_time, time_offset_seconds=60)
    cp3 = create_checkpoint(150, 0.15, 11, now_time, time_offset_seconds=120)
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    analyzer.add_checkpoint(cp3)
    assert analyzer._compute_exp_per_minute() == pytest.approx(100.0)


def test_compute_exp_per_minute_zero_time_diff(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.1, 10, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(
        200, 0.2, 10, now_time, time_offset_seconds=0
    )  # Same timestamp
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    assert analyzer._compute_exp_per_minute() is None


def test_compute_total_exp_no_checkpoints(analyzer: ExpAnalyzer):
    assert analyzer._compute_total_exp() is None


def test_compute_total_exp_zero_exp_ratio(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.0, 10, now_time)  # exp_ratio is 0
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_total_exp() is None


def test_compute_total_exp_valid(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.1, 10, now_time)  # Total exp = 100 / 0.1 = 1000
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_total_exp() == pytest.approx(1000.0)

    cp2 = create_checkpoint(
        500, 0.5, 10, now_time, time_offset_seconds=60
    )  # Total exp = 500 / 0.5 = 1000
    analyzer.add_checkpoint(cp2)  # Uses last checkpoint
    assert analyzer._compute_total_exp() == pytest.approx(1000.0)


def test_compute_minutes_to_next_level_no_checkpoints(analyzer: ExpAnalyzer):
    assert analyzer._compute_minutes_to_next_level(exp_per_minute=100.0) is None


def test_compute_minutes_to_next_level_total_exp_none(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.0, 10, now_time)  # exp_ratio is 0
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_minutes_to_next_level(exp_per_minute=100.0) is None


def test_compute_minutes_to_next_level_remaining_exp_zero(
    analyzer: ExpAnalyzer, now_time
):
    cp1 = create_checkpoint(1000, 1.0, 10, now_time)
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_minutes_to_next_level(
        exp_per_minute=100.0
    ) == pytest.approx(0.0)


def test_compute_minutes_to_next_level_exp_per_minute_zero(
    analyzer: ExpAnalyzer, now_time
):
    cp1 = create_checkpoint(500, 0.5, 10, now_time)
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_minutes_to_next_level(exp_per_minute=0.0) == float("inf")


def test_compute_minutes_to_next_level_exp_per_minute_negative(
    analyzer: ExpAnalyzer, now_time
):
    cp1 = create_checkpoint(500, 0.5, 10, now_time)
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_minutes_to_next_level(exp_per_minute=-10.0) == float("inf")


def test_compute_minutes_to_next_level_valid(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(500, 0.5, 10, now_time)
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_minutes_to_next_level(
        exp_per_minute=100.0
    ) == pytest.approx(5.0)


def test_get_result_valid_simple(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.1, 10, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(200, 0.2, 10, now_time, time_offset_seconds=60)
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)

    result = analyzer.get_result()
    assert result is not None
    assert isinstance(result, ExpAnalyzerResult)
    assert result.current_exp == 200
    assert result.current_level == 10
    assert result.exp_per_minute == pytest.approx(100.0)
    assert result.exp_ratio_per_minute == pytest.approx(0.1)
    assert result.minutes_to_next_level == pytest.approx(8.0)
    assert isinstance(result.ts, datetime.datetime)


def test_get_result_exp_per_minute_is_zero(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.1, 10, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(100, 0.1, 10, now_time, time_offset_seconds=60)
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)

    result = analyzer.get_result()
    assert result is not None
    assert result.current_exp == 100
    assert result.current_level == 10
    assert result.exp_per_minute == pytest.approx(0.0)
    assert result.exp_ratio_per_minute == pytest.approx(0.0)
    assert result.minutes_to_next_level == float("inf")


def test_get_result_total_exp_is_none(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 0.1, 10, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(
        200, 0.0, 10, now_time, time_offset_seconds=60
    )  # exp_ratio = 0
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)

    result = analyzer.get_result()
    assert result is not None
    assert result.current_exp == 200
    assert result.current_level == 10
    assert result.exp_per_minute == pytest.approx(100.0)
    assert math.isnan(result.exp_ratio_per_minute)
    assert math.isnan(result.minutes_to_next_level)


def test_get_result_already_leveled_up_in_data(analyzer: ExpAnalyzer, now_time):
    cp1 = create_checkpoint(900, 0.9, 10, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(1000, 1.0, 10, now_time, time_offset_seconds=60)
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)

    result = analyzer.get_result()
    assert result is not None
    assert result.current_exp == 1000
    assert result.current_level == 10
    assert result.exp_per_minute == pytest.approx(100.0)
    assert result.exp_ratio_per_minute == pytest.approx(0.1)
    assert result.minutes_to_next_level == pytest.approx(0.0)
