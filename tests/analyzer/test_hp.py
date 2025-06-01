# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import datetime

import pytest

from explab.analyzer.hp import HpAnalyzer, HpAnalyzerConfig, HpAnalyzerResult
from explab.maplestory.hp import HpCheckpoint


@pytest.fixture
def now_time():
    return datetime.datetime(2023, 1, 1, 12, 0, 0)


@pytest.fixture
def analyzer():
    return HpAnalyzer()


def create_checkpoint(
    current_hp: int,
    total_hp: int,
    now_time_fixture: datetime.datetime,
    time_offset_seconds: int = 0,
) -> HpCheckpoint:
    return HpCheckpoint(
        current_hp=current_hp,
        total_hp=total_hp,
        ts=now_time_fixture + datetime.timedelta(seconds=time_offset_seconds),
    )


def test_initialization(analyzer: HpAnalyzer):
    assert len(analyzer.checkpoints) == 0
    assert isinstance(analyzer.config, HpAnalyzerConfig)


def test_reset(analyzer: HpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 1000, now_time)
    analyzer.add_checkpoint(cp1)
    assert len(analyzer.checkpoints) == 1
    analyzer.reset()
    assert len(analyzer.checkpoints) == 0


def test_add_checkpoint(analyzer: HpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 1000, now_time)
    analyzer.add_checkpoint(cp1)
    assert len(analyzer.checkpoints) == 1
    assert analyzer.checkpoints[0] == cp1

    cp2 = create_checkpoint(200, 1000, now_time, time_offset_seconds=60)
    analyzer.add_checkpoint(cp2)
    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[1] == cp2


def test_add_checkpoint_max_limit(analyzer: HpAnalyzer, now_time):
    analyzer.config.max_checkpoints = 2
    cp1 = create_checkpoint(100, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(200, 1000, now_time, time_offset_seconds=60)
    cp3 = create_checkpoint(300, 1000, now_time, time_offset_seconds=120)

    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[0] == cp1
    assert analyzer.checkpoints[1] == cp2

    analyzer.add_checkpoint(cp3)
    assert len(analyzer.checkpoints) == 2
    assert analyzer.checkpoints[0] == cp2  # cp1 should be popped
    assert analyzer.checkpoints[1] == cp3


def test_get_result_not_enough_checkpoints(analyzer: HpAnalyzer, now_time):
    analyzer.config.min_checkpoints = 2
    cp1 = create_checkpoint(100, 1000, now_time)
    analyzer.add_checkpoint(cp1)
    result = analyzer.get_result()
    assert result is None


def test_get_result_one_checkpoint_but_min_is_one(analyzer: HpAnalyzer, now_time):
    analyzer.config.min_checkpoints = 1
    cp1 = create_checkpoint(100, 1000, now_time)
    analyzer.add_checkpoint(cp1)
    # _compute_hp_lost_per_minute will return None with only one checkpoint
    # and thus get_result will also return None
    result = analyzer.get_result()
    assert result is None


def test_compute_hp_lost_per_minute_no_checkpoints(analyzer: HpAnalyzer):
    assert analyzer._compute_hp_lost_per_minute() is None


def test_compute_hp_lost_per_minute_one_checkpoint(analyzer: HpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 1000, now_time)
    analyzer.add_checkpoint(cp1)
    assert analyzer._compute_hp_lost_per_minute() is None


def test_compute_hp_lost_per_minute_valid_loss(analyzer: HpAnalyzer, now_time):
    # Lost 100 HP in 1 minute (60 seconds)
    cp1 = create_checkpoint(200, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(100, 1000, now_time, time_offset_seconds=60)
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    assert analyzer._compute_hp_lost_per_minute() == pytest.approx(100.0)

    # Lost 50 HP, then another 100 HP. Total 150 HP lost over 2 minutes. Avg 75 HP/min
    # cp2 is (100, 1000, t+60)
    # cp3 is (50, 1000, t+120) -> 50 lost in 60s from cp2
    # cp4 is (0, 1000, t+180) -> 50 lost in 60s from cp3
    # Total lost from cp1: (200-100) + (100-50) + (50-0) = 100 + 50 + 50 = 200
    # Total time: 180s = 3 mins
    # Expected: 200 / 3 = 66.66
    analyzer.reset()
    cp1 = create_checkpoint(200, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(100, 1000, now_time, time_offset_seconds=60)  # 100 lost
    cp3 = create_checkpoint(50, 1000, now_time, time_offset_seconds=120)  # 50 lost
    cp4 = create_checkpoint(0, 1000, now_time, time_offset_seconds=180)  # 50 lost
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    analyzer.add_checkpoint(cp3)
    analyzer.add_checkpoint(cp4)
    # Total HP lost = (200-100) + (100-50) + (50-0) = 100 + 50 + 50 = 200
    # Total time diff = 180 seconds = 3 minutes
    # HP lost per minute = 200 / 3
    assert analyzer._compute_hp_lost_per_minute() == pytest.approx(200.0 / 3.0)


def test_compute_hp_lost_per_minute_no_loss(analyzer: HpAnalyzer, now_time):
    cp1 = create_checkpoint(100, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(200, 1000, now_time, time_offset_seconds=60)  # HP Gained
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    assert analyzer._compute_hp_lost_per_minute() == pytest.approx(
        0.0
    )  # No loss counted


def test_compute_hp_lost_per_minute_mixed_loss_gain(analyzer: HpAnalyzer, now_time):
    # Lost 100, Gained 50, Lost 20. Total lost = 100 + 20 = 120. Total time = 3 mins. Rate = 40
    cp1 = create_checkpoint(200, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(100, 1000, now_time, time_offset_seconds=60)  # Lost 100
    cp3 = create_checkpoint(150, 1000, now_time, time_offset_seconds=120)  # Gained 50
    cp4 = create_checkpoint(130, 1000, now_time, time_offset_seconds=180)  # Lost 20
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    analyzer.add_checkpoint(cp3)
    analyzer.add_checkpoint(cp4)
    # HP Lost: (200-100) + (150-130) = 100 + 20 = 120
    # Time diff: 180s = 3 minutes
    # Rate: 120 / 3 = 40
    assert analyzer._compute_hp_lost_per_minute() == pytest.approx(40.0)


def test_compute_hp_lost_per_minute_zero_time_diff(analyzer: HpAnalyzer, now_time):
    cp1 = create_checkpoint(200, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(
        100, 1000, now_time, time_offset_seconds=0
    )  # Same timestamp
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    assert analyzer._compute_hp_lost_per_minute() is None


def test_get_result_valid_simple(analyzer: HpAnalyzer, now_time):
    analyzer.config.min_checkpoints = (
        2  # Ensure enough checkpoints for rate calculation
    )
    cp1 = create_checkpoint(200, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(
        100, 1000, now_time, time_offset_seconds=60
    )  # Lost 100 HP in 60s
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)

    result = analyzer.get_result()
    assert result is not None
    assert isinstance(result, HpAnalyzerResult)
    assert result.current_hp == 100
    assert result.total_hp == 1000
    assert result.hp_lost_per_minute == pytest.approx(100.0)
    assert isinstance(result.ts, datetime.datetime)


def test_get_result_hp_lost_per_minute_is_zero(analyzer: HpAnalyzer, now_time):
    analyzer.config.min_checkpoints = 2
    cp1 = create_checkpoint(100, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(100, 1000, now_time, time_offset_seconds=60)  # No change
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)

    result = analyzer.get_result()
    assert result is not None
    assert result.current_hp == 100
    assert result.total_hp == 1000
    assert result.hp_lost_per_minute == pytest.approx(0.0)


def test_get_result_hp_lost_per_minute_is_none_due_to_time(
    analyzer: HpAnalyzer, now_time
):
    analyzer.config.min_checkpoints = 2
    cp1 = create_checkpoint(200, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(100, 1000, now_time, time_offset_seconds=0)  # Same time
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)

    result = analyzer.get_result()
    # HpAnalyzer.get_result returns None if _compute_hp_lost_per_minute is None
    assert result is None


def test_get_result_with_more_checkpoints(analyzer: HpAnalyzer, now_time):
    analyzer.config.min_checkpoints = 3
    cp1 = create_checkpoint(300, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(200, 1000, now_time, time_offset_seconds=60)  # Lost 100
    cp3 = create_checkpoint(150, 1000, now_time, time_offset_seconds=120)  # Lost 50
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    analyzer.add_checkpoint(cp3)

    result = analyzer.get_result()
    assert result is not None
    assert result.current_hp == 150
    assert result.total_hp == 1000
    # Total lost: (300-200) + (200-150) = 100 + 50 = 150
    # Total time: 120s = 2 minutes
    # Rate: 150 / 2 = 75
    assert result.hp_lost_per_minute == pytest.approx(75.0)


def test_get_result_with_gain_and_loss(analyzer: HpAnalyzer, now_time):
    analyzer.config.min_checkpoints = 3
    cp1 = create_checkpoint(300, 1000, now_time, time_offset_seconds=0)
    cp2 = create_checkpoint(
        350, 1000, now_time, time_offset_seconds=60
    )  # Gained 50 (not counted in loss)
    cp3 = create_checkpoint(
        250, 1000, now_time, time_offset_seconds=120
    )  # Lost 100 from cp2
    analyzer.add_checkpoint(cp1)
    analyzer.add_checkpoint(cp2)
    analyzer.add_checkpoint(cp3)

    result = analyzer.get_result()
    assert result is not None
    assert result.current_hp == 250
    assert result.total_hp == 1000
    # HP Lost: (350-250) = 100 (loss from cp2 to cp3)
    # Time diff: 120s = 2 minutes (total duration of checkpoints)
    # Rate: 100 / 2 = 50
    assert result.hp_lost_per_minute == pytest.approx(50.0)
