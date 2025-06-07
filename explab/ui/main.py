# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os
import platform
import sys
from pathlib import Path

# For explab
sys.path.append(Path(__file__).resolve().parents[2].as_posix())

import datetime

import dotenv
import numpy as np
from nicegui import app, ui

from explab.analyzer.exp import ExpAnalyzer, ExpAnalyzerResult
from explab.analyzer.hp import HpAnalyzer, HpAnalyzerResult
from explab.analyzer.mp import MpAnalyzer, MpAnalyzerResult
from explab.maplestory.exp import ExpCheckpoint
from explab.maplestory.hp import HpCheckpoint
from explab.maplestory.mp import MpCheckpoint
from explab.ocr import ocr
from explab.screen_capture import capture_app_window
from explab.utils.base import PROJECT_ROOT

dotenv.load_dotenv(PROJECT_ROOT / ".env")

on_top_env = os.getenv("ON_TOP", "false").lower()
app.native.window_args["on_top"] = on_top_env in ("true", "1", "t")


class UI:
    def __init__(self):
        ocr.initialize()  # Initialize OCR engine

        self.exp_analyzer = ExpAnalyzer()
        self.hp_analyzer = HpAnalyzer()
        self.mp_analyzer = MpAnalyzer()
        self.hp_capture_buffer: list[np.ndarray] = []
        self.hp_ts_buffer: list[datetime.datetime] = []
        self.mp_capture_buffer: list[np.ndarray] = []
        self.mp_ts_buffer: list[datetime.datetime] = []
        self.is_exp_running = False
        self.is_hp_running = False
        self.is_mp_running = False
        self.exp_timer = None
        self.hp_timer = None
        self.mp_timer = None

        # Define ECharts options for the experience plot
        self.exp_chart_options = {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["Current EXP", "Predicted Growth"], "top": "bottom"},
            "xAxis": {
                "type": "category",
                "boundaryGap": False,
                "data": [],  # Timestamps
            },
            "yAxis": {"type": "value"},
            "series": [
                {
                    "name": "Current EXP",
                    "type": "line",
                    "data": [],  # Actual EXP values
                    "smooth": True,
                    "showSymbol": False,
                },
                {
                    "name": "Predicted Growth",
                    "type": "line",
                    "data": [],  # Predicted EXP values
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"type": "dashed"},
                },
            ],
            "grid": {
                "left": "3%",
                "right": "4%",
                "bottom": "15%",
                "containLabel": True,
            },  # Adjust grid to make space for legend
        }

        if platform.system() == "Windows":
            self.app_name = os.getenv("WINDOWS_MAPLESTORY_EXE_NAME", "msw.exe")
        else:
            self.app_name = os.getenv("MAPLESTORY_APP_NAME", "MapleStory Worlds")

        with ui.row().classes("w-full h-full m-auto justify-center"):
            with ui.card().classes("w-[48%]"):
                ui.label("Experience Analyzer").classes("text-h6")
                with ui.row(align_items="center"):
                    self.exp_start_button = ui.button(
                        "Start", on_click=self.toggle_exp_analyzer
                    )
                    self.exp_status_label = ui.label("Status: Idle")
                self.exp_result_label = ui.markdown("Result: ")
                # Add the echart to the UI
                self.exp_chart = ui.echart(options=self.exp_chart_options).classes(
                    "w-full h-64 mt-4"
                )

            with ui.column().classes("w-[48%]"):
                with ui.card().classes("w-full"):
                    ui.label("HP Analyzer").classes("text-h6")
                    with ui.row(align_items="center"):
                        self.hp_start_button = ui.button(
                            "Start", on_click=self.toggle_hp_analyzer
                        )
                        self.hp_status_label = ui.label("Status: Idle")
                    self.hp_result_label = ui.markdown("Result: ")

                with ui.card().classes("w-full"):
                    ui.label("MP Analyzer").classes("text-h6")
                    with ui.row(align_items="center"):
                        self.mp_start_button = ui.button(
                            "Start", on_click=self.toggle_mp_analyzer
                        )
                        self.mp_status_label = ui.label("Status: Idle")
                    self.mp_result_label = ui.markdown("Result: ")

    def toggle_exp_analyzer(self):
        if self.is_exp_running:
            self.stop_exp_analyzer()
        else:
            self.start_exp_analyzer()

    def start_exp_analyzer(self):
        self.is_exp_running = True
        self.exp_status_label.set_text("Status: Running")
        self.exp_start_button.set_text("Stop")
        self.exp_analyzer.reset()
        # Reset chart data
        if hasattr(self, "exp_chart"):
            self.exp_chart.options["xAxis"]["data"] = []
            self.exp_chart.options["series"][0]["data"] = []
            self.exp_chart.options["series"][1]["data"] = []
            self.exp_chart.update()

        self.exp_timer = ui.timer(
            interval=self.exp_analyzer.config.interval,
            callback=self.update_exp_analysis,
            active=True,
        )
        ui.notify("Experience analyzer started.")

    def stop_exp_analyzer(self):
        self.is_exp_running = False
        if self.exp_timer:
            self.exp_timer.deactivate()
            self.exp_timer = None
        self.exp_status_label.set_text("Status: Idle")
        self.exp_start_button.set_text("Start")
        ui.notify("Experience analyzer stopped.")

    async def update_exp_analysis(self):
        if not self.is_exp_running:
            return

        try:
            capture = capture_app_window(self.app_name)
            if capture is None:
                self.exp_result_label.set_content(
                    "Error: Failed to capture APP window."
                )
                return

            checkpoint = ExpCheckpoint.from_app_capture(capture=capture)

            if checkpoint:
                self.exp_analyzer.add_checkpoint(checkpoint)
                try:
                    result: ExpAnalyzerResult | None = self.exp_analyzer.get_result()
                    if result:
                        markdown_lines = [
                            "Results:",
                            "",
                            f"- Level: **{result.current_level}**, Exp: **{result.current_exp}**",
                            f"- Gain: **{result.exp_per_minute:.2f}** exp/min (**{result.exp_ratio_per_minute * 100:.2f}** %/min)",
                            f"- Level Up: **{result.minutes_to_next_level:.2f}** min",
                        ]
                        markdown_content = "\n".join(markdown_lines)
                        self.exp_result_label.set_content(markdown_content)

                        # Update chart
                        all_checkpoints = self.exp_analyzer.checkpoints

                        if all_checkpoints:
                            timestamps_display = [
                                cp.ts.strftime("%H:%M:%S") for cp in all_checkpoints
                            ]
                            current_exp_values_display = [
                                cp.exp for cp in all_checkpoints
                            ]

                            predicted_exp_values_display = []
                            if all_checkpoints:  # Ensure there's at least one checkpoint for prediction base
                                first_checkpoint_session = all_checkpoints[0]
                                base_exp = first_checkpoint_session.exp
                                base_ts_seconds = (
                                    first_checkpoint_session.ts.timestamp()
                                )

                                # Use the overall exp_per_minute from the result for prediction
                                exp_per_second = result.exp_per_minute / 60.0

                                for cp_display in all_checkpoints:
                                    time_diff_seconds = (
                                        cp_display.ts.timestamp() - base_ts_seconds
                                    )
                                    predicted_exp = base_exp + (
                                        exp_per_second * time_diff_seconds
                                    )
                                    predicted_exp_values_display.append(
                                        round(predicted_exp)
                                    )

                            # Dynamically adjust Y-axis
                            all_y_values = (
                                current_exp_values_display
                                + predicted_exp_values_display
                            )
                            if all_y_values:
                                y_min = min(all_y_values)
                                y_max = max(all_y_values)
                                padding = (y_max - y_min) * 0.1  # 10% padding
                                if (
                                    padding == 0
                                ):  # Handle case where all values are the same
                                    padding = max(
                                        10, y_min * 0.1
                                    )  # Or 10% of the value, or 10 if value is small

                                self.exp_chart.options["yAxis"]["min"] = round(
                                    y_min - padding
                                )
                                self.exp_chart.options["yAxis"]["max"] = round(
                                    y_max + padding
                                )
                            else:
                                # Reset to default or auto if no data
                                if "min" in self.exp_chart.options["yAxis"]:
                                    del self.exp_chart.options["yAxis"]["min"]
                                if "max" in self.exp_chart.options["yAxis"]:
                                    del self.exp_chart.options["yAxis"]["max"]

                            self.exp_chart.options["xAxis"]["data"] = timestamps_display
                            self.exp_chart.options["series"][0]["data"] = (
                                current_exp_values_display
                            )
                            self.exp_chart.options["series"][1]["data"] = (
                                predicted_exp_values_display
                            )
                            self.exp_chart.update()

                    else:
                        self.exp_result_label.set_content("Result: Analyzing...")
                except ValueError as ve:
                    self.exp_result_label.set_content(f"Result: {ve}")
            else:
                self.exp_result_label.set_content("Error: Failed to capture EXP.")
        except Exception as e:
            self.exp_result_label.set_content(f"Error: {e}")
            ui.notify(f"Error during EXP analysis: {e}", type="negative")
            print(f"Error during EXP analysis: {e}")

    def toggle_hp_analyzer(self):
        if self.is_hp_running:
            self.stop_hp_analyzer()
        else:
            self.start_hp_analyzer()

    def start_hp_analyzer(self):
        self.is_hp_running = True
        self.hp_status_label.set_text("Status: Running")
        self.hp_start_button.set_text("Stop")
        self.hp_analyzer.reset()
        self.hp_capture_buffer.clear()
        self.hp_ts_buffer.clear()
        self.hp_timer = ui.timer(
            interval=self.hp_analyzer.config.interval,
            callback=self.update_hp_analysis,
            active=True,
        )
        ui.notify("HP analyzer started.")

    def stop_hp_analyzer(self):
        self.is_hp_running = False
        if self.hp_timer:
            self.hp_timer.deactivate()
            self.hp_timer = None
        self.hp_capture_buffer.clear()
        self.hp_ts_buffer.clear()
        self.hp_status_label.set_text("Status: Idle")
        self.hp_start_button.set_text("Start")
        ui.notify("HP analyzer stopped.")

    async def update_hp_analysis(self):
        if not self.is_hp_running:
            return

        try:
            capture = capture_app_window(self.app_name)
            if capture is None:
                self.hp_result_label.set_content("Error: Failed to capture APP window.")
                return

            self.hp_capture_buffer.append(capture)
            self.hp_ts_buffer.append(datetime.datetime.now())

            if len(self.hp_capture_buffer) >= self.hp_analyzer.config.batch_size:
                # Use the batch_size from analyzer config for ocr_batch_size as well
                checkpoints = HpCheckpoint.from_app_captures(
                    captures=self.hp_capture_buffer,
                    ts_list=self.hp_ts_buffer,
                    ocr_batch_size=self.hp_analyzer.config.batch_size,
                )
                self.hp_capture_buffer.clear()
                self.hp_ts_buffer.clear()

                processed_count = 0
                for ckpt in checkpoints:
                    if ckpt:
                        self.hp_analyzer.add_checkpoint(ckpt)
                        processed_count += 1

                if (
                    processed_count == 0 and not checkpoints
                ):  # No captures in batch were valid
                    self.hp_result_label.set_content("Error: Failed to capture HP.")
                    return

                result: HpAnalyzerResult | None = self.hp_analyzer.get_result()
                if result:
                    markdown_lines = [
                        "Results:",
                        "",
                        f"- Current HP: **{result.current_hp}**",
                        f"- Total HP: **{result.total_hp}**",
                        f"- HP Lost: **{result.hp_lost_per_minute:.2f}** HP/min",
                    ]
                    markdown_content = "\n".join(markdown_lines)
                    self.hp_result_label.set_content(markdown_content)
                else:
                    self.hp_result_label.set_content("Result: Analyzing...")

        except Exception as e:
            self.hp_result_label.set_content(f"Error: {e}")
            ui.notify(f"Error during HP analysis: {e}", type="negative")
            print(f"Error during HP analysis: {e}")

    def toggle_mp_analyzer(self):
        if self.is_mp_running:
            self.stop_mp_analyzer()
        else:
            self.start_mp_analyzer()

    def start_mp_analyzer(self):
        self.is_mp_running = True
        self.mp_status_label.set_text("Status: Running")
        self.mp_start_button.set_text("Stop")
        self.mp_analyzer.reset()
        self.mp_capture_buffer.clear()
        self.mp_ts_buffer.clear()
        self.mp_timer = ui.timer(
            interval=self.mp_analyzer.config.interval,
            callback=self.update_mp_analysis,
            active=True,
        )
        ui.notify("MP analyzer started.")

    def stop_mp_analyzer(self):
        self.is_mp_running = False
        if self.mp_timer:
            self.mp_timer.deactivate()
            self.mp_timer = None
        self.mp_capture_buffer.clear()
        self.mp_ts_buffer.clear()
        self.mp_status_label.set_text("Status: Idle")
        self.mp_start_button.set_text("Start")
        ui.notify("MP analyzer stopped.")

    async def update_mp_analysis(self):
        if not self.is_mp_running:
            return

        try:
            capture = capture_app_window(self.app_name)
            if capture is None:
                self.mp_result_label.set_content("Error: Failed to capture APP window.")
                return

            self.mp_capture_buffer.append(capture)
            self.mp_ts_buffer.append(datetime.datetime.now())

            if len(self.mp_capture_buffer) >= self.mp_analyzer.config.batch_size:
                checkpoints = MpCheckpoint.from_app_captures(
                    captures=self.mp_capture_buffer,
                    ts_list=self.mp_ts_buffer,
                    ocr_batch_size=self.mp_analyzer.config.batch_size,
                )
                self.mp_capture_buffer.clear()
                self.mp_ts_buffer.clear()

                processed_count = 0
                for ckpt in checkpoints:
                    if ckpt:
                        self.mp_analyzer.add_checkpoint(ckpt)
                        processed_count += 1

                if (
                    processed_count == 0 and not checkpoints
                ):  # No captures in batch were valid
                    self.mp_result_label.set_content("Error: Failed to capture MP.")
                    return

                result: MpAnalyzerResult | None = self.mp_analyzer.get_result()
                if result:
                    markdown_lines = [
                        "Results:",
                        "",
                        f"- Current MP: **{result.current_mp}**",
                        f"- Total MP: **{result.total_mp}**",
                        f"- MP Lost: **{result.mp_lost_per_minute:.2f}** MP/min",
                    ]
                    markdown_content = "\n".join(markdown_lines)
                    self.mp_result_label.set_content(markdown_content)
                else:
                    self.mp_result_label.set_content("Result: Analyzing...")
        except Exception as e:
            self.mp_result_label.set_content(f"Error: {e}")
            ui.notify(f"Error during MP analysis: {e}", type="negative")
            print(f"Error during MP analysis: {e}")


def main():
    UI()
    ui.run(title="MapleStory Artale ExpLab", native=True, reload=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
