# Copyright 2025 Yu-Kai Lin. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import os
import sys
from pathlib import Path

# For explab
sys.path.append(Path(__file__).resolve().parents[2].as_posix())

import dotenv
from nicegui import ui

from explab.analyzer.exp import (
    ExpAnalyzer,
    ExpAnalyzerResult,
)
from explab.analyzer.hp import (
    HpAnalyzer,
    HpAnalyzerResult,
)
from explab.analyzer.mp import (
    MpAnalyzer,
    MpAnalyzerResult,
)
from explab.maplestory.exp import ExpCheckpoint
from explab.maplestory.hp import HpCheckpoint
from explab.maplestory.mp import MpCheckpoint
from explab.ocr import ocr
from explab.screen_capture import capture_app_window
from explab.utils.base import PROJECT_ROOT

dotenv.load_dotenv(PROJECT_ROOT / ".env")


class UI:
    def __init__(self):
        ocr.initialize()  # Initialize OCR engine

        self.exp_analyzer = ExpAnalyzer()
        self.hp_analyzer = HpAnalyzer()
        self.mp_analyzer = MpAnalyzer()
        self.is_exp_running = False
        self.is_hp_running = False
        self.is_mp_running = False
        self.exp_timer = None
        self.hp_timer = None
        self.mp_timer = None

        self.app_name = os.getenv("MAPLESTORY_APP_NAME", "MapleStory Worlds")

        with ui.row():
            with ui.card():
                ui.label("Experience Analyzer").classes("text-h6")
                with ui.row(align_items="center"):
                    self.exp_start_button = ui.button(
                        "Start", on_click=self.toggle_exp_analyzer
                    )
                    self.exp_status_label = ui.label("Status: Idle")
                self.exp_result_label = ui.markdown("Result: ")

            with ui.card():
                ui.label("HP Analyzer").classes("text-h6")
                with ui.row(align_items="center"):
                    self.hp_start_button = ui.button(
                        "Start", on_click=self.toggle_hp_analyzer
                    )
                    self.hp_status_label = ui.label("Status: Idle")
                self.hp_result_label = ui.markdown("Result: ")

            with ui.card():
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

            checkpoint = HpCheckpoint.from_app_capture(capture=capture)

            if checkpoint:
                self.hp_analyzer.add_checkpoint(checkpoint)
                try:
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
                except ValueError as ve:
                    self.hp_result_label.set_content(f"Error: {ve}")
            else:
                self.hp_result_label.set_content("Error: Failed to capture HP.")
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

            checkpoint = MpCheckpoint.from_app_capture(capture=capture)

            if checkpoint:
                self.mp_analyzer.add_checkpoint(checkpoint)
                try:
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
                except ValueError as ve:
                    self.mp_result_label.set_content(f"Error: {ve}")
            else:
                self.mp_result_label.set_content("Error: Failed to capture MP.")
        except Exception as e:
            self.mp_result_label.set_content(f"Error: {e}")
            ui.notify(f"Error during MP analysis: {e}", type="negative")
            print(f"Error during MP analysis: {e}")


def main():
    UI()
    ui.run(title="MapleStory Artale ExpLab", native=True, reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
