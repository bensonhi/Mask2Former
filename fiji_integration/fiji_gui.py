#!/usr/bin/env python3
"""
Fiji Integration GUI Launcher

This script launches the new modular GUI with multi-tab support.
This demonstrates the refactored architecture with extensibility for future tabs.
"""

import sys
import os

# Add fiji_integration parent directory to path
fiji_integration_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(fiji_integration_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from fiji_integration.gui.main_window import MainWindow
from fiji_integration.gui.tabs.myotube_tab import MyotubeTab
from fiji_integration.gui.tabs.cellpose_tab import CellPoseTab
from fiji_integration.gui.tabs.analysis_tab import AnalysisTab
from fiji_integration.gui.tabs.max_projection_tab import MaxProjectionTab


def main():
    """Launch the Fiji Integration GUI."""
    # Create tabs
    tabs = [
        MaxProjectionTab(),
        MyotubeTab(),
        CellPoseTab(),
        AnalysisTab(),
    ]

    # Create and show main window
    window = MainWindow(tabs, window_title="Fiji Integration - Multi-Modal Segmentation")
    window.show()


if __name__ == '__main__':
    main()
