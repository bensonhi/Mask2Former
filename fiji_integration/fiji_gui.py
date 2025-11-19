#!/usr/bin/env python3
"""
Fiji Integration GUI Launcher

This script launches the new modular GUI with multi-tab support.
This demonstrates the refactored architecture with extensibility for future tabs.
"""

import sys
import os

# Setup Python path for imports
# This script can be run from two locations:
# 1. From fiji_integration/ directory (development)
# 2. From Fiji.app/macros/ directory (deployment)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check if we're in the fiji_integration directory or Fiji macros
if os.path.basename(script_dir) == 'fiji_integration':
    # Development mode: add parent directory
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
else:
    # Deployment mode: create fiji_integration as an alias to current directory
    # This allows all the existing imports to work without modification
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    # Create symlink-like behavior by adding script_dir with 'fiji_integration' name
    import importlib.util
    import types

    # Create fiji_integration as a module pointing to current directory
    fiji_integration = types.ModuleType('fiji_integration')
    fiji_integration.__path__ = [script_dir]
    fiji_integration.__file__ = os.path.join(script_dir, '__init__.py')
    sys.modules['fiji_integration'] = fiji_integration

# Now import normally
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
