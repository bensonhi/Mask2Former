"""
Base class for GUI tabs in the multi-functionality application.

This module defines the interface that all functionality tabs must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import tkinter as tk
from tkinter import ttk


__all__ = ['TabInterface']


class TabInterface(ABC):
    """
    Abstract base class for all functionality tabs.

    Each tab represents a distinct functionality (e.g., myotube segmentation,
    nuclei analysis, max projection) and must implement this interface.
    """

    def __init__(self):
        """Initialize the tab."""
        self.root = None
        self.console_text = None
        self.button_frame = None
        self.is_running = False
        self.stop_requested = False

    @abstractmethod
    def get_tab_name(self) -> str:
        """
        Return the display name for this tab.

        Returns:
            str: Tab name to show in the UI (e.g., "Myotube Segmentation")
        """
        pass

    @abstractmethod
    def build_ui(self, parent_frame: ttk.Frame, console_text: tk.Text) -> None:
        """
        Build the tab's user interface inside the parent frame.

        Args:
            parent_frame: The scrollable frame where UI elements should be added
            console_text: Shared console widget for output (optional, can be None if tab has own console)
        """
        pass

    @abstractmethod
    def get_button_frame_widgets(self) -> list:
        """
        Return a list of button widgets to display in the shared button area.

        Each button should be a tuple of (button_widget, side) where side is
        tk.LEFT or tk.RIGHT.

        Returns:
            list: List of (widget, side) tuples for button placement

        Example:
            return [
                (self.run_button, tk.LEFT),
                (self.stop_button, tk.LEFT),
            ]
        """
        pass

    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """
        Load saved configuration for this tab.

        Returns:
            dict: Configuration parameters
        """
        pass

    @abstractmethod
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration for this tab.

        Args:
            config: Configuration parameters to save
        """
        pass

    @abstractmethod
    def validate_parameters(self) -> tuple[bool, Optional[str]]:
        """
        Validate current parameters before running.

        Returns:
            tuple: (is_valid, error_message)
                is_valid: True if parameters are valid
                error_message: None if valid, error string if invalid
        """
        pass

    # Optional methods with default implementations

    def on_tab_selected(self):
        """
        Called when this tab is selected/activated.

        Override this to perform actions when tab becomes active.
        """
        pass

    def on_tab_deselected(self):
        """
        Called when this tab is deselected.

        Override this to perform cleanup when tab becomes inactive.
        """
        pass

    def can_close(self) -> tuple[bool, Optional[str]]:
        """
        Check if the application can be closed.

        Returns:
            tuple: (can_close, reason)
                can_close: True if safe to close
                reason: None if safe, warning message if not safe
        """
        if self.is_running:
            return False, f"{self.get_tab_name()} is currently running. Please stop it first."
        return True, None

    # Helper methods for console output

    def write_to_console(self, text: str):
        """
        Write text to the console widget.

        Args:
            text: Text to append to console
        """
        if self.console_text:
            self.console_text.config(state='normal')
            self.console_text.insert(tk.END, text)
            self.console_text.see(tk.END)  # Auto-scroll to bottom
            self.console_text.config(state='disabled')
            if self.root:
                self.root.update_idletasks()

    def clear_console(self):
        """Clear the console widget."""
        if self.console_text:
            self.console_text.config(state='normal')
            self.console_text.delete('1.0', tk.END)
            self.console_text.config(state='disabled')
