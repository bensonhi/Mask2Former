"""
Main application window with multi-tab support.

This module provides the main GUI window that can host multiple
functionality tabs (myotube segmentation, nuclei analysis, etc.).
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Optional

from fiji_integration.gui.base_tab import TabInterface


__all__ = ['MainWindow']


class MainWindow:
    """
    Main application window with tabbed interface.

    Manages multiple functionality tabs and provides shared infrastructure
    like console output and button area.
    """

    def __init__(self, tabs: List[TabInterface], window_title: str = "Fiji Integration"):
        """
        Initialize the main window.

        Args:
            tabs: List of TabInterface implementations to display
            window_title: Title for the application window
        """
        self.tabs = tabs
        self.window_title = window_title
        self.root = None
        self.notebook = None
        self.console_text = None
        self.button_frame = None
        self.current_tab = None

    def show(self) -> Optional[dict]:
        """
        Display the GUI and return results when closed.

        Returns:
            dict: Result dictionary (if applicable) or None
        """
        # Create main window
        self.root = tk.Tk()
        self.root.title(self.window_title)
        self.root.geometry("900x1000")

        # Create main container
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Create notebook (tab container) - takes most of the space
        self.notebook = ttk.Notebook(container)
        self.notebook.grid(row=0, column=0, sticky=(tk.N, tk.W, tk.E, tk.S), padx=5, pady=5)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=3)  # Notebook gets 3/4 of space

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # Create tabs
        self.tab_frames = {}
        for tab in self.tabs:
            # Create scrollable frame for each tab
            tab_container = ttk.Frame(self.notebook)

            # Create canvas with scrollbar
            canvas = tk.Canvas(tab_container)
            scrollbar = ttk.Scrollbar(tab_container, orient="vertical", command=canvas.yview)

            # Create scrollable frame inside canvas
            scrollable_frame = ttk.Frame(canvas, padding="10")

            # Configure canvas
            canvas.configure(yscrollcommand=scrollbar.set)

            # Pack scrollbar and canvas
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Create window in canvas
            canvas_frame = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

            # Configure scroll region when frame changes size
            def configure_scroll_region(event=None, c=canvas):
                c.configure(scrollregion=c.bbox("all"))

            scrollable_frame.bind("<Configure>", configure_scroll_region)

            # Bind mousewheel for scrolling
            def on_mousewheel(event, c=canvas):
                c.yview_scroll(int(-1*(event.delta/120)), "units")

            # Bind mousewheel to canvas and all child widgets
            def bind_mousewheel(widget, c=canvas):
                widget.bind("<MouseWheel>", lambda e, c=c: on_mousewheel(e, c))  # Windows/MacOS
                widget.bind("<Button-4>", lambda e, c=c: c.yview_scroll(-1, "units"))  # Linux scroll up
                widget.bind("<Button-5>", lambda e, c=c: c.yview_scroll(1, "units"))   # Linux scroll down
                for child in widget.winfo_children():
                    bind_mousewheel(child, c)

            # Initial bind
            self.root.after(100, lambda sf=scrollable_frame: bind_mousewheel(sf))

            # Update canvas width when container is resized
            def on_canvas_configure(event, c=canvas, cf=canvas_frame):
                c.itemconfig(cf, width=event.width)

            canvas.bind("<Configure>", on_canvas_configure)

            # Store references
            self.tab_frames[tab] = {
                'container': tab_container,
                'scrollable_frame': scrollable_frame,
                'canvas': canvas
            }

            # Add tab to notebook
            self.notebook.add(tab_container, text=tab.get_tab_name())

        # Create shared console output area
        console_container = ttk.LabelFrame(container, text="Console Output", padding="5")
        console_container.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        container.rowconfigure(1, weight=1)  # Console gets 1/4 of space

        # Create text widget with scrollbar
        console_frame = ttk.Frame(console_container)
        console_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(console_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.console_text = tk.Text(console_frame, height=15, width=80,
                                     yscrollcommand=scrollbar.set,
                                     bg='#1e1e1e', fg='#d4d4d4',
                                     font=('Consolas', 9))
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.console_text.yview)

        # Make console read-only
        self.console_text.config(state='disabled')

        # Create shared button frame
        self.button_frame = ttk.Frame(container)
        self.button_frame.grid(row=2, column=0, pady=10)

        # Build UI for all tabs
        for tab in self.tabs:
            scrollable_frame = self.tab_frames[tab]['scrollable_frame']
            tab.root = self.root
            tab.console_text = self.console_text
            tab.button_frame = self.button_frame  # Provide button frame to tabs
            tab.build_ui(scrollable_frame, self.console_text)

        # Add Close button (always present)
        ttk.Button(self.button_frame, text="Close", command=self._on_close).pack(side=tk.RIGHT, padx=5)

        # Initialize first tab
        if self.tabs:
            self.current_tab = self.tabs[0]
            self._update_button_frame()
            self.current_tab.on_tab_selected()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

        # Run the GUI
        self.root.mainloop()

        return None  # Can be extended to return results from tabs

    def _on_tab_changed(self, event):
        """Handle tab selection change."""
        # Get newly selected tab index
        selected_index = self.notebook.index(self.notebook.select())

        # Notify old tab it was deselected
        if self.current_tab:
            self.current_tab.on_tab_deselected()

        # Switch to new tab
        self.current_tab = self.tabs[selected_index]

        # Update button frame with new tab's buttons
        self._update_button_frame()

        # Notify new tab it was selected
        self.current_tab.on_tab_selected()

    def _update_button_frame(self):
        """Update the button frame with current tab's buttons."""
        # Hide all buttons except Close button (which is always last)
        for widget in list(self.button_frame.winfo_children())[:-1]:
            widget.pack_forget()

        # Show current tab's buttons
        if self.current_tab:
            buttons = self.current_tab.get_button_frame_widgets()
            for button, side in buttons:
                button.pack(side=side, padx=5)

    def _on_close(self):
        """Handle window close request."""
        # Check if any tab is running
        for tab in self.tabs:
            can_close, reason = tab.can_close()
            if not can_close:
                result = messagebox.askyesno(
                    "Confirm Close",
                    f"{reason}\n\nDo you want to close anyway? This may leave incomplete results."
                )
                if not result:
                    return

        # Save config for current tab
        if self.current_tab:
            try:
                config = self.current_tab.load_config()
                self.current_tab.save_config(config)
            except:
                pass  # Ignore save errors on close

        # Close window
        self.root.quit()
        self.root.destroy()
