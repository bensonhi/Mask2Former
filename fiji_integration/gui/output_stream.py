"""
Output stream helper for redirecting stdout to GUI console.
"""

__all__ = ['GUIOutputStream']


class GUIOutputStream:
    """
    Redirects stdout to GUI console widget.

    This class acts as a file-like object that can replace sys.stdout
    to capture print statements and display them in a GUI text widget.
    """

    def __init__(self, tab):
        """
        Initialize the output stream.

        Args:
            tab: The tab object that has a write_to_console() method
        """
        self.tab = tab

    def write(self, text):
        """
        Write text to the console.

        Args:
            text: Text to write
        """
        if text:  # Write all text including newlines
            self.tab.write_to_console(text)

    def flush(self):
        """Flush the stream (required for file-like object)."""
        pass  # Required for file-like object interface
