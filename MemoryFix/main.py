"""
main.py - Entry point for the MemoryFix application.
Creates the QApplication and launches the main window.
"""

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from gui import MemoryFixWindow


def main():
    """Initialize and run the MemoryFix application."""
    # Create the application
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("MemoryFix")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("MemoryFix")
    
    # Create and show the main window
    window = MemoryFixWindow()
    window.show()
    
    # Run the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
