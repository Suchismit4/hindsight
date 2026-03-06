#!/usr/bin/env python3
"""
Simple script to open the documentation in a web browser.
"""
import webbrowser
import os
from pathlib import Path

# Get the absolute path to the HTML documentation
docs_path = Path(__file__).parent / "build" / "html" / "index.html"

if docs_path.exists():
    print(f"Opening documentation at: {docs_path}")
    webbrowser.open(f"file://{docs_path.absolute()}")
else:
    print(f"Documentation not found at: {docs_path}")
    print("Please run 'make html' first to build the documentation.")
