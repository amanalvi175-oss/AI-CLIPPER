"""Streamlit Community Cloud entry point.

Execute the main application as the Streamlit script so the UI is rendered
reliably by Community Cloud.
"""

from pathlib import Path
import runpy

app_path = Path(__file__).with_name("app.py")
runpy.run_path(str(app_path), run_name="__main__")
