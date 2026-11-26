"""Streamlit Community Cloud entry point.

The main application is maintained in app.py. This wrapper keeps the
configured Streamlit Cloud entry-point path working without duplicating code.
"""

import app  # noqa: F401  (running app.py builds the Streamlit interface)
