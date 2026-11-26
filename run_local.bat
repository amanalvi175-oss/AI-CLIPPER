@echo off
setlocal
cd /d "%~dp0"
set AI_CLIPPER_LOCAL=1
echo Starting AI CLIPPER locally.
echo The app can read your local Chrome session for authorized YouTube downloads.
echo Your browser cookies are not uploaded to Streamlit Cloud.
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
pause
