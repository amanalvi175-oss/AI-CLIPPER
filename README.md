# AI CLIPPER

A Streamlit-based, local-first video clipping workspace. It uses Faster-Whisper for speech transcription and FFmpeg for clip rendering.

## Current features

- Import a video from a computer or a supported URL (yt-dlp).
- Transcribe speech and optionally translate it into English.
- Rank candidate highlights using transcript-based heuristics.
- Export vertical (9:16) and landscape (16:9) clips.
- Burn subtitles, add a logo, mix background music, and apply fades.
- Generate editable title, description, and hashtag drafts.
- Keep project metadata, transcripts, and exports in the app's project folder.

## Run locally (Windows)

1. Install Python 3.12.
2. Open PowerShell in this repository.
3. Create and activate a virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Start the app:
   ```powershell
   streamlit run app.py
   ```

The app uses the FFmpeg executable bundled by `imageio-ffmpeg` when a system FFmpeg installation is not found. The first transcription run downloads the selected Whisper model.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new app in Streamlit Community Cloud.
3. Select this repository, branch `main`, and file `app.py`.
4. Deploy. The `packages.txt` file requests the FFmpeg system package.

## Important deployment limits

- This is currently a single-user workspace, not a production multi-user SaaS.
- There is no account/password authentication or cloud database yet.
- Local project files are stored on the app server. Streamlit Community Cloud storage is not guaranteed to be permanent; download exports and keep backups.
- CPU transcription and video rendering can be slow on free cloud resources. Large uploads may exceed hosting limits.
- Highlight ranking is transcript-based; it does not currently perform visual scene understanding, emotion detection, face tracking, or guaranteed virality prediction.
- Direct publishing is not implemented. Use official platform APIs and approved authorization flows before adding it.

## Privacy and rights

Only upload or download videos you own or have permission to process. Do not put passwords, API keys, or other secrets in source code or public repository files.
