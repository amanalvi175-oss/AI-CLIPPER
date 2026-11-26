# AI CLIPPER

A Streamlit-based, local-first video clipping workspace. It uses Faster-Whisper for speech transcription and FFmpeg for clip rendering.

## Current features

- Import a video from a computer or a supported URL (yt-dlp).
- Transcribe speech and optionally translate it into English.
- Rank candidate highlights using transcript-based heuristics.
- Export vertical (9:16) and landscape (16:9) clips.
- Burn word-timed captions, add a logo, mix background music, and apply fades.\n- Add optional zoom, progress indicator, soft-vignette or flash-intro visual finishes.\n- Save and reuse editing preferences on the same persistent app instance.\n- Generate video-frame thumbnails with editable title overlays.\n- Experimental active-speaker framing estimates mouth movement from sampled faces; it is not voice diarization and may be inaccurate.
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
- Project files and editing preferences are stored on the app server. Streamlit Community Cloud storage is not guaranteed to be permanent; download exports and keep backups. Saved preferences persist only while the app's storage persists.
- CPU transcription and video rendering can be slow on free cloud resources. Large uploads may exceed hosting limits.
- Highlight ranking is transcript-based and does not predict virality. Active-speaker framing is a lightweight mouth-motion heuristic, not robust speaker identification or diarization. Visual results depend on lighting, camera motion, face visibility and CPU sampling.
- Direct publishing is not implemented. Use official platform APIs and approved authorization flows before adding it.

## Current feature boundaries\n\n- The transition controls apply visual finishing to each exported clip; they do not create a timeline between separate clip files.\n- Thumbnail title overlays use Pillow when available; the source frame can still be extracted if Pillow is unavailable.\n- Active-speaker framing is experimental and should be reviewed manually before publishing.\n- Live deployment status must be checked in the Streamlit Community Cloud dashboard for the connected app; a successful GitHub syntax check does not prove that a deployed app starts or that video rendering works on the cloud host.\n\n## Privacy and rights

Only upload or download videos you own or have permission to process. Do not put passwords, API keys, or other secrets in source code or public repository files.
