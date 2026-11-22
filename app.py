
import re
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import streamlit as st
import imageio_ffmpeg
from faster_whisper import WhisperModel

# ============================================================
# AI CLIPPER 5.1 - FINAL MASTER EDITION
# Windows | Python 3.12 | CPU | INT8 | 8 GB RAM
# ============================================================

APP_NAME = "AI CLIPPER 5.1"
ROOT = Path("AI_CLIPPER_PROJECTS")
ROOT.mkdir(exist_ok=True)

# ----------------------- FFMPEG -----------------------------

def find_ffmpeg():
    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if Path(candidate).exists():
            return candidate
    except Exception:
        pass

    return None

FFMPEG = find_ffmpeg()

def require_ffmpeg():
    if not FFMPEG:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and reopen PowerShell."
        )

def run_ffmpeg(args):
    require_ffmpeg()

    proc = subprocess.run(
        [FFMPEG, *map(str, args)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr[-5000:] or "FFmpeg operation failed."
        )

    return proc

# ----------------------- UTILITIES --------------------------

def safe_name(value):
    value = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]', "_", str(value)
    )
    return value.strip(" ._")[:100] or "video"

def get_duration(path):
    require_ffmpeg()

    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)",
        proc.stderr,
    )

    if not match:
        raise RuntimeError(
            "Could not read video duration. Try MP4, MOV or MKV."
        )

    h, m, s, fraction = match.groups()

    return (
        int(h) * 3600
        + int(m) * 60
        + int(s)
        + float("0." + fraction)
    )

def has_audio(path):
    require_ffmpeg()

    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    return bool(re.search(r"Audio:", proc.stderr))

def ffmpeg_filter_path(path):
    value = str(Path(path).resolve()).replace("\\", "/")
    value = value.replace("\\", r"\\")
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    value = value.replace(",", r"\,")
    return value

def srt_timestamp(seconds):
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)

    return (
        f"{hours:02}:{minutes:02}:"
        f"{secs:02},{ms:03}"
    )

def write_srt(segments, destination, offset=0, end=None):
    lines = []
    index = 1

    for seg in segments:
        start = max(0.0, float(seg["start"]) - offset)
        finish = float(seg["end"]) - offset

        if end is not None:
            finish = min(finish, end - offset)

        if finish <= start:
            continue

        lines.extend([
            str(index),
            f"{srt_timestamp(start)} --> "
            f"{srt_timestamp(finish)}",
            str(seg["text"]).strip(),
            "",
        ])

        index += 1

    Path(destination).write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )

# ----------------------- WHISPER ----------------------------

@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size):
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
    )

def transcribe(path, translate, progress, model_size):
    model = load_whisper_model(model_size)

    task = "translate" if translate else "transcribe"

    segments_iter, info = model.transcribe(
        str(path),
        task=task,
        beam_size=1,
        vad_filter=True,
        word_timestamps=False,
    )

    results = []

    for segment in segments_iter:
        results.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
        })

    progress.progress(100)

    if not results:
        raise RuntimeError("No speech detected in this video.")

    return results, info.language

# ----------------------- VIDEO IMPORT -----------------------

def import_from_url(url, project_dir):
    import yt_dlp

    require_ffmpeg()

    options = {
        "format": "bestvideo*+bestaudio/best",
        "outtmpl": str(project_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG,
        "retries": 3,
        "fragment_retries": 3,
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)

    candidates = [
        p for p in project_dir.glob("source.*")
        if p.is_file()
        and p.suffix.lower() not in {".part", ".ytdl"}
    ]

    if not candidates:
        raise RuntimeError("Video download did not produce a file.")

    candidates.sort(
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    return candidates[0], info.get("title") or candidates[0].stem

# ----------------------- HIGHLIGHTS -------------------------

HIGHLIGHT_WORDS = {
    "amazing", "important", "secret", "mistake",
    "never", "always", "best", "worst", "truth",
    "crazy", "wow", "remember", "learn", "believe",
    "problem", "solution", "success", "fail", "money",
    "powerful", "incredible", "actually", "why", "how",
    "tip", "lesson", "reason", "finally", "imagine",
    "warning", "because", "first", "three", "top",
    "avoid", "stop", "start", "change",
}

def segment_score(segment):
    text = segment["text"].lower()
    words = re.findall(r"\b[\w']+\b", text)

    score = min(len(words) / 16, 3.0)

    score += sum(
        1.8 for word in HIGHLIGHT_WORDS
        if re.search(rf"\b{re.escape(word)}\b", text)
    )

    score += 1.0 if "?" in text or "!" in text else 0

    score += (
        0.4
        if any(w in text for w in ("you", "your", "because"))
        else 0
    )

    return score

def find_highlights(
    segments, total_duration, clip_length, count,
    spacing_seconds=8,
):
    step = max(5, min(12, clip_length // 4))
    candidates = []
    start = 0.0

    while start < total_duration:
        end = min(total_duration, start + clip_length)

        if end - start < min(10, clip_length * 0.6):
            break

        selected = [
            s for s in segments
            if s["end"] > start and s["start"] < end
        ]

        if selected:
            score = sum(segment_score(s) for s in selected)

            text = " ".join(
                s["text"] for s in selected
            ).strip()

            speech_time = sum(
                max(
                    0,
                    min(s["end"], end) - max(s["start"], start),
                )
                for s in selected
            )

            coverage = min(
                1.0,
                speech_time / max(1.0, end - start),
            )

            score *= 0.65 + 0.35 * coverage

            candidates.append({
                "start": start,
                "end": end,
                "score": score,
                "text": text,
                "coverage": coverage,
            })

        start += step

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    chosen = []

    for candidate in candidates:
        overlaps = any(
            candidate["start"] < old["end"] + spacing_seconds
            and candidate["end"] > old["start"] - spacing_seconds
            for old in chosen
        )

        if not overlaps:
            chosen.append(candidate)

        if len(chosen) >= count:
            break

    return sorted(chosen, key=lambda item: item["start"])

# ----------------------- METADATA ---------------------------

def metadata_for(text):
    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    stop = {
        "this", "that", "with", "from", "your", "you",
        "the", "and", "for", "are", "was", "were",
        "have", "has", "had", "what", "when", "then",
        "they", "them", "about", "into", "just", "like",
        "very", "really", "because", "there", "their",
    }

    counts = {}

    for word in words:
        if len(word) > 3 and word not in stop:
            counts[word] = counts.get(word, 0) + 1

    topics = sorted(
        counts,
        key=counts.get,
        reverse=True,
    )[:5] or ["video", "highlights"]

    title = " ".join(w.title() for w in topics[:3])
    hashtags = " ".join("#" + w for w in topics)

    description = (
        f"Watch this highlight on {', '.join(topics[:3])}."
        f"\n\n{hashtags}"
    )

    return title, description, hashtags

# ----------------------- PROJECTS ---------------------------

DEFAULTS = {
    "project_dir": None,
    "video_path": None,
    "video_name": None,
    "transcript": [],
    "detected_language": "",
    "clips": [],
    "exported": [],
    "last_import": "",
}

def reset_project_state():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

def save_project():
    if not st.session_state.project_dir:
        return

    if not st.session_state.video_path:
        return

    project = Path(st.session_state.project_dir)
    project.mkdir(parents=True, exist_ok=True)

    data = {
        "video_file": Path(st.session_state.video_path).name,
        "video_name": st.session_state.video_name,
        "transcript": st.session_state.transcript,
        "detected_language": st.session_state.detected_language,
        "clips": st.session_state.clips,
        "exported": st.session_state.exported,
        "last_import": st.session_state.last_import,
    }

    temp_file = project / "project.json.tmp"

    temp_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temp_file.replace(project / "project.json")

def load_project(project_path):
    project_path = Path(project_path)

    metadata_path = project_path / "project.json"

    if not metadata_path.exists():
        raise FileNotFoundError("Project file is missing.")

    data = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    video = project_path / data.get("video_file", "")

    if not video.is_file():
        raise FileNotFoundError("Original source video is missing.")

    reset_project_state()

    st.session_state.project_dir = str(project_path)
    st.session_state.video_path = str(video)
    st.session_state.video_name = data.get("video_name", video.name)
    st.session_state.transcript = data.get("transcript", [])
    st.session_state.detected_language = data.get("detected_language", "")
    st.session_state.clips = data.get("clips", [])
    st.session_state.exported = data.get("exported", [])
    st.session_state.last_import = data.get("last_import", "")

def create_project_from_upload(uploaded):
    project = Path(
        tempfile.mkdtemp(prefix="clipper_", dir=ROOT)
    )

    destination = project / safe_name(uploaded.name)
    destination.write_bytes(uploaded.getbuffer())

    reset_project_state()

    st.session_state.project_dir = str(project)
    st.session_state.video_path = str(destination)
    st.session_state.video_name = uploaded.name
    st.session_state.last_import = "Computer upload"

    save_project()

# ----------------------- CLIP EXPORT -----------------------

def ass_color(hex_color):
    value = hex_color.lstrip("#")

    if len(value) != 6:
        return "&H00FFFFFF"

    r, g, b = value[0:2], value[2:4], value[4:6]

    return f"&H00{b}{g}{r}".upper()

def export_clip(
    video, clip, number, export_dir, all_segments, aspect,
    burn_subtitles=True, subtitle_size=22,
    subtitle_color="#FFFFFF", subtitle_outline=2,
    crop_mode="Center crop", fade_seconds=0.0,
    music_path=None, logo_path=None,
    mix_original_audio=True, music_volume=0.12,
):
    require_ffmpeg()

    export_dir.mkdir(parents=True, exist_ok=True)

    output = export_dir / f"clip_{number:02}.mp4"
    srt_path = export_dir / f"clip_{number:02}.srt"

    start = float(clip["start"])
    end = float(clip["end"])
    duration = end - start

    if start < 0 or duration <= 0:
        raise RuntimeError("Invalid clip start or end time.")

    write_srt(
        all_segments,
        srt_path,
        offset=start,
        end=end,
    )

    if aspect == "9:16":
        if crop_mode == "Fit with blurred background":
            graph = (
                "[0:v]split=2[bg][fg];"
                "[bg]scale=720:1280:"
                "force_original_aspect_ratio=increase,"
                "crop=720:1280,boxblur=20:10[blur];"
                "[fg]scale=720:1280:"
                "force_original_aspect_ratio=decrease[front];"
                "[blur][front]overlay=(W-w)/2:(H-h)/2,"
                "setsar=1[base]"
            )
        else:
            graph = (
                "[0:v]scale=720:1280:"
                "force_original_aspect_ratio=increase,"
                "crop=720:1280,setsar=1[base]"
            )
    else:
        graph = (
            "[0:v]scale=1280:720:"
            "force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1[base]"
        )

    current = "base"

    if burn_subtitles:
        sub_path = ffmpeg_filter_path(srt_path)
        color = ass_color(subtitle_color)

        graph += (
            f";[{current}]subtitles='{sub_path}':"
            f"force_style='FontName=Arial,"
            f"FontSize={int(subtitle_size)},Bold=1,"
            f"PrimaryColour={color},"
            f"OutlineColour=&H00000000,"
            f"Outline={int(subtitle_outline)},"
            f"Shadow=1,Alignment=2,MarginV=55'[subbed]"
        )

        current = "subbed"

    command = [
        "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video),
    ]

    music_index = None
    logo_index = None
    next_index = 1

    if music_path:
        music_index = next_index
        next_index += 1
        command += [
            "-stream_loop", "-1",
            "-i", str(music_path),
        ]

    if logo_path:
        logo_index = next_index
        next_index += 1
        command += ["-i", str(logo_path)]

        graph += (
            f";[{logo_index}:v]scale=180:-1[brand];"
            f"[{current}][brand]overlay=W-w-24:24[vlogo]"
        )

        current = "vlogo"

    if fade_seconds > 0:
        fade = min(float(fade_seconds), duration / 3)

        graph += (
            f";[{current}]fade=t=in:st=0:d={fade:.2f},"
            f"fade=t=out:st={max(0, duration-fade):.2f}:"
            f"d={fade:.2f}[vout]"
        )
    else:
        graph += f";[{current}]null[vout]"

    if music_path and mix_original_audio:
        if has_audio(video):
            graph += (
                ";[0:a:0]volume=1.0[orig];"
                f"[{music_index}:a]volume={float(music_volume)}[music];"
                "[orig][music]amix=inputs=2:"
                "duration=first:dropout_transition=2[aout]"
            )

            audio_map = ["-map", "[aout]"]

        else:
            graph += (
                f";[{music_index}:a]volume="
                f"{float(music_volume)}[aout]"
            )

            audio_map = ["-map", "[aout]"]

    elif music_path:
        audio_map = ["-map", f"{music_index}:a:0"]

    else:
        audio_map = ["-map", "0:a?"]

    command += [
        "-t", f"{duration:.3f}",
        "-filter_complex", graph,
        "-map", "[vout]",
        *audio_map,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output),
    ]

    run_ffmpeg(command)

    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Export file was not created.")

    return output, srt_path

# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f4f6fb;
}
section[data-testid="stSidebar"] {
    background: #121a2b;
}
div.stButton > button {
    background: #655cff;
    color: white;
    border: 0;
    border-radius: 9px;
    min-height: 42px;
    font-weight: 650;
}
div.stButton > button:hover {
    background: #817aff;
    color: white;
}
div[data-testid="stMetric"] {
    background: #151f34;
    padding: 14px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ----------------------- SIDEBAR ----------------------------

with st.sidebar:
    st.title("🎬 AI CLIPPER")

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Import Video",
            "Subtitles & Transcript",
            "AI Highlights",
            "Clip Editor & Export",
            "Content Studio",
            "Publishing",
            "Settings",
        ],
    )

    st.divider()
    st.caption("LOCAL CPU EDITION")
    st.caption("Faster-Whisper INT8")

    if FFMPEG:
        st.success("FFmpeg ready")
    else:
        st.error("FFmpeg missing")

st.title("🎬 AI CLIPPER 5.1")
st.caption(
    "Local-first video clipping and short-form content studio."
)

# ----------------------- DASHBOARD --------------------------

if page == "Dashboard":

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Video loaded",
        "Yes" if st.session_state.video_path else "No",
    )

    c2.metric("Highlights", len(st.session_state.clips))
    c3.metric("Exports", len(st.session_state.exported))

    st.subheader("Workflow")

    st.markdown("""
    1. Import a video from your computer or supported URL.
    2. Generate subtitles or translate speech into English.
    3. Detect candidate highlights.
    4. Create vertical or landscape clips.
    5. Add subtitles, background music and logo.
    6. Generate titles, descriptions and hashtags.
    7. Review and publish your exports.
    """)

    if st.session_state.video_path:
        st.success(
            f"Active project: {st.session_state.video_name}"
        )

        st.video(st.session_state.video_path)

        st.caption(st.session_state.project_dir)

    saved_projects = sorted(
        [
            p for p in ROOT.iterdir()
            if p.is_dir() and (p / "project.json").exists()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if saved_projects:
        st.subheader("Saved projects")

        for project in saved_projects[:10]:
            try:
                meta = json.loads(
                    (project / "project.json").read_text(
                        encoding="utf-8"
                    )
                )

                col1, col2 = st.columns([4, 1])

                col1.write(
                    meta.get("video_name", project.name)
                )

                if col2.button(
                    "Open", key=f"open_{project.name}"
                ):
                    load_project(project)
                    st.success("Project reopened.")
                    st.rerun()

            except Exception as exc:
                st.warning(str(exc))

# ----------------------- IMPORT VIDEO -----------------------

elif page == "Import Video":

    st.header("Import video")

    method = st.radio(
        "Source",
        ["Computer upload", "Video URL"],
        horizontal=True,
    )

    if method == "Computer upload":

        uploaded = st.file_uploader(
            "Choose video",
            type=["mp4", "mov", "mkv", "avi", "webm"],
        )

        if uploaded and st.button(
            "Import video", type="primary"
        ):
            try:
                create_project_from_upload(uploaded)
                st.success("Video imported and project saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")

    else:

        url = st.text_input("Paste a supported video URL")

        st.caption(
            "Only download videos you own or have permission to use."
        )

        if st.button(
            "Download and import", type="primary"
        ):

            if not url.strip():
                st.warning("Paste a URL first.")

            else:
                project = Path(
                    tempfile.mkdtemp(
                        prefix="clipper_", dir=ROOT
                    )
                )

                try:
                    with st.spinner("Downloading video..."):
                        video, title = import_from_url(
                            url.strip(), project
                        )

                    reset_project_state()

                    st.session_state.project_dir = str(project)
                    st.session_state.video_path = str(video)
                    st.session_state.video_name = title
                    st.session_state.last_import = "Video URL"

                    save_project()

                    st.success("Video imported successfully.")
                    st.rerun()

                except Exception as exc:
                    st.error(f"Import failed: {exc}")

    if st.session_state.video_path:
        st.divider()
        st.write(
            f"**Current video:** {st.session_state.video_name}"
        )
        st.video(st.session_state.video_path)

# ----------------------- SUBTITLES --------------------------

elif page == "Subtitles & Transcript":

    st.header("Subtitles and transcript")

    if not st.session_state.video_path:
        st.warning("Import or reopen a project first.")

    else:

        model_size = st.selectbox(
            "Whisper model",
            ["tiny", "base", "small"],
            index=0,
            help="Tiny is recommended for 8 GB RAM.",
        )

        translate = st.checkbox(
            "Translate speech into English",
            value=False,
        )

        st.caption(
            "The first run downloads the selected model."
        )

        if st.button(
            "Generate subtitles", type="primary"
        ):

            bar = st.progress(0)

            try:
                with st.spinner("Transcribing on your CPU..."):

                    segments, language = transcribe(
                        st.session_state.video_path,
                        translate,
                        bar,
                        model_size,
                    )

                st.session_state.transcript = segments
                st.session_state.detected_language = language

                project = Path(
                    st.session_state.project_dir
                )

                write_srt(
                    segments,
                    project / "subtitles.srt",
                )

                (project / "transcript.txt").write_text(
                    "\n".join(
                        x["text"] for x in segments
                    ),
                    encoding="utf-8",
                )

                save_project()

                st.success(
                    f"Subtitles generated. Language: {language}"
                )

            except Exception as exc:
                st.error(f"Transcription failed: {exc}")

        if st.session_state.transcript:

            st.info(
                "Detected language: "
                + st.session_state.detected_language
            )

            st.text_area(
                "Transcript",
                "\n".join(
                    x["text"]
                    for x in st.session_state.transcript
                ),
                height=280,
            )

            project = Path(
                st.session_state.project_dir
            )

            for filename in (
                "subtitles.srt", "transcript.txt"
            ):

                path = project / filename

                if path.exists():
                    st.download_button(
                        f"Download {filename}",
                        data=path.read_bytes(),
                        file_name=filename,
                        key=f"get_{filename}",
                    )

# ----------------------- AI HIGHLIGHTS ----------------------

elif page == "AI Highlights":

    st.header("Find highlights")

    if not st.session_state.transcript:

        st.warning("Generate subtitles first.")

    else:

        left, right = st.columns(2)

        with left:
            length = st.slider(
                "Target clip length (seconds)",
                15, 60, 45,
            )

        with right:
            count = st.slider(
                "Maximum number of clips",
                1, 10, 5,
            )

        spacing = st.slider(
            "Minimum gap between clips (seconds)",
            0, 30, 8,
        )

        st.caption(
            "Highlights are ranked using transcript-based scoring, "
            "not visual scene or emotion recognition."
        )

        if st.button(
            "Find highlights", type="primary"
        ):

            try:
                duration = get_duration(
                    st.session_state.video_path
                )

                clips = find_highlights(
                    st.session_state.transcript,
                    duration,
                    length,
                    count,
                    spacing,
                )

                st.session_state.clips = clips
                st.session_state.exported = []

                save_project()

                if clips:
                    st.success(
                        f"Found {len(clips)} candidate highlights."
                    )
                else:
                    st.warning(
                        "No suitable speech sections found."
                    )

            except Exception as exc:
                st.error(str(exc))

        for i, clip in enumerate(
            st.session_state.clips, 1
        ):

            with st.expander(
                f"Highlight {i} | "
                f"{clip['start']:.1f}s - "
                f"{clip['end']:.1f}s | "
                f"Score {clip['score']:.1f}"
            ):

                st.write(clip["text"])
                st.caption("Review before exporting.")

# ----------------------- CLIP EDITOR ------------------------

elif page == "Clip Editor & Export":

    st.header("Clip editor and export")

    if not st.session_state.clips:

        st.warning("Find highlights first.")

    else:

        aspect_label = st.selectbox(
            "Output format",
            ["9:16 vertical", "16:9 landscape"],
        )

        crop_mode = "Center crop"

        if aspect_label.startswith("9:16"):

            crop_mode = st.selectbox(
                "Vertical framing",
                [
                    "Center crop",
                    "Fit with blurred background",
                ],
            )

        burn = st.checkbox(
            "Burn subtitles into video",
            value=True,
        )

        subtitle_size = st.slider(
            "Subtitle font size", 14, 42, 22
        )

        subtitle_color = st.color_picker(
            "Subtitle colour", "#FFFFFF"
        )

        subtitle_outline = st.slider(
            "Subtitle outline thickness", 0, 5, 2
        )

        fade_seconds = st.slider(
            "Fade in/out seconds",
            0.0, 1.5, 0.3, 0.1,
        )

        music_upload = st.file_uploader(
            "Optional background music",
            type=["mp3", "wav", "m4a"],
        )

        logo_upload = st.file_uploader(
            "Optional logo",
            type=["png", "jpg", "jpeg"],
        )

        mix_audio = False
        music_volume = 0.12

        if music_upload:

            mix_audio = st.checkbox(
                "Mix music with original audio",
                value=True,
            )

            music_volume = st.slider(
                "Background music volume",
                0.0, 1.0, 0.12, 0.05,
            )

        if st.button(
            "Generate all clips", type="primary"
        ):

            project = Path(
                st.session_state.project_dir
            )

            export_dir = project / "EXPORTS"
            export_dir.mkdir(exist_ok=True)

            music_path = None
            logo_path = None

            if music_upload:

                music_path = project / (
                    "music"
                    + Path(music_upload.name).suffix.lower()
                )

                music_path.write_bytes(
                    music_upload.getbuffer()
                )

            if logo_upload:

                logo_path = project / (
                    "logo"
                    + Path(logo_upload.name).suffix.lower()
                )

                logo_path.write_bytes(
                    logo_upload.getbuffer()
                )

            progress = st.progress(0)
            status = st.empty()

            completed = []

            try:

                for i, clip in enumerate(
                    st.session_state.clips, 1
                ):

                    status.write(
                        f"Exporting clip {i} of "
                        f"{len(st.session_state.clips)}..."
                    )

                    output, _ = export_clip(
                        st.session_state.video_path,
                        clip,
                        i,
                        export_dir,
                        st.session_state.transcript,
                        (
                            "9:16"
                            if aspect_label.startswith("9:16")
                            else "16:9"
                        ),
                        burn,
                        subtitle_size,
                        subtitle_color,
                        subtitle_outline,
                        crop_mode,
                        fade_seconds,
                        music_path,
                        logo_path,
                        mix_audio,
                        music_volume,
                    )

                    completed.append(str(output))

                    progress.progress(
                        i / len(st.session_state.clips)
                    )

                    st.session_state.exported = completed.copy()

                    save_project()

                st.success("All clips exported successfully.")

            except Exception as exc:

                st.error(f"Export stopped: {exc}")

                st.info(
                    "Clips completed before the error remain saved."
                )

            status.empty()

        export_dir = (
            Path(st.session_state.project_dir) / "EXPORTS"
        )

        if export_dir.exists():

            files = sorted(
                export_dir.glob("clip_*.mp4")
            )

            for file in files:

                with st.expander(file.name):

                    st.video(str(file))

                    st.download_button(
                        f"Download {file.name}",
                        data=file.read_bytes(),
                        file_name=file.name,
                        key=f"download_{file.name}",
                    )

            if files:

                zip_path = (
                    export_dir / "AI_CLIPPER_ALL_CLIPS.zip"
                )

                if st.button("Create ZIP of all clips"):

                    with zipfile.ZipFile(
                        zip_path,
                        "w",
                        zipfile.ZIP_DEFLATED,
                    ) as archive:

                        for file in files:
                            archive.write(file, file.name)

                        for subtitle in sorted(
                            export_dir.glob("clip_*.srt")
                        ):
                            archive.write(
                                subtitle, subtitle.name
                            )

                    st.success("ZIP created.")

                if zip_path.exists():

                    st.download_button(
                        "Download all clips (ZIP)",
                        data=zip_path.read_bytes(),
                        file_name=zip_path.name,
                        key="download_all_zip",
                    )

# ----------------------- CONTENT STUDIO ---------------------

elif page == "Content Studio":

    st.header("Content studio")

    if not st.session_state.clips:

        st.warning("Find highlights first.")

    else:

        platform = st.selectbox(
            "Prepare metadata for",
            [
                "General",
                "YouTube Shorts",
                "Instagram Reels",
                "TikTok",
            ],
        )

        for i, clip in enumerate(
            st.session_state.clips, 1
        ):

            title, description, hashtags = metadata_for(
                clip["text"]
            )

            if platform == "YouTube Shorts":
                hashtags += " #Shorts"

            elif platform == "Instagram Reels":
                hashtags += " #Reels"

            elif platform == "TikTok":
                hashtags += " #FYP"

            with st.expander(
                f"Clip {i} metadata",
                expanded=(i == 1),
            ):

                st.text_input(
                    "Suggested title",
                    title,
                    key=f"meta_title_{i}",
                )

                st.text_area(
                    "Suggested description",
                    description,
                    key=f"meta_desc_{i}",
                )

                st.text_input(
                    "Hashtags",
                    hashtags,
                    key=f"meta_tags_{i}",
                )

                st.caption(
                    "Locally generated metadata drafts. "
                    "Review and edit before publishing."
                )

# ----------------------- PUBLISHING -------------------------

elif page == "Publishing":

    st.header("Publishing workspace")

    st.info(
        "This version prepares content and exports files. "
        "Direct publishing requires approved platform APIs, "
        "authentication and account permissions."
    )

    st.subheader("Publishing checklist")

    st.markdown("""
    - [ ] Review exported video and subtitles.
    - [ ] Confirm content usage rights.
    - [ ] Copy metadata from Content Studio.
    - [ ] Upload through an approved platform integration.
    """)

    st.caption(
        "Do not paste social-media passwords or API secrets here."
    )

    if st.session_state.project_dir:

        export_dir = (
            Path(st.session_state.project_dir) / "EXPORTS"
        )

        if export_dir.exists():

            files = sorted(
                export_dir.glob("clip_*.mp4")
            )

            st.write(
                f"Ready-to-review exports: {len(files)}"
            )

            for item in files:
                st.write(item.name)

# ----------------------- SETTINGS ---------------------------

elif page == "Settings":

    st.header("Settings and project information")

    st.write("**Application:**", APP_NAME)
    st.write("**AI model:** Faster-Whisper")
    st.write("**Device:** CPU")
    st.write("**Compute:** INT8")
    st.write("**Video engine:**", FFMPEG or "Not found")
    st.write("**Project storage:**", str(ROOT.resolve()))
    st.write("**Python project:** app.py")

    if st.session_state.project_dir:

        st.write("**Current project folder:**")

        st.code(st.session_state.project_dir)

        if st.button("Save project now"):

            try:
                save_project()
                st.success("Project saved.")
            except Exception as exc:
                st.error(str(exc))

    st.info(
        "Projects are stored inside AI_CLIPPER_PROJECTS. "
        "Keep this folder to preserve source videos, "
        "transcripts and exports."
    )

    st.warning(
        "Resetting the session does not delete project files. "
        "Back up your project folder before deleting it."
    )

    if st.button("Reset current project session"):

        reset_project_state()
        st.rerun()

# ----------------------- FOOTER -----------------------------

st.divider()

st.caption(
    "AI CLIPPER 5.1 | Single-file local edition | "
    "CPU INT8 | Review content before publishing."
)