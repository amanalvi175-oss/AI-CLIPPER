
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
from video_analysis import detect_scene_changes, detect_silence_ranges, build_scene_clips, score_highlight_text, build_keep_ranges, map_transcript_to_keep_ranges, compact_video_with_ffmpeg, detect_face_centers, build_face_crop_expression

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
        word_timestamps=True,
    )

    results = []

    for segment in segments_iter:
        word_items = []
        for word in (getattr(segment, "words", None) or []):
            word_items.append({"start": float(word.start), "end": float(word.end), "word": str(word.word)})
        results.append({
            "start": float(segment.start),
            "end": float(segment.end),
            "text": segment.text.strip(),
            "words": word_items,
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
    score, _ = score_highlight_text(segment.get("text", ""))
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


def write_word_ass(segments, destination, offset=0.0, end=None, font_size=30, primary="#FFFFFF", highlight="#35E6FF"):
    """Render timestamped words as individually highlighted ASS subtitle events."""
    def ass_rgb(value):
        value = str(value).lstrip("#")
        if len(value) != 6: value = "FFFFFF"
        return "&H00" + value[4:6] + value[2:4] + value[0:2]
    def stamp(value):
        value = max(0.0, float(value))
        return f"{int(value//3600)}:{int((value%3600)//60):02}:{value%60:05.2f}"
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 720\nPlayResY: 1280\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Word,Arial,{int(font_size)},{ass_rgb(primary)},{ass_rgb(highlight)},&H00000000,&H80000000,-1,0,1,3,1,2,35,35,105,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    events = []
    for segment in segments:
        words = segment.get("words") or []
        if words:
            for word in words:
                left = max(0.0, float(word.get("start", 0)) - offset)
                right = float(word.get("end", 0)) - offset
                token = str(word.get("word", "")).strip()
                if end is not None: right = min(right, float(end) - offset)
                if token and right > left:
                    events.append((left, right, token))
        else:
            left = max(0.0, float(segment.get("start", 0)) - offset)
            right = float(segment.get("end", 0)) - offset
            token = str(segment.get("text", "")).strip()
            if end is not None: right = min(right, float(end) - offset)
            if token and right > left: events.append((left, right, token))
    for left, right, token in sorted(events):
        safe = token.replace("{", "(").replace("}", ")")
        lines.append(f"Dialogue: 0,{stamp(left)},{stamp(right)},Word,,0,0,0,,{{\\c{ass_rgb(highlight)}\\fscx108\\fscy108}}{safe}\n")
    Path(destination).write_text("".join(lines), encoding="utf-8-sig")
    return destination


def create_thumbnail(ffmpeg, video_path, timestamp, destination):
    """Extract a representative frame for a clip thumbnail."""
    subprocess.run([ffmpeg, "-y", "-ss", f"{max(0.0, float(timestamp)):.3f}", "-i", str(video_path),
                    "-frames:v", "1", "-q:v", "2", str(destination)],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                   text=True, encoding="utf-8", errors="replace")
    if not Path(destination).exists() or Path(destination).stat().st_size == 0:
        raise RuntimeError("Thumbnail frame could not be generated.")
    return destination


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
    mix_original_audio=True, music_volume=0.12, crop_position=0.5, face_follow=False,
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
            crop_x = f"(in_w-out_w)*{max(0.0, min(1.0, float(crop_position))):.3f}"
            if face_follow:
                try:
                    face_points = detect_face_centers(video, start, end)
                    dynamic_x = build_face_crop_expression(face_points)
                    if dynamic_x:
                        crop_x = dynamic_x
                except Exception as face_error:
                    st.warning(f"Face-follow framing unavailable; using manual crop. {face_error}")
            graph = (
                "[0:v]scale=720:1280:"
                "force_original_aspect_ratio=increase,"
                f"crop=720:1280:x='{crop_x}':y=(in_h-out_h)/2,setsar=1[base]"
            )
    else:
        graph = (
            "[0:v]scale=1280:720:"
            "force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1[base]"
        )

    current = "base"
    zoom_effect = bool(st.session_state.get("export_zoom", False))
    progress_bar = bool(st.session_state.get("export_progress", False))
    emoji_accent = bool(st.session_state.get("export_emoji", False))

    if zoom_effect:
        graph += f";[{current}]scale=iw*1.06:ih*1.06,crop=iw/1.06:ih/1.06,setsar=1[zoomed]"
        current = "zoomed"

    if progress_bar:
        graph += f";[{current}]drawbox=x=0:y=ih-10:w='iw*t/{max(duration, 0.1):.3f}':h=10:color=0x35E6FF@0.95:t=fill[progress]"
        current = "progress"

    if emoji_accent:
        graph += f";[{current}]drawtext=text='✦':fontcolor=0x35E6FF:fontsize=34:x=w-70:y=70:enable='between(t,0,2)'[accent]"
        current = "accent"

    if burn_subtitles:
        word_ass = export_dir / f"clip_{number:02}_words.ass"
        write_word_ass(all_segments, word_ass, offset=start, end=end,
                       font_size=max(18, int(subtitle_size) + 8),
                       primary=subtitle_color)
        sub_path = ffmpeg_filter_path(word_ass)
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
        command += ["-loop", "1", "-i", str(logo_path)]

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
            "Video Analysis",
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


# ----------------------- VIDEO ANALYSIS PAGE -----------------

elif page == "Video Analysis":
    st.header("Video intelligence")
    if not st.session_state.video_path:
        st.warning("Import or reopen a project first.")
    else:
        st.caption("FFmpeg scene-change and silence analysis. Results are estimates; review before editing.")
        c1, c2 = st.columns(2)
        with c1:
            threshold = st.slider("Scene sensitivity", 0.15, 0.80, 0.35, 0.05)
        with c2:
            noise = st.slider("Silence threshold (dB)", -60, -15, -35, 1)
        min_silence = st.slider("Minimum silence duration", 0.2, 3.0, 0.5, 0.1)
        if st.button("Analyse scenes and silence", type="primary"):
            try:
                with st.spinner("Analysing video with FFmpeg..."):
                    st.session_state.scene_points = detect_scene_changes(FFMPEG, st.session_state.video_path, threshold)
                    duration = get_duration(st.session_state.video_path)
                    st.session_state.silence_ranges = detect_silence_ranges(FFMPEG, st.session_state.video_path, noise, min_silence, duration)
                st.success("Video analysis complete.")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
        scenes = st.session_state.get("scene_points", [])
        silences = st.session_state.get("silence_ranges", [])
        if scenes:
            st.subheader("Detected scene changes")
            st.write(", ".join(f"{p:.2f}s" for p in scenes))
            if st.button("Create candidate clips from scenes"):
                st.session_state.clips = build_scene_clips(scenes, get_duration(st.session_state.video_path))
                save_project()
                st.success(f"Created {len(st.session_state.clips)} scene-based candidates. Review them in Clip Editor & Export.")
        if silences:
            st.subheader("Detected silent sections")
            st.dataframe(silences, use_container_width=True)
        if not scenes and not silences:
            st.info("Run the analysis to see scene changes and silent sections.")

# ----------------------- CLIP EDITOR ------------------------

elif page == "Clip Editor & Export":

    st.header("Clip editor and export")

    if not st.session_state.clips:

        st.warning("Find highlights first.")

    else:

        st.subheader("Trim and review clips")
        try:
            source_duration = get_duration(st.session_state.video_path)
        except Exception:
            source_duration = max(float(c.get("end", 0)) for c in st.session_state.clips)

        st.caption("Adjust each clip's start and end time before exporting. Changes are saved to this project.")
        for clip_index, clip in enumerate(st.session_state.clips):
            clip_start = max(0.0, min(float(clip.get("start", 0.0)), source_duration))
            clip_end = max(clip_start, min(float(clip.get("end", source_duration)), source_duration))
            with st.expander(
                f"Edit clip {clip_index + 1} | {clip_start:.1f}s–{clip_end:.1f}s",
                expanded=False,
            ):
                start_col, end_col = st.columns(2)
                with start_col:
                    edited_start = st.number_input(
                        "Start time (seconds)",
                        min_value=0.0,
                        max_value=float(max(0.1, source_duration)),
                        value=float(clip_start),
                        step=0.1,
                        key=f"trim_start_{clip_index}",
                    )
                with end_col:
                    edited_end = st.number_input(
                        "End time (seconds)",
                        min_value=0.0,
                        max_value=float(max(0.1, source_duration)),
                        value=float(clip_end),
                        step=0.1,
                        key=f"trim_end_{clip_index}",
                    )
                if st.button("Apply trim", key=f"apply_trim_{clip_index}"):
                    if edited_end <= edited_start:
                        st.error("End time must be greater than start time.")
                    else:
                        st.session_state.clips[clip_index]["start"] = float(edited_start)
                        st.session_state.clips[clip_index]["end"] = float(edited_end)
                        save_project()
                        st.success("Clip timing updated and saved.")

        aspect_label = st.selectbox(
            "Output format",
            ["9:16 vertical", "16:9 landscape"],
        )

        crop_mode = "Center crop"
        crop_position = 0.5

        if aspect_label.startswith("9:16"):

            crop_mode = st.selectbox(
                "Vertical framing",
                [
                    "Center crop",
                    "Fit with blurred background",
                ],
            )
            face_follow = False
            if crop_mode == "Center crop":
                crop_position = st.slider(
                    "Horizontal subject position",
                    0, 100, 50, 1,
                    help="Manual fallback position when face-follow is disabled or no face is detected.",
                ) / 100.0
                face_follow = st.checkbox(
                    "Experimental AI face-follow framing",
                    value=False,
                    help="Uses lightweight CPU face detection to keep the largest visible face in frame. Sampling may miss profiles, poor lighting or multiple speakers.",
                )

        remove_silence = st.checkbox(
            "Automatically remove long silent gaps",
            value=False,
            help="Cuts detected silence while keeping a short pause around speech. This adds processing time.",
        )
        silence_threshold = -35
        silence_minimum = 0.7
        if remove_silence:
            silence_threshold = st.slider(
                "Silence detection threshold (dB)", -60, -15, -35, 1
            )
            silence_minimum = st.slider(
                "Only remove gaps longer than (seconds)", 0.3, 3.0, 0.7, 0.1
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

        zoom_effect = st.checkbox("Subtle cinematic zoom", value=False)
        progress_bar = st.checkbox("Add animated bottom progress bar", value=True)
        emoji_accent = st.checkbox("Add sparkle accent overlay", value=False,
                                   help="Uses FFmpeg drawtext; may depend on available fonts.")

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

                silence_ranges = []
                if remove_silence:
                    if has_audio(st.session_state.video_path):
                        with st.spinner("Detecting silent sections..."):
                            source_duration = get_duration(st.session_state.video_path)
                            silence_ranges = detect_silence_ranges(
                                FFMPEG, st.session_state.video_path,
                                silence_threshold, silence_minimum, source_duration
                            )
                    else:
                        st.warning("This video has no audio track; silence removal was skipped.")

                st.session_state["export_zoom"] = zoom_effect
                st.session_state["export_progress"] = progress_bar
                st.session_state["export_emoji"] = emoji_accent

                for i, clip in enumerate(
                    st.session_state.clips, 1
                ):

                    status.write(
                        f"Exporting clip {i} of "
                        f"{len(st.session_state.clips)}..."
                    )

                    source_for_export = st.session_state.video_path
                    clip_for_export = dict(clip)
                    transcript_for_export = st.session_state.transcript
                    compacted_path = export_dir / f"clip_{i:02}_compacted.mp4"

                    if remove_silence and silence_ranges:
                        keep_ranges = build_keep_ranges(
                            clip["start"], clip["end"], silence_ranges, pad=0.15,
                            min_gap=silence_minimum
                        )
                        kept_duration = sum(b - a for a, b in keep_ranges)
                        original_duration = float(clip["end"]) - float(clip["start"])
                        if keep_ranges and kept_duration < original_duration - 0.05:
                            with st.spinner(f"Removing silent gaps from clip {i}..."):
                                compact_video_with_ffmpeg(
                                    FFMPEG, st.session_state.video_path,
                                    keep_ranges, compacted_path, include_audio=True
                                )
                            source_for_export = str(compacted_path)
                            clip_for_export = {"start": 0.0, "end": kept_duration}
                            transcript_for_export = map_transcript_to_keep_ranges(
                                st.session_state.transcript, keep_ranges
                            )

                    output, _ = export_clip(
                        source_for_export,
                        clip_for_export,
                        i,
                        export_dir,
                        transcript_for_export,                        (
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
                        crop_position,
                        face_follow if aspect_label.startswith("9:16") and crop_mode == "Center crop" else False,
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

                if st.button("Generate thumbnail", key=f"thumb_{i}"):
                    try:
                        thumb_dir = Path(st.session_state.project_dir) / "THUMBNAILS"
                        thumb_dir.mkdir(exist_ok=True)
                        thumb_path = thumb_dir / f"clip_{i:02}_thumbnail.jpg"
                        create_thumbnail(FFMPEG, st.session_state.video_path,
                                         (float(clip["start"]) + float(clip["end"])) / 2,
                                         thumb_path)
                        st.image(str(thumb_path), caption=f"Thumbnail for clip {i}")
                        st.download_button("Download thumbnail", thumb_path.read_bytes(),
                                           file_name=thumb_path.name, mime="image/jpeg",
                                           key=f"thumb_dl_{i}")
                    except Exception as exc:
                        st.error(f"Thumbnail generation failed: {exc}")

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
    "AI CLIPPER 5.2 | Word captions and thumbnail tools | "
    "CPU INT8 | Review content before publishing."
)