import re
import subprocess


def detect_scene_changes(ffmpeg, video_path, threshold=0.35, max_points=200):
    """Return approximate scene-change timestamps using FFmpeg scene scores."""
    process = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path),
         "-vf", f"select='gt(scene,{float(threshold):.2f})',showinfo",
         "-an", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    timestamps = re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", process.stderr)
    points = []
    for timestamp in timestamps:
        point = float(timestamp)
        if not points or point - points[-1] >= 0.25:
            points.append(point)
        if len(points) >= int(max_points):
            break
    return [round(point, 2) for point in points]


def detect_silence_ranges(ffmpeg, video_path, noise_db=-35, min_duration=0.5, total_duration=0):
    """Return detected silence intervals; handles an unfinished trailing silence."""
    process = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video_path),
         "-af", f"silencedetect=noise={int(noise_db)}dB:d={float(min_duration):.2f}",
         "-vn", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    starts = re.findall(r"silence_start: ([0-9.]+)", process.stderr)
    ends = re.findall(r"silence_end: ([0-9.]+)", process.stderr)
    output = []
    for index, start in enumerate(starts):
        end = float(ends[index]) if index < len(ends) else float(total_duration)
        start = float(start)
        output.append({"start": round(start, 2), "end": round(end, 2),
                       "duration": round(max(0, end - start), 2)})
    return output


def build_scene_clips(scene_points, total_duration, before=2.0, after=18.0, limit=10):
    """Turn scene boundaries into reviewable candidate clip windows."""
    clips = []
    for point in scene_points:
        start = max(0.0, float(point) - float(before))
        end = min(float(total_duration), float(point) + float(after))
        if end - start >= 5:
            clips.append({"start": round(start, 2), "end": round(end, 2),
                          "score": 0.0, "text": "Scene-change candidate",
                          "coverage": 0.0})
        if len(clips) >= int(limit):
            break
    return clips



HOOK_PHRASES = (
    "the biggest mistake", "here is why", "here's why",
    "the truth is", "what if", "how to", "you need to",
    "stop doing", "never do", "the reason", "this is why",
    "the secret", "most people", "nobody tells you",
)
ENGAGEMENT_WORDS = {
    "you", "your", "because", "but", "however", "actually",
    "secret", "mistake", "warning", "important", "truth",
    "best", "worst", "never", "always", "why", "how",
    "learn", "tip", "lesson", "solution", "avoid", "stop",
    "start", "change", "success", "fail", "money",
}


def score_highlight_text(text):
    """Return a transparent editorial score; it is not a prediction of virality."""
    value = str(text).lower()
    words = re.findall(r"\b[\w']+\b", value)
    reasons = []
    score = min(len(words) / 12.0, 2.5)

    if len(words) >= 8:
        reasons.append("Substantial spoken content")
    if any(phrase in value for phrase in HOOK_PHRASES):
        score += 2.0
        reasons.append("Contains a strong hook phrase")
    matched = sorted({word for word in words if word in ENGAGEMENT_WORDS})
    if matched:
        score += min(len(matched) * 0.45, 2.25)
        reasons.append("Engaging language: " + ", ".join(matched[:4]))
    if "?" in value:
        score += 0.8
        reasons.append("Uses a question to create curiosity")
    if "!" in value:
        score += 0.4
        reasons.append("Emphatic delivery in transcript")
    if any(term in value for term in ("first", "second", "third", "step", "number one")):
        score += 0.5
        reasons.append("Has a clear list or step structure")

    return round(score, 2), reasons


def build_keep_ranges(start, end, silence_ranges, pad=0.15, min_gap=0.25):
    """Return source-time intervals to retain, preserving a small pause around speech."""
    start, end = float(start), float(end)
    if end <= start:
        return []
    cuts = []
    for item in silence_ranges:
        silence_start = max(start, float(item["start"]) + float(pad))
        silence_end = min(end, float(item["end"]) - float(pad))
        if silence_end - silence_start >= float(min_gap):
            cuts.append((silence_start, silence_end))
    cuts.sort()
    merged = []
    for left, right in cuts:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    keep = []
    cursor = start
    for left, right in merged:
        if left > cursor:
            keep.append((round(cursor, 3), round(left, 3)))
        cursor = max(cursor, right)
    if cursor < end:
        keep.append((round(cursor, 3), round(end, 3)))
    return [(a, b) for a, b in keep if b - a >= 0.08]


def map_transcript_to_keep_ranges(segments, keep_ranges):
    """Map original transcript timestamps onto the compacted output timeline."""
    mapped = []
    output_cursor = 0.0
    for keep_start, keep_end in keep_ranges:
        for segment in segments:
            left = max(float(segment["start"]), keep_start)
            right = min(float(segment["end"]), keep_end)
            if right > left:
                mapped.append({
                    "start": round(output_cursor + left - keep_start, 3),
                    "end": round(output_cursor + right - keep_start, 3),
                    "text": segment.get("text", ""),
                })
        output_cursor += keep_end - keep_start
    return sorted(mapped, key=lambda item: item["start"])


def compact_video_with_ffmpeg(ffmpeg, video_path, keep_ranges, destination, include_audio=True):
    """Render retained ranges and concatenate them into a single MP4."""
    if not keep_ranges:
        raise ValueError("No video sections remain after silence removal.")
    graph = []
    labels = []
    for index, (start, end) in enumerate(keep_ranges):
        graph.append(
            f"[0:v]trim=start={float(start):.3f}:end={float(end):.3f},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        if include_audio:
            graph.append(
                f"[0:a:0]atrim=start={float(start):.3f}:end={float(end):.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
            labels.extend([f"[v{index}]", f"[a{index}]"])
        else:
            labels.append(f"[v{index}]")
    if include_audio:
        graph.append(
            "".join(labels) + f"concat=n={len(keep_ranges)}:v=1:a=1[vout][aout]"
        )
    else:
        graph.append(
            "".join(labels) + f"concat=n={len(keep_ranges)}:v=1:a=0[vout]"
        )
    command = ["-y", "-i", str(video_path), "-filter_complex", ";".join(graph),
               "-map", "[vout]"]
    if include_audio:
        command += ["-map", "[aout]"]
    command += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-pix_fmt", "yuv420p"]
    if include_audio:
        command += ["-c:a", "aac", "-b:a", "128k"]
    command += ["-movflags", "+faststart", str(destination)]
    subprocess.run([ffmpeg, *command], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   text=True, encoding="utf-8", errors="replace", check=True)
    return destination



def detect_face_centers(video_path, start_time=0.0, end_time=None,
                        sample_interval=0.75, max_samples=180):
    """Detect the largest visible face at sampled times; returns (local_time, x_center)."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Face tracking needs opencv-python-headless. Install project requirements.") from exc
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open video for face detection.")
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        capture.release()
        raise RuntimeError("OpenCV face detector could not be loaded.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if frame_count else 0.0
    start = max(0.0, float(start_time))
    finish = min(float(end_time), duration) if end_time is not None and duration else (float(end_time) if end_time is not None else duration)
    if finish <= start:
        capture.release()
        return []
    count = min(int(max_samples), max(1, int((finish - start) / max(0.1, sample_interval)) + 1))
    step = (finish - start) / max(1, count - 1)
    points = []
    try:
        for index in range(count):
            timestamp = start + index * step
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            height, width = frame.shape[:2]
            if width > 960:
                scale = 960.0 / width
                frame = cv2.resize(frame, (960, max(1, int(height * scale))))
                height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(32, 32))
            if len(faces):
                x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
                center = max(0.0, min(1.0, (float(x) + float(w) / 2.0) / width))
                points.append((round(timestamp - start, 3), round(center, 4)))
    finally:
        capture.release()
    return points


def build_face_crop_expression(face_points):
    """Build an FFmpeg crop-x expression from local-time normalized face centers."""
    points = sorted((max(0.0, float(t)), max(0.0, min(1.0, float(x)))) for t, x in face_points)
    if not points:
        return None
    if len(points) == 1:
        return f"max(0,min(in_w-out_w,{points[0][1]:.4f}*in_w-out_w/2))"
    expression = f"{points[-1][1]:.4f}*in_w-out_w/2"
    for index in range(len(points) - 2, -1, -1):
        t0, x0 = points[index]
        t1, x1 = points[index + 1]
        if t1 <= t0:
            continue
        interpolated = f"({x0:.4f}+({x1-x0:.4f})*(t-{t0:.3f})/{t1-t0:.3f})*in_w-out_w/2"
        expression = f"if(lt(t,{t1:.3f}),{interpolated},{expression})"
    first_t, first_x = points[0]
    expression = f"if(lt(t,{first_t:.3f}),{first_x:.4f}*in_w-out_w/2,{expression})"
    return f"max(0,min(in_w-out_w,{expression}))"
