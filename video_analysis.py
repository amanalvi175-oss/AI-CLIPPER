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
    words = re.findall(r"\b[\w']+\b", value)
    return [round(float(value), 2) for value in values[:max_points]]


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
    words = re.findall(r"\\b[\\w']+\\b", value)
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
