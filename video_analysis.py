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
    values = re.findall(r"pts_time:([0-9.]+)", process.stderr)
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
