import unittest
from unittest.mock import patch, Mock

from video_analysis import build_scene_clips, detect_silence_ranges, score_highlight_text


class VideoAnalysisTests(unittest.TestCase):
    def test_scene_points_create_bounded_clips(self):
        clips = build_scene_clips([1, 20, 95], total_duration=100, before=2, after=10, limit=5)
        self.assertEqual(len(clips), 3)
        self.assertEqual(clips[0]["start"], 0.0)
        self.assertEqual(clips[-1]["end"], 100.0)
        self.assertTrue(all(item["end"] > item["start"] for item in clips))

    @patch("video_analysis.subprocess.run")
    def test_silence_intervals_are_parsed(self, run):
        run.return_value = Mock(stderr="silence_start: 2.000\nsilence_end: 4.500\nsilence_start: 8.000\n")
        result = detect_silence_ranges("ffmpeg", "video.mp4", total_duration=12)
        self.assertEqual(result[0], {"start": 2.0, "end": 4.5, "duration": 2.5})
        self.assertEqual(result[1], {"start": 8.0, "end": 12.0, "duration": 4.0})


    def test_highlight_score_rewards_hook_and_explains_signals(self):
        plain_score, _ = score_highlight_text("A short update about the meeting today.")
        hook_score, reasons = score_highlight_text(
            "Here is why you should never make this mistake! What if you could learn the secret?"
        )
        self.assertGreater(hook_score, plain_score)
        self.assertTrue(any("hook" in reason.lower() for reason in reasons))
        self.assertTrue(any("question" in reason.lower() for reason in reasons))



if __name__ == "__main__":
    unittest.main()
