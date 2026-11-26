import unittest
from unittest.mock import patch, Mock

from video_analysis import build_scene_clips, detect_scene_changes, detect_silence_ranges, score_highlight_text, build_face_crop_expression


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



    def test_keep_ranges_remove_silence_but_preserve_pause_edges(self):
        from video_analysis import build_keep_ranges
        keep = build_keep_ranges(0, 20, [{"start": 5, "end": 10}], pad=0.2)
        self.assertEqual(keep, [(0.0, 5.2), (9.8, 20.0)])

    def test_transcript_is_remapped_after_silence_removal(self):
        from video_analysis import map_transcript_to_keep_ranges
        source = [
            {"start": 1, "end": 3, "text": "hello"},
            {"start": 8, "end": 11, "text": "world"},
        ]
        result = map_transcript_to_keep_ranges(source, [(0, 5), (10, 15)])
        self.assertEqual(result[0], {"start": 1.0, "end": 3.0, "text": "hello"})
        self.assertEqual(result[1], {"start": 5.0, "end": 6.0, "text": "world"})

    @patch("video_analysis.subprocess.run")
    def test_scene_change_timestamps_are_parsed_and_deduplicated(self, run):
        run.return_value = Mock(stderr="pts_time:1.234\\npts_time:1.300\\npts_time:3.500\\n")
        result = detect_scene_changes("ffmpeg", "video.mp4", max_points=10)
        self.assertEqual(result, [1.23, 3.5])

    def test_highlight_score_counts_words(self):
        score, reasons = score_highlight_text("You should learn why this important lesson matters.")
        self.assertGreater(score, 0)
        self.assertTrue(any("Substantial spoken content" in reason for reason in reasons))

    def test_face_crop_expression_tracks_multiple_positions(self):
        expression = build_face_crop_expression([(0, 0.25), (2, 0.75), (4, 0.5)])
        self.assertIn("if(lt(t,2.000)", expression)
        self.assertIn("in_w-out_w", expression)
        self.assertTrue(expression.startswith("max(0,min("))

    def test_face_crop_expression_handles_no_detection(self):
        self.assertIsNone(build_face_crop_expression([]))


if __name__ == "__main__":
    unittest.main()
