import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from video_recording import Recording, cached_video


class VideoRecordingTests(unittest.TestCase):
    def test_complete_recording_reused_only_with_matching_key_and_intact_clip(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "report"
            folder.mkdir()
            report = folder / "video.json"
            report.write_text(json.dumps({"status": "completo", "frames": 3}))
            recording = Recording(root, 30)
            try:
                for _ in range(3):
                    recording.append(Image.new("RGB", (51, 35)))
                recording.finish(folder, "model-a", 3)
            finally:
                recording.close()
            clip, _ = cached_video(root, "model-a")
            self.assertIsNone(cached_video(root, "model-b"))
            data = json.loads(report.read_text())
            data["status"] = "detenido"
            report.write_text(json.dumps(data))
            self.assertIsNone(cached_video(root, "model-a"))
            data["status"] = "completo"
            report.write_text(json.dumps(data))
            clip.write_bytes(b"truncated")
            self.assertIsNone(cached_video(root, "model-a"))

    def test_interrupted_writer_discards_temporary_file(self):
        with TemporaryDirectory() as tmp:
            recording = Recording(Path(tmp), 30)
            recording.append(Image.new("RGB", (80, 60)))
            recording.close()
            self.assertEqual(list(Path(tmp).iterdir()), [])
