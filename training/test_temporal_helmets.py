import unittest
from unittest.mock import patch

from traffic_video import RiderTracks, Tracks
from temporal_head_search import LiveHelmetSearch
from helmet_flow import HelmetFlow
from edge_riders import EdgeRiders


class TemporalHelmetTests(unittest.TestCase):
    def track(self, helmet_frames, verified_frames, bare_frames=()):
        tracker = Tracks()
        for frame in range(1, 31):
            state = "indeterminado"
            if frame in helmet_frames:
                state = "con_casco"
            if frame in bare_frames:
                state = "sin_casco"
            tracker.update([dict(id=1, box=[0, 0, 40, 120], estado=state,
                                 cabezas=[{"verified_helmet": frame in verified_frames}])], frame)
        return tracker.state(1)

    def test_spaced_checks_can_resolve_sparse_observations(self):
        frames = [2, 3, 9, 10, 19, 20]
        self.assertEqual(self.track(frames, [2, 9, 19]), "con_casco")
        self.assertEqual(self.track(frames, []), "indeterminado")
        self.assertEqual(self.track(frames, [2, 9, 19], [30]), "indeterminado")

    def test_same_burst_does_not_count_as_three_checks(self):
        frames = list(range(2, 8))
        self.assertEqual(self.track(frames, frames), "indeterminado")

    def test_few_votes_and_bare_votes_do_not_trigger_recovery(self):
        self.assertEqual(self.track([1, 10, 20], [1, 10, 20]), "indeterminado")
        self.assertEqual(self.track([], [], range(1, 8)), "indeterminado")


class RiderContinuityTests(unittest.TestCase):
    @staticmethod
    def rider(index, box, head=None, state="indeterminado", verified=False, bike=1):
        heads = [] if head is None else [dict(box=head, verified_helmet=verified)]
        return dict(id=index, box=box, moto=bike, estado=state, cabezas=heads)

    def test_nested_body_does_not_inherit_later_helmet_votes(self):
        tracker = RiderTracks()
        body = self.rider(1, [0, 0, 60, 120])
        nested = self.rider(2, [10, 0, 50, 100])
        for frame in range(1, 4):
            self.assertEqual(tracker.update([body, nested], frame), {})
        rider = self.rider(2, nested["box"], [18, 5, 40, 24], "con_casco")
        for frame in range(4, 7):
            self.assertEqual(tracker.update([body, rider], frame), {2: 1})
        self.assertEqual(tracker.items[1]["first"], 4)
        self.assertEqual(tracker.state(1), "con_casco")

    def test_two_heads_and_distinct_bikes_preserve_overlapping_people(self):
        tracker = RiderTracks()
        box = [0, 0, 60, 120]
        a = self.rider(1, box, [5, 0, 20, 20], "con_casco")
        b = self.rider(2, box, [34, 0, 55, 20], "con_casco")
        ids = tracker.update([a, b], 1)
        self.assertEqual(tracker.update([b, a], 2), ids)
        separate = [self.rider(i, box, bike=i) for i in (1, 2)]
        self.assertEqual(len(tracker.update(separate, 3)), 2)

    def test_lone_headless_rider_can_still_be_counted(self):
        tracker = RiderTracks()
        for frame in range(1, 4):
            tracker.update([self.rider(1, [0, 0, 60, 120], bike=None)], frame)
        self.assertEqual(tracker.count(), 1)
        self.assertEqual(tracker.state(1), "indeterminado")

    def test_partial_rider_keeps_id_without_new_votes_and_stops_at_cut(self):
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(715)
        helmet = Image.fromarray(rng.integers(95, 245, (22, 23, 3), dtype=np.uint8))
        torso = Image.fromarray(rng.integers(20, 165, (48, 34, 3), dtype=np.uint8))
        search = EdgeRiders(30)
        tracker = RiderTracks()
        for frame in range(1, 14):
            left = 17 - 2 * frame
            image = Image.new("RGB", (150, 120), (110, 115, 120))
            if frame < 13:
                image.paste(torso, (left, 44))
                image.paste(helmet, (left + 6, 22))
            people = []
            if frame <= 4:
                people.append(self.rider(1, [left, 19, left + 34, 92],
                                         [left + 3, 20, left + 32, 46], "con_casco", True))
            ids = tracker.update(people, frame)
            search.refine(image, people, ids, tracker, frame)
            if frame == 12:
                self.assertEqual(list(ids.values()), [1])
                self.assertEqual(people[0]["cabezas"][0]["source"], "edge_flow")
                self.assertEqual(people[0]["box"][0], 0)
            if frame in (6, 13):
                self.assertFalse(people)
        self.assertEqual(tracker.count(), 1)
        self.assertEqual(tracker.items[1]["hits"], 4)
        self.assertEqual(tracker.items[1]["votes"], {"con_casco": 4})

    def test_patch_recovery_uses_same_track_and_expires(self):
        import numpy as np
        from PIL import Image

        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8))
        tracker = RiderTracks()
        search = LiveHelmetSearch(None)
        first = self.rider(1, [10, 15, 70, 95], [28, 23, 48, 43], "con_casco", verified=True)
        for n in range(1, 4):
            ids = tracker.update([first], n)
            search.refine(image, [first], ids, tracker, n)
        hidden = self.rider(1, first["box"])
        with patch("temporal_head_search.verify", return_value=.99):
            search.refine(image, [hidden], {1: 1}, tracker, 4)
            self.assertEqual(hidden["cabezas"][0]["source"], "temporal_patch_check")
            self.assertEqual(tracker.items[1]["votes"]["con_casco"], 3)
            missing = self.rider(1, first["box"])
            search.refine(image, [missing], {}, tracker, 5)
            self.assertEqual(missing["cabezas"], [])
            search.refine(image, [missing], {1: 1}, tracker, 19)
            self.assertEqual(missing["cabezas"], [])

    def test_patch_checks_cannot_confirm_a_track_from_one_verified_sighting(self):
        tracker = RiderTracks()
        for n in range(1, 11):
            rider = self.rider(1, [0, 0, 50, 100])
            if n <= 3:
                rider = self.rider(1, rider["box"], [5, 0, 25, 20], "con_casco", n == 1)
            tracker.update([rider], n)
        tracker.items[1]["patch_checks"] = 7
        self.assertEqual(tracker.state(1), "indeterminado")

    def test_short_unanimous_track_needs_a_separately_checked_frame(self):
        tracker = RiderTracks()
        for n in range(1, 11):
            rider = self.rider(1, [0, 0, 50, 100])
            if n in (1, 6, 7):
                rider = self.rider(1, rider["box"], [5, 0, 25, 20], "con_casco", n != 7)
            tracker.update([rider], n)
        self.assertEqual(tracker.state(1), "indeterminado")
        tracker.items[1]["patch_checks"] = 1
        self.assertEqual(tracker.state(1), "con_casco")
        tracker.update([self.rider(1, rider["box"], [5, 0, 25, 20], "sin_casco")], 11)
        self.assertEqual(tracker.state(1), "indeterminado")

    def test_motion_requires_body_and_current_detector_confirmation(self):
        import numpy as np
        from PIL import Image
        from types import SimpleNamespace
        texture = np.random.default_rng(53).integers(0, 255, (140, 120, 3), dtype=np.uint8)
        images = [Image.fromarray(np.roll(texture, n, axis=1)) for n in range(6)]
        tracker = RiderTracks()
        flow = HelmetFlow(SimpleNamespace(head_model=None), 60)
        for n in range(3):
            rider = self.rider(1, [4, 10, 94, 135], [30+n, 20, 60+n, 50], "con_casco", True)
            ids = tracker.update([rider], n+1)
            flow.refine(images[n], [rider], ids, tracker, n+1)
        hits = tracker.items[1]["hits"]
        with patch("helmet_flow.verify", return_value=.99), patch("helmet_flow.redetect") as check:
            check.return_value = False
            body = self.rider(7, rider["box"])
            ids = {}
            flow.refine(images[3], [body], ids, tracker, 4)
            self.assertFalse(body["cabezas"])
            self.assertFalse(ids)
            check.return_value = True
            flow.refine(images[4], [], {}, tracker, 5)
            self.assertEqual(tracker.items[1]["hits"], hits)
            flow.refine(images[5], [body], ids, tracker, 6)
            self.assertEqual(ids, {7: 1})
            self.assertEqual(body["cabezas"][0]["source"], "helmet_flow")
            self.assertEqual(tracker.count(), 1)
            self.assertEqual(tracker.items[1]["flow_returns"], 1)
            self.assertNotIn("flow_checks", tracker.items[1])

    def test_contextual_bare_evidence_vetoes_sparse_motion_confirmation(self):
        tracker = RiderTracks()
        for n in range(1, 41):
            rider = self.rider(1, [0, 0, 50, 100])
            if n in (1, 2, 3, 7, 8, 9, 10):
                rider = self.rider(1, rider["box"], [5, 0, 25, 20], "con_casco", True)
            if n in (25, 32):
                rider = self.rider(1, rider["box"], [5, 0, 25, 20], "sin_casco")
            tracker.update([rider], n)
        track = tracker.items[1]
        track.update(flow_checks=6, flow_anchors={1, 7})
        self.assertEqual(tracker.state(1), "con_casco")
        track.update(first_context_bare=25, last_context_bare=32)
        self.assertEqual(tracker.state(1), "indeterminado")

    def test_short_burst_cannot_confirm_bare_head(self):
        tracker = RiderTracks()
        rider = self.rider(1, [0, 0, 50, 140], [5, 0, 25, 20], "sin_casco")
        for frame in (1, 3, 4):
            tracker.update([rider], frame)
        self.assertEqual(tracker.state(1), "indeterminado")
        self.assertEqual(tracker.evidence()[0]["motivos"], ["votos_sin_casco_demasiado_proximos"])
        tracker.update([rider], 7)
        self.assertEqual(tracker.state(1), "sin_casco")

    def test_sparse_final_helmet_requires_spaced_checks_and_no_later_conflict(self):
        tracker = RiderTracks()
        for frame in range(1, 19):
            state = "con_casco" if frame in (2, 3, 8, 9, 15, 16) else "indeterminado"
            head = None if state == "indeterminado" else [5, 0, 20, 20]
            rider = self.rider(1, [0, 0, 60, 120], head, state, frame in (2, 15))
            tracker.update([rider], frame)
        self.assertEqual(tracker.state(1), "con_casco")
        conflict = self.rider(1, rider["box"], [5, 0, 20, 20], "sin_casco")
        tracker.update([conflict], 19)
        self.assertEqual(tracker.state(1), "indeterminado")
