import unittest

from traffic_video import RiderTracks, Tracks


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
