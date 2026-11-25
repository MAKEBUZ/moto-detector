import unittest

from traffic_video import Tracks


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
