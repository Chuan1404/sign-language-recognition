import unittest
from unittest.mock import patch

import numpy as np
from flask import Flask

from app.candidate_picker import CandidatePicker
from app.index import SignDetectionApp


CANDIDATES = [{"gloss": word, "probability": probability}
              for word, probability in zip(["book", "drink", "help", "yes", "no"],
                                           [0.5, 0.2, 0.1, 0.08, 0.05])]


class PickerTests(unittest.TestCase):
    def setUp(self):
        self.picker = CandidatePicker()
        self.picker.show(CANDIDATES, 0)

    def point(self, index):
        x, y, w, h = self.picker.boxes(640, 480)[index]
        return x + w // 2, y + h // 2

    def update(self, point, now, hand="right"):
        return self.picker.update(point, hand, now, 640, 480)

    def test_mirrored_fingertip_and_left_hand(self):
        left = np.zeros((21, 3))
        left[8, :2] = [0.75, 0.5]
        self.assertEqual(self.picker.pointer(left, left, True, False, 640, 480),
                         ((159, 240), "left"))

    def test_dwell_does_not_confirm_early_and_only_confirms_once(self):
        point = self.point(1)
        self.assertEqual(self.update(point, 0.1), (False, None))
        self.assertEqual(self.update(point, 0.6), (False, None))
        self.assertEqual(self.update(point, 1.5), (False, None))
        self.assertEqual(self.update(point, 1.7), (True, "drink"))
        self.assertEqual(self.update(point, 4), (False, None))

    def test_lost_hand_target_change_and_hand_switch_reset_progress(self):
        self.update(self.point(0), 0.6)
        self.update(None, 1.4, None)
        self.assertEqual(self.update(self.point(0), 1.5), (False, None))
        self.update(self.point(1), 2)
        self.assertEqual(self.update(self.point(1), 2.9, "left"), (False, None))
        self.assertEqual(self.update(self.point(1), 3.8, "left"), (False, None))
        self.assertEqual(self.update(self.point(1), 4, "left"), (True, "drink"))

    def test_cancel_and_clear(self):
        point = self.point(5)
        self.update(point, 0.6)
        self.assertEqual(self.update(point, 1.7), (True, None))
        self.picker.show(CANDIDATES, 2)
        self.update(self.point(0), 2.6)
        self.picker.clear()
        self.assertEqual(self.update(point, 4), (False, None))

    def test_overlay_stays_in_frame(self):
        for x, y, w, h in self.picker.boxes(640, 480):
            self.assertLessEqual(x + w, 640)
            self.assertLessEqual(y + h, 480)
        image = np.zeros((480, 640, 3), np.uint8)
        self.assertGreater(self.picker.draw(image, self.point(0)).sum(), 0)


class StreamTests(unittest.TestCase):
    def test_stream_waits_for_selection_and_requires_release(self):
        clock = [0.0]

        class Camera:
            index = -1

            def read(self):
                self.index += 1
                clock[0] = self.index / 10 if self.index < 17 else [2.5, 3.6, 4.5, 4.7, 4.9][self.index - 17]
                return True, np.zeros((480, 640, 3), np.uint8)

        camera = Camera()

        class Service:
            calls = 0

            def detect(self, frame, timestamp):
                pose = np.zeros((33, 3), np.float32)
                pose[11, :2] = [0.4, 0.4]
                pose[12, :2] = [0.6, 0.4]
                hand = np.zeros((21, 3), np.float32)
                in_zone = camera.index <= 5 or camera.index in (17, 18, 19, 21)
                hand[0, :2] = [0.5, 0.5] if in_zone else [0.95, 0.95]
                # Mirrored display point lies in the second candidate row.
                hand[8, :2] = [(639 - 150) / 640, 140 / 480]
                return (None, np.zeros_like(hand), hand, False, True), (None, pose, True)

            def predict_topk(self, *args):
                self.calls += 1
                return CANDIDATES

            def draw(self, frame, *args):
                return frame

        service = Service()
        application = SignDetectionApp(Flask(__name__), service, camera, [], ".")
        with patch("app.index.time.monotonic", side_effect=lambda: clock[0]):
            stream = application.generate_frames()
            for _ in range(17):
                next(stream)
            self.assertTrue(application.picker.active)
            self.assertEqual(application.predicted_text, [])
            next(stream)
            next(stream)
            self.assertEqual(application.predicted_text, ["drink"])
            self.assertFalse(application.picker.active)
            next(stream)
            self.assertTrue(application.wait_for_release)
            next(stream)
            self.assertFalse(application.wait_for_release)
            next(stream)
            self.assertEqual(service.calls, 1)
            self.assertEqual(application.predicted_text, ["drink"])
            stream.close()
        application.picker.show(CANDIDATES, 5)
        response = application.app.test_client().post("/clear_text")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(application.picker.active)
        self.assertEqual(application.predicted_text, [])

    def test_short_zone_entries_never_call_model(self):
        # A slow camera may spend seconds in the exit grace period; that time
        # must not turn a brief intrusion into a valid sign. Repeated brief
        # entries must not accumulate across gaps either.
        for times, inside in [
            ([0, 0.1] + list(range(1, 12)), {0, 1}),
            ([0, 0.2, 0.3, 0.4, 0.6] + list(range(1, 12)), {0, 1, 3, 4}),
        ]:
            with self.subTest(times=times):
                clock = [0.0]

                class Camera:
                    index = -1

                    def read(self):
                        self.index += 1
                        if self.index == len(times):
                            return False, None
                        clock[0] = times[self.index]
                        return True, np.zeros((480, 640, 3), np.uint8)

                camera = Camera()

                class Service:
                    def detect(self, frame, timestamp):
                        pose = np.zeros((33, 3), np.float32)
                        pose[11, :2] = [0.4, 0.4]
                        pose[12, :2] = [0.6, 0.4]
                        hand = np.zeros((21, 3), np.float32)
                        hand[0, :2] = [0.5, 0.5] if camera.index in inside else [0.95, 0.95]
                        return (None, np.zeros_like(hand), hand, False, True), (None, pose, True)

                    def predict_topk(self, *args):
                        raise AssertionError("A brief zone entry must not call the model")

                    def draw(self, frame, *args):
                        return frame

                application = SignDetectionApp(Flask(__name__), Service(), camera, [], ".")
                with patch("app.index.time.monotonic", side_effect=lambda: clock[0]):
                    list(application.generate_frames())
                self.assertFalse(application.picker.active)
                self.assertEqual(application.predicted_text, [])
                self.assertFalse(application.wait_for_release)


if __name__ == "__main__":
    unittest.main()
