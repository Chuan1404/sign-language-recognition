"""Camera selection in mirrored display coordinates, timed independently of FPS."""
import cv2 as cv


class CandidatePicker:
    HOLD_SECONDS = 1.0
    ARM_SECONDS = 0.5

    def __init__(self):
        self.clear()

    def clear(self):
        self.candidates = []
        self.hovered = None
        self.hover_started = None
        self.pointer_hand = None
        self.ready_at = 0.0
        self.progress = 0.0

    @property
    def active(self):
        return bool(self.candidates)

    def show(self, candidates, now):
        self.clear()
        self.candidates = list(candidates[:5])
        self.ready_at = now + self.ARM_SECONDS

    def boxes(self, width, height):
        count = len(self.candidates) + 1  # Last row cancels the prediction.
        gap = 8
        row_height = min(52, (height - 76 - gap * (count - 1)) // count)
        return [(12, 60 + i * (row_height + gap), min(285, width - 24), row_height)
                for i in range(count)]

    @staticmethod
    def pointer(left, right, left_detected, right_detected, width, height):
        # Follow one hand consistently; a hand change restarts the dwell timer.
        if right_detected:
            hand, name = right, "right"
        elif left_detected:
            hand, name = left, "left"
        else:
            return None, None
        x, y = hand[8, :2]  # Index fingertip, before horizontal flip.
        if not (0 <= x <= 1 and 0 <= y <= 1):
            return None, None
        return (min(width - 1, max(0, int(width - 1 - x * width))),
                min(height - 1, max(0, int(y * height)))), name

    def update(self, pointer, hand, now, width, height):
        """Return (confirmed, word); word=None means cancel."""
        target = None
        if self.active and pointer is not None and now >= self.ready_at:
            px, py = pointer
            for index, (x, y, w, h) in enumerate(self.boxes(width, height)):
                if x <= px < x + w and y <= py < y + h:
                    target = index
                    break
        if target != self.hovered or hand != self.pointer_hand:
            self.hovered = target
            self.pointer_hand = hand
            self.hover_started = now if target is not None else None
        self.progress = 0.0
        if target is None:
            return False, None
        self.progress = min(1.0, (now - self.hover_started) / self.HOLD_SECONDS)
        if self.progress < 1.0:
            return False, None
        word = (self.candidates[target]["gloss"]
                if target < len(self.candidates) else None)
        self.clear()
        return True, word

    def draw(self, frame, pointer):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv.rectangle(overlay, (0, 0), (min(width - 1, 310), height - 1), (22, 25, 32), -1)
        cv.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv.putText(frame, "CHOOSE A WORD", (14, 25), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv.putText(frame, "Hold index finger in a box for 1s", (14, 45), cv.FONT_HERSHEY_SIMPLEX, 0.43, (210, 215, 225), 1)
        for index, (x, y, w, h) in enumerate(self.boxes(width, height)):
            is_candidate = index < len(self.candidates)
            active = index == self.hovered
            color = (80, 220, 140) if active else (100, 110, 130)
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if is_candidate:
                candidate = self.candidates[index]
                label = f"{index + 1}. {candidate['gloss']}"
                scale = 0.58
                while cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > w - 82 and scale > 0.25:
                    scale -= 0.025
                cv.putText(frame, label, (x + 10, y + 31), cv.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(frame, f"{candidate['probability']:.0%}", (x + w - 63, y + 31), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
            else:
                cv.putText(frame, "CANCEL / SIGN AGAIN", (x + 10, y + 31), cv.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 255), 1, cv.LINE_AA)
            if active:
                cv.rectangle(frame, (x + 2, y + h - 7), (x + 2 + int((w - 4) * self.progress), y + h - 2), color, -1)
        if pointer is not None:
            cv.circle(frame, pointer, 9, (255, 255, 255), 2, cv.LINE_AA)
            cv.circle(frame, pointer, 3, (80, 220, 140), -1, cv.LINE_AA)
        return frame
