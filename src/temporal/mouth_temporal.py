# temporal/yawn_temporal.py

class YawnTemporalTracker:
    def __init__(
        self,
        yawn_th=0.5,
        min_yawn_duration=0.3,
        refractory_time=2.0
    ):
        self.yawn_th = yawn_th
        self.min_yawn_duration = min_yawn_duration
        self.refractory_time = refractory_time

        self.yawn_start = None
        self.last_yawn_time = None
        self.yawn_active = False

    def update(self, yawn_signal, t):
        if yawn_signal is None:
            self.yawn_start = None
            return False

        p = yawn_signal["prob"]

        # Refractory period
        if self.last_yawn_time and (t - self.last_yawn_time) < self.refractory_time:
            return False

        if p >= self.yawn_th:
            if self.yawn_start is None:
                self.yawn_start = t
            elif (t - self.yawn_start) >= self.min_yawn_duration:
                self.last_yawn_time = t
                self.yawn_start = None
                return True
        else:
            self.yawn_start = None

        return False
