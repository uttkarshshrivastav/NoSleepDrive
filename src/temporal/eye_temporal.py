# temporal/eye_temporal.py

class EyeTemporalTracker:
   #reducing false alarm and tracing alerts

    def __init__(self, close_th=0.4, open_th=0.2, ema_alpha=0.1, min_close_evidence=0.25, drowsy_duration=1.5):
        self.close_th = close_th
        self.open_th = open_th
        self.ema_alpha = ema_alpha
        self.min_close_evidence = min_close_evidence
        self.drowsy_duration = drowsy_duration

        self._ema = None
        self.eye_state = "OPEN"

        self.closed_evidence = 0.0
        self.closed_start = None
        self.drowsy = False
        self.last_t = None

    def _ema_update(self, x):
        #reducing the effect of spikes in reading 
        if self._ema is None:
            self._ema = x
        else:
            self._ema = (1 - self.ema_alpha) * self._ema + self.ema_alpha * x
        return self._ema

    def update(self, eye_signal, t):
        
        # Update tracker with new eye signal
        # eye_signal: dict with 'prob' and 'conf'
        # t: current timestamp
        # Returns: True if drowsy, False otherwise
        
        # no previous frames
        if eye_signal is None:
            self.last_t = t
            return self.drowsy

        # first frame after starting 
        if self.last_t is None:
            self.last_t = t
            return self.drowsy

        # time difference between the previous eye closure
        dt = t - self.last_t
        self.last_t = t

        # Smooth the probability
        p = self._ema_update(eye_signal["prob"])

        # logic for making decision
        if self.eye_state == "OPEN":
            if p >= self.close_th:
                self.closed_evidence = self.closed_evidence + dt
                if self.closed_evidence >= self.min_close_evidence:
                    # Confirming eye closure
                    self.eye_state = "CLOSED"
                    self.closed_start = t
            else:
                # eye still open 
                self.closed_evidence = 0.0

        elif self.eye_state == "CLOSED":
            if p <= self.open_th:
                # eye opened
                self.eye_state = "OPEN"
                self.closed_evidence = 0.0
                self.closed_start = None
                self.drowsy = False

        # Check for drowsiness (eyes closed too long)
        if self.eye_state == "CLOSED" and self.closed_start is not None:
            closed_time = t - self.closed_start
            if closed_time >= self.drowsy_duration:
                self.drowsy = True

        return self.drowsy