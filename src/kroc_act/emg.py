"""Shared offline/online causal sampling; timestamps must use the SAME clock."""
from dataclasses import dataclass

import numpy as np


@dataclass
class EMGStream:
    time_ns: np.ndarray
    available_ns: np.ndarray
    values: np.ndarray
    valid: np.ndarray

    def __post_init__(self):
        self.time_ns = np.asarray(self.time_ns, dtype=np.int64)
        self.available_ns = np.asarray(self.available_ns, dtype=np.int64)
        self.values = np.asarray(self.values, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        n = len(self.time_ns)
        if (self.values.ndim != 2 or self.values.shape[0] != n
                or self.available_ns.shape != (n,) or self.valid.shape != (n,)):
            raise ValueError("Invalid EMG array shapes")
        if n == 0 or np.any(np.diff(self.time_ns) <= 0):
            raise ValueError("EMG acquisition timestamps must be nonempty and strictly increasing")
        if np.any(self.available_ns < self.time_ns):
            raise ValueError("Availability cannot precede acquisition in the aligned clock")

    def window(self, now_ns: int, samples: int, rate_hz: int,
               max_gap_ms: float = 2.0, max_age_ms: float = 100.0) -> np.ndarray:
        """Backward sample-and-hold. Never select unavailable/future samples.

        Reject warmup, incomplete history and gaps; caller uses identical exclusion
        rules for every variant. No episode boundary crossing or zero-filled history.
        """
        if samples <= 0 or rate_hz <= 0:
            raise ValueError("Positive samples/rate required")
        # Availability cutoff reflects the sensor data the policy really had at now.
        end = np.searchsorted(self.time_ns, now_ns, side="right")
        ids = np.flatnonzero(self.valid[:end] & (self.available_ns[:end] <= now_ns))
        if len(ids) == 0 or now_ns - self.time_ns[ids[-1]] > max_age_ms * 1e6:
            raise ValueError("No fresh available EMG sample")
        # Anchor at the latest received acquisition, not an unavailable point at now.
        # This preserves a real 200ms history instead of filling the terminal delay.
        anchor = self.time_ns[ids[-1]]
        targets = anchor - np.rint(np.arange(samples - 1, -1, -1)
                                  * (1e9 / rate_hz)).astype(np.int64)
        selected = np.searchsorted(self.time_ns[ids], targets, side="right") - 1
        if np.any(selected < 0):
            raise ValueError("Insufficient causal EMG history")
        indices = ids[selected]
        if np.any(targets - self.time_ns[indices] > max_gap_ms * 1e6):
            raise ValueError("Missing, delayed or invalid EMG in requested window")
        window = self.values[indices]
        if not np.isfinite(window).all():
            raise ValueError("Non-finite EMG")
        return window.copy()


def rms(window: np.ndarray, samples: int) -> np.ndarray:
    """Trailing per-channel RMS on the SAME filtered signal used by model 3."""
    if samples <= 0 or len(window) < samples:
        raise ValueError("Insufficient RMS history")
    return np.sqrt(np.mean(np.square(window[-samples:], dtype=np.float64), axis=0)).astype(np.float32)
