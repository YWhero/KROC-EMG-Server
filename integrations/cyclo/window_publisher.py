"""Build KROC causal windows from the collector's native-rate data."""
from collections import deque
import fcntl

import numpy as np

from emg_acq.postprocess import runs
from emg_acq.processing import Processor
from kroc_act.emg import EMGStream

from .emg_file import publish_emg_window
from .wire import EMG_SHAPE

WINDOW_SAMPLES, CHANNELS = EMG_SHAPE
RATE = 2000


class WindowPublisher:
    def __init__(self, output):
        self.output = output
        self.processor = Processor(RATE, CHANNELS, notch=True)
        self.rows = deque()
        self.previous_index = None
        self.latest = -1
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o770)
        self.lock = output.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            output.unlink(missing_ok=True)
        except Exception:
            self.lock.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.lock.closed:
            return
        try:
            self.output.unlink(missing_ok=True)
            self.output.with_suffix(".tmp").unlink(missing_ok=True)
        finally:
            self.lock.close()

    def add(self, batch):
        mapping = batch.clock_map
        if batch.adc_bits != 14 or batch.period_us != 500:
            raise ValueError("KROC EMG requires 14-bit ADC scans at 2 kHz")
        mapped = mapping["host_anchor_ns"] + np.rint(
            (batch.device_us.astype(float) * 1000 - mapping["device_anchor_ns"]) * mapping["slope"]
        ).astype(np.int64)
        scan_end = mapped + batch.duration_us.astype(np.int64) * 1000
        for lo, hi, reset in runs(batch.sample_index, self.previous_index):
            if reset:
                self.processor.reset()
                self.rows.clear()
                self.latest = -1
                self.output.unlink(missing_ok=True)
            self.previous_index = int(batch.sample_index[hi - 1])
            filtered, _, valid = self.processor.apply(batch.adc[lo:hi])
            times = scan_end[lo:hi]
            usable = np.flatnonzero(
                (mapping["uncertainty_ns"] >= 0) & (times <= batch.host_receive_ns)
            )
            times = times[usable]
            if not len(times):
                continue
            previous_max = np.maximum.accumulate(np.r_[self.latest, times[:-1]])
            keep = times > previous_max
            if not keep.any():
                continue
            times = times[keep]
            selected = usable[keep]
            self.latest = int(times[-1])
            self.rows.append(
                (times, np.full(len(times), batch.host_receive_ns, dtype=np.int64),
                 filtered[selected].astype(np.float32), valid[selected])
            )
        while self.rows and self.rows[0][0][-1] < self.latest - 700_000_000:
            self.rows.popleft()

    def publish(self, now_ns):
        if not self.rows:
            return False
        stream = EMGStream(*(np.concatenate(arrays) for arrays in zip(*self.rows, strict=True)))
        try:
            window = stream.window(now_ns, WINDOW_SAMPLES, RATE, max_gap_ms=2.0, max_age_ms=100.0)
        except ValueError:
            self.output.unlink(missing_ok=True)
            return False
        anchor = int(stream.time_ns[stream.valid & (stream.available_ns <= now_ns)][-1])
        publish_emg_window(self.output, anchor, window)
        return True
