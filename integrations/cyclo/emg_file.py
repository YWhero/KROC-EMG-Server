"""Share the most recent causal EMG window on the local Cyclo host."""
from pathlib import Path
import struct
import time

import numpy as np

from .wire import EMG_SHAPE, FLOAT

EMG_PATH = Path("/workspace/kroc_emg/window.bin")
SENSOR_AGE_S = 0.100
HEADER = struct.Struct("<4sQ")
MAGIC = b"KEM1"
SIZE = HEADER.size + FLOAT.itemsize * int(np.prod(EMG_SHAPE))


def read_emg_window(path):
    try:
        with path.open("rb") as fd:
            data = fd.read(SIZE + 1)
    except FileNotFoundError as exc:
        raise ValueError("KROC EMG is warming up or the 4-channel RAW collector is offline") from exc
    if len(data) != SIZE:
        raise ValueError("Invalid EMG window file")
    magic, scan_end_ns = HEADER.unpack_from(data)
    if magic != MAGIC or not 0 <= time.monotonic_ns() - scan_end_ns <= SENSOR_AGE_S * 1e9:
        raise ValueError("EMG window is stale or has an invalid timestamp")
    return np.frombuffer(data, dtype=FLOAT, offset=HEADER.size).reshape(EMG_SHAPE)


def publish_emg_window(path, scan_end_ns, window):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as fd:
        fd.write(HEADER.pack(MAGIC, scan_end_ns))
        fd.write(np.asarray(window, dtype=FLOAT).tobytes())
    temporary.replace(path)
