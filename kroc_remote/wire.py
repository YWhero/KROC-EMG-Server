"""Binary request and response bodies carried over a persistent HTTP/1.1 connection."""
import struct

import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 17665
MODES = ("baseline", "concat", "window")
MODEL_PREFIX = "kroc://"
UNAVAILABLE = "KROC observation unavailable: "
JOINTS = tuple(f"left_joint{i}" for i in range(1, 6)) + ("left_gripper_joint_1",)
CAMERAS = ("top", "wrist")
IMAGE_HW = (180, 320)
IMAGE_HWC = (*IMAGE_HW, 3)
IMAGE_BYTES = int(np.prod(IMAGE_HWC))
EMG_SHAPE = (400, 4)
CHUNK_SHAPE = (15, 6)
FLOAT = np.dtype("<f4")
HEADER = struct.Struct("<4sBII")
MAGIC = b"KAC1"
JPEG_PARAMS = (cv2.IMWRITE_JPEG_QUALITY, 90)
MAX_REQUEST_BYTES = HEADER.size + FLOAT.itemsize * (len(JOINTS) + int(np.prod(EMG_SHAPE))) + 2 * IMAGE_BYTES
CHUNK_BYTES = FLOAT.itemsize * int(np.prod(CHUNK_SHAPE))
BINARY = "application/octet-stream"


def encode_observation(state, images, window=None, *, jpeg=True):
    state = np.asarray(state, dtype=FLOAT)
    if state.shape != (len(JOINTS),) or not np.isfinite(state).all():
        raise ValueError("Expected six finite left-arm positions")
    data = [state.tobytes()]
    if window is not None:
        window = np.asarray(window, dtype=FLOAT)
        if window.shape != EMG_SHAPE or not np.isfinite(window).all():
            raise ValueError("Expected a finite 400×4 EMG window")
        data.append(window.tobytes())
    frames = []
    for camera in CAMERAS:
        img = images[camera]
        if img.shape != IMAGE_HWC or img.dtype != np.uint8:
            raise ValueError(f"Expected a 320×180 BGR uint8 image for {camera}")
        if jpeg:
            ok, encoded = cv2.imencode(".jpg", img, JPEG_PARAMS)
            if not ok or encoded.nbytes > IMAGE_BYTES:
                raise ValueError(f"Could not encode {camera}")
            frames.append(encoded.tobytes())
        else:
            frames.append(img.tobytes())
    return b"".join([HEADER.pack(MAGIC, int(jpeg), *map(len, frames)), *data, *frames])


def decode_observation(body, needs_emg):
    if len(body) < HEADER.size:
        raise ValueError("Missing observation header")
    magic, jpeg, top_size, wrist_size = HEADER.unpack_from(body)
    if magic != MAGIC or jpeg not in (0, 1):
        raise ValueError("Unsupported observation format")
    if any(not (0 < size <= IMAGE_BYTES) for size in (top_size, wrist_size)):
        raise ValueError("Invalid camera content length")
    float_count = len(JOINTS) + (int(np.prod(EMG_SHAPE)) if needs_emg else 0)
    offset = HEADER.size + float_count * FLOAT.itemsize
    if len(body) != offset + top_size + wrist_size:
        raise ValueError("Observation length does not match the camera and EMG layout")
    values = np.frombuffer(body, dtype=FLOAT, count=float_count, offset=HEADER.size)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite state or EMG values")
    images = {}
    for camera, size in zip(CAMERAS, (top_size, wrist_size), strict=True):
        pixels = np.frombuffer(body, dtype=np.uint8, count=size, offset=offset)
        if jpeg:
            if not pixels.tobytes().startswith(b"\xff\xd8"):
                raise ValueError(f"Invalid JPEG header for {camera}")
            bgr = cv2.imdecode(pixels, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        elif size == IMAGE_BYTES:
            bgr = pixels.reshape(IMAGE_HWC)
        else:
            raise ValueError(f"Invalid raw image size for {camera}")
        if bgr is None or bgr.shape != IMAGE_HWC:
            raise ValueError(f"Invalid image for {camera}")
        images[camera] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()
        offset += size
    state = values[:len(JOINTS)].copy()
    window = values[len(JOINTS):].reshape(EMG_SHAPE).copy() if needs_emg else None
    return state, images, window
