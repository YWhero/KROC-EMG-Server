"""Persistent client for the local SSH forward."""
from http.client import HTTPConnection
import json
import time

import numpy as np

from .wire import BINARY, CHUNK_BYTES, CHUNK_SHAPE, FLOAT, HOST, MODES, PORT, encode_observation


class RemotePolicy:
    def __init__(self, mode, *, port=PORT, timeout=0.150, jpeg=True):
        if mode not in MODES:
            raise ValueError(f"Choose one of {', '.join(MODES)}")
        self.mode = mode
        self.timeout = timeout
        self.jpeg = jpeg
        self.connection = HTTPConnection(HOST, port, timeout=timeout)

    def close(self):
        self.connection.close()

    def _request(self, method, path, body=None):
        started = time.monotonic()
        try:
            headers = {"Content-Type": BINARY} if body is not None else {}
            self.connection.request(method, path, body, headers)
            response = self.connection.getresponse()
            limit = CHUNK_BYTES if method == "POST" and response.status == 200 else 16_384
            data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError("KROC response exceeds expected size")
            if response.status != 200:
                raise ValueError(f"KROC HTTP {response.status}: {data.decode(errors='replace')}")
            if time.monotonic() - started > self.timeout:
                raise TimeoutError("KROC response exceeded the inference deadline")
            return data, response.headers
        except Exception:
            self.close()
            raise

    def info(self):
        data, _ = self._request("GET", "/v1/models")
        return json.loads(data)[self.mode]

    def predict(self, state, images, window=None):
        body = encode_observation(state, images, window, jpeg=self.jpeg)
        data, headers = self._request("POST", f"/v1/actions/{self.mode}", body)
        if headers.get_content_type() != BINARY or len(data) != CHUNK_BYTES:
            raise ValueError("Invalid action chunk response")
        chunk = np.frombuffer(data, dtype=FLOAT).reshape(CHUNK_SHAPE)
        if not np.isfinite(chunk).all():
            raise ValueError("KROC returned a non-finite action")
        return chunk, float(headers["X-Inference-Ms"]), len(body)
