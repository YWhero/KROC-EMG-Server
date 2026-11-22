"""Serve the three KROC ACT policies on the GPU over a loopback-only endpoint."""
import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import socket
import threading
import time

import cv2
import numpy as np
import torch

from kroc_act.emg import rms
from kroc_act.inference import CycloACTAdapter

from .wire import (
    BINARY, CAMERAS, CHUNK_SHAPE, EMG_SHAPE, FLOAT, HOST, IMAGE_HW, JOINTS,
    MAX_REQUEST_BYTES, MODES, PORT, decode_observation,
)


class Policies:
    def __init__(self, checkpoints):
        self.adapters = {}
        self.info = {}
        self.lock = threading.Lock()
        self._validate_device()
        for mode in MODES:
            folder = checkpoints / mode
            metadata = json.loads((folder / "metadata.json").read_text())
            adapter = CycloACTAdapter(folder / "checkpoint.pt")
            cfg = adapter.cfg
            expected_cameras = {f"observation.images.{name}": [3, *IMAGE_HW] for name in CAMERAS}
            if (
                cfg["variant"] != mode
                or cfg["state_dim"] != len(JOINTS)
                or cfg["action_dim"] != CHUNK_SHAPE[1]
                or cfg["n_action_steps"] != CHUNK_SHAPE[0]
                or cfg["chunk_size"] != 30
                or cfg["camera_shapes"] != expected_cameras
                or (cfg["sample_rate_hz"], cfg["window_ms"], cfg["rms_ms"], cfg["emg_channels"]) != (2000, 200, 50, 4)
                or metadata["experiment"]["variant"] != mode
                or metadata["step"] != 100_000
            ):
                raise ValueError(f"Unexpected checkpoint contract for {mode}")
            self.adapters[mode] = adapter
            self.info[mode] = {
                "step": metadata["step"], "cameras": expected_cameras, "joints": JOINTS,
                "predicted_steps": cfg["chunk_size"], "returned_steps": cfg["n_action_steps"],
                "emg_window": EMG_SHAPE if mode != "baseline" else None,
            }
            state = np.asarray(metadata["statistics"]["state"]["mean"], dtype=FLOAT)
            images = {name: np.zeros((3, *IMAGE_HW), dtype=np.uint8) for name in CAMERAS}
            window = np.zeros(EMG_SHAPE, dtype=FLOAT) if mode != "baseline" else None
            for _ in range(8):
                self.predict(mode, state, images, window)
            logging.info("Ready: %s step %d", mode, metadata["step"])

    @staticmethod
    def _validate_device():
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required")
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        logging.info("CUDA device: %s", torch.cuda.get_device_name())

    @torch.inference_mode()
    def predict(self, mode, state, images, window):
        adapter = self.adapters[mode]
        batch = {"observation.state": torch.from_numpy(state)[None].to(adapter.device)}
        for camera, image in images.items():
            batch[f"observation.images.{camera}"] = torch.from_numpy(image)[None].to(adapter.device)
        if mode == "concat":
            cfg = adapter.cfg
            count = round(cfg["rms_ms"] * cfg["sample_rate_hz"] / 1000)
            batch["observation.emg"] = torch.from_numpy(rms(window, count))[None].to(adapter.device)
        if mode == "window":
            batch["observation.emg_window"] = torch.from_numpy(window)[None].to(adapter.device)
        # Use the external action buffer in Cyclo. The checkpoint's per-observation
        # chunk method preserves normalization and keeps the RPC stateless.
        chunk = adapter.model.predict_action_chunk(batch)[0, :CHUNK_SHAPE[0]].cpu().numpy()
        if chunk.shape != CHUNK_SHAPE or not np.isfinite(chunk).all():
            raise RuntimeError("KROC produced an invalid action chunk")
        return chunk.astype(FLOAT, copy=False)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 10.0

    def setup(self):
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        super().setup()

    def log_message(self, *_):
        pass

    def reply(self, status, data, media_type="text/plain", inference_ms=None):
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        if inference_ms is not None:
            self.send_header("X-Inference-Ms", f"{inference_ms:.3f}")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self.reply(HTTPStatus.OK, self.server.model_info, "application/json")
        else:
            self.reply(HTTPStatus.NOT_FOUND, b"Unknown endpoint")

    def do_POST(self):
        mode = self.path.rsplit("/", 1)[-1]
        if mode not in MODES or self.path != f"/v1/actions/{mode}":
            self.close_connection = True
            self.reply(HTTPStatus.NOT_FOUND, b"Unknown model")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES or self.headers.get_content_type() != BINARY:
                raise ValueError("Invalid content type or length")
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError("Truncated observation")
            state, images, window = decode_observation(body, mode != "baseline")
        except (ValueError, cv2.error) as exc:
            self.close_connection = True
            self.reply(HTTPStatus.BAD_REQUEST, str(exc).encode())
            return
        if not self.server.policies.lock.acquire(blocking=False):
            self.reply(HTTPStatus.SERVICE_UNAVAILABLE, b"GPU busy")
            return
        try:
            started = time.perf_counter()
            chunk = self.server.policies.predict(mode, state, images, window)
            inference_ms = (time.perf_counter() - started) * 1_000
        except Exception:
            logging.exception("Inference failed for %s", mode)
            self.reply(HTTPStatus.INTERNAL_SERVER_ERROR, b"GPU inference failed")
            return
        finally:
            self.server.policies.lock.release()
        self.reply(HTTPStatus.OK, chunk.tobytes(), BINARY, inference_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policies = Policies(args.checkpoints)
    with ThreadingHTTPServer((HOST, args.port), Handler) as server:
        server.policies = policies
        server.model_info = json.dumps(policies.info).encode()
        logging.info("Listening on %s:%d", HOST, args.port)
        server.serve_forever()


if __name__ == "__main__":
    main()
