"""Exercise actual policy inference through the SSH tunnel without robot I/O."""
import argparse
import json
import time

import cv2
import numpy as np

from kroc_remote.client import RemotePolicy
from kroc_remote.wire import CAMERAS, CHUNK_SHAPE, EMG_SHAPE, IMAGE_HW, MODES, PORT


def stats(values):
    return {name: round(float(value), 2)
            for name, value in zip(("median", "p95", "max"), np.percentile(values, [50, 95, 100]), strict=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--codec", choices=("jpeg", "raw", "both"), default="both")
    parser.add_argument("--mode", choices=MODES)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261122)
    height, width = IMAGE_HW
    images = {
        name: cv2.resize(rng.integers(0, 256, (height // 3, width // 3, 3), dtype=np.uint8), (width, height))
        for name in CAMERAS
    }
    state = np.zeros(6, dtype=np.float32)
    for mode in (args.mode,) if args.mode else MODES:
        for codec in ("jpeg", "raw") if args.codec == "both" else (args.codec,):
            policy = RemotePolicy(mode, port=args.port, jpeg=codec == "jpeg")
            try:
                assert policy.info()["step"] == 100_000
                window = rng.normal(size=EMG_SHAPE).astype(np.float32) if mode != "baseline" else None
                reference, _, body_bytes = policy.predict(state, images, window)
                assert reference.shape == CHUNK_SHAPE
                end_to_end, inference, errors = [], [], 0
                for _ in range(args.samples):
                    started = time.perf_counter_ns()
                    try:
                        chunk, gpu_ms, _ = policy.predict(state, images, window)
                    except (OSError, TimeoutError):
                        errors += 1
                        continue
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    np.testing.assert_allclose(chunk, reference, atol=1e-6, rtol=1e-5)
                    end_to_end.append(elapsed)
                    inference.append(gpu_ms)
                    time.sleep(0.016)
                if not end_to_end:
                    raise RuntimeError(f"No successful requests: {mode}/{codec}")
                print(json.dumps({
                    "model": mode, "codec": codec, "request_bytes": body_bytes,
                    "samples": len(end_to_end), "deadline_errors": errors,
                    "end_to_end_ms": stats(end_to_end), "server_inference_ms": stats(inference),
                }), flush=True)
            finally:
                policy.close()


if __name__ == "__main__":
    main()
