"""Use Cyclo's RobotClient and action pipeline with a remote KROC policy."""
import time

import numpy as np

from robot_client import RobotClient

from ..image_preprocessing import prepare_policy_image
from .client import RemotePolicy
from .emg_file import EMG_PATH, SENSOR_AGE_S, read_emg_window
from .wire import CAMERAS, CHUNK_SHAPE, EMG_SHAPE, IMAGE_HW, JOINTS, UNAVAILABLE

LEFT_PROFILES = ("omx_ai_left", "omx_ai_left_emg")


class KrocEngine:
    def __init__(self, mode, robot_type):
        self.robot = None
        self.policy = RemotePolicy(mode)
        try:
            self._connect(robot_type)
        except Exception:
            self.close()
            raise

    def _connect(self, robot_type):
        if robot_type not in LEFT_PROFILES:
            raise ValueError("KROC requires the OMX-AI left-arm profile used for training")
        info = self.policy.info()
        self.uses_emg = self.policy.mode != "baseline"
        expected_emg = list(EMG_SHAPE) if self.uses_emg else None
        if (
            info["step"] != 100_000
            or tuple(info["joints"]) != JOINTS
            or info["returned_steps"] != CHUNK_SHAPE[0]
            or info["emg_window"] != expected_emg
        ):
            raise ValueError("KROC server returned an incompatible model contract")
        if self.uses_emg:
            read_emg_window(EMG_PATH)
        self.robot = RobotClient(robot_type)
        self.action_keys = list(self.robot.action_keys)
        if len(self.action_keys) != 1:
            raise ValueError("KROC requires exactly one six-joint action group")
        action_cfg = self.robot._action_groups[self.action_keys[0]]
        if (
            tuple(action_cfg.get("joint_names", ())) != JOINTS
            or action_cfg.get("topic") != "/left/leader/joint_trajectory"
        ):
            raise ValueError("Cyclo left command topic or joint order differs from KROC training")
        if not all(name in self.robot.camera_names for name in CAMERAS):
            raise ValueError("KROC requires the top and left wrist camera")
        self.rotations = {
            name: self.robot._config["cameras"][name].get("rotation_deg", 0)
            for name in CAMERAS
        }
        if not self.robot.wait_for_ready(timeout=10.0):
            raise ValueError("Cyclo cameras and joint state did not become ready")
        self._snapshot()

    def _snapshot(self):
        # The SDK replaces cached image arrays on receipt. Take both images and
        # named joints under one lock, then resize/encode outside the callback lock.
        with self.robot._lock:
            now = time.time()
            now_ns = time.monotonic_ns()
            images = {}
            for camera in CAMERAS:
                stamp = self.robot._image_timestamps.get(camera)
                if stamp is None or not 0 <= now - stamp <= SENSOR_AGE_S:
                    raise ValueError(f"Missing or stale camera: {camera}")
                images[camera] = self.robot._images[camera]
            for joint in JOINTS:
                stamp = self.robot._joint_position_timestamps_by_name.get(joint)
                if stamp is None or not 0 <= now_ns * 1e-9 - stamp <= SENSOR_AGE_S:
                    raise ValueError(f"Missing or stale joint: {joint}")
            state = [self.robot._joint_positions_by_name[name] for name in JOINTS]
        return now_ns, images, state

    def predict(self):
        try:
            captured_ns, images, state = self._snapshot()
            window = read_emg_window(EMG_PATH) if self.uses_emg else None
            height, width = IMAGE_HW
            images = {
                name: prepare_policy_image(
                    frame, rotation_deg=self.rotations[name], target_size=(width, height)
                )
                for name, frame in images.items()
            }
            chunk, _, _ = self.policy.predict(state, images, window)
            if time.monotonic_ns() - captured_ns > self.policy.timeout * 1e9:
                raise TimeoutError("Camera-to-action deadline exceeded")
            return {
                "success": True,
                "action_chunk": np.ascontiguousarray(chunk.reshape(-1), dtype=np.float64),
                "chunk_size": CHUNK_SHAPE[0],
                "action_dim": CHUNK_SHAPE[1],
            }
        except Exception as exc:
            return {"success": False, "message": f"{UNAVAILABLE}{exc}"}

    def close(self):
        self.policy.close()
        if self.robot is not None:
            self.robot.close()
            self.robot = None
