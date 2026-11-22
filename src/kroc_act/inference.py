"""Cyclo-facing bridge. The actual Cyclo runtime registration remains deployment work."""
import torch

from .emg import rms
from .model import ACTExperiment


class CycloACTAdapter:
    def __init__(self, checkpoint, device="cuda"):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.cfg = dict(saved["experiment"], device=device, backbone_weights=None)
        self.model = ACTExperiment(self.cfg, saved["statistics"]).to(device)
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.device = device

    def reset(self):
        """Call at each episode/task boundary and clear upstream sensor buffers too."""
        self.model.reset()

    def action(self, state, images, now_ns, stream=None):
        """Images: camera-key -> RGB CHW uint8 or float [0,1].

        stream: EMGStream containing only this episode's causal, identically filtered
        data in the verified robot clock. Runtime must keep a bounded ring buffer.
        Missing EMG raises; the controller must handle hold/stop, never feed zeros.
        """
        batch = {"observation.state": torch.as_tensor(state, dtype=torch.float32)[None]}
        for key, shape in self.cfg["camera_shapes"].items():
            image = torch.as_tensor(images[key])
            if tuple(image.shape) != tuple(shape):
                raise ValueError(f"Camera shape mismatch: {key}")
            batch[key] = image[None]
        if self.cfg["variant"] != "baseline":
            if stream is None:
                raise ValueError("EMG stream required")
            n = round(self.cfg["window_ms"] * self.cfg["sample_rate_hz"] / 1000)
            window = stream.window(now_ns, n, self.cfg["sample_rate_hz"], self.cfg["max_gap_ms"],
                                   self.cfg.get("max_emg_age_ms", 100.0))
            summary = rms(window, round(self.cfg["rms_ms"] * self.cfg["sample_rate_hz"] / 1000))
            batch["observation.emg_window"] = torch.from_numpy(window)[None]
            batch["observation.emg"] = torch.from_numpy(summary)[None]
        batch = {k: v.to(self.device) for k, v in batch.items()}
        return self.model.select_action(batch)[0].cpu().numpy()
