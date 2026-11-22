"""Use upstream ACT with an optional learned EMG environment-state token."""
import torch
from torch import nn


class WindowEncoder(nn.Module):
    def __init__(self, channels, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, 64, 9, stride=2, padding=4), nn.GELU(),
            nn.Conv1d(64, 128, 9, stride=2, padding=4), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, width), nn.LayerNorm(width))

    def forward(self, x):
        return self.network(x.transpose(1, 2))  # [B,N,C] -> [B,D]


class SpectrogramEncoder(nn.Module):
    """A causal four-channel log-power image projected to one ACT token."""

    def __init__(self, experiment):
        super().__init__()
        stft = experiment["spectrogram"]
        rate = experiment["sample_rate_hz"]
        self.frame_samples = round(stft["frame_ms"] * rate / 1000)
        self.hop_samples = round(stft["hop_ms"] * rate / 1000)
        self.n_fft = stft["n_fft"]
        bins_hz = torch.fft.rfftfreq(self.n_fft, d=1 / rate)
        bins = (bins_hz >= stft["frequency_hz"][0]) & (bins_hz <= stft["frequency_hz"][1])
        self.first_bin = int(torch.where(bins)[0][0])
        self.last_bin = int(torch.where(bins)[0][-1]) + 1
        taper = torch.hann_window(self.frame_samples)
        self.register_buffer("taper", taper, persistent=False)
        self.power_scale = float(taper.square().sum())
        self.network = nn.Sequential(
            nn.Conv2d(experiment["emg_channels"], 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=(2, 1), padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(),
            nn.Linear(128 * 4 * 3, experiment["dim_model"]),
            nn.LayerNorm(experiment["dim_model"]),
        )

    def image(self, x):
        # Use only complete frames from the observed history.
        offset = (x.shape[1] - self.frame_samples) % self.hop_samples
        frames = x[:, offset:].transpose(1, 2).unfold(-1, self.frame_samples, self.hop_samples)
        spectrum = torch.fft.rfft(frames.float() * self.taper, n=self.n_fft)
        power = spectrum[..., self.first_bin:self.last_bin].abs().square()
        return torch.log1p(power / self.power_scale).transpose(-2, -1)

    def forward(self, x):
        return self.network(self.image(x))


class ACTExperiment(nn.Module):
    def __init__(self, experiment, statistics):
        super().__init__()
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        self.experiment = experiment
        self.variant = experiment["variant"]
        if self.variant not in {"baseline", "concat", "window", "spectrogram"}:
            raise ValueError("Unknown ACT variant")
        for key, values in statistics.items():
            for name in ["mean", "std"]:
                self.register_buffer(f"{key}_{name}", torch.tensor(values[name], dtype=torch.float32))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        state_dim = experiment["state_dim"]
        if self.variant == "concat":
            state_dim += experiment["emg_channels"]
        inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (state_dim,))}
        inputs.update({k: PolicyFeature(FeatureType.VISUAL, tuple(shape))
                       for k, shape in experiment["camera_shapes"].items()})
        if self.variant in {"window", "spectrogram"}:
            inputs["observation.environment_state"] = PolicyFeature(FeatureType.ENV, (experiment["dim_model"],))
            if self.variant == "window":
                self.emg_encoder = WindowEncoder(experiment["emg_channels"], experiment["dim_model"])
            else:
                self.emg_encoder = SpectrogramEncoder(experiment)
        cfg = ACTConfig(
            input_features=inputs,
            output_features={"action": PolicyFeature(FeatureType.ACTION, (experiment["action_dim"],))},
            chunk_size=experiment["chunk_size"], n_action_steps=experiment["n_action_steps"],
            dim_model=experiment["dim_model"], dim_feedforward=experiment["dim_feedforward"],
            n_heads=experiment["n_heads"], n_encoder_layers=experiment["n_encoder_layers"],
            n_decoder_layers=experiment["n_decoder_layers"], n_vae_encoder_layers=experiment["n_vae_encoder_layers"],
            pretrained_backbone_weights=experiment["backbone_weights"],
            optimizer_lr=experiment["lr"], optimizer_lr_backbone=experiment["lr"],
            device=experiment["device"], push_to_hub=False)
        self.policy = ACTPolicy(cfg)

    def normalize(self, x, key):
        return (x - getattr(self, f"{key}_mean")) / getattr(self, f"{key}_std")

    def prepare(self, batch, training=False):
        state = self.normalize(batch["observation.state"].float(), "state")
        if self.variant == "concat":
            state = torch.cat([state, self.normalize(batch["observation.emg"].float(), "emg")], -1)
        result = {"observation.state": state}
        for key in self.experiment["camera_shapes"]:
            image = batch[key]
            image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
            result[key] = (image - self.image_mean) / self.image_std
        if self.variant in {"window", "spectrogram"}:
            window = self.normalize(batch["observation.emg_window"].float(), "window")
            result["observation.environment_state"] = self.emg_encoder(window)
        if training:
            result["action"] = self.normalize(batch["action"].float(), "action")
            result["action_is_pad"] = batch["action_is_pad"].bool()
        return result

    def forward(self, batch):
        return self.policy(self.prepare(batch, training=True))

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        self.eval()
        out = self.policy.predict_action_chunk(self.prepare(batch))
        return out * self.action_std + self.action_mean

    @torch.no_grad()
    def select_action(self, batch):
        self.eval()
        out = self.policy.select_action(self.prepare(batch))
        return out * self.action_std + self.action_mean

    def reset(self):
        self.policy.reset()

    def optimizer_groups(self):
        groups = self.policy.get_optim_params()
        if self.variant in {"window", "spectrogram"}:
            groups.append({"params": self.emg_encoder.parameters()})
        return groups
