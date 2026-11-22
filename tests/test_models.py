import importlib.util
import unittest

import torch

from kroc_act.model import ACTExperiment, WindowEncoder


class EncoderTests(unittest.TestCase):
    def test_gradient_reaches_temporal_encoder(self):
        encoder = WindowEncoder(4, 32)
        x = torch.randn(2, 400, 4, requires_grad=True)
        out = encoder(x)
        self.assertEqual(out.shape, (2, 32))
        out.square().mean().backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0)


@unittest.skipUnless(importlib.util.find_spec("lerobot"), "Requires server Python 3.12 / pinned LeRobot environment")
class ACTIntegrationTests(unittest.TestCase):
    def test_three_variants_backward_and_checkpoint_roundtrip(self):
        stats = {k: {"mean": [0]*d, "std": [1]*d}
                 for k, d in [("state", 6), ("action", 6), ("emg", 4), ("window", 4)]}
        for variant in ["baseline", "concat", "window"]:
            with self.subTest(variant=variant):
                cfg = dict(variant=variant, state_dim=6, action_dim=6, emg_channels=4,
                           camera_shapes={"observation.images.top": [3,64,64]},
                           chunk_size=4, n_action_steps=4, dim_model=32, dim_feedforward=64,
                           n_heads=4, n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
                           backbone_weights=None, lr=1e-5, device="cpu")
                model = ACTExperiment(cfg, stats)
                batch = {"observation.state": torch.randn(2,6), "action": torch.randn(2,4,6),
                         "action_is_pad": torch.zeros(2,4,dtype=torch.bool),
                         "observation.images.top": torch.rand(2,3,64,64),
                         "observation.emg": torch.rand(2,4), "observation.emg_window": torch.rand(2,400,4)}
                model.train()
                loss, _ = model(batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                if variant == "window":
                    g = model.emg_encoder.network[0].weight.grad
                    self.assertIsNotNone(g)
                    self.assertGreater(g.abs().sum().item(), 0)
                model.eval()
                predicted = model.predict_action_chunk(batch)
                loaded = ACTExperiment(cfg, stats)
                loaded.load_state_dict(model.state_dict())
                torch.testing.assert_close(predicted, loaded.predict_action_chunk(batch))
                changed = dict(batch)
                changed["observation.emg"] = batch["observation.emg"] + 10
                changed["observation.emg_window"] = torch.randn(2,400,4) * 10
                other = loaded.predict_action_chunk(changed)
                if variant == "baseline":
                    torch.testing.assert_close(predicted, other)
                else:
                    self.assertFalse(torch.allclose(predicted, other))


if __name__ == "__main__":
    unittest.main()
