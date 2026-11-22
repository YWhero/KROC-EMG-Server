import tempfile
import unittest
from pathlib import Path

import torch

from kroc_act.paths import PROJECT_ROOT
from kroc_act.train import save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_full_state_and_bounded_retention(self):
        output = PROJECT_ROOT / "outputs"
        output.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="kroc_checkpoint_test_", dir=output) as directory:
            run = Path(directory)
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters())
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()
            generator = torch.Generator().manual_seed(1000)
            for step in range(1, 5):
                save_checkpoint(run, step, model, optimizer, {"keep_checkpoints": 3},
                                {"state": {"mean": [0], "std": [1]}}, {"test": True}, generator)
            self.assertEqual(sorted(p.name for p in (run / "checkpoints").iterdir() if (p / 'checkpoint.pt').is_file()),
                             ["00000002", "00000003", "00000004"])
            saved = torch.load(run / "checkpoints/00000004/checkpoint.pt", weights_only=False)
            self.assertEqual(saved["step"], 4)
            self.assertTrue(saved["optimizer"]["state"])
            clone = torch.nn.Linear(4, 2)
            clone.load_state_dict(saved["model"])
            torch.testing.assert_close(model.weight, clone.weight)
            self.assertIn("statistics", saved)
            self.assertIn("loader_rng", saved)


if __name__ == "__main__":
    unittest.main()
