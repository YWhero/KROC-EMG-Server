import json
import tempfile
import unittest
from pathlib import Path

import torch

from kroc_act.checkpoints import save_checkpoint, wait_for_receipt
from kroc_act.paths import PROJECT_ROOT
from kroc_act.relay_server import acknowledge


class RelayTests(unittest.TestCase):
    def setUp(self):
        outputs = PROJECT_ROOT/'outputs'
        outputs.mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix='test_relay_', dir=outputs)
        self.run = Path(self.directory.name)
        self.assertTrue(self.run.resolve().is_relative_to(outputs.resolve()))
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.cfg = {'variant': 'baseline', 'keep_checkpoints': 3,
                    'checkpoint_archive': {'mode':'relay','repo_id':'test/test','private':False}}
        save_checkpoint(self.run, 10, self.model, self.optimizer, self.cfg, {}, {},
                        torch.Generator(), is_best=True, wait_for_archive=False)
        self.relative = str((self.run/'transfer_queue/00000010.json').relative_to(PROJECT_ROOT))
        self.job = json.loads((PROJECT_ROOT/self.relative).read_text())
        self.checkpoint = self.run/'checkpoints/00000010/checkpoint.pt'

    def tearDown(self):
        self.directory.cleanup()

    def receipt(self):
        return {'job_id':self.job['job_id'], 'repo_id':self.job['repo_id'], 'commit':'a'*40,
                'verified':True,'files':{f['remote_path']:{'size':f['size'],'sha256':f['sha256']}
                                        for f in self.job['files']}}

    def test_no_receipt_timeout_retains_checkpoint(self):
        with self.assertRaises(TimeoutError):
            wait_for_receipt(self.run, self.job, {'wait_timeout_seconds':0.01,'poll_seconds':0.005})
        self.assertTrue(self.checkpoint.is_file())

    def test_wrong_hash_receipt_cannot_delete(self):
        receipt = self.receipt()
        receipt['files'][self.job['files'][0]['remote_path']]['sha256'] = '0'*64
        with self.assertRaises(ValueError):
            acknowledge(self.relative, receipt)
        self.assertTrue(self.checkpoint.is_file())

    def test_verified_receipt_deletes_only_bundle_and_preserves_best_locator(self):
        result = acknowledge(self.relative, self.receipt())
        self.assertTrue(result['checkpoint_removed'])
        self.assertTrue((self.run/'checkpoints/00000010/metadata.json').is_file())
        best = json.loads((self.run/'best.json').read_text())
        self.assertEqual(best['step'], 10)
        self.assertTrue(best['remote']['cleanup_complete'])
        # Retrying a completed confirmation is safe and does not touch other files.
        self.assertTrue(acknowledge(self.relative, self.receipt())['checkpoint_removed'])

    def test_changed_checkpoint_cannot_be_deleted(self):
        with self.checkpoint.open('ab') as handle:
            handle.write(b'changed')
        with self.assertRaises(ValueError):
            acknowledge(self.relative, self.receipt())
        self.assertTrue(self.checkpoint.is_file())
