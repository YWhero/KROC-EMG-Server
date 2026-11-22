import copy
import json
import unittest

import torch

from kroc_act.evaluation import evaluate
from kroc_act.paths import PROJECT_ROOT, project_path
from kroc_act.study import stratified_split, validate_study


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.annotations = json.loads((PROJECT_ROOT / 'data/study_annotations.json').read_text(encoding='utf-8'))

    def test_frozen_split_covers_all_conditions_without_episode_leakage(self):
        split = stratified_split(self.annotations)
        self.assertEqual(split, stratified_split(self.annotations))
        counts = validate_study(self.annotations, split, range(30))
        self.assertEqual(counts['val'], {'as_is': 2, 'wrist_left': 2, 'wrist_right': 2})
        self.assertEqual(len(split['train']), 24)
        self.assertEqual(json.loads((PROJECT_ROOT/'data/splits.json').read_text()), split)
        split['val'][0] = split['train'][0]
        with self.assertRaises(ValueError):
            validate_study(self.annotations, split, range(30))

    def test_scope_blocks_outside_path(self):
        with self.assertRaises(ValueError):
            project_path(PROJECT_ROOT / '..' / 'other_workspace' / 'file')


class FakePolicy:
    def predict_action_chunk(self, batch):
        return torch.zeros_like(batch['action'])


class FakeModel:
    action_std = torch.tensor([2.0])
    policy = FakePolicy()
    def eval(self):
        return self
    def prepare(self, batch, training=False):
        return batch


class EvaluationTests(unittest.TestCase):
    def test_all_batches_padding_weight_and_equal_condition_reporting(self):
        batches = [dict(action=torch.tensor([[[1.0], [99.0]]]),
                        action_is_pad=torch.tensor([[False, True]]),
                        episode_index=torch.tensor([1]), study_condition=['as_is']),
                   dict(action=torch.tensor([[[3.0], [3.0]]]),
                        action_is_pad=torch.tensor([[False, False]]),
                        episode_index=torch.tensor([2]), study_condition=['wrist_right'])]
        result = evaluate(FakeModel(), batches, 'cpu')
        self.assertEqual(result['val_frames'], 2)
        self.assertAlmostEqual(result['val_l1_zero_latent'], 7/3)
        self.assertEqual(result['val_condition_macro_l1'], 2.0)
        self.assertAlmostEqual(result['val_action_mae_original_units'][0], 14/3, places=5)
        self.assertEqual(evaluate(FakeModel(), batches, 'cpu', 1)['val_frames'], 1)
