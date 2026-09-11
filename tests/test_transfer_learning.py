"""Run with: python -m unittest discover -s tests -v"""
import tempfile
import unittest
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from configs.config import compose_config
from net.ProTok import ProTok, ProTok_lightning
from scripts.transfer_learning import optimizer_parameter_groups, regression_metrics
from src.common.lr_scheduler import get_cosine_scheduler_with_warmup
from src.common.prediction import merge_indexed_arrays
from src.common.transfer_data import ProteinDataModule
from src.common.utils import DecoderUtils
from src.common.label_weights import (
    DistributedWeightedSampler, parse_label_weights, sample_label_weights, weighted_mse_loss,
)


class OptimizerTests(unittest.TestCase):
    def test_decay_is_applied_only_to_weights(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)
                self.norm = nn.LayerNorm(2)
                self.embedding_table = nn.Embedding(5, 2)
                self.embedding_dense = nn.Linear(3, 2, bias=False)
                self.frozen = nn.Linear(2, 2).requires_grad_(False)
        model = Model()
        groups = optimizer_parameter_groups(model, 0.1)
        ids = [id(p) for group in groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        before = {name: param.detach().clone() for name, param in model.named_parameters()}
        opt = torch.optim.AdamW(groups, lr=0.2)
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.zeros_like(param)
        opt.step()
        for name, param in model.named_parameters():
            expected = before[name] * 0.98 if name == 'linear.weight' else before[name]
            torch.testing.assert_close(param, expected)
        self.assertEqual([g['weight_decay'] for g in optimizer_parameter_groups(model, 0.1, True)], [0, 0.1])

    def test_cosine_endpoints_and_no_rebound(self):
        opt = torch.optim.SGD([nn.Parameter(torch.ones(1))], lr=1.0)
        scheduler = get_cosine_scheduler_with_warmup(opt, 2, 6, 1.0, 0.1, 0.01)
        rates = [opt.param_groups[0]['lr']]
        for _ in range(10):
            opt.step()
            scheduler.step()
            rates.append(opt.param_groups[0]['lr'])
        self.assertAlmostEqual(rates[0], 0.01)
        self.assertAlmostEqual(rates[2], 1.0)
        np.testing.assert_allclose(rates[6:], 0.1)
        with self.assertRaises(ValueError):
            get_cosine_scheduler_with_warmup(opt, 2, 2, 1.0, 0.1)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'custom.csv'
        self.frame = pd.DataFrame({'protein': ['ACDEFG' + aa for aa in 'ARNDCQEGHILKMFPSTWYV'],
                                   'activity': np.arange(20) / 2,
                                   'category': ['high', 'low'] * 10})
        self.frame.to_csv(self.path, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def module(self, **kwargs):
        return ProteinDataModule(self.path, sequence_column='protein', target_column='activity',
                                 label_column='category', num_workers=0, max_len=80, **kwargs)

    def test_custom_csv_and_reproducible_legacy_split(self):
        dm = self.module()
        dm.setup()
        expected, val = train_test_split(np.arange(20), train_size=17, random_state=42)
        np.testing.assert_array_equal(dm.train_ds.row_indices, expected)
        np.testing.assert_array_equal(dm.val_ds.row_indices, val)
        self.assertEqual(dm.label_metadata['classes'], ['high', 'low'])
        self.assertEqual(dm.export_labels.dtype, np.int64)
        expected_labels = (self.frame.iloc[expected]['category'] == 'low').astype(int)
        np.testing.assert_array_equal(dm.export_labels, expected_labels)
        self.assertEqual(dm.train_ds[0]['input']['label'].shape, (80,))
        dm2 = self.module(seed=7)
        dm2.setup()
        self.assertFalse(np.array_equal(dm2.train_ds.row_indices, expected))
        saved = dm.train_ds
        dm.setup('predict')
        self.assertIs(dm.train_ds, saved)

    def test_optional_labels_and_training_only_bins(self):
        self.frame.drop(columns='category').to_csv(self.path, index=False)
        dm = self.module()
        dm.setup()
        self.assertIsNone(dm.export_labels)
        dm = self.module(num_bins=4)
        dm.setup()
        expected = np.quantile(dm.train_ds.data_targets.numpy(), np.linspace(0, 1, 5))
        np.testing.assert_allclose(dm.label_metadata['edges'], expected)
        self.assertTrue((dm.export_labels >= 0).all() and (dm.export_labels < 4).all())

    def test_validation_without_class_labels(self):
        val = Path(self.temp.name) / 'val.csv'
        pd.DataFrame({'protein': ['WWWW', 'YYYY'], 'activity': [-100, 100]}).to_csv(val, index=False)
        dm = self.module(val_csv_path=val, num_bins=4)
        dm.setup()
        self.assertEqual(len(dm.train_ds), 20)
        self.assertEqual(dm.label_metadata['edges'][0], 0)
        self.assertEqual(dm.label_metadata['edges'][-1], 9.5)

    def test_duplicate_sequences_stay_in_one_split(self):
        pd.concat([self.frame, self.frame.iloc[:3]]).to_csv(self.path, index=False)
        dm = self.module()
        dm.setup()
        self.assertFalse(set(dm.train_ds.data_seqs) & set(dm.val_ds.data_seqs))
        self.assertEqual(len(dm.train_ds) + len(dm.val_ds), 23)

    def test_reject_leakage_and_invalid_values(self):
        with self.assertRaisesRegex(ValueError, 'shared sequences'):
            self.module(val_csv_path=self.path).setup()
        for value, message in [(np.nan, 'empty'), ('ACD*', 'unsupported'), ('A' * 50, 'maximum')]:
            frame = self.frame.copy()
            frame.loc[0, 'protein'] = value
            frame.to_csv(self.path, index=False)
            with self.assertRaisesRegex(ValueError, message):
                self.module().setup()
        self.frame.loc[0, 'activity'] = np.inf
        self.frame.to_csv(self.path, index=False)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.module().setup()

    def test_explicit_sequence_policies(self):
        self.frame.loc[0, 'protein'] = 'acdBJ'
        self.frame.to_csv(self.path, index=False)
        dm = self.module(unknown_residues='map-to-x', strip_characters='J')
        dm.setup()
        self.assertIn('ACDX', dm.train_ds.data_seqs + dm.val_ds.data_seqs)

    def test_class_weights_apply_only_to_training(self):
        dm = self.module(sampler_label_weights=(1, 2), reg_loss_label_weights=(3, 1))
        dm.setup()
        expected = self.frame.iloc[dm.train_ds.row_indices]['category'].to_numpy()
        np.testing.assert_allclose(dm.sampling_weights, np.where(expected == 'high', 0.5, 1.0))
        np.testing.assert_allclose(dm.train_ds.regression_weights, np.where(expected == 'high', 1.0, 1 / 3))
        self.assertIsInstance(dm.train_dataloader().sampler, DistributedWeightedSampler)
        self.assertIn('regression_weight', dm.train_ds[0])
        self.assertNotIn('regression_weight', dm.val_ds[0])
        exported_indices = torch.cat([batch['index'] for batch in dm.predict_dataloader()])
        torch.testing.assert_close(exported_indices, torch.arange(len(dm.train_ds)))

    def test_default_class_weights_preserve_shuffling(self):
        for weights in [(1,), (1, 1)]:
            dm = self.module(sampler_label_weights=weights, reg_loss_label_weights=weights)
            dm.setup()
            self.assertIsNone(dm.sampling_weights)
            self.assertIsNone(dm.train_ds.regression_weights)
            self.assertIsInstance(dm.train_dataloader().sampler, torch.utils.data.RandomSampler)


class LabelWeightTests(unittest.TestCase):
    def test_weight_parsing_and_validation(self):
        self.assertEqual(parse_label_weights('1, 2, 0'), (1.0, 2.0, 0.0))
        for value in ['', '1,', 'NaN', 'inf', '-1,2', '0,0']:
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_label_weights(value)
        labels = np.array([0, 1, 1])
        with self.assertRaisesRegex(ValueError, 'expected 2'):
            sample_label_weights((1, 1, 1), labels, 2, 'weights')
        with self.assertRaisesRegex(ValueError, 'require'):
            sample_label_weights((1, 2), None, 0, 'weights')
        with self.assertRaisesRegex(ValueError, 'observed'):
            sample_label_weights((0, 1), np.array([0, 0]), 2, 'weights')

    def test_weighted_mse_value_and_gradients(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        loss = weighted_mse_loss(pred, torch.zeros(2), torch.tensor([1.0, 3.0]))
        self.assertAlmostEqual(loss.item(), 7.0)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([0.5, 4.5]))
        self.assertEqual(weighted_mse_loss(pred, torch.zeros(2)).item(), 5.0)
        self.assertEqual(weighted_mse_loss(pred, torch.zeros(2), torch.ones(2)).item(), 5.0)

    def test_zero_weight_batch_has_zero_finite_gradients(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        loss = weighted_mse_loss(pred, torch.zeros(2), torch.zeros(2))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(pred.grad, torch.zeros(2))

    def test_weighted_sampler_distribution_and_epoch_seed(self):
        weights = np.array([1.0] * 2000 + [3.0] * 2000)
        sampler = DistributedWeightedSampler(range(4000), weights, seed=42)
        indices = list(sampler)
        self.assertEqual(len(indices), 4000)
        self.assertTrue(0.72 < np.mean(np.array(indices) >= 2000) < 0.78)
        self.assertLess(len(set(indices)), len(indices))
        self.assertEqual(indices, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(indices, list(sampler))

    def test_weighted_sampler_shards_one_shared_stream(self):
        weights = [1, 2, 0, 3, 1]
        samplers = [DistributedWeightedSampler(range(5), weights, num_replicas=2, rank=rank, seed=17)
                    for rank in range(2)]
        shards = [list(sampler) for sampler in samplers]
        self.assertEqual([len(shard) for shard in shards], [3, 3])
        stream = torch.multinomial(torch.tensor(weights, dtype=torch.double), 6, replacement=True,
                                   generator=torch.Generator().manual_seed(17)).tolist()
        self.assertEqual([index for pair in zip(*shards) for index in pair], stream)
        self.assertNotIn(2, stream)


class PredictionTests(unittest.TestCase):
    def test_distributed_padding_preserves_original_order(self):
        parts = [{'index': np.array([0, 2, 4]), 'embedding': np.array([[0], [2], [4]])},
                 {'index': np.array([1, 3, 0]), 'embedding': np.array([[1], [3], [0]])}]
        result = merge_indexed_arrays(parts, 5)
        np.testing.assert_array_equal(result['embedding'].flatten(), np.arange(5))
        with self.assertRaises(RuntimeError):
            merge_indexed_arrays(parts, 6)

    def test_global_metrics_ignore_sampler_padding(self):
        def records(ids):
            return {'index': np.array(ids), 'target': np.array(ids, dtype=float),
                    'pred': np.array(ids, dtype=float) * 2,
                    'recon_numerator': np.array(ids, dtype=float) + 1,
                    'recon_weight': np.ones(len(ids))}
        metrics = regression_metrics(merge_indexed_arrays([records([0, 2, 4]), records([1, 3, 0])], 5))
        self.assertAlmostEqual(metrics['mse'], 6)
        self.assertAlmostEqual(metrics['pearson'], 1)
        self.assertAlmostEqual(metrics['spearman'], 1)
        self.assertAlmostEqual(metrics['recon_loss_epoch'], 3)
        self.assertTrue(np.isnan(regression_metrics(records([0, 0]))['pearson']))

    def test_batched_regression_prediction(self):
        class TinyModel(nn.Module):
            def __init__(self, *args):
                super().__init__()
            def encode(self, x):
                return torch.ones(3, 64, 12), None
        cfg = compose_config()
        model = ProTok_lightning(TinyModel, cfg.model, cfg.global_cfg)
        self.assertEqual(model.predict({}).shape, (3,))

    def test_sampling_helpers(self):
        logits = torch.ones(1, 25)
        tokens = torch.tensor([[22, 0, 1, 2, 0, 1]])
        DecoderUtils.block_no_repeat_ngram_(logits, tokens, 3)
        self.assertTrue(torch.isneginf(logits[0, 2]))
        logits = torch.ones(1, 25)
        logits[0, 1] = -2
        DecoderUtils.apply_repetition_penalty_(logits, tokens, 2, 22)
        self.assertEqual(logits[0, 0], 0.5)
        self.assertEqual(logits[0, 1], -4)
        self.assertEqual(logits[0, 22], 1)

    def test_beam_search_stops_when_all_inputs_finish(self):
        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.prefix_position_embedding_table = nn.Embedding(1, 26)

        class FakeDecoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, latent, tokens, **kwargs):
                self.calls += 1
                result = torch.full((len(latent), tokens.shape[1], 26), -100.0)
                for row in range(len(latent)):
                    # One input terminates on step one, the other on step two.
                    token = 23 if latent[row, 0, 0] == 0 or tokens.shape[1] >= 2 else 0
                    result[row, -1, token] = 0
                return result

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.arr_dtype = torch.float32
                self.num_prefix_tokens = 1
                self.token_codebook = torch.eye(26)[:22]
                self.protoken_embedding_dense = nn.Identity()
                self.common_embedding_table = torch.eye(26)[22:]
                self.encoder = FakeEncoder()
                self.decoder = FakeDecoder()

        model = FakeModel()
        results = ProTok.decode_batch(model, torch.tensor([[[0.0]], [[1.0]]]),
                                      num_beams=2, min_len=0)
        self.assertEqual(model.decoder.calls, 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result['best_sequence'][0, -1] == 23 for result in results))


if __name__ == '__main__':
    unittest.main()
