"""Input-length behavior is independent of the model's fixed latent length."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from net.ProTok import ProTok
from src.common.transfer_data import ProteinDataModule
from src.common.utils import build_feature, create_feature_list, feature_generate, restype_order


class SequenceLengthTests(unittest.TestCase):
    def test_default_features_keep_long_inputs(self):
        sequence = 'A' * 1535 + 'V'
        batch = build_feature([sequence], restype_order, num_prefix=64)
        self.assertEqual(batch['encoder_protokens'].shape, (1, 1538))
        self.assertEqual(batch['encoder_mask'].sum().item(), 1538)
        self.assertEqual(batch['encoder_protokens'][0, 1536].item(), restype_order['V'])
        self.assertEqual(batch['label_mask'].sum().item(), 1537)
        single = create_feature_list([sequence])[0]
        self.assertEqual(single['encoder_protokens'].shape, (1, 1538))
        self.assertEqual(single['encoder_mask'].sum().item(), 1538)

    def test_short_input_defaults_are_unchanged(self):
        sequences = ['ACDEFG', 'ARND']
        inferred = build_feature(sequences, restype_order)
        fixed = build_feature(sequences, restype_order, max_len=1024)
        for key in inferred:
            torch.testing.assert_close(inferred[key], fixed[key])
        raw = {'seq_len': 4, 'aatype': np.array([0, 1, 2, 3]), 'residue_index': np.arange(1, 5)}
        inferred = feature_generate(raw, num_prefix_tokens=64)
        fixed = feature_generate(raw, num_prefix_tokens=64, max_len=1024)
        for key in inferred:
            np.testing.assert_array_equal(inferred[key], fixed[key])

    def test_transfer_data_accepts_larger_configured_length(self):
        with tempfile.TemporaryDirectory() as directory:
            train_path, val_path = Path(directory) / 'train.csv', Path(directory) / 'val.csv'
            pd.DataFrame({'seq': ['A' * 1400, 'C' * 1500], 'fitness': [0.1, 0.2]}).to_csv(train_path, index=False)
            pd.DataFrame({'seq': ['D' * 1200, 'E' * 1300], 'fitness': [0.3, 0.4]}).to_csv(val_path, index=False)
            data = ProteinDataModule(train_path, val_csv_path=val_path, max_len=2048, num_workers=0)
            data.setup()
            item = data.train_ds[1]
            self.assertEqual(item['input']['label'].shape, (2048,))
            self.assertEqual(item['input']['encoder_mask'].sum(), 1502)
            self.assertEqual(len(data.train_ds.data_seqs[1]), 1500)

    def test_decode_request_is_not_capped_at_1024(self):
        class Decoder(nn.Module):
            def forward(self, latent, tokens, **kwargs):
                output = torch.full((len(latent), tokens.shape[1], 26), -100.0)
                output[:, -1, 0] = 0  # Always emit alanine, never EOS.
                return output

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.prefix_position_embedding_table = nn.Embedding(1, 26)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.arr_dtype = torch.float32
                self.num_prefix_tokens = 1
                self.token_codebook = torch.eye(26)[:22]
                self.protoken_embedding_dense = nn.Identity()
                self.common_embedding_table = torch.eye(26)[22:]
                self.encoder = Encoder()
                self.decoder = Decoder()

        tokens = ProTok.decode_batch(Model(), torch.zeros(1, 1, 1), method='greedy_search',
                                     min_len=1100, max_len=1100)
        self.assertEqual(tokens.shape, (1, 1101))  # BOS plus 1100 residues


if __name__ == '__main__':
    unittest.main()
