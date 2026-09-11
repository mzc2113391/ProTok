#!/usr/bin/env python3
"""Supervised ProTok transfer learning from user-supplied CSV files."""

import argparse
import json
import math
from pathlib import Path
import pickle
import time

import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
from torch import nn
import torch.nn.functional as F

from configs.config import compose_config
from net.ProTok import ProTok
from src.common.loss import reduce_loss
from src.common.prediction import merge_indexed_arrays
from src.common.lr_scheduler import get_cosine_scheduler_with_warmup
from src.common.transfer_data import ProteinDataModule
from src.common.label_weights import parse_label_weights, weighted_mse_loss


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--train_csv_path', required=True)
    p.add_argument('--val_csv_path', help='Optional independent validation CSV; otherwise split training CSV.')
    p.add_argument('--test_csv_path', help='Optional held-out test CSV, evaluated after model selection.')
    p.add_argument('--sequence_column', default='seq')
    p.add_argument('--target_column', default='fitness')
    p.add_argument('--label_column', default='label', help='Optional class column for weighting and diffusion export; empty string disables it.')
    p.add_argument('--num_bins', type=int, help='Create this many target quantile bins using training rows only; overrides label_column.')
    p.add_argument('--sampler_label_weights', type=parse_label_weights, default=(1.0,),
                   help='Relative sampling weights in class-ID order, comma-separated. A single 1 applies to every class.')
    p.add_argument('--reg_loss_label_weights', type=parse_label_weights, default=(1.0,),
                   help='Relative regression-loss weights in class-ID order, comma-separated. A single 1 applies to every class.')
    p.add_argument('--val_fraction', type=float, default=0.15)
    p.add_argument('--dataseed', type=int, default=42)
    p.add_argument('--unknown_residues', choices=['error', 'map-to-x'], default='error')
    p.add_argument('--long_sequences', choices=['error', 'truncate'], default='error')
    p.add_argument('--strip_characters', default='', help='Explicitly remove these characters, e.g. J for legacy GFP padding.')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size per device.')
    p.add_argument('--num_prefix', type=int, default=None, help='Defaults to checkpoint value; must match it.')
    p.add_argument('--max_len', type=int, default=1024, help='Total token length including prefix and special tokens.')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--init_ckpt', default='./checkpoint/ProTok_main.ckpt')
    p.add_argument('--ckpt_path', default='./results/transfer_runs/checkpoints')
    p.add_argument('--save_embedding_path', default='./results/transfer_runs/train_embeddings.pkl')
    p.add_argument('--skip_export', action='store_true')
    p.add_argument('--project_name', default='transfer')
    p.add_argument('--logger_name', default=None)
    p.add_argument('--logger_save_dir', default='./results/transfer_runs/tensorboard')
    p.add_argument('--log_graph', action='store_true', help='Enable TensorBoard graph logging.')
    p.add_argument('--num_epoch', type=int, default=50)
    p.add_argument('--num_gpus', type=int, default=1, help='GPU count; 0 selects CPU.')
    p.add_argument('--precision', default=None, help='Defaults to bf16-mixed on supported GPUs, otherwise 16-mixed; CPU uses 32-true.')
    p.add_argument('--gradient_clip_val', type=float, default=1.0)
    p.add_argument('--accumulate_grad_batches', type=int, default=1)
    p.add_argument('--check_val_every_n_epoch', type=int, default=1)
    p.add_argument('--log_every_n_steps', type=int, default=10)
    p.add_argument('--progress_refresh_rate', type=int, default=10)
    p.add_argument('--max_lr', type=float, default=1e-4)
    p.add_argument('--min_lr', type=float, default=1e-7)
    p.add_argument('--init_lr', type=float, default=1e-7)
    p.add_argument('--warmup_ratio', type=float, default=0.01)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--legacy_weight_decay', action='store_true', help='Reproduce the old inverted decay groups; not recommended for new runs.')
    p.add_argument('--recon_loss_weight', type=float, default=1.0)
    p.add_argument('--reg_loss_weight', type=float, default=2.0)
    p.add_argument('--monitor', choices=['val_mse', 'val_pearson', 'val_spearman'], default='val_mse')
    p.add_argument('--save_top_k', type=int, default=1, help='Keep the best K checkpoints plus last; -1 keeps all.')
    p.add_argument('--max_steps', type=int, default=-1, help='Optional optimizer-step cap for short runs.')
    p.add_argument('--validate_data_only', action='store_true', help='Check CSVs and show split/label metadata without loading weights.')
    return p


class LatentRegressor(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=64, output_dim=1, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        return self.fc2(self.dropout(F.relu(self.fc1(z))))


def optimizer_parameter_groups(module, weight_decay, legacy=False):
    """Keep biases, normalization and embedding parameters out of weight decay.

    embedding_dense follows the original ProTok embedding-projection convention.
    named_parameters deduplicates shared parameters and frozen weights are excluded.
    """
    decay, no_decay = [], []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        exempt = param.ndim < 2 or any(key in name for key in ('bias', 'norm', 'embedding_table', 'embedding_dense'))
        (no_decay if exempt else decay).append(param)
    return [
        {'params': decay, 'weight_decay': 0.0 if legacy else weight_decay},
        {'params': no_decay, 'weight_decay': weight_decay if legacy else 0.0},
    ]


def regression_metrics(records):
    y, pred = records['target'], records['pred']
    varying = len(y) >= 2 and np.ptp(y) > 0 and np.ptp(pred) > 0
    return {
        'mse': float(np.mean((pred.astype(np.float64) - y) ** 2)),
        'pearson': float(pearsonr(pred, y).statistic) if varying else float('nan'),
        'spearman': float(spearmanr(pred, y).statistic) if varying else float('nan'),
        'recon_loss_epoch': float(records['recon_numerator'].sum() / records['recon_weight'].sum()),
    }


class ProTok_FT(L.LightningModule):
    def __init__(self, model, config, global_config, protoken_codebook=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['protoken_codebook'])
        self.register_buffer('protoken_codebook', torch.zeros(22, 1280) if protoken_codebook is None else protoken_codebook)
        self.model = model(config, global_config, self.protoken_codebook)
        self.config, self.global_config = config, global_config
        latent_dim = config.encoder.num_prefix_tokens * config.vq_config.latent_dim
        self.regressor = LatentRegressor(latent_dim=latent_dim)
        self.model.clip_projection_layer.requires_grad_(False)
        self._eval_records = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch['input'])
        pred = self.regressor(outputs['quantized'].flatten(1)).flatten()
        reg_loss = weighted_mse_loss(pred, batch['fitness'], batch.get('regression_weight'))
        recon_loss = reduce_loss(outputs['decode_protokens_logits'], batch['input']['label_mask'], batch['input']['label'])
        loss = self.hparams.get('recon_loss_weight', 1.0) * recon_loss + self.hparams.get('reg_loss_weight', 2.0) * reg_loss
        for key, value in {'train_recon_loss': recon_loss, 'train_reg_loss': reg_loss, 'train_loss': loss}.items():
            self.log(key, value, on_step=True, on_epoch=True, sync_dist=True, batch_size=len(pred), prog_bar=key == 'train_loss')
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False)
        return loss

    def on_validation_epoch_start(self):
        self._eval_records = []

    def on_test_epoch_start(self):
        self._eval_records = []

    def _evaluation_step(self, batch):
        outputs = self.model(batch['input'])
        pred = self.regressor(outputs['quantized'].flatten(1)).flatten().float()
        logits = outputs['decode_protokens_logits'].float()
        labels, mask = batch['input']['label'], batch['input']['label_mask']
        token_loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none', label_smoothing=0.05).reshape_as(mask)
        lengths = mask.sum(1).float()
        weights = lengths.sqrt()
        records = {
            'index': batch['index'], 'pred': pred, 'target': batch['fitness'].float().flatten(),
            'recon_numerator': (token_loss * mask).sum(1) / lengths * weights,
            'recon_weight': weights,
        }
        self._eval_records.append({key: value.detach().cpu().numpy() for key, value in records.items()})

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch)

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch)

    def _evaluation_epoch_end(self, stage):
        local = {key: np.concatenate([r[key] for r in self._eval_records]) for key in self._eval_records[0]}
        parts = [local]
        if torch.distributed.is_initialized():
            parts = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(parts, local)
        # Metrics are computed once per real row, even when DDP pads the last batch.
        metrics = regression_metrics(merge_indexed_arrays(parts))
        for name, value in metrics.items():
            # Every rank has the same global scalar; averaging it again preserves its value.
            self.log(f'{stage}_{name}', value, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self._eval_records.clear()

    def on_validation_epoch_end(self):
        self._evaluation_epoch_end('val')

    def on_test_epoch_end(self):
        self._evaluation_epoch_end('test')

    def configure_optimizers(self):
        h = self.hparams
        groups = optimizer_parameter_groups(self, h.weight_decay, h.get('legacy_weight_decay', False))
        optimizer = torch.optim.AdamW(groups, lr=h.max_lr)
        steps = self.trainer.estimated_stepping_batches
        if not math.isfinite(steps) or steps < 1:
            raise ValueError('Training must have a finite positive number of optimizer steps.')
        total_steps = int(steps)
        scheduler = get_cosine_scheduler_with_warmup(
            optimizer, warmup_steps=int(total_steps * h.warmup_ratio), total_steps=total_steps,
            max_lr=h.max_lr, min_lr=h.min_lr, init_lr=h.init_lr,
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

    def encode(self, x):
        return self.model.encode(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        latent, _ = self.encode(batch['input'])
        return {'index': batch['index'].detach().cpu(), 'embedding': latent.float().detach().cpu()}


def load_initial_model(path, **training_options):
    """Load published checkpoints, rejecting missing backbone weights instead of hiding them."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    hparams = checkpoint.get('hyper_parameters', {})
    defaults = compose_config()
    config = hparams.get('config', defaults.model)
    global_config = hparams.get('global_config', defaults.global_cfg)
    model = ProTok_FT(model=ProTok, config=config, global_config=global_config, **training_options)
    state = checkpoint['state_dict']
    # Older training checkpoints may contain metric accumulators; these are not model weights.
    state = {key: value for key, value in state.items()
             if key.startswith(('model.', 'regressor.')) or key == 'protoken_codebook'}
    missing, unexpected = model.load_state_dict(state, strict=False)
    backbone_missing = [key for key in missing if not key.startswith('regressor.')]
    if backbone_missing or unexpected:
        raise ValueError(f'Incompatible checkpoint: missing={backbone_missing}, unexpected={unexpected}')
    if missing:
        print(f'Initialized new regression head: {missing}')
    return model


def export_embeddings(trainer, model, data_module, path):
    predictions = trainer.predict(model, datamodule=data_module, ckpt_path='best')
    local = {key: torch.cat([part[key] for part in predictions]).numpy() for key in predictions[0]} if predictions else {
        'index': np.empty(0, dtype=np.int64),
        'embedding': np.empty((0, data_module.hparams.num_prefix, model.config.vq_config.latent_dim), dtype=np.float32),
    }
    parts = [local]
    if torch.distributed.is_initialized():
        parts = [None] * trainer.world_size if trainer.is_global_zero else None
        torch.distributed.gather_object(local, parts, dst=0)
    if trainer.is_global_zero:
        merged = merge_indexed_arrays(parts, len(data_module.predict_ds))
        payload = {
            'embedding': merged['embedding'],
            'row_indices': data_module.predict_ds.row_indices,
            'targets': data_module.predict_ds.data_targets.numpy(),
            'metadata': data_module.split_manifest(),
        }
        if data_module.export_labels is not None:
            payload['labels'] = data_module.export_labels
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Saved {len(payload["embedding"])} training embeddings to {path}; label metadata: {data_module.label_metadata}')


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.num_gpus < 0 or args.num_epoch < 1 or args.accumulate_grad_batches < 1:
        parser.error('num_gpus must be nonnegative; num_epoch and accumulate_grad_batches must be positive.')
    if not 0 <= args.warmup_ratio < 1 or args.weight_decay < 0:
        parser.error('warmup_ratio must be in [0, 1); weight_decay must be nonnegative.')
    if args.save_top_k == 0 or args.save_top_k < -1:
        parser.error('save_top_k must be -1 or positive, so a best checkpoint is available.')
    if not 1 <= args.check_val_every_n_epoch <= args.num_epoch:
        parser.error('check_val_every_n_epoch must be between 1 and num_epoch.')
    if args.recon_loss_weight <= 0 or args.reg_loss_weight <= 0:
        parser.error('Both loss weights must be positive for joint transfer learning.')
    seed_everything(args.dataseed, workers=True)
    model = None
    num_prefix = args.num_prefix or 64
    if not args.validate_data_only:
        model = load_initial_model(args.init_ckpt, **{
            name: getattr(args, name) for name in ('max_lr', 'min_lr', 'init_lr', 'warmup_ratio', 'weight_decay',
                                                  'legacy_weight_decay', 'recon_loss_weight', 'reg_loss_weight')})
        num_prefix = model.config.encoder.num_prefix_tokens
        if args.num_prefix is not None and args.num_prefix != num_prefix:
            parser.error(f'num_prefix must match checkpoint ({num_prefix}).')
    dm = ProteinDataModule(
        train_csv_path=args.train_csv_path, val_csv_path=args.val_csv_path, test_csv_path=args.test_csv_path,
        batch_size=args.batch_size, num_prefix=num_prefix, max_len=args.max_len, num_workers=args.num_workers,
        sequence_column=args.sequence_column, target_column=args.target_column, label_column=args.label_column,
        val_fraction=args.val_fraction, seed=args.dataseed, num_bins=args.num_bins,
        unknown_residues=args.unknown_residues, long_sequences=args.long_sequences, strip_characters=args.strip_characters,
        sampler_label_weights=args.sampler_label_weights, reg_loss_label_weights=args.reg_loss_label_weights,
    )
    dm.setup()
    print(f'Data: train={len(dm.train_ds)}, val={len(dm.val_ds)}, test={len(dm.test_ds) if dm.test_ds is not None else 0}; labels={dm.label_metadata}')
    if args.validate_data_only:
        return
    logger = TensorBoardLogger(save_dir=args.logger_save_dir, name=args.logger_name or args.project_name, log_graph=args.log_graph)
    checkpoint = ModelCheckpoint(monitor=args.monitor, mode='min' if args.monitor == 'val_mse' else 'max',
                                 dirpath=args.ckpt_path, filename='ProTok-{epoch:03d}-{step:06d}',
                                 save_top_k=args.save_top_k, save_last=True, save_on_train_epoch_end=False)
    precision = args.precision or (
        ('bf16-mixed' if torch.cuda.is_bf16_supported() else '16-mixed') if args.num_gpus else '32-true'
    )
    trainer = L.Trainer(
        logger=logger, callbacks=[checkpoint, TQDMProgressBar(refresh_rate=args.progress_refresh_rate)],
        max_epochs=args.num_epoch, max_steps=args.max_steps, accelerator='gpu' if args.num_gpus else 'cpu',
        devices=args.num_gpus or 1, precision=precision,
        gradient_clip_val=args.gradient_clip_val, accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch, log_every_n_steps=args.log_every_n_steps,
    )
    if trainer.is_global_zero:
        output_dir = Path(logger.log_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'data_manifest.json').write_text(json.dumps(dm.split_manifest(), indent=2) + '\n')
        run_config = dict(vars(args), resolved_precision=precision, resolved_num_prefix=num_prefix)
        (output_dir / 'run_config.json').write_text(json.dumps(run_config, indent=2) + '\n')
    start = time.time()
    trainer.fit(model, datamodule=dm)
    if not checkpoint.best_model_path:
        raise RuntimeError('No validation checkpoint was saved. Run through a validation epoch before evaluation/export.')
    if trainer.is_global_zero:
        print(f'Training time: {time.time() - start:.1f}s; best checkpoint: {checkpoint.best_model_path}')
    # Keep the same strategy and process group for every stage; all ranks participate.
    if args.test_csv_path:
        trainer.test(model, datamodule=dm, ckpt_path='best')
    if not args.skip_export:
        export_embeddings(trainer, model, dm, args.save_embedding_path)


if __name__ == '__main__':
    main()
