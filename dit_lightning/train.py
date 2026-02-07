import argparse
import os
import pickle as pkl

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

try:
    from .config import DiTConfig, TrainConfig
    from .data import DiTDataModule, DataConfig
    from .lightning_module import DiTLightning
except ImportError:  # allow running as a script: python dit_lightning/train.py
    from config import DiTConfig, TrainConfig
    from data import DiTDataModule, DataConfig
    from lightning_module import DiTLightning
    
from lightning.pytorch import Trainer, seed_everything


DATASEED = 7272

seed_everything(DATASEED, workers=True)

def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument('--train_pkl', type=str, default='/lustre/grp/gyqlab/share/mzc/ProTok/GFP_data/GFP_nonorm_0904_train.pkl', help='Pickle path with {embedding, labels?} for eps-pred')
    p.add_argument('--train_list', type=str, default=None, help='Text file with one pkl path per line for x0-pred')
    p.add_argument('--seq_len', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_labels', action='store_true')
    # Objective mode
    p.add_argument('--x_prediction', action='store_true', help='Use x0-pred instead of eps-pred')
    # Embeddings for x0 mode
    p.add_argument('--protoken_emb', type=str, default=None)
    p.add_argument('--aatype_emb', type=str, default=None)
    # Model
    p.add_argument('--hidden_size', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=16)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.01)
    p.add_argument('--use_rope', action='store_true', default=True)
    p.add_argument('--num_classes', type=int, default=8)
    # Optim
    p.add_argument('--lr_max', type=float, default=1e-4)
    p.add_argument('--lr_min', type=float, default=1e-5)
    p.add_argument('--warmup_steps', type=int, default=2000)
    p.add_argument('--lr_decay_steps', type=int, default=2_000_000)
    p.add_argument('--weight_decay', type=float, default=1e-3)
    p.add_argument('--grad_clip', type=float, default=0.1)
    # Diffusion
    p.add_argument('--timesteps', type=int, default=500)
    p.add_argument('--t_min', type=int, default=0)
    p.add_argument('--t_max', type=int, default=500)
    # Trainer
    p.add_argument('--max_epochs', type=int, default=20000)
    p.add_argument('--save_dir', type=str, default='./runs/dit_lightning')
    p.add_argument('--precision', type=str, default='16-mixed', help='bf16-mixed, 16-mixed, 32-true')
    p.add_argument('--devices', type=int, default=1)
    p.add_argument('--accumulate_grad_batches', type=int, default=1)
    # TensorBoard logger
    p.add_argument('--tb_log_dir', type=str, default='tensorboard_logger', help='TensorBoard save_dir')
    p.add_argument('--project_name', type=str, default='DiT', help='TensorBoard project name')
    return p.parse_args()


def infer_in_features(args) -> int:
    if not args.x_prediction:
        assert args.train_pkl, 'train_pkl is required for epsilon mode'
        if not os.path.exists(args.train_pkl):
            raise FileNotFoundError(f'train_pkl not found: {args.train_pkl}. Please pass --train_pkl or place your data there.')
        with open(args.train_pkl, 'rb') as f:
            d = pkl.load(f)
        if 'embedding' not in d:
            raise KeyError('Expected key "embedding" in train_pkl')
        emb = d['embedding']
        # emb can be np.ndarray or list of arrays
        try:
            return int(emb.shape[0]), int(emb.shape[-1])  # (N,T,F)
        except Exception:
            # fall back: inspect first valid sample
            if isinstance(emb, (list, tuple)) and len(emb) > 0:
                import numpy as np
                sample = emb[0]
                sample = np.asarray(sample)
                return len(sample), int(sample.shape[-1])
            else:
                raise ValueError('Cannot infer feature dimension from embedding; please pass consistent arrays')
    else:
        assert args.protoken_emb and args.aatype_emb, 'protoken_emb and aatype_emb are required for x0 mode'
        with open(args.protoken_emb, 'rb') as f:
            pe = pkl.load(f)
        with open(args.aatype_emb, 'rb') as f:
            ae = pkl.load(f)
        return int(pe.shape[-1] + ae.shape[-1])



def main():
    # torch.set_float32_matmul_precision('high')
    
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log_dir, exist_ok=True)

    N_samples, in_features = infer_in_features(args)

    dit_cfg = DiTConfig(
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_rope=args.use_rope,
        num_classes=args.num_classes,
        time_embed_dim=args.hidden_size,
        label_embed_hidden=args.hidden_size,
    )

    train_cfg = TrainConfig(
        diffusion_timesteps=args.timesteps,
        train_t_min=args.t_min,
        train_t_max=args.t_max,
        x_prediction=args.x_prediction,
        lr_max=args.lr_max,
        lr_min=args.lr_min,
        lr_warmup_steps=args.warmup_steps,
        lr_decay_steps=args.lr_decay_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip,
    )

    data_cfg = DataConfig(
        train_pkl=args.train_pkl,
        train_list=args.train_list,
        batch_size=min(N_samples//args.devices, args.batch_size),
        num_workers=args.num_workers,
        seq_len=args.seq_len,
        use_labels=not args.no_labels,
    )

    lit = DiTLightning(
        dit_cfg=dit_cfg,
        train_cfg=train_cfg,
        in_features=in_features,
        num_classes=args.num_classes,
        protoken_emb_path=args.protoken_emb,
        aatype_emb_path=args.aatype_emb,
    )

    dm = DiTDataModule(data_cfg, x_prediction=args.x_prediction)

    ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(args.save_dir, 'checkpoints'), save_top_k = -1, monitor='train_loss', every_n_epochs = 500, filename="DiT-{epoch:03d}-{step:06d}-{train_loss_epoch:.6f}")
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    tb_logger = TensorBoardLogger(save_dir=args.tb_log_dir, name=args.project_name)

    trainer = pl.Trainer(
        default_root_dir=args.save_dir,
        max_epochs=args.max_epochs,
        precision=args.precision,
        devices=args.devices,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ckpt_cb, lr_monitor],
        logger=tb_logger,
        log_every_n_steps=10,
    )

    trainer.fit(lit, datamodule=dm)


if __name__ == '__main__':
    main()