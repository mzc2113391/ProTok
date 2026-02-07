from typing import Optional, Dict

import math
import pickle as pkl

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

try:
    from .config import DiTConfig, TrainConfig
    from .model import DiT
    from .scheduler import GaussianDiffusion
except ImportError:  # allow import when run as a script
    from config import DiTConfig, TrainConfig
    from model import DiT
    from scheduler import GaussianDiffusion


class WarmupCosineLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps: int, total_decay_steps: int, lr_min: float, lr_max: float):
        def lr_lambda(step):
            if step < warmup_steps:
                return (lr_max - lr_min) * (step / max(1, warmup_steps)) / lr_max + lr_min / lr_max
            # cosine from lr_max -> lr_min over total_decay_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (lr_min + (lr_max - lr_min) * cosine) / lr_max
        super().__init__(optimizer, lr_lambda)


class DiTLightning(pl.LightningModule):
    def __init__(self,
                 dit_cfg: DiTConfig,
                 train_cfg: TrainConfig,
                 in_features: int,
                 num_classes: int = 8,
                 protoken_emb_path: Optional[str] = None,
                 aatype_emb_path: Optional[str] = None,
                 ):  # in_features is the token feature dim for epsilon mode
        super().__init__()
        self.save_hyperparameters(ignore=["protoken_emb_path", "aatype_emb_path"])  # cleaner checkpoints
        self.dit_cfg = dit_cfg
        self.train_cfg = train_cfg
        self.in_features = in_features
        self.dit_cfg.num_classes = num_classes
        self.model = DiT(dit_cfg, in_features=in_features)
        self.diffusion = GaussianDiffusion(diffusion_timesteps=train_cfg.diffusion_timesteps)

        # x0 prediction setup
        self.x_prediction = train_cfg.x_prediction
        if self.x_prediction:
            assert protoken_emb_path and aatype_emb_path, 'Embedding paths required in x_prediction mode'
            with open(protoken_emb_path, 'rb') as f:
                prot = pkl.load(f)
            with open(aatype_emb_path, 'rb') as f:
                aat = pkl.load(f)
            prot = torch.tensor(prot, dtype=torch.float32)
            aat = torch.tensor(aat, dtype=torch.float32)
            self.register_buffer('protoken_emb', prot)  # (V_p, Dp)
            self.register_buffer('aatype_emb', aat)     # (V_a, Da)
            self.protoken_indicator = nn.Parameter(torch.zeros(1, prot.shape[-1]))
            self.aatype_indicator = nn.Parameter(torch.zeros(1, aat.shape[-1]))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.train_cfg.lr_max, betas=(0.9, 0.99), weight_decay=self.train_cfg.weight_decay)
        sch = WarmupCosineLR(
            opt,
            warmup_steps=self.train_cfg.lr_warmup_steps,
            total_decay_steps=self.train_cfg.lr_decay_steps,
            lr_min=self.train_cfg.lr_min,
            lr_max=self.train_cfg.lr_max,
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "step"}}

    def _seq_len_weight(self, seq_mask: torch.Tensor) -> torch.Tensor:
        # seq_mask: (B,T) bool
        seq_len = seq_mask.float().sum(dim=-1)  # (B,)
        return torch.pow(seq_len.clamp(min=1.0), self.train_cfg.seq_len_power)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        if not self.x_prediction:
            x = batch['embedding']  # (B,T,F)
            seq_mask = batch['seq_mask']  # (B,T) bool
            residue_index = batch['residue_index'].long()
            labels = batch['labels'].long() if 'labels' in batch else torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

            B = x.shape[0]
            t = torch.randint(self.train_cfg.train_t_min, self.train_cfg.train_t_max, (B,), device=x.device)
            eps = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x, t, eps)

            pred_eps = self.model(x_t, seq_mask, t, label=labels, tokens_rope_index=residue_index)
            mse = (pred_eps - eps).pow(2).mean(dim=(-2, -1))  # (B,)
            w = self._seq_len_weight(seq_mask)
            loss = (mse * (w / (w.sum().clamp_min(1e-6)))).sum()

            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True,logger=True)
            self.log('train_mse_eps', mse.mean(), on_step=True, on_epoch=True,sync_dist=True,logger=True)
            return loss
        else:
            # Build x0 from tables
            protokens = batch['protokens'].long()
            aatypes = batch['aatypes'].long()
            seq_mask = batch['seq_mask']
            residue_index = batch['residue_index'].long()
            prot = self.protoken_emb[protokens]  # (B,T,Dp)
            aat = self.aatype_emb[aatypes]       # (B,T,Da)
            x = torch.cat([prot, aat], dim=-1)

            B = x.shape[0]
            t = torch.randint(self.train_cfg.train_t_min, self.train_cfg.train_t_max, (B,), device=x.device)
            eps = torch.randn_like(x)
            x_t = self.diffusion.q_sample(x, t, eps)

            indicator = torch.cat([self.protoken_indicator, self.aatype_indicator], dim=-1)  # (1,Dp+Da)
            x0_pred = self.model(x_t + indicator[None, None, :], seq_mask, t, label=None, tokens_rope_index=residue_index)

            x0_pred = x0_pred.float()
            x = x.float()

            # Split losses
            Dp = self.protoken_emb.shape[-1]
            mse = (x0_pred - x).pow(2)
            mse_prot = mse[..., :Dp].mean(dim=-1)  # (B,T)
            mse_aat = mse[..., Dp:].mean(dim=-1)   # (B,T)
            # mask-average per sequence
            denom = seq_mask.float().sum(dim=-1).clamp_min(1.0)
            loss_prot = (mse_prot * seq_mask).sum(dim=-1) / denom
            loss_aat = (mse_aat * seq_mask).sum(dim=-1) / denom
            loss = self.train_cfg.w_protoken * loss_prot + self.train_cfg.w_aatype * loss_aat

            if self.train_cfg.w_protoken_cos > 0:
                # cosine distance: 2 - dot(u,v) with l2 normalized per token
                def l2_norm(z, mask):
                    z = F.normalize(z + (~mask[..., None]).float() * 1e-6, dim=-1)
                    return z
                z_pred = l2_norm(x0_pred[..., :Dp], seq_mask)
                z_true = l2_norm(x[..., :Dp], seq_mask)
                cos_dist = 2.0 - (z_pred * z_true).sum(dim=-1)  # (B,T)
                loss_cos = (cos_dist * seq_mask).sum(dim=-1) / denom
                loss = loss + self.train_cfg.w_protoken_cos * loss_cos

            w = self._seq_len_weight(seq_mask)
            loss = (loss * (w / (w.sum().clamp_min(1e-6)))).sum()

            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True,sync_dist=True)
            self.log('train_loss_protoken', loss_prot.mean(), on_step=True, on_epoch=True,sync_dist=True)
            self.log('train_loss_aatype', loss_aat.mean(), on_step=True, on_epoch=True,sync_dist=True)
            return loss
