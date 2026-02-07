#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from scipy.stats import pearsonr, spearmanr
from src.common.dataset import GFP_dataset
from net.ProTok import ProTok
from src.common.loss import reduce_loss
from configs.config import compose_config
from src.common.lr_scheduler import get_cosine_scheduler_with_warmup
from tqdm import tqdm
import pickle as pkl
from torchmetrics.regression import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef
torch._dynamo.config.optimize_ddp = False
torch.multiprocessing.set_sharing_strategy("file_system")
cfg = compose_config()
ProTok_config = cfg.model
global_config = cfg.global_cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ProTok GFP Fine-tuning (argparse refactor, logic/output preserved)"
    )

    p.add_argument("--dataseed", type=int, default=42)

    # Project / logging
    p.add_argument("--project_name", type=str, default="GFP")
    p.add_argument("--logger_name", type=str, default="GFP")
    p.add_argument(
        "--logger_save_dir",
        type=str,
        default="./results/transfer_runs/tensorbord_logger",
    )
    p.add_argument("--log_graph", action="store_true", default=True)

    # Checkpoints
    p.add_argument("--ckpt_path", type=str, default="./results/transfer_runs/GFP/")
    p.add_argument(
        "--init_ckpt",
        type=str,
        default="./checkpoint/ProTok_main.ckpt",
        help="Checkpoint to load for fine-tuning",
    )

    # Data
    p.add_argument(
        "--train_csv_path",
        type=str,
        default="./data/Generation_data/DMS/GFP/GFP-train.csv",
    )
    p.add_argument(
        "--test_csv_path",
        type=str,
        default="./data/Generation_data/DMS/GFP/GFP-test.csv",
    )
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_prefix", type=int, default=64)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=4)

    # Training
    p.add_argument("--num_epoch", type=int, default=50)
    p.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Number of GPUs (used by Trainer and warmup_steps scaling)",
    )
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--check_val_every_n_epoch", type=int, default=1)
    p.add_argument("--log_every_n_steps", type=int, default=1)
    p.add_argument("--progress_refresh_rate", type=int, default=1)

    # LR schedule / optimizer
    p.add_argument("--max_lr", type=float, default=1.0e-4)
    p.add_argument("--min_lr", type=float, default=1.0e-7)
    p.add_argument("--init_lr", type=float, default=1.0e-7)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Misc (kept for completeness; does not change your current logic)
    p.add_argument("--val_check_interval", type=int, default=2000)
    p.add_argument("--every_n_train_steps", type=int, default=10000)
    p.add_argument("--bucket_size", type=int, default=50000)
    p.add_argument("--log_frequency", type=int, default=10)

    # save embedding path for diffusion training
    p.add_argument("--save_embedding_path", type=str, default="./example/cond_traindit_exp.pkl")

    return p


class LatentRegressor(nn.Module):
    def __init__(self, latent_dim=768, hidden_dim=64, output_dim=1, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.dropout(F.relu(self.fc1(z)))
        return self.fc2(h)


class ProTok_FT(L.LightningModule):
    def __init__(self, model, config, global_config, protoken_codebook=None, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["protoken_codebook"])

        if protoken_codebook is None:
            placeholder = torch.zeros((22, 1280))
            self.register_buffer("protoken_codebook", placeholder)
        else:
            self.register_buffer("protoken_codebook", protoken_codebook)

        self.model = model(config, global_config, self.protoken_codebook)
        self.config = config
        self.global_config = global_config

        self.regressor = LatentRegressor()
        self.no_decay_modulename = ["bias", "norm", "embedding_table", "embedding_dense"]

        for p in self.model.clip_projection_layer.parameters():
            p.requires_grad = False

        self.recon_loss_weight = 1
        self.reg_loss_weight = 2

        # ===== Metrics (DDP-safe) =====
        # validation
        self.val_pearson_metric = PearsonCorrCoef()
        self.val_spearman_metric = SpearmanCorrCoef()
        self.val_mse_metric = MeanSquaredError()

        # test
        self.test_pearson_metric = PearsonCorrCoef()
        self.test_spearman_metric = SpearmanCorrCoef()
        self.test_mse_metric = MeanSquaredError()

        # mean recon loss (use torch tensor accumulation to be DDP-friendly)
        self.register_buffer("val_recon_sum", torch.tensor(0.0))
        self.register_buffer("val_recon_count", torch.tensor(0.0))
        self.register_buffer("test_recon_sum", torch.tensor(0.0))
        self.register_buffer("test_recon_count", torch.tensor(0.0))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch["input"])

        latent = outputs["quantized"]
        B, num_prefix, feature_dim = latent.shape

        targets = batch["fitness"]
        y_hat = self.regressor(latent.reshape(B, -1))

        y_hat_gathered = self.all_gather(y_hat, sync_grads=True)
        targets_gathered = self.all_gather(targets, sync_grads=True)

        if self.trainer.world_size > 1:
            global_batch_size = y_hat_gathered.size(0) * y_hat_gathered.size(1)
            y_hat_global = y_hat_gathered.view(global_batch_size, -1)
            targets_global = targets_gathered.view(global_batch_size, -1)
        else:
            if isinstance(y_hat_gathered, list) and len(y_hat_gathered) == 1:
                y_hat_global = y_hat_gathered[0]
                targets_global = targets_gathered[0]
            elif (
                y_hat_gathered.dim() == y_hat_gathered.dim() + 1
                and y_hat_gathered.size(0) == 1
            ):
                y_hat_global = y_hat_gathered.squeeze(0)
                targets_global = targets_gathered.squeeze(0)
            else:
                y_hat_global = y_hat_gathered
                targets_global = targets_gathered

        logits = outputs["decode_protokens_logits"]
        reduced_reconstruct_loss = reduce_loss(
            logits, batch["input"]["label_mask"], batch["input"]["label"]
        )
## You can also try smoothl1 loss, may be more robust!
        reg_loss = nn.MSELoss()(y_hat_global.flatten(), targets_global.flatten())
        loss = self.recon_loss_weight * reduced_reconstruct_loss + self.reg_loss_weight * reg_loss

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        log_frequency = self.hparams.get("log_frequency", 10)
        if batch_idx % log_frequency == 0:
            self.log("reconstruct_loss", reduced_reconstruct_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("Total_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("reg_loss", reg_loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("lr", current_lr, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        self.log("train_loss_epoch", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=True)
        return loss

    # ------------------------
    # Validation (DDP-correct)
    # ------------------------
    def on_validation_epoch_start(self):
        # reset accumulators + metrics
        self.val_recon_sum.zero_()
        self.val_recon_count.zero_()
        self.val_pearson_metric.reset()
        self.val_spearman_metric.reset()
        self.val_mse_metric.reset()

    def validation_step(self, val_batch, batch_idx):
        outputs = self.model(val_batch["input"])

        latent = outputs["quantized"]
        B, num_prefix, feature_dim = latent.shape

        targets = val_batch["fitness"].unsqueeze(-1).float()
        y_hat = self.regressor(latent.reshape(B, -1))

        logits = outputs["decode_protokens_logits"]
        recon_loss = reduce_loss(
            logits, val_batch["input"]["label_mask"], val_batch["input"]["label"]
        )

        reg_loss = nn.MSELoss()(y_hat.flatten(), targets.flatten())

        # step logging (keep your keys)
        self.log("val_loss", recon_loss, sync_dist=True, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        self.log("reg_loss", reg_loss, sync_dist=True, on_epoch=True, on_step=True, prog_bar=True, logger=True)

        # update recon mean (tensor accum, safe)
        self.val_recon_sum += recon_loss.detach()
        self.val_recon_count += 1.0

        # update global metrics (torchmetrics handles DDP)
        y = targets.flatten()
        ypred = y_hat.flatten()

        # torchmetrics expects float tensor
        self.val_mse_metric.update(ypred, y)
        # Pearson/Spearman require at least 2 points; torchmetrics will handle but can output nan for tiny batches
        self.val_pearson_metric.update(ypred, y)
        self.val_spearman_metric.update(ypred, y)

        return {"pred": y_hat, "label": targets}

    def on_validation_epoch_end(self):
        # mean recon loss over steps
        mean_recon = self.val_recon_sum / torch.clamp(self.val_recon_count, min=1.0)

        val_mse = self.val_mse_metric.compute()
        val_pearson = self.val_pearson_metric.compute()
        val_spearman = self.val_spearman_metric.compute()

        # IMPORTANT: sync_dist=True here is now correct because compute() returns a DDP-synced metric
        self.log("val_recon_loss_epoch", mean_recon, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_pearson", val_pearson, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_spearman", val_spearman, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_mse", val_mse, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if self.trainer.is_global_zero:
            try:
                print("pearson", float(val_pearson))
                print("spearman", float(val_spearman))
            except Exception:
                print("pearson", val_pearson)
                print("spearman", val_spearman)

    # -------------------
    # Test (DDP-correct)
    # -------------------
    def on_test_epoch_start(self):
        self.test_recon_sum.zero_()
        self.test_recon_count.zero_()
        self.test_pearson_metric.reset()
        self.test_spearman_metric.reset()
        self.test_mse_metric.reset()

    def test_step(self, test_batch, batch_idx):
        outputs = self.model(test_batch["input"])

        latent = outputs["quantized"]
        B, num_prefix, feature_dim = latent.shape

        targets = test_batch["fitness"].unsqueeze(-1).float()
        y_hat = self.regressor(latent.reshape(B, -1))

        logits = outputs["decode_protokens_logits"]
        recon_loss = reduce_loss(
            logits, test_batch["input"]["label_mask"], test_batch["input"]["label"]
        )

        # recon mean
        self.test_recon_sum += recon_loss.detach()
        self.test_recon_count += 1.0

        # metrics update
        y = targets.flatten()
        ypred = y_hat.flatten()
        self.test_mse_metric.update(ypred, y)
        self.test_pearson_metric.update(ypred, y)
        self.test_spearman_metric.update(ypred, y)

        return {"pred": y_hat, "label": targets}

    def on_test_epoch_end(self):
        mean_recon = self.test_recon_sum / torch.clamp(self.test_recon_count, min=1.0)

        test_mse = self.test_mse_metric.compute()
        test_pearson = self.test_pearson_metric.compute()
        test_spearman = self.test_spearman_metric.compute()

        self.log("test_recon_loss_epoch", mean_recon, sync_dist=True)
        self.log("test_pearson_final", test_pearson)
        self.log("test_spearman_final", test_spearman)
        self.log("test_mse_final", test_mse)
        if self.trainer.is_global_zero:
            print(
                f"\nTest Results - Pearson: {float(test_pearson):.4f}, Spearman: {float(test_spearman):.4f}, MSE: {float(test_mse):.4f}"
            )

    # --------------
    # Optimizers
    # --------------
    def configure_optimizers(self):
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if not any(nd in n for nd in self.no_decay_modulename) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in self.named_parameters()
                    if any(nd in n for nd in self.no_decay_modulename) and p.requires_grad
                ],
                "weight_decay": self.hparams.weight_decay,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.max_lr)

        train_batches = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.hparams.num_epochs * train_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio) // self.hparams.num_gpus

        scheduler = get_cosine_scheduler_with_warmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_lr=self.hparams.max_lr,
            min_lr=self.hparams.min_lr,
            init_lr=self.hparams.init_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # -------
    # Encode / Predict
    # -------
    def encode(self, x):
        return self.model.encode(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        latent, clip = self.model.encode(batch["input"])
        return {
            "index": batch["index"].detach().cpu(),
            "embedding": latent.detach().cpu(),
        }



class MyDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        test_csv_path,
        batch_size,
        num_prefix,
        max_len,
        num_workers=8,
    ):
        super().__init__()
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.num_workers = num_workers
        self.num_prefix = num_prefix
        self.max_len = max_len
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = GFP_dataset(self.train_csv_path, "train", num_prefix_tokens=self.num_prefix, max_len=self.max_len)
            self.val_ds = GFP_dataset(self.train_csv_path, "val", num_prefix_tokens=self.num_prefix, max_len=self.max_len)
        if stage == "test" or stage is None:
            self.test_ds = GFP_dataset(self.test_csv_path, "test", num_prefix_tokens=self.num_prefix, max_len=self.max_len)
        if stage == "predict":
            base_ds = GFP_dataset(self.train_csv_path, "train", num_prefix_tokens=self.num_prefix, max_len=self.max_len)
            self.predict_ds = IndexedDataset(base_ds)

    def train_dataloader(self): return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self): return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    def test_dataloader(self): return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers)
    def predict_dataloader(self): return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    
class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        item["index"] = idx
        return item


def main():
    args = build_parser().parse_args()

    seed_everything(args.dataseed, workers=True)

    logger = pl_loggers.TensorBoardLogger(
        name=args.logger_name,
        save_dir=args.logger_save_dir,
        log_graph=args.log_graph,
    )

    data_module = MyDataModule(
        train_csv_path=args.train_csv_path,
        test_csv_path=args.test_csv_path,
        batch_size=args.batch_size,
        num_prefix=args.num_prefix,
        max_len=args.max_len,
        num_workers=args.num_workers
    )

    lightning_model = ProTok_FT.load_from_checkpoint(
        args.init_ckpt,
        config = ProTok_config,
        global_config =global_config,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epoch,
        init_lr=args.init_lr,
        num_gpus=args.num_gpus,  
        strict=False,
        log_frequency=args.log_frequency,
    )

    lightning_model.hparams.update(
        {"model_class": ProTok, "config": ProTok_config, "global_config": global_config}
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_pearson",
        mode="max",
        dirpath=args.ckpt_path,
        filename="StablePT-{epoch:03d}-{step:06d}-{val_pearson:.3f}-{val_recon_loss_epoch:.3f}-{val_spearman:.3f}",
        save_top_k=-1,
        save_last=True,
    )

    trainer = L.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=args.progress_refresh_rate)],
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.num_epoch,
        accelerator="cuda",
        devices=str(args.num_gpus),
        precision=args.precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
    )

    start_time = time.time()
    trainer.fit(model=lightning_model, datamodule=data_module)

    print("traning time:", time.time() - start_time)

    best_path = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score

    print(f"--- Training finished ---")
    print(f"best model path: {best_path}")
    print(f"best val_pearson score: {best_score:.4f}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


    if trainer.is_global_zero:
        print("\n" + "="*30)
        print("Starting Single-GPU Evaluation & Encoding")
        print("="*30)


        eval_trainer = L.Trainer(accelerator="cuda", devices=1, logger=False, precision=args.precision)
        best_model_path = checkpoint_callback.best_model_path


        print(f"--- Testing on: {best_model_path} ---")
        eval_trainer.test(lightning_model, datamodule=data_module, ckpt_path=best_model_path)
        

        print(f"Test Pearson: {lightning_model.test_pearson_metric.compute():.4f}")


        print(f"--- Encoding training sequences to {args.save_embedding_path} ---")
        data_module.setup("predict")
        predictions = eval_trainer.predict(lightning_model, datamodule=data_module, ckpt_path=best_model_path)


        all_indices = []
        all_embeddings = []
        for batch in predictions:
            all_indices.append(batch["index"])
            all_embeddings.append(batch["embedding"])

        full_indices = torch.cat(all_indices).numpy()
        full_embeddings = torch.cat(all_embeddings).numpy()


        sort_idx = np.argsort(full_indices)
        final_embeddings = full_embeddings[sort_idx]
        

        train_labels = data_module.predict_ds.base_ds.data_labels - 1

        with open(args.save_embedding_path, "wb") as f:
            pkl.dump({"embedding": final_embeddings, "labels": train_labels}, f)

        print(f"Success! Saved {len(final_embeddings)} embeddings.")

if __name__ == "__main__":
    main()









