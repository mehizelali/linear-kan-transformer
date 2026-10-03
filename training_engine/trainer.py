from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler  # type: ignore

from dataset_loader import get_data_loaders
from models.build import build_model

from .distributed import (
    barrier,
    cleanup_distributed,
    fsdp_load_state_dict,
    fsdp_state_dict,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    wrap_fsdp,
)
from .engine import build_criterion, build_mixup, evaluate, train_one_epoch
from .optimizer import create_optimizer
from .results import (
    append_aggregate_results,
    save_resolved_config,
    write_epoch_table_md,
    write_run_results_md,
)
from .utils import (
    abs_pos_embed_target_from_cfg,
    append_log,
    cosine_lr,
    count_parameters,
    interpolate_abs_pos_embed,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    sqrt_scaled_lr,
)

_RECIPE: dict[str, Any] = {
    "recipe": "fast",
    "aug": "deit",
    "input_size": 224,
    "hflip": 0.5,
    "mixup": 0.8,
    "cutmix": 1.0,
    "mixup_prob": 1.0,
    "mixup_switch_prob": 0.5,
    "mixup_mode": "batch",
    "smoothing": 0.1,
    "dropout": 0.0,
    "clip_grad": None,
    "batch_size": 256,
    "epochs": 50,
    "opt": "adamw",
    "opt_eps": 1.0e-8,
    "weight_decay": 0.0,
    "lr": 3.0e-4,
    "unscale_lr": True,
    "warmup_lr": 1.0e-6,
    "min_lr": 1.0e-5,
    "warmup_epochs": 5,
    "amp": True,
    "amp_dtype": "bf16",
    "deterministic": False,
    "fused_optim": False,
    "prefetch_factor": 4,
    "model_ema": False,
}


class Trainer:
    def __init__(self, cfg: dict[str, Any]):
        cfg = {**_RECIPE, **cfg, "recipe": "fast"}
        cfg.setdefault("aug", "deit")
        cfg["device"] = "cuda"
        cfg["fsdp"] = bool(cfg.get("fsdp", True))
        if cfg.get("seed") is None:
            # No fixed seed in config → draw one and log it (do not default to 0).
            cfg["seed"] = int(torch.randint(0, 2**31 - 1, ()).item())
        else:
            cfg["seed"] = int(cfg["seed"])

        dist_info = init_distributed(cfg)
        try:
            self._init_after_distributed(cfg, dist_info)
        except Exception:
            cleanup_distributed()
            raise

    def _init_after_distributed(self, cfg: dict[str, Any], dist_info: dict[str, Any]) -> None:
        self.cfg = cfg
        self.device = dist_info["device"]
        self.local_rank = dist_info["local_rank"]
        self.world_size = dist_info["world_size"]

        # Per-rank seed offset keeps DDP/FSDP workers desynchronized on data order
        # while remaining reproducible; base seed is logged in results.
        set_seed(
            int(cfg["seed"]) + get_rank(),
            deterministic=bool(cfg.get("deterministic", True)),
        )

        self.output_dir = Path(cfg.get("output_dir", "outputs/exp"))
        self.repo_root = Path(__file__).resolve().parents[1]
        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            save_resolved_config(self.output_dir, cfg)
        barrier()

        aug = str(cfg.get("aug", "deit")).lower()
        if aug not in ("deit", "basic"):
            aug = "deit"
        (
            self.loader_train,
            self.loader_val,
            self.nb_classes,
            _smooth,
            mixup_from_loader,
        ) = get_data_loaders(
            dataset_name=str(cfg.get("dataset", "cifar10")),
            batch_size=int(cfg.get("batch_size", 256)),
            num_workers=int(cfg.get("num_workers", 4)),
            img_size=int(cfg.get("input_size", 224)),
            aug_type=aug,
            data_path=cfg.get("data_path"),
            mixup=float(cfg.get("mixup", 0.8)),
            cutmix=float(cfg.get("cutmix", 1.0)),
        )
        if is_main_process():
            print(
                f"[data] {cfg.get('dataset')} aug={aug} batch={cfg.get('batch_size')} "
                f"train={len(self.loader_train.dataset)} val={len(self.loader_val.dataset)}"
            )

        model = build_model(cfg, self.nb_classes)
        self.n_parameters = count_parameters(model)
        self.model = wrap_fsdp(model, cfg, self.local_rank)
        self.peak_mem_gb = 0.0
        if self.device.type != "cuda" or next(self.model.parameters()).device.type != "cuda":
            raise RuntimeError("Training must run on GPU (CUDA); CPU is not supported.")

        base_lr = float(cfg.get("lr", 5e-4))
        world = max(get_world_size(), int(cfg.get("num_gpus", 1)))
        # Optional sqrt(batch×world/512) scaling; fast recipe uses unscale_lr: true
        if not cfg.get("unscale_lr", False):
            cfg["lr"] = sqrt_scaled_lr(
                base_lr,
                batch_size=int(cfg.get("batch_size", 256)),
                world_size=world,
            )
        self.base_lr = cfg["lr"]
        self.cfg["world_size"] = world
        self.cfg["base_lr_unscaled"] = base_lr

        self.optimizer = create_optimizer(self.model, cfg)
        self.use_amp = bool(cfg.get("amp", True))
        amp_dtype_name = str(cfg.get("amp_dtype", "fp16")).lower()
        self.amp_dtype = (
            torch.bfloat16
            if amp_dtype_name in ("bf16", "bfloat16")
            else torch.float16
            if amp_dtype_name in ("fp16", "float16", "half")
            else None
        )
        # GradScaler only for fp16; bf16 uses fp32-range exponents
        try:
            self.loss_scaler = GradScaler(
                "cuda",
                enabled=self.use_amp and amp_dtype_name in ("fp16", "float16", "half"),
            )
        except TypeError:  # older torch.cuda.amp.GradScaler
            self.loss_scaler = GradScaler(
                enabled=self.use_amp and amp_dtype_name in ("fp16", "float16", "half")
            )

        # Mixup/CutMix already applied in the DataLoader collate.
        self.mixup_fn = None
        self.soft_targets = mixup_from_loader is not None
        self.criterion = build_criterion(cfg, mixup_active=self.soft_targets)

        self.model_ema = None
        if bool(cfg.get("model_ema", False)) and not bool(cfg.get("fsdp", True)):
            from timm.utils import ModelEma

            self.model_ema = ModelEma(
                self.model,
                decay=float(cfg.get("model_ema_decay", 0.99996)),
                device="",
                resume="",
            )

        self.start_epoch = int(cfg.get("start_epoch", 0))
        self.epochs = int(cfg.get("epochs", 300))
        self.max_accuracy = 0.0
        self.best_epoch = -1
        self.epoch_history: list[dict[str, Any]] = []
        self.final_acc1 = 0.0
        self.final_acc5 = 0.0

        # Fine-tune: load weights only (skip head on shape mismatch). Not a full resume.
        if cfg.get("pretrained") and not cfg.get("resume"):
            self._load_pretrained(str(cfg["pretrained"]))

        if cfg.get("resume"):
            self._resume(cfg["resume"])

        if is_main_process():
            print(
                f"[ready] {cfg.get('model')} {self.n_parameters / 1e6:.2f}M | "
                f"{cfg.get('dataset')} | {self.device} | "
                f"gpus={world} | epochs={self.epochs} | "
                f"bs={cfg.get('batch_size')} | lr={self.base_lr:.3e} | "
                f"amp={amp_dtype_name} | mixup={self.soft_targets} | "
                f"fsdp={get_world_size() > 1 and bool(cfg.get('fsdp', True))} | "
                f"wrap_blocks={bool(cfg.get('fsdp_auto_wrap_blocks', True))} | "
                f"act_ckpt={bool(cfg.get('activation_checkpointing', False))} | "
                f"seed={cfg.get('seed')}"
            )

    def _load_pretrained(self, path: str) -> None:
        """Load classification checkpoint for fine-tuning (model weights only)."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        ckpt = load_checkpoint(path, map_location="cpu")
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        # Drop classifier when transferring across num_classes (e.g. IN100 → CIFAR-10).
        filtered = {k: v for k, v in state.items() if not k.startswith("head.")}

        # Absolute pos_embed length depends on input_size (224→384 FT).
        # Target size from cfg — FSDP shards pos_embed so model.pos_embed.shape is unusable.
        target = abs_pos_embed_target_from_cfg(self.cfg)
        if "pos_embed" in filtered and target is not None:
            num_patches, num_extra = target
            src_shape = tuple(filtered["pos_embed"].shape)
            dst_len = num_patches + num_extra
            if src_shape[1] != dst_len:
                filtered["pos_embed"] = interpolate_abs_pos_embed(
                    filtered["pos_embed"],
                    num_patches=num_patches,
                    num_extra=num_extra,
                )
                if is_main_process():
                    print(
                        f"[pretrained] interpolated pos_embed {src_shape} → "
                        f"(1, {dst_len}, {src_shape[2]})"
                    )

        if isinstance(self.model, FSDP):
            current = fsdp_state_dict(self.model)
            current.update(filtered)
            fsdp_load_state_dict(self.model, current)
        else:
            missing, unexpected = self.model.load_state_dict(filtered, strict=False)
            if is_main_process():
                print(
                    f"[pretrained] strict=False missing={len(missing)} unexpected={len(unexpected)}"
                )
        self.start_epoch = 0
        self.max_accuracy = 0.0
        self.best_epoch = -1
        if is_main_process():
            meta = ""
            if isinstance(ckpt, dict):
                meta = (
                    f" | ckpt_epoch={ckpt.get('epoch')} "
                    f"best_epoch={ckpt.get('best_epoch')} "
                    f"max_accuracy={ckpt.get('max_accuracy')}"
                )
            print(f"[pretrained] loaded {path} (weights only, head skipped){meta}")

    def _resume(self, path: str) -> None:
        ckpt = load_checkpoint(path, map_location="cpu")
        fsdp_load_state_dict(self.model, ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and self.loss_scaler is not None and ckpt["scaler"] is not None:
            self.loss_scaler.load_state_dict(ckpt["scaler"])
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.max_accuracy = float(ckpt.get("max_accuracy", 0.0))
        self.best_epoch = int(ckpt.get("best_epoch", -1))
        if is_main_process():
            print(f"[resume] {path} → epoch {self.start_epoch}")
        barrier()

    def _save(self, epoch: int, is_best: bool = False) -> None:
        barrier()
        state_dict = fsdp_state_dict(self.model)
        if not is_main_process():
            barrier()
            return
        state = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "scaler": self.loss_scaler.state_dict() if self.loss_scaler is not None else None,
            "max_accuracy": self.max_accuracy,
            "seed": self.cfg.get("seed"),
            "cfg": self.cfg,
        }
        save_checkpoint(state, self.output_dir / "checkpoint.pth")
        if is_best:
            save_checkpoint(state, self.output_dir / "best_checkpoint.pth")
        barrier()

    def _write_markdown_results(self, train_time_h: float) -> None:
        best_err = 100.0 - float(self.max_accuracy) if self.max_accuracy is not None else None
        final_err = 100.0 - float(self.final_acc1) if self.final_acc1 is not None else None
        metrics = {
            "best_acc1": self.max_accuracy,
            "best_error": best_err,
            "best_epoch": self.best_epoch,
            "final_acc1": self.final_acc1,
            "final_error": final_err,
            "final_acc5": self.final_acc5,
            "epochs": self.epochs,
            "train_time_h": train_time_h,
            "train_time_s": train_time_h * 3600.0,
            "peak_mem_gb": self.peak_mem_gb,
            "params_m": self.n_parameters / 1e6,
        }
        report = write_run_results_md(
            self.output_dir,
            self.cfg,
            metrics,
            n_parameters=self.n_parameters,
            repo_root=self.repo_root,
        )
        metrics["report_file"] = report.name
        write_epoch_table_md(self.output_dir, self.epoch_history)
        append_aggregate_results(
            self.repo_root, self.cfg, metrics, n_parameters=self.n_parameters
        )
        print(f"[results] {report}")

    def fit(self) -> None:
        cfg = self.cfg
        t0 = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        try:
            for epoch in range(self.start_epoch, self.epochs):
                sampler = getattr(self.loader_train, "sampler_obj", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

                cosine_lr(
                    self.optimizer,
                    epoch=epoch,
                    epochs=self.epochs,
                    warmup_epochs=int(cfg.get("warmup_epochs", 5)),
                    lr=self.base_lr,
                    min_lr=float(cfg.get("min_lr", 1e-5)),
                    warmup_lr=float(cfg.get("warmup_lr", 1e-6)),
                )

                train_stats = train_one_epoch(
                    self.model,
                    self.criterion,
                    self.loader_train,
                    self.optimizer,
                    self.device,
                    epoch,
                    self.loss_scaler,
                    mixup_fn=None,
                    model_ema=self.model_ema,
                    clip_grad=cfg.get("clip_grad", None),
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    show_pbar=is_main_process(),
                    kan_prune_reg=float(cfg.get("kan_prune_reg", 0.0)),
                )

                test_stats = evaluate(
                    self.loader_val,
                    self.model,
                    self.device,
                    use_amp=self.use_amp,
                    amp_dtype=self.amp_dtype,
                    show_pbar=is_main_process(),
                    epoch=epoch,
                )
                self.final_acc1 = test_stats["acc1"]
                self.final_acc5 = test_stats["acc5"]

                if is_main_process():
                    print(
                        f"Epoch {epoch:03d} | train_loss {train_stats['loss']:.4f} | "
                        f"acc1 {test_stats['acc1']:.2f} | acc5 {test_stats['acc5']:.2f} | "
                        f"lr {train_stats['lr']:.6e}"
                    )

                is_best = test_stats["acc1"] > self.max_accuracy
                if is_best:
                    self.max_accuracy = test_stats["acc1"]
                    self.best_epoch = epoch
                self._save(epoch, is_best=is_best)

                if is_main_process():
                    self.epoch_history.append(
                        {
                            "epoch": epoch,
                            "train_loss": train_stats["loss"],
                            "lr": train_stats["lr"],
                            "test_acc1": test_stats["acc1"],
                            "test_acc5": test_stats["acc5"],
                        }
                    )
                    log_stats = {
                        **{f"train_{k}": v for k, v in train_stats.items()},
                        **{f"test_{k}": v for k, v in test_stats.items()},
                        "epoch": epoch,
                        "max_accuracy": self.max_accuracy,
                        "best_epoch": self.best_epoch,
                        "seed": cfg.get("seed"),
                        "num_gpus": cfg.get("num_gpus"),
                        "world_size": get_world_size(),
                    }
                    append_log(self.output_dir, log_stats)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                self.peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)

            if is_main_process():
                elapsed = time.time() - t0
                print(
                    f"[done] best Acc@1 {self.max_accuracy:.2f}% "
                    f"(epoch {self.best_epoch}) | {elapsed / 3600:.2f}h | "
                    f"peak_mem {self.peak_mem_gb:.2f} GiB"
                )
                self._write_markdown_results(train_time_h=elapsed / 3600.0)
        finally:
            cleanup_distributed()
