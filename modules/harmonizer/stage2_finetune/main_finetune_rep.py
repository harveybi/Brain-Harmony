
import argparse
import csv
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pickle
import lmdb
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

import modules.harmonizer.util.misc as misc
from datasets.datasets import GenerateEmbedDataset_downstream
from modules.harmonizer.util.misc import NativeScalerWithGradNormCount as NativeScaler

try:
    import timm as _timm
except Exception as exc:
    _timm = None
    _timm_import_error = exc
else:
    _timm_import_error = None

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {name}: {value}") from exc


ADNI_NUM_REGIONS = 400
ADNI_TIME_TOKENS = _env_int("ADNI_TIME_TOKENS", 18)
ADNI_TOKEN_DIM = 768
ADNI_PAD_TOKENS = _env_int("ADNI_PAD_TOKENS", 1200)

DATASET_CONFIG = {
    "ADNI": {
        "loader_name": "ADNI",
        "task": "binary",
        "nb_classes": 2,
        "metric_key": "bac",
    },
    "ADHD200": {
        "loader_name": "ADHD200",
        "task": "binary",
        "nb_classes": 2,
        "metric_key": "bac",
    },
    "LEMON_fMRI": {
        "loader_name": "LEMON_fMRI",
        "task": "binary",
        "nb_classes": 2,
        "metric_key": "bac",
    },
    "SEEDV": {
        "loader_name": "SEEDV",
        "task": "multiclass",
        "nb_classes": 5,
        "metric_key": "kappa",
    },
}

DEFAULT_ADAPT_CONFIG = {
    "target_regions": ADNI_NUM_REGIONS,
    "target_time": ADNI_TIME_TOKENS,
    "token_dim": ADNI_TOKEN_DIM,
    "pad_tokens": ADNI_PAD_TOKENS,
}
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "")


def _target_token_count(adapt_config):
    return int(
        adapt_config["target_regions"] * adapt_config["target_time"]
        + adapt_config["pad_tokens"]
    )


def maybe_resize_pos_embed(model, target_tokens):
    pos_embed = getattr(model, "pos_embed", None)
    if pos_embed is None or not hasattr(pos_embed, "shape"):
        return
    expected = target_tokens + 1
    current = pos_embed.shape[1]
    if current == expected:
        return
    if expected <= 1:
        raise ValueError(f"Invalid target token count: {expected}")
    cls_token = pos_embed[:, :1, :]
    tokens = pos_embed[:, 1:, :].transpose(1, 2)
    tokens = F.interpolate(tokens, size=expected - 1, mode="linear", align_corners=False)
    tokens = tokens.transpose(1, 2)
    model.pos_embed = torch.nn.Parameter(torch.cat([cls_token, tokens], dim=1))
    print(
        "Resized pos_embed from {current} to {expected} tokens to match adapted input.".format(
            current=current, expected=expected
        )
    )


def adapt_adni_signal(
    signal,
    target_regions=ADNI_NUM_REGIONS,
    target_time=ADNI_TIME_TOKENS,
    token_dim=ADNI_TOKEN_DIM,
    pad_tokens=ADNI_PAD_TOKENS,
):
    signal = torch.as_tensor(signal, dtype=torch.float32)
    if signal.ndim == 1:
        signal = signal.unsqueeze(0)
    elif signal.ndim > 2:
        signal = signal.squeeze()
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal, got shape {tuple(signal.shape)}")

    num_regions, seq_len = signal.shape
    if num_regions < target_regions:
        pad = torch.zeros((target_regions - num_regions, seq_len), dtype=signal.dtype)
        signal = torch.cat([signal, pad], dim=0)
    elif num_regions > target_regions:
        signal = signal[:target_regions, :]

    signal = signal.unsqueeze(0)
    signal = F.interpolate(signal, size=target_time, mode="linear", align_corners=False)
    signal = signal.squeeze(0)

    tokens = signal.reshape(-1, 1)
    tokens = tokens.repeat(1, token_dim)

    attn_mask = torch.ones(target_regions * target_time, dtype=torch.int64)
    if pad_tokens:
        pad = torch.zeros((pad_tokens, token_dim), dtype=tokens.dtype)
        tokens = torch.cat([tokens, pad], dim=0)

    return tokens, attn_mask


class BrainSignalFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, adapt_config=None):
        self.base_dataset = base_dataset
        self.adapt_config = adapt_config or DEFAULT_ADAPT_CONFIG

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        signal, target = self.base_dataset[idx]
        tokens, attn_mask = adapt_adni_signal(signal, **self.adapt_config)
        target_tensor = torch.as_tensor(target)
        if target_tensor.numel() > 1:
            target = int(torch.argmax(target_tensor).item())
        else:
            target = int(target_tensor.item())
        sample_id = str(idx)
        if hasattr(self.base_dataset, "keys"):
            key_info = self.base_dataset.keys[idx]
            sample_id = f"{key_info['dataset']}:{key_info['key']}"
        return tokens, target, attn_mask, sample_id


def collate_brain_signals(batch):
    tokens, targets, attn_masks, sample_ids = zip(*batch)
    tokens = torch.stack(tokens, dim=0)
    targets = torch.tensor(targets, dtype=torch.long)
    attn_masks = torch.stack(attn_masks, dim=0)
    return tokens, targets, attn_masks, list(sample_ids)


def resolve_sample_id(base_dataset, idx):
    if hasattr(base_dataset, "keys"):
        key_info = base_dataset.keys[idx]
        return f"{key_info['dataset']}:{key_info['key']}"
    return str(idx)


def _preview_target(target, target_tensor=None):
    if target_tensor is not None:
        try:
            return target_tensor.flatten().tolist()
        except Exception:
            return repr(target)
    return repr(target)


def ensure_lmdb_split(data_root, dataset_name, split_name):
    lmdb_path = os.path.join(data_root, dataset_name, split_name, "BrainSignal.lmdb")
    if not os.path.isdir(lmdb_path):
        raise FileNotFoundError(
            f"Missing LMDB split: {lmdb_path}. Expected {dataset_name}/{split_name}/BrainSignal.lmdb"
        )
    return lmdb_path


def ensure_split_non_empty(base_dataset, split_name, dataset_name):
    if len(base_dataset) == 0:
        raise ValueError(
            f"{dataset_name} split {split_name} is empty. Check LMDB contents."
        )


def validate_binary_label_contract(base_dataset, split_name, dataset_name):
    counts = {0: 0, 1: 0}
    invalid = []

    for idx in range(len(base_dataset)):
        _, target = base_dataset[idx]
        try:
            target_tensor = torch.as_tensor(target).detach().cpu()
        except Exception:
            invalid.append((resolve_sample_id(base_dataset, idx), repr(target)))
            continue

        if target_tensor.numel() == 0:
            invalid.append((resolve_sample_id(base_dataset, idx), []))
            continue

        flat = target_tensor.flatten().to(torch.float32)

        if flat.numel() == 1:
            value = float(flat.item())
            if not np.isfinite(value) or value not in (0.0, 1.0):
                invalid.append(
                    (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
                )
            else:
                counts[int(value)] += 1
        elif flat.numel() == 2:
            values = flat.numpy()
            if not np.all(np.isfinite(values)) or not np.all(
                np.isin(values, [0.0, 1.0])
            ):
                invalid.append(
                    (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
                )
            elif int(values.sum()) != 1:
                invalid.append(
                    (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
                )
            else:
                counts[int(np.argmax(values))] += 1
        else:
            invalid.append(
                (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
            )

    if invalid:
        samples = ", ".join(f"{sid}={val}" for sid, val in invalid[:5])
        raise ValueError(
            f"{dataset_name} label contract failed for {split_name}. "
            "Expected scalar 0/1 or one-hot [1,0]/[0,1] with finite values. "
            f"Examples: {samples}"
        )

    print(
        f"{dataset_name} label histogram ({split_name}): 0={counts[0]} 1={counts[1]}"
    )
    return counts


def validate_multiclass_label_contract(
    base_dataset, split_name, dataset_name, nb_classes
):
    counts = {cls: 0 for cls in range(nb_classes)}
    invalid = []

    for idx in range(len(base_dataset)):
        _, target = base_dataset[idx]
        try:
            target_tensor = torch.as_tensor(target).detach().cpu()
        except Exception:
            invalid.append((resolve_sample_id(base_dataset, idx), repr(target)))
            continue

        if target_tensor.numel() == 0:
            invalid.append((resolve_sample_id(base_dataset, idx), []))
            continue

        flat = target_tensor.flatten().to(torch.float32)
        if flat.numel() == 1:
            value = float(flat.item())
            if not np.isfinite(value) or int(value) != value:
                invalid.append(
                    (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
                )
                continue
            value = int(value)
            if value < 0 or value >= nb_classes:
                invalid.append(
                    (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
                )
                continue
            counts[value] += 1
        elif flat.numel() == nb_classes:
            values = flat.numpy()
            if not np.all(np.isfinite(values)) or not np.all(
                np.isin(values, [0.0, 1.0])
            ):
                invalid.append(
                    (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
                )
            elif int(values.sum()) != 1:
                invalid.append(
                    (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
                )
            else:
                counts[int(np.argmax(values))] += 1
        else:
            invalid.append(
                (resolve_sample_id(base_dataset, idx), _preview_target(target, flat))
            )

    if invalid:
        samples = ", ".join(f"{sid}={val}" for sid, val in invalid[:5])
        raise ValueError(
            f"{dataset_name} label contract failed for {split_name}. "
            "Expected class index or one-hot vector with finite values. "
            f"Examples: {samples}"
        )

    summary = " ".join(f"{cls}={count}" for cls, count in counts.items())
    print(f"{dataset_name} label histogram ({split_name}): {summary}")
    return counts


def validate_label_cardinality(counts, split_name, dataset_name, nb_classes):
    missing = [cls for cls, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"{dataset_name} label cardinality failed for {split_name}. "
            f"Missing classes: {missing} (expected {nb_classes} classes)."
        )


def validate_adni_label_contract(base_dataset, split_name):
    return validate_binary_label_contract(base_dataset, split_name, "ADNI")


def validate_dataset_splits(data_root, dataset_cfg, dataset_name, splits):
    for split_name, base_dataset in splits:
        ensure_lmdb_split(data_root, dataset_cfg["loader_name"], split_name)
        ensure_split_non_empty(base_dataset, split_name, dataset_name)
        if dataset_cfg["task"] == "binary":
            if dataset_name == "ADNI":
                counts = validate_adni_label_contract(base_dataset, split_name)
            else:
                counts = validate_binary_label_contract(
                    base_dataset, split_name, dataset_name
                )
            validate_label_cardinality(
                counts, split_name, dataset_name, dataset_cfg["nb_classes"]
            )
        elif dataset_cfg["task"] == "multiclass":
            counts = validate_multiclass_label_contract(
                base_dataset,
                split_name,
                dataset_name,
                dataset_cfg["nb_classes"],
            )
            validate_label_cardinality(
                counts, split_name, dataset_name, dataset_cfg["nb_classes"]
            )

def resolve_git_commit(repo_root):
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception:
        git_commit = "unknown"
    return git_commit


def _select_metric(metric_name, y_true, y_pred):
    if metric_name == "bac":
        return balanced_accuracy_score(y_true, y_pred)
    if metric_name == "kappa":
        return cohen_kappa_score(y_true, y_pred)
    raise ValueError(f"Unsupported metric for run artifacts: {metric_name}")


class BrainSignalDatasetFallback(torch.utils.data.Dataset):
    def __init__(self, root, split="train", dataset=None):
        self.root = root
        self.split = split
        self.dataset = dataset
        self.env = {}
        self.keys = []

        if dataset is None:
            raise ValueError("dataset must be provided for BrainSignalDatasetFallback")

        path = os.path.join(root, dataset, split, "BrainSignal.lmdb")
        if os.path.isdir(os.path.join(root, dataset)):
            self.env[dataset] = lmdb.open(
                path, readonly=True, lock=False, readahead=False, meminit=False
            )
            with self.env[dataset].begin(write=False) as txn:
                self.keys.extend(pickle.loads(txn.get("__keys__".encode("ascii"))))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        dataset = self.keys[idx]["dataset"]
        key = self.keys[idx]["key"]

        with self.env[dataset].begin(write=False) as txn:
            sample = pickle.loads(txn.get(key.encode("ascii")))
            signal = sample["signal"]
            target = sample["y"]
            if "CamCAN" in dataset:
                target = [target["age"][0]]
            elif "LEMON" in dataset:
                target = target[0]

        mean = signal.mean(axis=-1, keepdims=True)
        std = signal.std(axis=-1, keepdims=True)
        signal = (signal - mean) / (std + 1e-8)

        orig_num_region, orig_signal_length = signal.shape
        signal_length = 200
        if signal_length < orig_signal_length:
            signal = signal[:, :signal_length]
            orig_signal_length = signal_length

        padding_size_l = (-orig_signal_length % 200 + 1) // 2
        padding_size_r = (-orig_signal_length % 200) - padding_size_l
        signal = np.pad(
            signal,
            pad_width=((0, 0), (padding_size_l, padding_size_r)),
            mode="constant",
            constant_values=0,
        )

        return torch.FloatTensor(signal), torch.FloatTensor(target)


def prepare_Brain_dataset_fallback(root, dataset):
    train_dataset = BrainSignalDatasetFallback(root, "train", dataset)
    val_dataset = BrainSignalDatasetFallback(root, "val", dataset)
    test_dataset = BrainSignalDatasetFallback(root, "test", dataset)
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    return train_dataset, test_dataset, val_dataset


def load_brain_datasets(data_root, dataset_name):
    try:
        from rep_scripts.utils import prepare_Brain_dataset
    except Exception as exc:
        if dataset_name == "ADNI":
            raise RuntimeError(
                "ADNI contract requires rep_scripts.utils.prepare_Brain_dataset; "
                "fix its import dependencies before running."
            ) from exc
        print(
            "Warning: rep_scripts.utils import failed; using fallback loader. "
            f"Original error: {exc}"
        )
        return prepare_Brain_dataset_fallback(data_root, dataset_name)
    return prepare_Brain_dataset(data_root, dataset_name)


def load_training_deps():
    if _timm is None:
        raise RuntimeError(
            "timm import failed; fix timm/PyTorch compatibility before training. "
            f"Original error: {_timm_import_error}"
        )
    if _timm.__version__ != "0.9.12":
        raise RuntimeError(
            f"Expected timm==0.9.12, found {_timm.__version__}. Update the env."
        )
    from timm.data.mixup import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
    from timm.models.layers import trunc_normal_

    import modules.harmonizer.stage2_finetune.models as models_enc_one_tok_reg
    import modules.harmonizer.util.lr_decay as lrd
    from modules.harmonizer.stage2_finetune.engine_finetune import (
        evaluate,
        train_one_epoch,
    )

    return {
        "Mixup": Mixup,
        "LabelSmoothingCrossEntropy": LabelSmoothingCrossEntropy,
        "SoftTargetCrossEntropy": SoftTargetCrossEntropy,
        "trunc_normal_": trunc_normal_,
        "models_enc_one_tok_reg": models_enc_one_tok_reg,
        "lrd": lrd,
        "evaluate": evaluate,
        "train_one_epoch": train_one_epoch,
    }


@torch.no_grad()
def collect_predictions(data_loader, model, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_ids = []
    all_probs = []

    for batch in data_loader:
        if len(batch) == 4:
            samples, targets, attn_mask, sample_ids = batch
        else:
            raise ValueError("Expected batch with 4 elements for prediction collection.")

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(samples, attn_mask)

        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(-1)
        probs = torch.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_ids.extend(sample_ids)
        all_probs.append(probs.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    prob = np.concatenate(all_probs)

    return y_true, y_pred, prob, all_ids


def write_run_artifact(
    output_dir,
    run_id,
    split,
    dataset_name,
    seed,
    metric_name,
    metric_value,
    predictions,
    git_commit,
):
    payload = {
        "dataset_name": dataset_name,
        "seed": int(seed),
        "split": split,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "predictions": predictions,
        "git_commit": git_commit,
    }
    path = os.path.join(output_dir, f"run-{run_id}.{split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_overfit_sanity(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    loss_scaler,
    args,
    tolerance,
    train_one_epoch,
):
    model.train()
    batch = next(iter(data_loader))
    samples, targets, attn_mask, _ = batch
    samples = samples.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    attn_mask = attn_mask.to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        outputs = model(samples, attn_mask)
        initial_loss = criterion(outputs, targets).item()

    for _ in range(args.overfit_epochs):
        train_one_epoch(
            model,
            criterion,
            data_loader,
            optimizer,
            device,
            args.start_epoch,
            loss_scaler,
            args.clip_grad,
            None,
            log_writer=None,
            args=args,
        )

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        outputs = model(samples, attn_mask)
        final_loss = criterion(outputs, targets).item()

    probs = torch.softmax(outputs, dim=-1)
    preds = torch.argmax(probs, dim=-1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    bac = balanced_accuracy_score(targets_np, preds)

    print(
        f"Overfit sanity: initial_loss={initial_loss:.6f} final_loss={final_loss:.6f} bac={bac:.4f}"
    )
    if not np.isfinite(bac):
        raise RuntimeError("Overfit sanity failed: balanced accuracy is NaN/inf.")
    if final_loss > (initial_loss - tolerance):
        raise RuntimeError(
            "Overfit sanity failed: loss did not decrease within tolerance."
        )


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=5, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    (
        parser.add_argument(
            "--aa",
            type=str,
            default="rand-m9-mstd0.5-inc1",
            metavar="NAME",
            help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
        ),
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )

    parser.add_argument("--encoders_freeze", action="store_true", default=False)

    parser.add_argument(
        "--data_path",
        default=DEFAULT_DATA_ROOT,
        type=str,
        help="dataset root (defaults to DATA_ROOT if set)",
    )
    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--dataset_name", default="", type=str, help="dataset_name")
    parser.add_argument("--split_seed", default="0", type=str, help="dataset_name")
    parser.add_argument(
        "--dataset_init_only",
        action="store_true",
        help="Initialize dataset + print shapes, then exit",
    )
    parser.add_argument(
        "--shape_print_only",
        action="store_true",
        help="Print raw/adapted shapes + run one forward pass, then exit",
    )
    parser.add_argument(
        "--overfit_batches",
        default=0,
        type=int,
        help="Number of batches to use for 1-batch overfit sanity mode (0 disables).",
    )
    parser.add_argument(
        "--overfit_epochs",
        default=3,
        type=int,
        help="Epochs to run in overfit sanity mode.",
    )
    parser.add_argument(
        "--overfit_tolerance",
        default=1e-4,
        type=float,
        help="Minimum loss decrease required for overfit sanity mode.",
    )
    parser.add_argument(
        "--run_id",
        default="",
        type=str,
        help="Optional run id for artifact files (default: timestamp).",
    )

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    dataset_cfg = DATASET_CONFIG.get(args.dataset_name)
    train_base = None

    if dataset_cfg is not None:
        if not args.data_path:
            raise ValueError(
                f"{args.dataset_name} requires --data_path or DATA_ROOT to be set."
            )
        if args.nb_classes == 1000:
            args.nb_classes = dataset_cfg["nb_classes"]
        elif args.nb_classes != dataset_cfg["nb_classes"]:
            print(
                "Warning: nb_classes mismatch for {dataset}. Using nb_classes={nb}.".format(
                    dataset=args.dataset_name, nb=args.nb_classes
                )
            )
        train_base, test_base, val_base = load_brain_datasets(
            args.data_path, dataset_cfg["loader_name"]
        )
        validate_dataset_splits(
            args.data_path,
            dataset_cfg,
            args.dataset_name,
            [
                ("train", train_base),
                ("val", val_base),
                ("test", test_base),
            ],
        )
        dataset_train = BrainSignalFinetuneDataset(train_base)
        dataset_val = BrainSignalFinetuneDataset(val_base)
        dataset_test = BrainSignalFinetuneDataset(test_base)
        collate_fn = collate_brain_signals
    elif args.dataset_name == "AbideI":
        root_dir = "experiments/stage0_embed/downstream_embed/AbideI"
        splits_file = f"/scratch/Projects/project_312_HelenZhou/ABIDE1_fMRI_T1/data_splits_seed{args.split_seed}.json"
        dataset_train = GenerateEmbedDataset_downstream(
            root_dir=root_dir, splits_file=splits_file, split="train"
        )
        dataset_test = GenerateEmbedDataset_downstream(
            root_dir=root_dir, splits_file=splits_file, split="val"
        )
        dataset_val = GenerateEmbedDataset_downstream(
            root_dir=root_dir, splits_file=splits_file, split="test"
        )
        collate_fn = None
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")

    if args.dataset_init_only and args.shape_print_only:
        raise ValueError(
            "Use only one of --dataset_init_only or --shape_print_only."
        )

    if True:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if len(dataset_train) < args.batch_size:
        drop_last_train = False
    else:
        drop_last_train = True
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last_train,
        collate_fn=collate_fn,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=collate_fn,
    )

    if args.dataset_init_only:
        print(f"Dataset root: {args.data_path}")
        print(
            f"Splits: train={len(dataset_train)} val={len(dataset_val)} test={len(dataset_test)}"
        )
        sample_batch = next(iter(data_loader_train))
        if len(sample_batch) == 4:
            samples, targets, attn_mask, _ = sample_batch
        else:
            samples, targets, attn_mask = sample_batch
        print(
            f"Batch shapes: samples={tuple(samples.shape)} targets={tuple(targets.shape)} attn_mask={tuple(attn_mask.shape)}"
        )
        return

    if args.shape_print_only:
        print(f"Dataset root: {args.data_path}")
        print(
            f"Splits: train={len(dataset_train)} val={len(dataset_val)} test={len(dataset_test)}"
        )
        if train_base is not None and len(train_base) > 0:
            raw_signal, raw_target = train_base[0]
            raw_signal = np.asarray(raw_signal)
            raw_target = np.asarray(raw_target)
            print(
                "Raw sample shapes: signal={signal} target={target}".format(
                    signal=tuple(raw_signal.shape), target=tuple(raw_target.shape)
                )
            )
        adapted_sample = dataset_train[0]
        if len(adapted_sample) == 4:
            tokens, target, attn_mask, _ = adapted_sample
        else:
            tokens, target, attn_mask = adapted_sample
        print(
            "Adapted sample shapes: tokens={tokens} target={target} attn_mask={attn}".format(
                tokens=tuple(tokens.shape),
                target=tuple(np.asarray(target).shape),
                attn=tuple(attn_mask.shape),
            )
        )
        sample_batch = next(iter(data_loader_train))
        if len(sample_batch) == 4:
            samples, targets, attn_mask, _ = sample_batch
        else:
            samples, targets, attn_mask = sample_batch
        print(
            "Batch shapes: samples={samples} targets={targets} attn_mask={attn}".format(
                samples=tuple(samples.shape),
                targets=tuple(targets.shape),
                attn=tuple(attn_mask.shape),
            )
        )
        deps = load_training_deps()
        models_enc_one_tok_reg = deps["models_enc_one_tok_reg"]
        model = models_enc_one_tok_reg.__dict__[args.model](
            img_size=(160, 192, 160),
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        if dataset_cfg is not None:
            maybe_resize_pos_embed(model, _target_token_count(DEFAULT_ADAPT_CONFIG))
        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(
                samples.to(device, non_blocking=True),
                attn_mask.to(device, non_blocking=True),
            )
        print(f"Forward pass output shape: {tuple(outputs.shape)}")
        return

    deps = load_training_deps()
    Mixup = deps["Mixup"]
    LabelSmoothingCrossEntropy = deps["LabelSmoothingCrossEntropy"]
    SoftTargetCrossEntropy = deps["SoftTargetCrossEntropy"]
    trunc_normal_ = deps["trunc_normal_"]
    models_enc_one_tok_reg = deps["models_enc_one_tok_reg"]
    lrd = deps["lrd"]
    evaluate = deps["evaluate"]
    train_one_epoch = deps["train_one_epoch"]

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = models_enc_one_tok_reg.__dict__[args.model](
        img_size=(160, 192, 160),
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    if dataset_cfg is not None:
        maybe_resize_pos_embed(model, _target_token_count(DEFAULT_ADAPT_CONFIG))

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu", weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        if isinstance(checkpoint_model, dict):
            if any(k.startswith("encoder.") for k in checkpoint_model):
                mapped = {}
                for k, v in checkpoint_model.items():
                    if not k.startswith("encoder."):
                        continue
                    new_key = k[len("encoder.") :]
                    if new_key.startswith("norm.") and args.global_pool:
                        new_key = "fc_norm." + new_key[len("norm.") :]
                    mapped[new_key] = v
                checkpoint_model = mapped
            filtered = {
                k: v
                for k, v in checkpoint_model.items()
                if k in state_dict and getattr(v, "shape", None) == state_dict[k].shape
            }
            if len(filtered) != len(checkpoint_model):
                print(
                    "Checkpoint filtering: loaded {loaded}/{total} keys (shape/name match).".format(
                        loaded=len(filtered), total=len(checkpoint_model)
                    )
                )
            checkpoint_model = filtered
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]


        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        expected_missing = {"head.weight", "head.bias"}
        if args.global_pool:
            expected_missing |= {"fc_norm.weight", "fc_norm.bias"}
        extra_missing = set(msg.missing_keys) - expected_missing
        if extra_missing:
            print(
                "Warning: checkpoint missing unexpected keys: {}".format(
                    sorted(extra_missing)
                )
            )

        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("number of training samples: %d" % (len(dataset_train)))
    print("number of evaluation samples: %d" % (len(dataset_val)))

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.dataset_name)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)

    if args.overfit_batches > 0:
        subset_size = min(len(dataset_train), args.overfit_batches * args.batch_size)
        subset_indices = list(range(subset_size))
        subset = torch.utils.data.Subset(dataset_train, subset_indices)
        data_loader_overfit = torch.utils.data.DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            collate_fn=collate_fn,
        )
        run_overfit_sanity(
            model,
            criterion,
            data_loader_overfit,
            optimizer,
            device,
            loss_scaler,
            args,
            args.overfit_tolerance,
            train_one_epoch,
        )
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_metric = -float("inf")
    metric_key = (
        dataset_cfg["metric_key"] if dataset_cfg is not None else "f1score"
    )
    for epoch in range(args.start_epoch, args.epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args,
        )
        epoch_time = time.time() - epoch_start
        step_time = train_stats.get("iter_time")
        data_time = train_stats.get("data_time")
        max_mem_mb = 0.0
        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        print(
            "Perf epoch={epoch} epoch_time_s={epoch_time:.2f} step_time_s={step_time:.4f} "
            "data_time_s={data_time:.4f} max_mem_mb={max_mem:.0f} eff_batch_size={eff_bs}".format(
                epoch=epoch,
                epoch_time=epoch_time,
                step_time=step_time or 0.0,
                data_time=data_time or 0.0,
                max_mem=max_mem_mb,
                eff_bs=eff_batch_size,
            )
        )

        test_stats = evaluate(data_loader_val, model, device, args.dataset_name)
        val_kappa = test_stats.get("kappa")
        val_kappa_msg = f" kappa={val_kappa:.3f}" if val_kappa is not None else ""
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}% {test_stats['f1score']:.1f}% bac={test_stats['bac']:.3f}{val_kappa_msg}"
        )

        test_test_stats = evaluate(data_loader_test, model, device, args.dataset_name)
        test_kappa = test_test_stats.get("kappa")
        test_kappa_msg = f" kappa={test_kappa:.3f}" if test_kappa is not None else ""
        print(
            f"Accuracy of the network on the test dataset {len(dataset_test)} test images: {test_test_stats['acc1']:.1f}% {test_test_stats['f1score']:.1f}% bac={test_test_stats['bac']:.3f}{test_kappa_msg}"
        )

        if args.output_dir:
            if test_stats[metric_key] >= max_metric:
                val_stats = test_stats
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    latest=False,
                )
            else:
                misc.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    latest=True,
                )
        max_metric = max(max_metric, test_stats[metric_key])
        print(f"Max {metric_key}: {max_metric:.4f}")

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_f1score", test_stats["f1score"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)
            log_writer.add_scalar("perf/test_bac", test_stats["bac"], epoch)
            if "kappa" in test_stats:
                log_writer.add_scalar("perf/test_kappa", test_stats["kappa"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    args.resume = args.output_dir + "/checkpoint-best.pth"
    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    test_stats = evaluate(data_loader_test, model, device, args.dataset_name)
    print(
        f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}% bac={test_stats['bac']:.3f}"
    )

    header = [
        "name",
        "val_loss",
        "val_acc1",
        "val_f1score",
        "val_bac",
        "test_loss",
        "test_acc1",
        "test_f1score",
        "test_bac",
    ]
    csv_file = os.path.join(args.output_dir, "results.csv")
    write_header = not os.path.exists(csv_file)

    row_name = f"{args.dataset_name}_split{args.split_seed}"

    with open(csv_file, mode="a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                row_name,
                val_stats["loss"],
                val_stats["acc1"],
                val_stats["f1score"],
                val_stats["bac"],
                test_stats["loss"],
                test_stats["acc1"],
                test_stats["f1score"],
                test_stats["bac"],
            ]
        )
        fcntl.flock(f, fcntl.LOCK_UN)

    if (
        args.output_dir
        and dataset_cfg is not None
        and dataset_cfg["task"] in {"binary", "multiclass"}
    ):
        run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        git_commit = resolve_git_commit(PROJECT_ROOT)
        metric_name = dataset_cfg["metric_key"]

        train_true, train_pred, train_prob, train_ids = collect_predictions(
            data_loader_train, model, device
        )
        val_true, val_pred, val_prob, val_ids = collect_predictions(
            data_loader_val, model, device
        )
        test_true, test_pred, test_prob, test_ids = collect_predictions(
            data_loader_test, model, device
        )

        splits = [
            ("train", train_true, train_pred, train_prob, train_ids),
            ("val", val_true, val_pred, val_prob, val_ids),
            ("test", test_true, test_pred, test_prob, test_ids),
        ]

        def _select_probabilities(probabilities, task):
            prob_array = np.asarray(probabilities)
            if task == "binary":
                if prob_array.ndim == 1:
                    return prob_array
                if prob_array.shape[1] == 1:
                    return prob_array[:, 0]
                return prob_array[:, 1]
            return prob_array

        for split_name, y_true, y_pred, prob, ids in splits:
            y_true = np.asarray(y_true).reshape(-1)
            y_pred = np.asarray(y_pred).reshape(-1)
            prob_values = _select_probabilities(prob, dataset_cfg["task"])
            metric_value = _select_metric(metric_name, y_true, y_pred)
            predictions = [
                {
                    "id": sample_id,
                    "y_true": int(y_t),
                    "y_pred": int(y_p),
                    "prob": (
                        [float(value) for value in np.asarray(p).tolist()]
                        if dataset_cfg["task"] == "multiclass"
                        else float(p)
                    ),
                }
                for sample_id, y_t, y_p, p in zip(ids, y_true, y_pred, prob_values)
            ]
            write_run_artifact(
                args.output_dir,
                run_id,
                split_name,
                args.dataset_name,
                args.seed,
                metric_name,
                metric_value,
                predictions,
                git_commit,
            )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
