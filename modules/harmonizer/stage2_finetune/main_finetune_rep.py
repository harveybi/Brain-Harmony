
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
from sklearn.metrics import balanced_accuracy_score

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


ADNI_NUM_REGIONS = 400
ADNI_TIME_TOKENS = 18
ADNI_TOKEN_DIM = 768
ADNI_PAD_TOKENS = 1200


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


class AdniFinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        signal, target = self.base_dataset[idx]
        tokens, attn_mask = adapt_adni_signal(signal)
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


def collate_adni(batch):
    tokens, targets, attn_masks, sample_ids = zip(*batch)
    tokens = torch.stack(tokens, dim=0)
    targets = torch.tensor(targets, dtype=torch.long)
    attn_masks = torch.stack(attn_masks, dim=0)
    return tokens, targets, attn_masks, list(sample_ids)


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


def load_adni_datasets(data_root):
    try:
        from rep_scripts.utils import prepare_Brain_dataset
    except Exception as exc:
        print(
            "Warning: rep_scripts.utils import failed; using fallback ADNI loader. "
            f"Original error: {exc}"
        )
        return prepare_Brain_dataset_fallback(data_root, "ADNI")
    return prepare_Brain_dataset(data_root, "ADNI")


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

        probs = torch.softmax(outputs, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_ids.extend(sample_ids)
        all_probs.append(probs[:, 1].detach().cpu().numpy())

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
        default="/datasets01/imagenet_full_size/061417/",
        type=str,
        help="dataset path",
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

    if args.dataset_name == "ADNI":
        train_base, test_base, val_base = load_adni_datasets(args.data_path)
        dataset_train = AdniFinetuneDataset(train_base)
        dataset_val = AdniFinetuneDataset(val_base)
        dataset_test = AdniFinetuneDataset(test_base)
        collate_fn = collate_adni
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

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location="cpu", weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]


        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


        if args.global_pool:
            assert set(msg.missing_keys) == {
                "head.weight",
                "head.bias",
                "fc_norm.weight",
                "fc_norm.bias",
            }
        else:
            assert set(msg.missing_keys) == {"head.weight", "head.bias"}

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
    metric_key = "bac" if args.dataset_name == "ADNI" else "f1score"
    for epoch in range(args.start_epoch, args.epochs):
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

        test_stats = evaluate(data_loader_val, model, device, args.dataset_name)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}% {test_stats['f1score']:.1f}% bac={test_stats['bac']:.3f}"
        )

        test_test_stats = evaluate(data_loader_test, model, device, args.dataset_name)
        print(
            f"Accuracy of the network on the test dataset {len(dataset_test)} test images: {test_test_stats['acc1']:.1f}% {test_test_stats['f1score']:.1f}% bac={test_test_stats['bac']:.3f}"
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

    if args.output_dir and args.dataset_name == "ADNI":
        run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        git_commit = resolve_git_commit(PROJECT_ROOT)

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

        for split_name, y_true, y_pred, prob, ids in splits:
            bac = balanced_accuracy_score(y_true, y_pred)
            predictions = [
                {
                    "id": sample_id,
                    "y_true": int(y_t),
                    "y_pred": int(y_p),
                    "prob": float(p),
                }
                for sample_id, y_t, y_p, p in zip(ids, y_true, y_pred, prob)
            ]
            write_run_artifact(
                args.output_dir,
                run_id,
                split_name,
                args.dataset_name,
                args.seed,
                "bac",
                bac,
                predictions,
                git_commit,
            )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
