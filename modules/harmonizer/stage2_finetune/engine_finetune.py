
import math
import sys
from typing import Iterable, Optional

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from timm.data import Mixup
from timm.utils import accuracy

import modules.harmonizer.util.lr_sched as lr_sched
import modules.harmonizer.util.misc as misc


def compute_balanced_accuracy(y_true, y_pred):
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    if y_true_np.size == 0 or y_pred_np.size == 0:
        return 0.0
    return balanced_accuracy_score(y_true_np, y_pred_np)


def compute_mae(y_true, y_pred):
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    if y_true_np.size == 0 or y_pred_np.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true_np - y_pred_np)))


def compute_mse(y_true, y_pred):
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    if y_true_np.size == 0 or y_pred_np.size == 0:
        return 0.0
    return float(np.mean((y_true_np - y_pred_np) ** 2))


def compute_rmse(y_true, y_pred):
    mse = compute_mse(y_true, y_pred)
    return float(math.sqrt(mse))


def compute_r2(y_true, y_pred):
    y_true_np = np.asarray(y_true, dtype=float)
    y_pred_np = np.asarray(y_pred, dtype=float)
    if y_true_np.size == 0 or y_pred_np.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true_np - y_pred_np) ** 2))
    mean_true = float(np.mean(y_true_np))
    ss_tot = float(np.sum((y_true_np - mean_true) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    mixup_fn: Optional[Mixup] = None,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if len(batch) == 3:
            samples, targets, attn_mask = batch
        elif len(batch) == 4:
            samples, targets, attn_mask, _ = batch
        else:
            raise ValueError("Expected batch with 3 or 4 elements.")
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, attn_mask)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, dataset_name, task="classification"):
    if task == "regression":
        criterion = torch.nn.MSELoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()

    batch_idx = 0
    all_preds = []
    all_targets = []
    multiclass_datasets = {"SEEDV"}

    for batch in metric_logger.log_every(data_loader, 10, header):
        if len(batch) == 3:
            images, target, attn_mask = batch
        elif len(batch) == 4:
            images, target, attn_mask, _ = batch
        else:
            raise ValueError("Expected batch with 3 or 4 elements.")
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        attn_list = None
        with torch.amp.autocast("cuda"):
            outputs = model(images, attn_mask)
            if isinstance(outputs, (tuple, list)):
                if len(outputs) == 2:
                    output, attn_list = outputs
                else:
                    output = outputs[0]
            else:
                output = outputs

            if task == "regression":
                output = output.squeeze(-1)
                target = target.to(output.dtype)
                if target.shape != output.shape:
                    target = target.view_as(output)

            try:
                loss = criterion(output, target)
            except:
                print(f"output shape: {output.shape}, target shape: {target.shape}")
                raise

        if attn_list:
            import os

            os.makedirs(f"vis_attn_map/harmonizer/{dataset_name}", exist_ok=True)
            torch.save(
                attn_list,
                f"vis_attn_map/harmonizer/{dataset_name}/attn_list_batch_{batch_idx}.pt",
            )
            print(f"Saved attn_list to attn_list_batch_{batch_idx}.pt")
            batch_idx += 1

        if task == "regression":
            preds = output.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            all_preds.append(preds)
            all_targets.append(target_np)
        else:
            acc1 = accuracy(output, target, topk=(1,))[0]

            predict = np.argmax(output.detach().cpu().numpy(), axis=1)

            target_np = target.detach().cpu().numpy()
            if dataset_name in multiclass_datasets or dataset_name == "PPMI":
                f1score = f1_score(
                    target_np, predict, average="weighted", zero_division=0
                )
            else:
                f1score = f1_score(target_np, predict, zero_division=0)

            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["f1score"].update(f1score, n=batch_size)

            all_preds.append(predict)
            all_targets.append(target_np)
    metric_logger.synchronize_between_processes()

    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_targets) if all_targets else np.array([])

    def _gather_concat(array):
        if not misc.is_dist_avail_and_initialized():
            return array
        local = torch.as_tensor(array, device=device)
        local_size = torch.tensor([local.numel()], device=device, dtype=torch.long)
        size_list = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
        dist.all_gather(size_list, local_size)
        sizes = [int(s.item()) for s in size_list]
        max_size = max(sizes) if sizes else 0
        if local.numel() < max_size:
            padded = torch.zeros(max_size, device=device, dtype=local.dtype)
            if local.numel() > 0:
                padded[: local.numel()] = local
        else:
            padded = local
        gather_list = [
            torch.zeros(max_size, device=device, dtype=local.dtype)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gather_list, padded)
        arrays = [g[: sizes[i]].cpu().numpy() for i, g in enumerate(gather_list)]
        return np.concatenate(arrays) if arrays else np.array([])

    y_true = _gather_concat(y_true)
    y_pred = _gather_concat(y_pred)

    if task == "regression":
        mae = compute_mae(y_true, y_pred)
        rmse = compute_rmse(y_true, y_pred)
        r2 = compute_r2(y_true, y_pred)
        print(
            f"* MAE {mae:.3f} RMSE {rmse:.3f} R2 {r2:.3f} "
            f"loss {metric_logger.loss.global_avg:.3f}"
        )
        metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        metrics["mae"] = mae
        metrics["rmse"] = rmse
        metrics["r2"] = r2
        return metrics

    if y_true.size == 0 or y_pred.size == 0:
        acc1 = 0.0
        f1score = 0.0
        bac = 0.0
        kappa = None
    else:
        acc1 = 100.0 * float(np.mean(y_true == y_pred))
        if dataset_name in multiclass_datasets or dataset_name == "PPMI":
            f1score = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        else:
            f1score = f1_score(y_true, y_pred, zero_division=0)
        bac = compute_balanced_accuracy(y_true, y_pred)
        kappa = None
        if dataset_name in multiclass_datasets and y_true.size and y_pred.size:
            kappa = cohen_kappa_score(y_true, y_pred)

    print(f"* Acc@1 {acc1:.3f} f1score {f1score:.3f} loss {metric_logger.loss.global_avg:.3f}")

    if kappa is None:
        print(f"* Balanced accuracy {bac:.3f}")
    else:
        print(f"* Balanced accuracy {bac:.3f} kappa {kappa:.3f}")

    metrics = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    metrics["acc1"] = acc1
    metrics["f1score"] = f1score
    metrics["bac"] = bac
    if kappa is not None:
        metrics["kappa"] = kappa
    return metrics
