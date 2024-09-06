import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import randomname
import segmentation_models_pytorch as smp
import torch
import wandb
from adabelief_pytorch import AdaBelief
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
from cv2 import BORDER_CONSTANT
from torch import nn
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, ConcatDataset
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex
from tqdm import tqdm

from dataset import MGRDatasetColumns, WatershedFromLabels
from loss import LOSS_FN, SobelFilter

ARCH = {
    "unet": smp.Unet,
    "unet+": smp.UnetPlusPlus,
    "manet": smp.MAnet,
}


def get_model(arch: str, encoder: str, num_targets: int, down_scale_factor: int):
    model = ARCH[arch](
        encoder,
        in_channels=3,
        classes=num_targets,
    )
    if down_scale_factor > 1:
        dims = model.segmentation_head[0].weight.shape[1]
        model.segmentation_head = nn.Sequential(
            nn.AvgPool2d(down_scale_factor, down_scale_factor),
            nn.Conv2d(dims, num_targets, 1)
        )

    return model


class Trainer:
    def __init__(self, args):
        self.args = args
        self.in_memory = not args.no_in_memory
        self.debug = args.debug
        self.wandb = not args.no_wandb
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.amp = not args.no_amp and str(self.device) != "cpu"
        self.val_epochs = args.val_epochs
        self.batch_size = args.batch_size
        self.batch_size_val = args.batch_size_val
        self.arch = args.arch.lower()
        args.no_amp = not self.amp

        if args.run is None:
            name = randomname.get_name()
        else:
            name = args.run

        # setup exp folder
        self.folder_exp = Path(args.exp_root) / name
        self.folder_exp.mkdir(parents=True, exist_ok=True)
        self.path_model = self.folder_exp / "model.pt"

        self.encoder_warmup_epochs = args.encoder_warmup_epochs
        self.down_scale_factor = args.down_scale_factor
        self.resize_size = args.resize_size

        if self.wandb:
            wandb_keys = [
                "lr", "batch_size", "batch_size_val", "num_epochs", "weight_decay", "arch",
                "train_on_all", "eps", "encoder", "loss", "selection_metric", "no_amp",
                "encoder_warmup_epochs", "column_ids", "pretrain_weights",
                "resize_size", "down_scale_factor", "sobel_alpha"
            ]
            wandb_args = {k: getattr(args, k) for k in wandb_keys}
            # Initialize Weights & Biases
            wandb.init(project=args.project_name, entity=args.entity, name=name, config=wandb_args, dir=self.folder_exp)

        self.num_energy_levels = args.num_energy_levels

        resize_transform = [Resize(args.resize_size, args.resize_size)] if args.resize_size is not None else []

        standard_transform = [
            WatershedFromLabels(self.num_energy_levels, self.down_scale_factor),
            ToTensorV2(transpose_mask=True),
        ]

        # Define data transformations for training and validation
        self.transform = A.Compose(resize_transform + standard_transform)
        self.train_transform = A.Compose(resize_transform + [
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.5, border_mode=BORDER_CONSTANT),
            A.ElasticTransform(alpha=0.25, border_mode=BORDER_CONSTANT, p=0.25, approximate=True),
            A.ColorJitter(),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
        ] + standard_transform)

        self.column_ids = [int(idx) for idx in args.column_ids.split(" ")]
        self.num_targets = len(self.column_ids) * self.num_energy_levels  # number of columns times energy_levels
        self.split_file = pd.read_csv(args.split_csv)
        train = self.split_file.query("split == 'train'")["folders"].values
        val = self.split_file.query("split == 'val'")["folders"].values
        test = self.split_file.query("split == 'test'")["folders"].values

        # Create dataset instances for training and validation
        self.train_dataset = MGRDatasetColumns(
            folders=train, column_ids=self.column_ids, transform=self.train_transform,
            in_memory=self.in_memory, memory_file=f"train_{args.memory_file}", column_count=args.column_count
        )

        self.val_dataset = MGRDatasetColumns(
            folders=val, column_ids=self.column_ids, transform=self.transform,
            in_memory=self.in_memory, memory_file=f"val_{args.memory_file}", column_count=args.column_count
        )
        self.test_dataset = MGRDatasetColumns(
            folders=test, column_ids=self.column_ids, transform=self.transform,
            in_memory=self.in_memory, memory_file=f"test_{args.memory_file}", column_count=args.column_count
        )

        # Define data loaders for training and validation
        if args.train_on_all:
            train_dataset = ConcatDataset([self.train_dataset, self.val_dataset, self.test_dataset])
        else:
            train_dataset = self.train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=args.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size_val, num_workers=args.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size_val, num_workers=args.num_workers)

        # Define the UNet model
        self.encoder = args.encoder

        self.model = get_model(self.arch, self.encoder, self.num_targets, self.down_scale_factor)
        self.model.to(self.device)

        self.sobel_alpha = args.sobel_alpha
        if self.sobel_alpha > 0.0:
            self.sobel = SobelFilter()
            self.sobel.to(self.device)

        self.criterion = LOSS_FN[args.loss]()

        # Define optim
        model_params = [
            {"params": self.model.encoder.parameters(), "lr": args.lr_encoder},
            {"params": self.model.decoder.parameters()},
            {"params": self.model.segmentation_head.parameters()},
        ]
        self.optim = AdaBelief(model_params, lr=args.lr, eps=args.eps, betas=(0.9, 0.999),
                               weight_decay=args.weight_decay, weight_decouple=False, rectify=False,
                               print_change_log=False)

        # Define gradient scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        # Initialize metrics

        self.metrics = {
            'accuracy': [Accuracy("binary").to(self.device) for _ in range(self.num_targets)],
            'precision': [Precision("binary").to(self.device) for _ in range(self.num_targets)],
            'recall': [Recall("binary").to(self.device) for _ in range(self.num_targets)],
            'f1_score': [F1Score("binary").to(self.device) for _ in range(self.num_targets)],
            'iou': [JaccardIndex("binary").to(self.device) for _ in range(self.num_targets)],
        }
        self.best_val_score = -np.inf
        self.selection_metric = args.selection_metric
        self.epoch = 0

        if args.pretrain_weights is not None:
            pretrain_weights = Path(args.pretrain_weights)
            self.load_model(pretrain_weights)

        if self.path_model.exists():
            self.load_model()

        # freeze weights on decoder by default
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def train_loop(self, dataloader):
        self.model.train()
        train_loss = 0.0

        pbar = tqdm(dataloader, desc="Training")
        for images, y_energy, n_rows in pbar:
            with torch.autocast(device_type=str(self.device), dtype=torch.float16, enabled=self.amp):
                images, y_energy = images.to(self.device), y_energy.to(self.device)
                preds = self.model(images)

                w = None

                if self.sobel_alpha > 0.0:
                    with torch.no_grad():
                        shape = y_energy.shape
                        y_energy_c = y_energy.reshape(shape[0] * len(self.column_ids), self.num_energy_levels,
                                                      *shape[2:]).sum(1, keepdims=True).float()
                        y_sobel = self.sobel(y_energy_c)
                        pred_energy_c = preds.reshape(
                            shape[0] * len(self.column_ids), self.num_energy_levels, *shape[2:]
                        )
                        pred_energy_d = (pred_energy_c > 0.).cumprod(1).sum(1, keepdims=True).float()
                        # pred_energy_c = (pred_energy_c.sigmoid() * pred_energy_d).cumprod(1).sum(1, keepdims=True)
                        pred_sobel = self.sobel(pred_energy_d)

                        w = 1 + self.sobel_alpha * mse_loss(y_sobel, pred_sobel, reduction="none")
                        w = w.repeat_interleave(self.num_energy_levels, 1).reshape(shape)

                loss = self.criterion(preds, y_energy, w)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            pbar.set_postfix_str(f"loss: {loss.item():0.3f}")
            train_loss += loss.item() * images.size(0)
            if self.debug:
                break

        train_loss /= len(dataloader.dataset)
        return train_loss

    @torch.no_grad()
    def val_loop(self, dataloader, name):
        self.model.eval()
        val_loss = 0.0
        [[self.metrics[k][c].reset() for c in range(self.num_targets)] for k in self.metrics]

        for images, y_energy, n_rows in tqdm(dataloader, desc=f"Validation on {name} set"):
            with torch.autocast(device_type=str(self.device), dtype=torch.float16, enabled=self.amp):
                images, y_energy = images.to(self.device), y_energy.to(self.device)

                preds = self.model(images)

                preds = preds.reshape(y_energy.shape)

                loss = self.criterion(preds, y_energy, None)

            val_loss += loss.item() * images.size(0)

            # Update metrics
            for metric in self.metrics.values():
                for c, _ in enumerate(self.column_ids):
                    for i in range(self.num_energy_levels):
                        c_i = c * self.num_energy_levels + i
                        metric[c_i].update(preds[:, c_i], y_energy[:, c_i])

            if self.debug:
                break

        val_loss /= len(dataloader)
        metrics = {}
        for metric in self.metrics:
            for c, idx in enumerate(self.column_ids):
                for i in range(self.num_energy_levels):
                    c_i = c * self.num_energy_levels + i
                    metrics[f"{name}_{metric}_{idx}_{i + 1}"] = self.metrics[metric][c_i].compute().cpu().numpy()

        mask_list = []
        if self.wandb:
            # also log some images
            num_log_images = min(5, self.batch_size)

            # Convert logits to binary predictions to visualize and summarize energy levels
            preds_ = preds >= 0.0
            shape = preds_.shape
            preds_ = preds_.reshape(shape[0], len(self.column_ids), self.num_energy_levels, *shape[-2:])
            labels_ = y_energy.reshape(*preds_.shape)
            preds_ = preds_.cumprod(2).sum(2)
            preds_ = nn.functional.interpolate(preds_.float(), scale_factor=self.down_scale_factor).cpu().int().numpy()
            labels_ = labels_.sum(2)
            labels_ = nn.functional.interpolate(labels_.float(),
                                                scale_factor=self.down_scale_factor).cpu().int().numpy()

            for i in range(num_log_images):
                class_labels = {0: "bg"}
                class_labels.update({
                    i + 1: str(i + 1) for i in range(self.num_energy_levels)
                })

                masks = {
                    f"pred_{idx}": {
                        "mask_data": preds_[i, c], "class_labels": class_labels,
                    } for c, idx in enumerate(self.column_ids)
                }
                masks.update({
                    f"ground_truth_{idx}": {
                        "mask_data": labels_[i, c], "class_labels": class_labels
                    } for c, idx in enumerate(self.column_ids)
                })

                mask_list.append(wandb.Image(
                    images[i],
                    masks=masks,
                ))

        return val_loss, metrics, mask_list

    def load_model(self, path_model: Path = None):
        if path_model is None:
            path_model = self.path_model
        load = torch.load(path_model, map_location="cpu")
        assert self.encoder == load["encoder"], f"loaded and initialized encoder are not the same:" \
                                                f"{load['encoder']} != {self.encoder}"
        assert self.arch == load["arch"], f"loaded and initialized architecture are not the same:" \
                                          f"{load['arch']} != {self.arch}"
        assert self.num_energy_levels == load["num_energy_levels"], \
            (f"loaded model uses different number of energy levels: "
             f"{load['num_energy_levels']} != {self.num_energy_levels}")
        assert self.down_scale_factor == load["down_scale_factor"], \
            (f"loaded model uses different down scale factor: "
             f"{load['down_scale_factor']} != {self.down_scale_factor}")
        assert self.resize_size == load["resize_size"], \
            (f"loaded model uses different resize size: "
             f"{load['resize_size']} != {self.resize_size}")
        self.model.load_state_dict(load["model"])
        self.optim.load_state_dict(load["optim"])
        self.epoch = load["epoch"]
        self.best_val_score = load["best_metrics"]
        self.scaler.load_state_dict(load["scaler"])

    def save_model(self):
        torch.save({
            "model": self.model.state_dict(),
            "encoder": self.encoder,
            "arch": self.arch,
            "optim": self.optim.state_dict(),
            "epoch": self.epoch,
            "best_metrics": self.best_val_score,
            "scaler": self.scaler.state_dict(),
            "num_energy_levels": self.num_energy_levels,
            "down_scale_factor": self.down_scale_factor,
            "resize_size": self.resize_size,
        }, self.path_model)

    def train(self):
        # Training loop
        for epoch in range(self.epoch, self.args.num_epochs):
            self.epoch = epoch

            if epoch >= self.encoder_warmup_epochs:
                print("unfreezing encoder")
                for p in self.model.encoder.parameters():
                    p.requires_grad = True

            train_loss = self.train_loop(self.train_loader)

            wandb_log = {
                'train_loss': train_loss,
                "epoch": self.epoch,
            }

            if epoch % self.val_epochs == 0:
                val_loss, val_metrics, val_mask_list = self.val_loop(self.val_loader, "val")

                # Log metrics to wandb
                wandb_log.update({
                    'val_loss': val_loss,
                    "val_predictions": val_mask_list,
                })
                wandb_log.update(val_metrics)

                print(f"Epoch {epoch + 1}/{self.args.num_epochs},"
                      f" Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                for idx in self.column_ids:
                    for i in range(self.num_energy_levels):
                        print(
                            f"Accuracy {idx} {i + 1}: {val_metrics[f'val_accuracy_{idx}_{i + 1}']:.4f}, "
                            f"Precision {idx} {i + 1}: {val_metrics[f'val_precision_{idx}_{i + 1}']:.4f}, "
                            f"Recall {idx} {i + 1}: {val_metrics[f'val_recall_{idx}_{i + 1}']:.4f}, "
                            f"F1 Score {idx} {i + 1}: {val_metrics[f'val_f1_score_{idx}_{i + 1}']:.4f}, "
                            f"IOU {idx} {i + 1}: {val_metrics[f'val_iou_{idx}_{i + 1}']:.4f}"
                        )

                val_score = np.mean([wandb_log[key] for key in wandb_log if self.selection_metric in key])
                if val_score > self.best_val_score:
                    self.best_val_score = val_score
                    wandb_log["best_val_score"] = val_score
                    self.save_model()

            # Log metrics to wandb
            if self.wandb:
                wandb.log(wandb_log)

        # load the trained model
        self.load_model()
        test_loss, test_metrics, test_mask_list = self.val_loop(self.test_loader, "test")

        # Log metrics to wandb
        if self.wandb:
            wandb_log.update(test_metrics)
            wandb_log["test_predictions"] = test_mask_list
            wandb.log(wandb_log)

        for idx in self.column_ids:
            for i in range(self.num_energy_levels):
                print(
                    f"Accuracy {idx} {i + 1}: {test_metrics[f'test_accuracy_{idx}_{i + 1}']:.4f}, "
                    f"Precision {idx} {i + 1}: {test_metrics[f'test_precision_{idx}_{i + 1}']:.4f}, "
                    f"Recall {idx} {i + 1}: {test_metrics[f'test_recall_{idx}_{i + 1}']:.4f}, "
                    f"F1 Score {idx} {i + 1}: {test_metrics[f'test_f1_score_{idx}_{i + 1}']:.4f}, "
                    f"IOU {idx} {i + 1}: {test_metrics[f'test_iou_{idx}_{i + 1}']:.4f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training UNet model for binary segmentation")
    parser.add_argument("--project_name", type=str, default="mgr_columns",
                        help="Name of the project in Weights & Biases")
    parser.add_argument("--entity", type=str, default=None, help="Your Weights & Biases username")
    parser.add_argument("--run", type=str, default=None, help="Name of run (will be generated if not give)")
    parser.add_argument("--split_csv", type=str, required=True,
                        help="Path to the csv file containing paths to train/val/test folders "
                             "(run create_datasplits.py to create this if missing)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--batch_size_val", type=int, default=16, help="Batch size for validation")
    parser.add_argument("--val_epochs", type=int, default=5, help="How many training epochs until validation.")
    parser.add_argument("--resize_size", type=int, default=512, help="Resizing of images to this size.")
    parser.add_argument("--down_scale_factor", type=int, default=2, help="How much should we downscale the labels?")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lr_encoder", type=float, default=1e-5, help="Learning rate for encoder")
    parser.add_argument("--num_energy_levels", type=int, default=5, help="Number of energy levels to learn per column")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers per dataloader")
    parser.add_argument("--eps", type=float, default=1e-8, help="Optimizer eps")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Optimizer weight decay")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--exp_root", type=str, default="exp_column", help="Path to main experiment folder")
    parser.add_argument("--debug", action='store_true', default=False, help="Flag to enable debug mode")
    parser.add_argument(
        "--train_on_all", action='store_true', default=False, help="Flag to train on all data (train/val/test)"
    )
    parser.add_argument("--no_amp", action='store_true', default=False, help="Flag to disable mixed precision training")
    parser.add_argument("--no_wandb", action='store_true', default=False, help="Flag to disable wandb")
    parser.add_argument("--no_in_memory", action='store_true', default=False,
                        help="Flag to disable in_memory saving of data")
    parser.add_argument("--memory_file", type=str, default="dataset.pt",
                        help="File to store memory of dataset")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Which encoder to use for UNet (e.g., resnet34, efficientnet-b4, ...)")
    parser.add_argument("--pretrain_weights", type=str, default=None,
                        help="Pretrained weights to use (overrides epochs as well).")
    parser.add_argument("--loss", type=str, default="softbce",
                        help="Which loss to use (e.g. bce, softbce, softfocalbce)")
    parser.add_argument("--sobel_alpha", type=float, default=0.0,
                        help="How much sobel loss should be used (loss + alpha*sobel_loss.")
    parser.add_argument("--column_ids", type=str, default="1 2 4 5",
                        help="Which columns to predict (WARNING will change task). Starting at 1.")
    parser.add_argument("--column_count", type=int, default=9,
                        help="How many columns are in the labeled tables.")
    parser.add_argument("--selection_metric", type=str, default="iou",
                        help="Which metric to use for model selection (e.g. f1_score, accuracy, precision, recall)")
    parser.add_argument("--encoder_warmup_epochs", type=int, default=100,
                        help="How many epochs should the encoder be frozen?")
    parser.add_argument("--arch", type=str, default="unet",
                        help="Which segmentation architecture to use ('unet', 'unet+', 'manet').")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
