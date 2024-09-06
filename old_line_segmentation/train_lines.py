import argparse
from functools import partial
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import randomname
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import wandb
from adabelief_pytorch import AdaBelief
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch.losses import SoftBCEWithLogitsLoss, TverskyLoss
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score, JaccardIndex
from tqdm import tqdm

from dataset import MGRDataset


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

        if self.wandb:
            wandb_keys = ["lr", "batch_size", "num_epochs", "weight_decay",
                          "eps", "encoder", "loss", "selection_metric", "no_amp", "encoder_warmup_epochs"]
            wandb_args = {k: getattr(args, k) for k in wandb_keys}
            # Initialize Weights & Biases
            wandb.init(project=args.project_name, entity=args.entity, name=name, config=wandb_args, dir=self.folder_exp)

        # Define data transformations for training and validation
        self.transform = A.Compose([
            ToTensorV2(transpose_mask=True),
        ])
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=.5),
            A.VerticalFlip(p=.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=5, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.5),
            ToTensorV2(transpose_mask=True),
        ])

        self.split_file = pd.read_csv(args.split_csv)
        train = self.split_file.query("split == 'train'")["folders"].values
        val = self.split_file.query("split == 'val'")["folders"].values
        test = self.split_file.query("split == 'test'")["folders"].values

        # Create dataset instances for training and validation
        self.train_dataset = MGRDataset(folders=train, transform=self.train_transform,
                                        in_memory=self.in_memory, memory_file=f"train_{args.memory_file}")

        self.val_dataset = MGRDataset(folders=val, transform=self.transform,
                                      in_memory=self.in_memory, memory_file=f"val_{args.memory_file}")
        self.test_dataset = MGRDataset(folders=test, transform=self.transform,
                                       in_memory=self.in_memory, memory_file=f"test_{args.memory_file}")

        # Define data loaders for training and validation
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        # Define the UNet model
        self.encoder = args.encoder
        self.model = smp.Unet(
            self.encoder,
            in_channels=3,
            classes=1,
        )
        self.model.to(self.device)

        LOSS_FN = {
            "bce": nn.BCEWithLogitsLoss,
            "softbce": partial(SoftBCEWithLogitsLoss, smooth_factor=0.1),
            "tversky": partial(TverskyLoss, mode="binary", alpha=0.3, beta=0.7),
            "softtversky": partial(TverskyLoss, mode="binary", alpha=0.3, beta=0.7, smooth=0.1),
        }
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
            'accuracy': Accuracy("binary").to(self.device),
            'precision': Precision("binary").to(self.device),
            'recall': Recall("binary").to(self.device),
            'f1_score': F1Score("binary").to(self.device),
            'iou': JaccardIndex("binary").to(self.device),
        }
        self.best_val_score = -np.inf
        self.selection_metric = args.selection_metric
        self.epoch = 0

        # freeze weights on decoder by default
        for p in self.model.encoder.parameters():
            p.requires_grad = False

        if self.path_model.exists():
            self.load_model()

    def train_loop(self, dataloader):
        self.model.train()
        train_loss = 0.0

        for images, labels in tqdm(dataloader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optim.zero_grad()
            with torch.autocast(device_type=str(self.device), dtype=torch.float16, enabled=self.amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            train_loss += loss.item() * images.size(0)
            if self.debug:
                break

        train_loss /= len(dataloader.dataset)
        return train_loss

    @torch.no_grad()
    def val_loop(self, dataloader, name):
        self.model.eval()
        val_loss = 0.0
        [self.metrics[k].reset() for k in self.metrics]

        for images, labels in tqdm(dataloader, desc=f"Validation on {name} set"):
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.autocast(device_type=str(self.device), dtype=torch.float16, enabled=self.amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            # Convert logits to binary predictions
            preds = outputs >= 0.0

            # Update metrics
            for metric in self.metrics.values():
                metric.update(preds, labels)

            if self.debug:
                break

        val_loss /= len(dataloader)
        metrics = {}
        for metric in self.metrics:
            metrics[f"{name}_{metric}"] = self.metrics[metric].compute()

        mask_list = []
        if self.wandb:
            # also log some images
            num_log_images = min(5, self.batch_size)
            for i in range(num_log_images):
                mask_data = preds[i, 0].cpu().int().numpy()
                ground_truth = labels[i, 0].cpu().int().numpy()

                class_labels = {0: "bg", 1: "line"}

                mask_list.append(wandb.Image(
                    images[i],
                    masks={
                        "predictions": {"mask_data": mask_data, "class_labels": class_labels},
                        "ground_truth": {"mask_data": ground_truth, "class_labels": class_labels},
                    },
                ))

        return val_loss, metrics, mask_list

    def load_model(self):
        load = torch.load(self.path_model, map_location="cpu")
        self.model.load_state_dict(load["model"])
        self.optim.load_state_dict(load["optim"])
        assert self.encoder == load["encoder"], f"loaded encoder and initialized encoder are not the same:" \
                                                f"{load['encoder']} != {self.encoder}"
        self.epoch = load["epoch"]
        self.best_val_score = load["best_metrics"]
        self.scaler.load_state_dict(load["scaler"])

    def save_model(self):
        torch.save({
            "model": self.model.state_dict(),
            "encoder": self.encoder,
            "optim": self.optim.state_dict(),
            "epoch": self.epoch,
            "best_metrics": self.best_val_score,
            "scaler": self.scaler.state_dict()
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

                print(
                    f"Epoch {epoch + 1}/{self.args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(
                    f"Accuracy: {val_metrics['val_accuracy']:.4f}, Precision: {val_metrics['val_precision']:.4f}, "
                    f"Recall: {val_metrics['val_recall']:.4f}, F1 Score: {val_metrics['val_f1_score']:.4f}"
                    f"IOU {val_metrics['val_iou']}"
                )
                if val_metrics[f'val_{self.selection_metric}'] > self.best_val_score:
                    self.best_val_score = val_metrics[f'val_{self.selection_metric}']
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

        print(
            f"Accuracy: {test_metrics['test_accuracy']:.4f}, Precision: {test_metrics['test_precision']:.4f}, "
            f"Recall: {test_metrics['test_recall']:.4f}, F1 Score: {test_metrics['test_f1_score']:.4f}, "
            f"IOU {test_metrics['test_iou']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training UNet model for binary segmentation")
    parser.add_argument("--project_name", type=str, default="mgr",
                        help="Name of the project in Weights & Biases")
    parser.add_argument("--entity", type=str, default=None, help="Your Weights & Biases username")
    parser.add_argument("--run", type=str, default=None, help="Name of run (will be generated if not give)")
    parser.add_argument("--split_csv", type=str, required=True,
                        help="Path to the csv file containing paths to train/val/test folders "
                             "(run create_datasplits.py to create this if missing)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--val_epochs", type=int, default=5, help="How many training epochs until validation.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lr_encoder", type=float, default=1e-5, help="Learning rate for encoder")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers per dataloader")
    parser.add_argument("--eps", type=float, default=1e-8, help="Optimizer eps")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Optimizer weight decay")
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--exp_root", type=str, default="exp", help="Path to main experiment folder")
    parser.add_argument("--debug", action='store_true', default=False, help="Flag to enable debug mode")
    parser.add_argument("--no_amp", action='store_true', default=False, help="Flag to disable mixed precision training")
    parser.add_argument("--no_wandb", action='store_true', default=False, help="Flag to disable wandb")
    parser.add_argument("--no_in_memory", action='store_true', default=False,
                        help="Flag to disable in_memory saving of data")
    parser.add_argument("--memory_file", action='store_true', default="dataset.pt",
                        help="File to store memory of dataset")
    parser.add_argument("--encoder", type=str, default="resnet34",
                        help="Which encoder to use for UNet (e.g., resnet34, efficientnet-b4, ...)")
    parser.add_argument("--loss", type=str, default="softbce",
                        help="Which loss to use (e.g. bce, softbce)")
    parser.add_argument("--selection_metric", type=str, default="iou",
                        help="Which metric to use for model selection (e.g. f1_score, accuracy, precision, recall)")
    parser.add_argument("--encoder_warmup_epochs", type=int, default=100,
                        help="How many epochs should the encoder be frozen?")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
