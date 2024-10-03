import argparse
import logging
import os
from glob import glob
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import randomname
import torch
import wandb
from PIL import Image
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
from joblib import Parallel, delayed
from skimage.color import label2rgb
from skimage.morphology import convex_hull_image, binary_dilation
from skimage.segmentation import watershed
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm

from dataset import MGRDatasetInference
from train import get_model

logger = logging.getLogger(__name__)

# This is hardcoded and only applies to birth book pages
COLUMN_NAMES = ["ID", "BIRTH_YEAR", "CHILD_NAME", "PARENTS"]
COLUMN_ID = [1, 2, 4, 5]


def run(args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.run is None:
        name = randomname.get_name()
    else:
        name = args.run

    logging.info(f"Run is called: {name}")

    # Define and create (sub-)folders

    # Base output dir
    output_dir = Path(args.output_dir) / name
    output_dir.mkdir(exist_ok=True, parents=True)

    # Folders for main segmentation outputs (cells)
    cells_output_dir = None
    if args.save_cutouts:
        cells_output_dir = output_dir / "cells"
        cells_output_dir.mkdir(exist_ok=True)

        for column_name in COLUMN_NAMES:
            (cells_output_dir / column_name.lower()).mkdir(exist_ok=True, parents=True)

    # Folder for U-Net predictions
    predictions_dir = None
    if args.save_predict:
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)

    inconsistency_dir = output_dir / "inconsistency"
    inconsistency_dir.mkdir(exist_ok=True)

    # Define default config for segmenter
    config = {
        "cells_dir": cells_output_dir,
        "pred_dir": predictions_dir,
        "inconsistency_dir": inconsistency_dir,
        "cell_dilation": args.cell_dilation,
        "cell_margin": args.cell_margin,
        "empty_threshold": args.empty_threshold,
        "min_size": args.min_size,
    }

    if not args.no_wandb:
        wandb_keys = [
            "batch_size", "no_amp", "acc_steps",
            "cell_dilation", "min_size", "pred_threshold",
            "empty_threshold", "voting",
        ]
        wandb_args = {k: getattr(args, k) for k in wandb_keys}
        # Initialize Weights & Biases
        wandb.init(project=args.project_name, entity=args.entity, name=name, config=wandb_args, dir=output_dir)

    # Log settings
    logger.info(f"Segmentation arguments {args.__dict__}")
    logger.info(f"Config {config}")

    # Start timer    t_start = timer()

    logger.info("Finding images")

    file_paths = Path(args.img_csv)
    if not file_paths.is_dir() and file_paths.suffix in [".csv", ".txt"]:
        file_paths = list(pd.read_csv(file_paths, header=None).values.flatten())
    else:
        file_paths = [file_paths]

    resize_transform = [Resize(args.resize_size, args.resize_size)] if args.resize_size is not None else []
    transform = Compose(resize_transform + [
        ToTensorV2(transpose_mask=True),
    ])

    # Create dataset and data loader
    dataset = MGRDatasetInference(folders=file_paths, transform=transform)

    def collate_fn(batch):
        img_or = [sample[-1] for sample in batch]
        return *default_collate([sample[:-1] for sample in batch]), img_or

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False, shuffle=False
    )

    # Load models
    model_paths = glob(os.path.join(args.model_dir, "*.pt"))
    models = []
    threshold = args.pred_threshold  # class threshold
    for m in model_paths:
        if any([s in m for s in ["train", "val", "test"]]):
            continue
        # Create model
        logger.info(f"loading {m}")
        load = torch.load(m, map_location="cpu")
        encoder = load["encoder"]
        arch = load["arch"]
        num_energy_levels = load["num_energy_levels"]
        down_scale_factor = load["down_scale_factor"]
        assert args.resize_size == load["resize_size"], \
            (f"loaded model uses different resize size: "
             f"{load['resize_size']} != {args.resize_size}")
        model = get_model(arch, encoder, len(COLUMN_ID) * num_energy_levels, down_scale_factor)
        model.load_state_dict(load["model"])
        model.num_energy_levels = num_energy_levels
        model.to(device)
        model.eval()
        models.append(model)

    amp = not args.no_amp and device != "cpu"
    pred_acc = []
    ori_img_acc = []
    name_acc = []
    for images, names, images_ori in tqdm(dataloader, "Inference (in number of batches)", total=len(dataloader)):
        name_acc.append(names)
        ori_img_acc.extend(images_ori)
        with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=amp):
            images = images.to(device)
            with torch.no_grad():
                preds = []
                # ensemble predictions by majority vote
                for model in models:
                    pred = model(images)
                    pred_shape = pred.shape
                    pred = pred.reshape(
                        pred_shape[0], len(COLUMN_ID), model.num_energy_levels, pred_shape[2], pred_shape[3]
                    )
                    pred = (pred.sigmoid() > threshold).float()
                    pred_energy_cumprod = pred.cumprod(2).sum(2)

                    preds.append(pred_energy_cumprod)

                preds = torch.stack(preds, 0)
                if args.voting == "majority":
                    preds = preds.mean(0).round()
                elif args.voting == "max":
                    preds = preds.amax(0)

                pred_acc.append(preds.cpu())

        if len(pred_acc) % args.acc_steps == 0:
            logging.info("Start post processing of images")

            postprocess_images(ori_img_acc, name_acc, pred_acc, config, args.num_post_workers)

            pred_acc = []
            ori_img_acc = []
            name_acc = []

        if args.debug:
            break

    if len(pred_acc) > 0:
        logging.info("Start post processing of images")
        postprocess_images(ori_img_acc, name_acc, pred_acc, config, args.num_post_workers)

    # finally, report number of inconsistencies
    inconsistency_files = glob(os.path.join(inconsistency_dir, "*"))
    inconsistency_count = len(inconsistency_files)
    inconsistency_rate = inconsistency_count / len(dataset) * 100.

    logging.info(f"Total of {inconsistency_count}/{len(dataset)} ({inconsistency_rate}%) inconsistencies")
    if not args.no_wandb:
        wandb.log({"inconsistency_rate": inconsistency_rate})


@torch.jit.script
def remove_small_cells(labels: torch.Tensor, min_size: int):
    new_labels = []
    for i in range(labels.shape[0]):
        label = labels[i]
        idx, counts = torch.unique(label, return_counts=True)
        mask = counts > min_size
        if (~mask).sum() > 0:
            bg = torch.prod(torch.stack([~(label == id) for id in idx[~mask]], 0).float(), dim=0)
            label *= bg
        new_labels.append(label)
    new_labels = torch.stack(new_labels, 0)
    return new_labels


def remove_small_cells_np(label: np.ndarray, min_size: int):
    idx, counts = np.unique(label, return_counts=True)
    mask = counts > min_size
    if (~mask).sum() > 0:
        bg = np.prod(np.stack([~(label == id) for id in idx[~mask]], 0), axis=0)
        label *= bg
    return label


def postprocess_images(images, names, preds, config, num_workers):
    names = [item for row in names for item in row]
    preds = torch.cat(preds, 0).numpy()

    if num_workers > 0:
        bs = max(ceil(len(images) / num_workers), 1)

        Parallel(n_jobs=num_workers)(delayed(postprocess_images_batch)(
            images[i * bs:(i + 1) * bs], names[i * bs:(i + 1) * bs], preds[i * bs:(i + 1) * bs], config
        ) for i in range(num_workers))
    else:
        postprocess_images_batch(images, names, preds, config)


def postprocess_images_batch(images, names, preds, config):
    for img, name, y_energy in zip(images, names, preds):
        # apply watershed to get label
        pred = watershed(-y_energy, connectivity=1, mask=y_energy != 0)
        pred = remove_small_cells_np(pred, config["min_size"])

        # save prediction
        if config["pred_dir"] is not None:
            pred_name = config["pred_dir"] / f"{name}.jpg"
            small_img = np.asarray(Image.fromarray((img)).resize(pred.shape[1:][::-1])).astype(np.single) / 255.
            pred_img = pred.sum(0)
            small_img = label2rgb(pred_img, small_img)
            small_img_pil = Image.fromarray((small_img * 255).astype(np.uint8))
            small_img_pil.save(pred_name)

        column_count = []
        column_min_h = []
        column_max_h = []
        img_std = img.std()
        for i, column_name in enumerate(COLUMN_NAMES):
            column = pred[i]
            u_column = np.unique(column)
            c_count = 0
            c_min_h = []
            c_max_h = []
            for j, idx in enumerate(u_column):
                if idx == 0: continue
                # create convex hull to remove holes from mask
                cell = convex_hull_image(column == idx)

                # get height bounds (do that now before resizing for consistency across images)
                cell_height_bounds = np.flatnonzero(cell.any(1))

                # increase size of cell
                cell = binary_dilation(
                    cell, [(np.ones((config["cell_dilation"], 1)), 1), (np.ones((1, config["cell_dilation"])), 1)]
                )

                # increase to same size as img
                cell = np.asarray(Image.fromarray(cell).resize(img.shape[:2][::-1]))

                # cut out from real image
                cell_img = img.copy()

                cell_std = cell_img[cell].std()
                if cell_std / img_std < config["empty_threshold"]:
                    continue

                if config["cells_dir"] is not None:
                    cell_img[~cell] = 0
                    cell_img = Image.fromarray(cell_img)
                    bbox = cell_img.getbbox()  # remove obvious black outside area
                    cell_img = cell_img.crop(bbox)
                    cell_name = config["cells_dir"] / column_name.lower() / f"{name}_{column_name}-r{j}.jpg"
                    cell_img.save(cell_name)
                c_count += 1
                c_min_h.append(cell_height_bounds[0])
                c_max_h.append(cell_height_bounds[-1])
            column_count.append(c_count)
            column_min_h.append(c_min_h)
            column_max_h.append(c_max_h)

        # check for inconsistencies in number of rows or missing columns and write to file
        first_len = column_count[0]
        inconsistency_msg = f""

        if first_len < 2 or not all([first_len == c_count for c_count in column_count]):
            for i, c_name in enumerate(COLUMN_NAMES):
                inconsistency_msg += f"{c_name.lower()}: {column_count[i]}\n"
        else:
            column_min_h = np.array(column_min_h)
            column_max_h = np.array(column_max_h)
            for i in range(first_len):
                std_h_min = np.std([h for h in column_min_h[:, i]])
                std_h_max = np.std([h for h in column_max_h[:, i]])

                if std_h_max > config["cell_margin"] and i != first_len - 1:  # ignore for last column
                    inconsistency_msg += f"row {i + 1} has large bottom height standard deviation: {std_h_max:.3f}"

                if std_h_min > config["cell_margin"]:
                    inconsistency_msg += f"row {i + 1} has large top height standard deviation: {std_h_min:.3f}"

        if len(inconsistency_msg) > 0:
            with open(config["inconsistency_dir"] / f"{name}.txt", "w") as text_file:
                text_file.write(inconsistency_msg)
            logging.info(inconsistency_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for UNet model for column/row line detection")
    parser.add_argument("--img_csv", type=str, required=True,
                        help="Path to the csv file containing paths to images to predict or folders with images "
                             "(no header just one folder/image per line)."
                             "If parameter is a folder or does not end with 'csv' or 'txt', "
                             "we assume that an image/folder is given directly")
    parser.add_argument("--acc_steps", type=int, default=1,
                        help="Number of postprocessed image batches at the same time")
    parser.add_argument("--voting", type=str, default="majority",
                        help="How to combine ensemble members (either 'majority' or 'max'")
    parser.add_argument("--resize_size", type=int, default=512, help="Resizing of images to this size.")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for DL model")
    parser.add_argument("--cell_dilation", type=int, default=2,
                        help="How many pixel to dilate the cell prediction")
    parser.add_argument("--cell_margin", type=int, default=20 // 4,
                        help="Margin to detect cells at different heights that "
                             "should be the same row.")
    parser.add_argument("--min_size", type=int, default=98, help="Minimum number of pixels in a cell.")
    parser.add_argument("--pred_threshold", type=float, default=0.6575329696782404,
                        help="Threshold for ensembled class predictions")
    parser.add_argument("--empty_threshold", type=float, default=0.02860938009670584,
                        help="Threshold for empty cell detection")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of workers per dataloader")
    parser.add_argument("--num_post_workers", type=int, default=10, help="Number of workers for postprocessing")
    parser.add_argument("--model_dir", type=str, default="./", help="Path to model files (pt files)")
    parser.add_argument("--output_dir", type=str, default="out", help="Path to output directory")
    parser.add_argument("--save_predict", action='store_true', default=False,
                        help="Flag to enable saving of predictions")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="Flag to enable debug mode (only one batch)")
    parser.add_argument("--save_cutouts", action='store_true', default=False,
                        help="Flag to save of cutouts")
    parser.add_argument("--no_wandb", action='store_true', default=False, help="Flag to disable wandb logging")
    parser.add_argument("--project_name", type=str, default="mgr_columns_eval",
                        help="Name of the project in Weights & Biases")
    parser.add_argument("--entity", type=str, default=None, help="Your Weights & Biases username")
    parser.add_argument("--run", type=str, default=None, help="Name of run (will be generated if not give)")
    parser.add_argument("--no_amp", action='store_true', default=False, help="Flag to disable mixed precision training")

    args = parser.parse_args()

    run(args)
