import argparse
import glob
import logging
import os
from collections import defaultdict
from math import ceil
from pathlib import Path
from sys import exc_info

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from joblib import Parallel, delayed
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MGRDatasetInference
from image_utils import illustrate_layout, ImageSegmentation

logger = logging.getLogger(__name__)

# This is hardcoded and only applies to birth book pages
# TODO: Implement solution for the other book types (deaths, weddings, confirmations)
COLUMN_NAMES = ["BIRTHDATE", "CHILD_NAME", "PARENTS"]
COLUMN_PAIRS = [(0, 1), (2, 3), (3, 4)]


def run(args):
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define and create (sub-)folders

    # Base output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Folders for main segmentation outputs (cells)
    cells_output_dir = output_dir / "cells"
    cells_output_dir.mkdir(exist_ok=True)

    for column_name in COLUMN_NAMES:
        (cells_output_dir / column_name.lower() / "image").mkdir(exist_ok=True, parents=True)
    (cells_output_dir / "id" / "image").mkdir(exist_ok=True, parents=True)

    # Folder for U-Net predictions
    if args.save_predict:
        unet_predictions_dir = output_dir / "unet_predictions"
        unet_predictions_dir.mkdir(exist_ok=True)
    else:
        unet_predictions_dir = None

    # Folder for illustrations, i.e. bounding boxes drawn on resized versions of the input images
    if args.save_illustrations:
        illustrations_output_dir = output_dir / "illustrations"
        illustrations_output_dir.mkdir(exist_ok=True)
    else:
        illustrations_output_dir = None

    # Folders for illustrations of failed images
    if args.save_failed:
        failed_output_dir = output_dir / "failed"
        failed_output_dir.mkdir(exist_ok=True)
        for column_name in COLUMN_NAMES:
            (failed_output_dir / column_name.lower()).mkdir(exist_ok=True)
    else:
        failed_output_dir = None

    # Define default config for segmenter
    config = {
        "illustration_dir": illustrations_output_dir,
        "cells_dir": cells_output_dir,
        "unet_pred_dir": unet_predictions_dir,
        "failed_dir": failed_output_dir,
    }

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

    transform = Compose([
        ToTensorV2(transpose_mask=True),
    ])

    # Create dataset and data loader
    dataset = MGRDatasetInference(folders=file_paths, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False, shuffle=False
    )

    # Load models
    model_paths = glob.glob(os.path.join(args.model_dir, "*.pt"))
    models = []
    for m in model_paths:
        # Create model
        logger.info(f"loading {m}")
        load = torch.load(m, map_location="cpu")
        encoder = load["encoder"]
        model = smp.Unet(encoder, in_channels=3, classes=1)
        model.load_state_dict(load["model"])
        model.to(device)
        model.eval()
        models.append(model)

    amp = not args.no_amp and device != "cpu"
    pred_acc = []
    img_acc = []
    name_acc = []
    for images, names in tqdm(dataloader, "Inference (in number of batches)", total=len(dataloader)):
        with torch.autocast(device_type=str(device), dtype=torch.float16, enabled=amp):
            img_acc.append(images)
            name_acc.append(names)
            images = images.to(device)
            with torch.no_grad():
                preds = []
                # ensemble predictions by majority vote
                for model in models:
                    preds.append(model(images) > 0.5)
                preds = torch.stack(preds, 0).float().mean(0) > 0.5
                pred_acc.append(preds.cpu().int())

        if len(pred_acc) % args.acc_steps == 0:
            logging.info("Start post processing of images")
            postprocess_images(img_acc, name_acc, pred_acc, config, args.num_post_workers)

            pred_acc = []
            img_acc = []
            name_acc = []

    if len(pred_acc) > 0:
        postprocess_images(img_acc, name_acc, pred_acc, config, args.num_post_workers)


def postprocess_images(images, names, preds, config, num_workers):
    images = torch.cat(images, 0).numpy()
    images *= 255
    images = images.astype(np.uint8)
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
    images = np.moveaxis(images, 1, -1)

    # Finally, a segmented image
    # b, c, h, w -> b, h, w, c
    segmented = np.moveaxis(preds, 1, -1)

    # Create version with alpha channel
    segmented = segmented.repeat(4, -1).astype(float)
    segmented *= np.array([0, 1.0, 1.0, 0.7])
    # Convert to uint8 to save as png without warning
    # jpg does not support alpha channel
    segmented = (segmented * 255).astype(np.uint8)

    for img, name, seg in zip(images, names, segmented):
        segmentation = ImageSegmentation(
            config=config,
            preloaded_img=img,
            preloaded_seg=seg,
        )

        label_strings_dict = defaultdict(list)

        # Save prediction from U-Net for speedy re-running
        if config["unet_pred_dir"] is not None:
            unet_prediction_name = config["unet_pred_dir"] / f"{name}.png"
            unet_prediction_img = Image.fromarray(segmentation.seg)
            unet_prediction_img.save(unet_prediction_name)

        illustration = img
        for column_name, column_pair in zip(COLUMN_NAMES, COLUMN_PAIRS):
            try:
                cells, ids, coords = segmentation.crop_cells(column_pair=column_pair)

                labels = ["n/a"] * len(cells)

                for idx, cell_img in enumerate(cells):
                    cell_name = config["cells_dir"] / column_name.lower() / "image" / f"{name}_{column_name}-r{idx}.jpg"

                    cell_img = Image.fromarray(cell_img)
                    cell_img.save(cell_name)

                    label_strings_dict[column_name].append(f"{cell_name}\t{labels[idx]}" + "\n")

                if column_pair == (0, 1):
                    for idx, cell_img in enumerate(ids):
                        cell_name = config["cells_dir"] / "id" / "image" / f"{name}_ID-r{idx}.jpg"
                        cell_img = Image.fromarray(cell_img)
                        cell_img.save(cell_name)

                        label_strings_dict["ID"].append(f"{cell_name}\tn/a\n")

                # Also save an illustration of the detected layout
                if config["illustration_dir"] is not None:
                    illustration = illustrate_layout(illustration, coords=coords)

            except Exception as e:
                exc_type, exc_obj, exc_tb = exc_info()

                failed_img_path = name.replace("(", "/").replace("[", ".")
                logger.warning(f"Image {failed_img_path} failed for column {column_name.lower()} with: {e} "
                               f"in: {exc_tb.tb_lineno}")
                if config["failed_dir"] is not None:
                    try:
                        failed_image_name = config["failed_dir"] / column_name.lower() / f"{name}.jpg"
                        failed_image = Image.fromarray(img)
                        failed_image.save(failed_image_name)
                    except:
                        pass

        if config["illustration_dir"] is not None:
            illustration = Image.fromarray(illustration)
            illustration_name = config["illustration_dir"] / f"{name}.jpg"
            illustration.save(illustration_name, optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for UNet model for column/row line detection")
    parser.add_argument("--img_csv", type=str, required=True,
                        help="Path to the csv file containing paths to images to predict or folders with images "
                             "(no header just one folder/image per line)."
                             "If parameter is a folder or does not end with 'csv' or 'txt', "
                             "we assume that an image/folder is given directly")
    parser.add_argument("--acc_steps", type=int, default=64,
                        help="Number of postprocessed image batches at the same time")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and validation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers per dataloader")
    parser.add_argument("--num_post_workers", type=int, default=16, help="Number of workers for postprocessing")
    parser.add_argument("--model_dir", type=str, default="./", help="Path to model files (pt files)")
    parser.add_argument("--output_dir", type=str, default="out", help="Path to output directory")
    parser.add_argument("--save_predict", action='store_true', default=False,
                        help="Flag to enable saving of predictions")
    parser.add_argument("--save_failed", action='store_true', default=False,
                        help="Flag to enable saving of failed images")
    parser.add_argument("--save_illustrations", action='store_true', default=False,
                        help="Flag to enable saving of illustration images")

    parser.add_argument("--no_amp", action='store_true', default=False, help="Flag to disable mixed precision training")

    args = parser.parse_args()

    run(args)
