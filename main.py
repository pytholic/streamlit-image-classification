from datetime import datetime

import albumentations as A
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from clearml import Task
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from simple_parsing import ArgumentParser
from torch.utils.data import Subset

from config import config
from config.args import Args
from config.config import logger
from dataloader import *
from model import Classifier


# Preprocessing function
def get_transform(dataset):

    resize = A.Resize(224, 224)
    normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    bright_contrast = A.RandomBrightnessContrast(
        brightness_limit=0.1, contrast_limit=0.1, p=0.2
    )
    rgb_shift = A.RGBShift(p=0.05)
    hue_sat = A.HueSaturationValue(p=0.05)
    channel_shuffle = A.ChannelShuffle(p=0.05)
    gamma = A.RandomGamma(p=0.05)
    gray = A.ToGray(p=0.05)
    jitter = A.ColorJitter(p=0.05)
    hor_flip = A.HorizontalFlip(p=0.1)
    rotate = A.Rotate(limit=15, p=0.1)
    crop = A.RandomCrop(height=100, width=100, p=0.1)
    translate = A.Affine(translate_percent=0.1, p=0.1)
    shear = A.Affine(shear=20, p=0.05)
    scale = A.Affine(scale=1.2, p=0.1)
    to_tensor = ToTensorV2()

    if dataset == "train":
        return A.Compose([resize, normalize, to_tensor])
    elif dataset == "val":
        return A.Compose([resize, normalize, to_tensor])


# Dataset function
def prepare_dataset(data_dir):
    try:
        train_dir = str(data_dir) + "/train"
        logger.debug(train_dir)
        val_dir = str(data_dir) + "/val"
        trainset = CustomDataset(data_dir=train_dir, transforms=get_transform("train"))
        valset = CustomDataset(data_dir=val_dir, transforms=get_transform("val"))
        return trainset, valset
    except Exception as e:
        logger.error(f"Got an exception: {e}")


# Sanity check subset function
def create_subset(trainset, valset):

    train_subset = Subset(trainset, range(100))
    val_subset = Subset(valset, range(5))

    return train_subset, val_subset


# Dataloaders function
def create_dataloaders(args, train_subset, val_subset):
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_subset, batch_size=args.batch_size, num_workers=24, shuffle=True
    )
    val_dataloader = DataLoader(
        val_subset, batch_size=args.batch_size, num_workers=24, shuffle=False
    )
    return train_dataloader, val_dataloader


def set_device():
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"
    return device


def main():

    device = set_device()
    logger.info(f"Currently using {device} device...")

    # Read args
    logger.info("Reading arguments...")
    parser = ArgumentParser()
    parser.add_arguments(Args, dest="options")
    args_namespace = parser.parse_args()
    args = args_namespace.options

    # Prepare dataset
    logger.info("Preparing datasets...")
    trainset, valset = prepare_dataset(data_dir=config.DATA_DIR)
    logger.info(
        f"""Total training images: {len(trainset)}
Total validation images: {len(valset)}"""
    )

    # Create subset
    # train_subset, val_subset = create_subset(trainset, valset)

    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(args, trainset, valset)

    # Initialize clearml task
    logger.info("Initializing clearml task...")
    task = Task.init(
        project_name="streamlit/image-classification",
        task_name=f"streamlit-image-classification-{datetime.now()}",
    )
    task.connect(args)

    # Saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="loss/val_loss",
        mode="min",
        filename="streamlit-image-classification-{epoch:02d}-{val_loss:.2f}",
    )

    # Create progress bar
    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="white",
            progress_bar="#50C878",
            progress_bar_finished="#50C878",
            progress_bar_pulse="#50C878",
            batch_progress="orange",
            time="grey54",
            processing_speed="grey70",
            metrics="white",
        ),
        console_kwargs=None,
        refresh_rate=10,
    )

    # Train
    # configure trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, progress_bar],
        default_root_dir=config.LOGS_DIR,
        accelerator=device,
        devices=1,
        log_every_n_steps=1,
    )

    # define classifier
    classifier = Classifier()

    # fit the model
    logger.info("Starting training...")
    trainer.fit(classifier, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
