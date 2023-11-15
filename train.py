import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from typing import *
import argparse
import yaml
from pathlib import Path

from tasks.classification.models import Classifier, build_model
from tasks.data.pacbed import build_datamodule
from tasks.data.augmentation import Augmenter

import warnings

warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser(description='Train a PACBED model')

    parser.add_argument('--config', '-c', type=str, default='')

    parser.add_argument('--simulated_metadata_file', type=str, default='')
    parser.add_argument('--simulated_src_path', type=str, default='')
    parser.add_argument('--experimental_metadata_file', type=str, default='')
    parser.add_argument('--experimental_src_path', type=str, default='')
    parser.add_argument('--logs_root', type=str, default='')
    parser.add_argument('--precision', type=str, default='medium')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--target', type=str, default='Phase index')

    parser.add_argument('--eta', type=float, default=0.8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--p_occlusion', type=float, default=0.4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--original_size', type=int, default=1024)
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--crop', type=int, default=510)

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=32)
    parser.add_argument('--n_train_samples', type=int, default=1000)
    parser.add_argument('--n_valid_samples', type=int, default=100)
    parser.add_argument('--n_test_samples', type=int, default=1000)

    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained', type=bool, default=True)

    args = parser.parse_args()

    if args.config != '':
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            args.__dict__.update(config)

    fit(args)


def fit(args: argparse.Namespace):

    # Set the random seed
    L.seed_everything(args.seed)

    # Create the loggers
    loggers = [
        CSVLogger(args.logs_root, name=args.backbone),
        WandbLogger(name=args.backbone, project='pacbed-classification'),
    ]

    # Setup checkpointing
    checkpointing = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        dirpath=args.logs_root,
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
    )

    ################### DATA ###################
    # Create the data module
    dm = build_datamodule(args)

    ################### MODEL ###################
    # Create the model
    model = build_model(args)

    ################### TRAINER ###################
    torch.set_float32_matmul_precision(args.precision)

    trainer = L.Trainer(
        accelerator             = args.device,
        devices                 = [0],
        logger                  = loggers,
        callbacks               = [ checkpointing ],
        max_epochs              = args.n_epochs,
        fast_dev_run            = True if args.debug else False,
    )

    ################### FIT ###################
    trainer.fit(
        model, 
        dm,
        ckpt_path = args.checkpoint if args.checkpoint != '' else None,
        )

    ################### TEST ###################
    trainer.test(model, dm)

if __name__ == '__main__':
    main()




