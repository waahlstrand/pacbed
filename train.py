import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from typing import *
import argparse
import yaml
from pathlib import Path

from tasks.classification.models import build_model
from tasks.data.pacbed import build_datamodule
from tasks.callbacks import PlotImageCallback

import warnings

warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser(description='Train a PACBED model')

    parser.add_argument('--config', '-c', type=str, default='')

    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--simulated_metadata_file', type=str, default=None)
    parser.add_argument('--simulated_src_path', type=str, default=None)
    parser.add_argument('--experimental_metadata_file', type=str, default=None)
    parser.add_argument('--experimental_src_path', type=str, default=None)
    parser.add_argument('--logs_root', type=str, default=None)
    parser.add_argument('--precision', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)

    parser.add_argument('--eta', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--momentum', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--p_occlusion', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--original_size', type=int, default=None)
    parser.add_argument('--target_size', type=int, default=None)
    parser.add_argument('--crop', type=int, default=None)
    parser.add_argument('--convergence_angle_index', type=int, default=None)
    parser.add_argument('--energy_index', type=int, default=None)

    parser.add_argument('--n_epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_workers', type=int, default=None)
    parser.add_argument('--n_train_samples', type=int, default=None)
    parser.add_argument('--n_valid_samples', type=int, default=None)
    parser.add_argument('--n_test_samples', type=int, default=None)

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--debug', type=bool, default=None)
    parser.add_argument('--log', type=bool, default=None)

    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--pretrained', type=bool, default=None)

    args = parser.parse_args()

    # If there is a config file, load it
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            # args.__dict__.update(config)
    
    params = {}
    for k, v in args.__dict__.items():
        if v is not None:
            params[k] = v

    # Add the config to a namespace
    args = argparse.Namespace(**config)

    # Update the namespace with the command line arguments
    args.__dict__.update(params)

    fit(args)


def fit(args: argparse.Namespace):

    name = "_".join([
        args.name, 
        args.target.replace(" ", "_"),
        str(args.crop), 
        str(args.convergence_angle_index), 
        str(args.energy_index)])

    # Set the random seed
    L.seed_everything(args.seed)

    # Create the loggers
    loggers = [
        CSVLogger(args.logs_root, name=name),
    ]
    
    if args.log:
        loggers += [WandbLogger(name=name, project='pacbed-classification')]
        
    # Setup checkpointing
    checkpointing = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=loggers[0].log_dir,
        filename='{epoch:02d}',
        save_top_k=1,
    )

    # plot_callback = PlotImageCallback()

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
    
    # Save last checkpoint
    trainer.save_checkpoint(f"{loggers[0].log_dir}/last.ckpt")

    ################### TEST ###################
    trainer.test(model, dm)

if __name__ == '__main__':
    main()




