from models import Augmenter, PACBED
from datasets import PACBEDDataset, FixedPACBEDDataset, InMemoryPACBEDDataset, generate_test_dataset_into_directory
from torch.utils.data import DataLoader, RandomSampler
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import warnings
import argparse
import warnings
import time
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

def main():
    
    parser = argparse.ArgumentParser(description='Train a PACBED model')

    N_SAMPLES_PER_FILE  = 165
    N_PIXELS_ORIGINAL   = 1040
    N_PIXELS_TARGET     = 256
    CROP                = 510
    ETA                 = 1
    FILES               = [""]
    TEST_ROOT           = Path("./data/test")
    BATCH_SIZE          = 2
    N_WORKERS           = 8
    N_SAMPLES           = int(1e3 * BATCH_SIZE)
    N_TEST              = int(1e2)
    N_VALIDATION        = int(1e2 * BATCH_SIZE)
    LR                  = 1e-4
    SEED                = 42
    DEVICE              = 'gpu'
    N_DEVICES           = 1
    N_EPOCHS            = 100
    BACKBONE            = 'resnet34'
    LOG_DIR             = './logs'

    parser.add_argument('--n_samples_per_file', type=int, default=N_SAMPLES_PER_FILE)
    parser.add_argument('--n_pixels_original', type=int, default=N_PIXELS_ORIGINAL)
    parser.add_argument('--n_pixels_target', type=int, default=N_PIXELS_TARGET)
    parser.add_argument('--crop', type=int, default=CROP)
    parser.add_argument('--eta', type=float, default=ETA)
    parser.add_argument('--files', type=str, nargs='+', default=FILES)
    parser.add_argument('--test_root', type=str, default=TEST_ROOT)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--n_workers', type=int, default=N_WORKERS)
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES)
    parser.add_argument('--n_test', type=int, default=N_TEST)
    parser.add_argument('--n_validation', type=int, default=N_VALIDATION)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', type=str, default=DEVICE)
    parser.add_argument('--n_devices', type=str, default=N_DEVICES)
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--backbone', type=str, default=BACKBONE)
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    start = time.time()

    # Human readable time
    name = time.strftime("%Y%m%d-%H%M%S")
    
    augmenter = Augmenter(
        n_pixels_original=args.n_pixels_original, 
        n_pixels_target=args.n_pixels_target, 
        crop=args.crop, 
        eta=args.eta,
        translate=(0.01, 0.01)
        )

    logger     = CSVLogger(args.log_dir, name="PACBED")
    model_dir  = logger.log_dir + "/checkpoints"
    checkpoint = ModelCheckpoint(model_dir, monitor='val_loss', save_top_k=2, mode='min')

    
    # Create training set
    train_set       = PACBEDDataset(files = args.files, n_samples = args.n_samples_per_file, n_pixels=args.n_pixels_original, transforms=augmenter)
    train_sampler   = RandomSampler(train_set, replacement=True, num_samples=args.n_samples)
    train_loader    = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.n_workers, sampler=train_sampler, pin_memory=True)

    validation_sampler          = RandomSampler(train_set, replacement=True, num_samples=args.n_validation)
    validation_initial_loader   = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.n_workers, sampler=validation_sampler, pin_memory=True)
    validation_set              = InMemoryPACBEDDataset.from_dataloader(validation_initial_loader)
    validation_loader           = DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=True, shuffle=True)

    # Create test set
    # If test set does not exist, create it
    # if not Path(TEST_ROOT).exists():

    #     generate_test_dataset_into_directory(files = args.files, 
    #                                         target_dir = args.test_root, 
    #                                         n_samples = args.n_test, 
    #                                         n_pixels=args.n_pixels_original, 
    #                                         n_samples_per_file=args.n_samples_per_file, 
    #                                         augmenter=augmenter,
    #                                         n_workers=args.n_workers)

    # test_set    = FixedPACBEDDataset(root = args.test_root)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.n_workers)

    # Define model
    model       = PACBED(backbone=args.backbone, n_pixels=args.n_pixels_original, lr=args.lr)
    
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(
        accelerator=args.device, 
        devices=args.n_devices, 
        max_epochs=args.n_epochs, 
        logger=logger,
        callbacks=[checkpoint])
    
    # Train model
    trainer.fit(model, 
                train_dataloaders=train_loader, 
                val_dataloaders=validation_loader)
    
    trainer.save_checkpoint(f"{model_dir}/{name}_final.ckpt")

    end = time.time()

    runtime = end - start

    return runtime

if __name__ == "__main__":

    runtime = main()

    print(f"\n Runtime: {runtime}")