from models.classification import Augmenter, PhaseClassifier
from data.classification import PACBEDPhaseDataset, InMemoryPACBEDPhaseDataset
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from pathlib import Path
import warnings
import argparse
import warnings
import time
import matplotlib
import utils
from torchvision.models import resnet50
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score
import pandas as pd
matplotlib.use('Agg')

warnings.filterwarnings("ignore")

def main():
    
    parser = argparse.ArgumentParser(description='Train a PACBED model')

    N_SAMPLES_PER_FILE  = 165
    N_PIXELS_ORIGINAL   = 1024
    N_PIXELS_TARGET     = 256
    CROP                = 510
    ETA                 = 1
    FILES               = ["*.bin"]
    TEST_ROOT           = Path("./data/test")
    BATCH_SIZE          = 2
    N_WORKERS           = 8
    N_SAMPLES           = int(1e3 * BATCH_SIZE)
    N_TEST              = int(1e2)
    N_VALIDATION        = int(1e2 * BATCH_SIZE)
    LR                  = 1e-4
    MOMENTUM            = 0.99
    SEED                = 42
    DEVICE              = 'gpu'
    N_DEVICES           = 1
    N_EPOCHS            = 10
    BACKBONE            = 'resnet34'
    LOG_DIR             = './logs'
    P_OCCLUSION         = 0.4
    PRECISION           = "high"

    parser.add_argument('--n_samples_per_file', type=int, default=N_SAMPLES_PER_FILE)
    parser.add_argument('--n_pixels_original', type=int, default=N_PIXELS_ORIGINAL)
    parser.add_argument('--n_pixels_target', type=int, default=N_PIXELS_TARGET)
    parser.add_argument('--crop', type=int, default=CROP)
    parser.add_argument('--eta', type=float, default=ETA)
    parser.add_argument('--root', type=str, default=FILES)
    parser.add_argument('--test_root', type=str, default=TEST_ROOT)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--n_workers', type=int, default=N_WORKERS)
    parser.add_argument('--n_samples', type=int, default=N_SAMPLES)
    parser.add_argument('--n_test', type=int, default=N_TEST)
    parser.add_argument('--n_validation', type=int, default=N_VALIDATION)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', type=str, default=DEVICE)
    parser.add_argument('--n_devices', type=str, default=N_DEVICES)
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--log_dir', type=str, default=LOG_DIR)
    parser.add_argument('--p_occlusion', type=float, default=P_OCCLUSION)
    parser.add_argument('--checkpoint', type=str, default='')
    # parser.add_argument('--exp_dir', type=str, required=True)
    # parser.add_argument('--exp_cfg', type=str, required=True)
    parser.add_argument('--precision', type=str, default=PRECISION)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument("--debug", default=False)
    parser.add_argument("--metadata", type=str, default="")
    parser.add_argument("--energy", type=float, default=200)
    parser.add_argument("--convergence_angle", type=float, default=26.69)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    start = time.time()
    max_duration = "07:00:00:00"

    # Human readable time
    name = time.strftime("%Y%m%d-%H%M%S")

    metadata = pd.read_csv(args.metadata)

    train_augmenter = Augmenter(
        n_pixels_original=args.n_pixels_original, 
        n_pixels_target=args.n_pixels_target, 
        crop=args.crop, 
        eta=args.eta,
        translate=(0.01, 0.01),
        p_occlusion=args.p_occlusion
        )
    
    val_augmenter = Augmenter(
        n_pixels_original=args.n_pixels_original, 
        n_pixels_target=args.n_pixels_target, 
        crop=args.crop, 
        eta=args.eta,
        translate=(0.01, 0.01),
        p_occlusion=0
        )

    loggers = []
    checkpoints = [Timer(duration=max_duration)]
    if not args.debug:
        logger     = CSVLogger(args.log_dir, name=args.name)
        model_dir  = logger.log_dir + "/checkpoints"
        checkpoint = ModelCheckpoint(model_dir, monitor='val_loss', save_top_k=2, mode='min')

        loggers.append(logger)
        checkpoints.append(checkpoint)
    else:
        print("Debugging.")
    
    # Create training set
    # Filter metadata to selection
    metadata["Filename"] = metadata["Filename"].apply(lambda x: "/".join([args.root, x]))
    metadata = metadata[(metadata["Energy index"] == args.energy) & (metadata["Convergence angle index"] == args.convergence_angle)]
    train_set       = PACBEDPhaseDataset(source=metadata, transforms=train_augmenter)

    # Count number of samples in training set for each class
    weights = [1/row["DimZ"] for _, row in metadata.iterrows()]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_sampler   = RandomSampler(train_set, replacement=True, num_samples=args.n_samples)
    train_loader    = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.n_workers, sampler=train_sampler, pin_memory=True)

    # Create a distribution of realistically augmented images (no occlusion, et c.)
    realistic_set           = PACBEDPhaseDataset(source=metadata, transforms=val_augmenter)
    
    # Create a validation set
    validation_sampler          = RandomSampler(realistic_set, replacement=True, num_samples=args.n_validation)
    validation_initial_loader   = DataLoader(realistic_set, batch_size=args.batch_size, num_workers=args.n_workers, sampler=validation_sampler, pin_memory=True)
    validation_set              = InMemoryPACBEDPhaseDataset.from_dataloader(validation_initial_loader)
    validation_loader           = DataLoader(validation_set, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=True, shuffle=True)

    # Create a test set
    test_sampler          = RandomSampler(realistic_set, replacement=True, num_samples=args.n_test)
    test_initial_loader   = DataLoader(realistic_set, batch_size=args.batch_size, num_workers=args.n_workers, sampler=test_sampler, pin_memory=True)
    test_set              = InMemoryPACBEDPhaseDataset.from_dataloader(test_initial_loader)
    test_loader           = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=True, shuffle=True)

    # Create an experimental test set
    # experimental_loader   = utils.experimental_dataloader(data_dir=Path(args.exp_dir), results_file=Path(args.exp_cfg), batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=True)

    # Define model
    backbone    = resnet50(pretrained=False, num_classes=len(metadata["Phase index"].unique()))
    optimizer   = torch.optim.SGD
    loss        = torch.nn.CrossEntropyLoss()
    model       = PhaseClassifier(
        backbone=backbone,
        optimizer=optimizer,
        loss=loss,
        optimizer_params={
            'lr': args.lr,
            'momentum': args.momentum
        },
        training_params={
            'optimizer': optimizer.__name__,
            'loss': loss.__class__.__name__,
            'backbone': backbone.__class__.__name__,
            **vars(args)
        })
    
    # Turn on grad for all parameters
    # torch.set_grad_enabled(True)
    
    # Compile the model
    compiled = torch.compile(model)
    # compiled = model

    torch.set_float32_matmul_precision(args.precision)
    trainer = pl.Trainer(
        accelerator=args.device, 
        devices=args.n_devices, 
        max_epochs=args.n_epochs, 
        logger=loggers,
        callbacks=checkpoints)
    
    if args.checkpoint == "best":
        chkpt = checkpoint.best_model_path
        trainer.fit(compiled, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=validation_loader,
                    ckpt_path=chkpt)
        
    elif args.checkpoint != '':
        trainer.fit(compiled, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=validation_loader,
                    ckpt_path=args.checkpoint)
    else:
        # Train model
        trainer.fit(compiled, 
                    train_dataloaders=train_loader, 
                    val_dataloaders=validation_loader)
    
    # Save final checkpoint
    if not args.debug:
        trainer.save_checkpoint(f"{model_dir}/{name}_final.ckpt")

    ###########################################
    # Test on data drawn from test set
    best_checkpoint = checkpoint.best_model_path
    model       = PhaseClassifier.load_from_checkpoint(
        best_checkpoint,
        hparams_file=args.log_dir + "/hparams.yaml",
        backbone=backbone,
        optimizer=optimizer,
        loss=loss
        )
    
    model.eval()

    y_true = []
    y_pred = []

    for x, y in test_loader:

        logits = model(x.to(model.device))
        pred = torch.softmax(logits, dim=1).argmax(dim=1)

        y_true.extend([true.cpu().numpy() for true in y])
        y_pred.extend([p.cpu().numpy() for p in pred])


    accuracy = accuracy_score(y_true, y_pred)    
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy}")

    disp = ConfusionMatrixDisplay(cm, 
                                display_labels=[c for c in metadata["Phase"].unique()])
    disp.plot()
    plt.savefig(f"{args.log_dir}/{name}_confusion_matrix.png")

    csv_path = loggers[0].experiment.metrics_file_path
    f, ax, df = utils.visualize_classification_metrics_csv(csv_path, figsize=(5, 5), dpi=300)
    ax.set_title("Cross-entropy loss for phase classification")
    plt.savefig(f"{args.log_dir}/{name}_loss.png")


    end = time.time()

    runtime = end - start

    return runtime

if __name__ == "__main__":

    runtime = main()

    print(f"\n Runtime: {runtime}")