#%%
from models.regression import PACBED, Augmenter
from data.regression import PACBEDDataset, FixedPACBEDDataset, InMemoryPACBEDDataset, generate_test_dataset_into_directory
from torch.utils.data import DataLoader, RandomSampler
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

N_SAMPLES_PER_FILE  = 165
N_PIXELS_ORIGINAL   = 1040
N_PIXELS_TARGET     = 256
CROP                = 225
ETA                 = 1
FILES               = [""]
TEST_ROOT           = Path("./data/test")
BATCH_SIZE          = 128
N_WORKERS           = 32
N_SAMPLES           = int(1e3 * BATCH_SIZE)
N_TEST              = int(1e2)
N_VALIDATION        = int(1e2 * BATCH_SIZE)
LR                  = 1e-4
SEED                = 42

torch.manual_seed(SEED)
#%%
augmenter   = Augmenter(n_pixels=N_PIXELS_TARGET, crop=CROP, eta=0)
# logger      = TensorBoardLogger("./logs", name="test")
logger      = CSVLogger("./logs", name="test")


#%%
# Create training set
train_set       = PACBEDDataset(files = FILES, n_samples = N_SAMPLES_PER_FILE, n_pixels=N_PIXELS_ORIGINAL, transforms=augmenter)
train_sampler   = RandomSampler(train_set, replacement=True, num_samples=N_SAMPLES)
train_loader    = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, sampler=train_sampler, pin_memory=True)

# Create validation set
# validation_sampler          = RandomSampler(train_set, replacement=True, num_samples=N_VALIDATION)
# validation_initial_loader   = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, sampler=validation_sampler, pin_memory=True)
# validation_set              = InMemoryPACBEDDataset.from_dataloader(validation_initial_loader)
# validation_loader           = DataLoader(validation_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS, pin_memory=True)

# Create test set
# If test set does not exist, create it
if not Path(TEST_ROOT).exists():

    generate_test_dataset_into_directory(files = FILES, 
                                        target_dir = TEST_ROOT, 
                                        n_samples = N_TEST, 
                                        n_pixels=N_PIXELS_ORIGINAL, 
                                        n_samples_per_file=N_SAMPLES_PER_FILE, 
                                        augmenter=augmenter,
                                        n_workers=N_WORKERS)

test_set       = FixedPACBEDDataset(root = TEST_ROOT)
test_loader    = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

#%%
# Visualize some samples from training set
# x, y = train_set[100]
# x, y = next(iter(train_loader))
# plt.imshow(x[0, :, :, :].squeeze())

#%%
# Define model
model       = PACBED(backbone="resnet34", n_pixels=N_PIXELS_TARGET, lr=LR)
#%%
torch.set_float32_matmul_precision('medium')
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10, logger=logger)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

# %%
