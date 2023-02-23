#%%
from models import PACBED, Augmenter
from datasets import PACBEDDataset
from torch.utils.data import DataLoader, RandomSampler
import pytorch_lightning as pl
import einops
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import TensorBoardLogger

N_SAMPLES_PER_FILE  = 165
N_PIXELS            = 1040
CROP                = 266
ETA                 = 0
FILES               = "/home/victor/research/pacbed/data/data_6.85.bin"
BATCH_SIZE          = 8
N_WORKERS           = 16
N_SAMPLES           = int(1e3 * BATCH_SIZE)

#%%
augmenter   = Augmenter(n_pixels=N_PIXELS, crop=CROP, eta=0)
logger      = TensorBoardLogger("./logs", name="test")

dataset     = PACBEDDataset(files = FILES, n_samples = N_SAMPLES_PER_FILE, n_pixels=N_PIXELS, transforms=augmenter)
sampler     = RandomSampler(dataset, replacement=True, num_samples=N_SAMPLES)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, sampler=sampler)
model       = PACBED(n_pixels=N_PIXELS)
#%%
trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10, precision=32, logger=logger)
trainer.fit(model, data_loader)

# %%
