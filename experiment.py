#%%
import os
from pathlib import Path
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import json
from models import Augmenter, PACBED
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

torch.manual_seed(42)
# %%

def preprocess(x: np.ndarray, scaling=0.25, size=256):

    # Reshape 
    x = x.reshape((size, size))

    x = np.abs(scaling * x / np.mean(np.abs(x)))

    return x


def bin_to_image(file: Path):

    # Read binary file
    with open(file, "rb") as f:
        data = np.fromfile(f, dtype=np.float32)

    # Preprocess
    data = preprocess(data)

    return data
    # Convert to PIL image
    # img = torchvision.transforms.ToPILImage()(data)

    # return img
# %%
torch.set_float32_matmul_precision('medium')
data_dir = Path("exp_data_20230421")
results_file = Path("experiment_results.json")

with open(results_file, "r") as f:
    results = json.load(f)

images = torch.stack([torch.tensor(bin_to_image(data_dir / result['file'])) for result in results['data']])
y = torch.tensor([result['thickness'] for result in results['data']])
# %%
X = images.unsqueeze(1)

plt.imshow(X[0, 0, :, :])

model       = PACBED.load_from_checkpoint("version_2/checkpoints/epoch=136-step=978944.ckpt")
model.eval()
model.to(X.device)

#%%
with torch.no_grad():
    pred = model(X.to(model.device))
    d = torch.exp(pred)
    print(pred)

#%%
trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0]
        )



# pred = model(X)
# print(pred)
# d = torch.exp(pred)

# %%
