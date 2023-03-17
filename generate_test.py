#%%
from models import Augmenter
from datasets import generate_test_dataset_into_directory
import torch
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser(description='Generate fixed test dataset')
    parser.add_argument('--n-samples-per-file', type=int, default=165, help='Number of samples per file')
    parser.add_argument('--n-pixels', type=int, default=1040, help='Number of pixels')
    parser.add_argument('--crop', type=int, default=225, help='Crop size')
    parser.add_argument('--eta', type=float, default=1, help='Eta')
    parser.add_argument('--files', type=str, help='Files')
    parser.add_argument('--test-root', type=str, default=Path("./data/test"), help='Test root')
    parser.add_argument('--n-test', type=int, default=int(1e2), help='Number of test samples')
    parser.add_argument('--n-workers', type=int, default=16, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Seed')

    args = parser.parse_args()

    N_SAMPLES_PER_FILE  = args.n_samples_per_file
    N_PIXELS            = args.n_pixels
    CROP                = args.crop
    ETA                 = args.eta
    FILES               = args.files
    TEST_ROOT           = args.test_root
    N_TEST              = args.n_test
    N_WORKERS           = args.n_workers
    SEED                = args.seed

    torch.manual_seed(SEED)

    augmenter   = Augmenter(n_pixels=N_PIXELS, crop=CROP, eta=0)

    generate_test_dataset_into_directory(files = FILES, 
                                    target_dir = TEST_ROOT, 
                                    n_samples = N_TEST, 
                                    n_pixels=N_PIXELS, 
                                    n_samples_per_file=N_SAMPLES_PER_FILE, 
                                    augmenter=augmenter,
                                    n_workers=N_WORKERS)
    

if __name__ == "__main__":

    main()