import glob
import os

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SegFormerDataset(Dataset):
    def __init__(self, data_df, transforms=None):
        self.df = data_df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.loc[idx, "images_paths"]
        mask_path = self.df.loc[idx, "masks_paths"]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = preprocess_mask(mask)

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask


def preprocess_mask(mask):
    mask_processed = np.zeros((mask.shape[0], mask.shape[1]))
    mask_processed[np.where((mask==[34]))] = 1
    mask_processed[np.where((mask==[38]))] = 2
    mask_processed[np.where((mask==[57]))] = 3
    mask_processed[np.where((mask==[75]))] = 4
    mask_processed[np.where((mask==[79]))] = 5
    mask_processed[np.where((mask==[90]))] = 6
    mask_processed[np.where((mask==[113]))] = 7
    return mask_processed


def get_csv_dataset(data_path):
    images = glob.glob(f"{data_path}/Images/*png", recursive=True)
    images_names = [os.path.basename(x) for x in images]

    labels = glob.glob(f"{data_path}/Labels/*png", recursive=True)
    labels_names = [os.path.basename(x) for x in labels]

    df = pd.DataFrame(zip(images, images_names, labels, labels_names))
    df.columns = ["images_paths", "images_names", "masks_paths", "masks_names"]
    return df
