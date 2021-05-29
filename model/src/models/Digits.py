from torch.utils.data import Dataset
from PIL import Image
from model.data.labels import labels

import pandas as pd
import os


class DigitsDataset(Dataset):
    """Digits captcha dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        # Load the image.
        digit_path = self.landmarks_frame.iloc[index, 0]
        image_path = os.path.join(self.root_dir, digit_path)
        image = Image.open(image_path).convert('L')

        # Loading the class of the image.
        label = labels[self.landmarks_frame.iloc[index, 1]]
        
        if self.transform:
            image = self.transform(image)

        return image, label
