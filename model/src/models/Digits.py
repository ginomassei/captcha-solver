from torch.utils.data import Dataset
from skimage import io
from labels import labels

import pandas as pd
import os


class DigitsDataset(Dataset):
    """Digits captcha dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, index):
        # Load the image.
        image_path = os.path.join(self.root_dir, self.landmarks_frame.iloc[index, 0])
        image = io.imread(image_path, as_gray=True)

        # Loading the class of the image.
        label = labels[self.landmarks_frame.iloc[index, 1]]
        
        if self.transform:
            image = self.transform(image)
        
        return (image, label)
