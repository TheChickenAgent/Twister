import numpy as np

np.random.seed(21)

from torch.utils.data import Dataset


class CustomTextDataset(Dataset):
    """
    CustomTextDataset: Custom Dataset for text classification
    Functions:
    - len: returns length of dataset
    - getitem: returns text and label of index
    """

    def __init__(
        self, file, label_name, text_name, transform=None, target_transform=None
    ):
        self.labels = file[label_name]
        self.texts = file[text_name]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        Returns:
            int: returns length of dataset
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns item of dataset at index idx

        Args:
            idx (int): index to return the text and label of

        Returns:
            (str, str|int): text and label
        """
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label
