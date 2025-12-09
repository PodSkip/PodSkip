# Custom Dataset

import os
import torch
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class podDataset(Dataset):
    """Dataset for podcast audio"""

    def __init__(self, metadata, audio_dir, transform=None):
        """Initialisation of Dataset"""

        self.audio_metadata = pd.read_csv(metadata)
        self.audio_labels = self._convert_labels(self.audio_metadata)
        self.audio_dir = audio_dir
        self.transform = transform

    def _convert_labels(self, metadata):
        labels = {}
        for _, row in metadata.iterrows():
            audio_name = row["filename"]
            segments = []

            for i in range(1, len(row) - 1, 2):
                start = row.iloc[i]
                end = row.iloc[i + 1]

                if pd.notna(start) and pd.notna(end) and start != "" and end != "":
                    segments.append((start, end))

            if segments:
                labels[audio_name] = segments

        print(labels)
        return labels

    def __len__(self):
        return len(self.audio_metadata)

    def __getitem__(self, idx):
        audio_name = self.audio_metadata.iloc[idx, 0]
        audio_path = os.path.join(self.audio_dir, audio_name)
        audio = torchaudio.load(audio_path)
        label = self.audio_labels.get(audio_name)
        return audio, label
