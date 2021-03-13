import torch
from .base import Datasets
from .DSADataset import DSAttDatasets
from torchvision import transforms, set_image_backend
import random, os
from PIL import Image
import numpy as np
import accimage
set_image_backend('accimage')

class IsoGDData(Datasets):
    def __init__(self, args, dataset_root, ground_truth, typ, sample_duration=16, sample_size=224, phase='train'):
        super(IsoGDData, self).__init__(args, dataset_root, ground_truth, typ, sample_duration, sample_size, phase)
        self.dsatt = DSAttDatasets(args, dataset_root, ground_truth, typ, sample_duration, sample_size, phase)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])
        # self.data_path = os.path.join(self.dataset_root, self.typ, self.inputs[index][0])
        self.data_path = os.path.join(self.dataset_root, self.inputs[index][0])

        if self.args.Network == 'RAAR3D':
            self.clip, skgmaparr = self.dsatt.image_propose(self.data_path, sl)
        else:
            self.clip, skgmaparr = self.image_propose(self.data_path, sl), []

        return self.clip.permute(0, 3, 1, 2), self.inputs[index][2], skgmaparr

    def __len__(self):
        return len(self.inputs)
