import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, set_image_backend
from PIL import Image
import os
import math
import random
import numpy as np
import logging
import cv2
# import functools
import accimage
set_image_backend('accimage')

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, Image):
        new_video_x = (Image - 127.5) / 128
        return new_video_x

class Datasets(Dataset):
    def __init__(self, args, dataset_root, ground_truth, typ, sample_duration=16, sample_size=224, phase='train'):

        def get_data_list_and_label(data_df):
            T = 0  # if typ == 'M' else 1
            return [(lambda arr: ('/'.join(arr[T].split('/')), int(arr[1]), int(arr[2])))(i[:-1].split(' '))
                    for i in open(data_df).readlines()]
        self.dataset_root = dataset_root
        self.sample_duration = sample_duration
        self.sample_size = sample_size
        self.phase = phase
        self.typ = typ
        self.args = args

        self.transform = transforms.Compose([Normaliztion(), transforms.ToTensor()])

        lines = filter(lambda x: x[1] > 1, get_data_list_and_label(ground_truth))
        self.inputs = list(lines)
        if phase == 'train':
            while len(self.inputs) % (args.batch_size * args.nprocs) != 0:
                sample = random.choice(self.inputs)
                self.inputs.append(sample)
            logging.info('Training Data Size is: {}'.format(len(self.inputs)))
        else:
            logging.info('Validation Data Size is: {} '.format(len(self.inputs)))

    def transform_params(self, resize=(320, 240), crop_size=224, flip=0.5):
        if self.phase == 'train':
            left, top = random.randint(0, resize[0] - crop_size), random.randint(0, resize[1] - crop_size)
            is_flip = True if random.uniform(0, 1) < flip else False
        else:
            left, top = (resize[0] - crop_size) // 2, (resize[1] - crop_size) // 2

            is_flip = False
        return (left, top, left + crop_size, top + crop_size), is_flip

    def rotate(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]
        if center is None:
            center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    def image_propose(self, data_path, sl):
        sample_size = self.sample_size
        if self.phase == 'train':
            resize = eval(self.args.resize)
        else:
            resize = (256, 256)
        crop_rect, is_flip = self.transform_params(resize=resize, flip=self.args.flip)  # no flip
        if random.uniform(0, 1) < self.args.rotated and self.phase == 'train':
            rotated = random.randint(-10, 10)
        else:
            rotated = 0
        def image_to_np(image):
            """
            Returns:
                np.ndarray: Image converted to array with shape (width, height, channels)
            """
            image_np = np.empty([image.channels, image.height, image.width], dtype=np.uint8)
            image.copyto(image_np)
            image_np = np.transpose(image_np, (1, 2, 0))
            return image_np

        def transform(img):
            img = self.rotate(np.asarray(img), rotated)
            img = Image.fromarray(img)
            img = img.resize(resize)
            img = img.crop(crop_rect)
            if is_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img.resize((sample_size, sample_size)))

        def Sample_Image(imgs_path, sl):
            frams = []
            for a in sl:
                img = transform(image_to_np(accimage.Image(os.path.join(imgs_path, "%06d.jpg" % a))))
                frams.append(self.transform(img).view(3, sample_size, sample_size, 1))
            return torch.cat(frams, dim=3).type(torch.FloatTensor)
        return Sample_Image(data_path, sl)

    def get_sl(self, clip):
        sn = self.sample_duration
        if self.phase == 'train':
            f = lambda n: [(lambda n, arr: n if arr == [] else random.choice(arr))(n * i / sn,
                                                                                   range(int(n * i / sn),
                                                                                         max(int(n * i / sn) + 1,
                                                                                             int(n * (
                                                                                                     i + 1) / sn))))
                           for i in range(sn)]
        else:
            f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                    max(int(
                                                                                                        n * i / sn) + 1,
                                                                                                        int(n * (
                                                                                                                i + 1) / sn))))
                           for i in range(sn)]
        return f(clip)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])
        self.data_path = os.path.join(self.dataset_root, self.inputs[index][0])
        self.clip = self.image_propose(self.data_path, sl)
        return self.clip.permute(0, 3, 1, 2), self.inputs[index][2]
    def __len__(self):
        return len(self.inputs)
