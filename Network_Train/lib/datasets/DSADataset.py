import torch
from .base import Datasets as dataset
from torchvision import transforms, set_image_backend
import random, os
from PIL import Image
import numpy as np
import accimage
set_image_backend('accimage')
from scipy.ndimage.filters import gaussian_filter
import json
import matplotlib.pyplot as plt
mycmap = plt.cm.get_cmap('jet')

class DSAttDatasets(dataset):
    def __init__(self, args, dataset_root, ground_truth, typ, sample_duration=16, sample_size=224, phase='train'):
        super(DSAttDatasets, self).__init__(args, dataset_root, ground_truth, typ, sample_duration, sample_size, phase)

    def image_propose(self, data_path, sl):
        sample_size = self.sample_size
        if self.phase == 'train':
            resize = eval(self.args.resize)
        else:
            resize = (256, 256)
        crop_rect, is_flip = self.transform_params(resize=resize, flip=self.args.flip)  # no flip

        if random.uniform(0, 1) < self.args.rotated and self.phase == 'train':
            rotated = random.randint(0, 10)
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
            img = self.rotate(img, rotated)
            img = Image.fromarray(img)
            img = img.resize(resize)
            img = img.crop(crop_rect)
            if is_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img.resize((sample_size, sample_size)))
            # return image_to_np(img.resize((sample_size, sample_size)))
        def get_gmap(kp_path, img, a):
            with open(os.path.join(kp_path, '%06d_keypoints.json' % a), 'r') as f:
                kps = json.loads(f.read())
            pose = kps['people'][0]['pose_keypoints_2d']
            r_hand = kps['people'][0]['hand_right_keypoints_2d']
            l_hand = kps['people'][0]['hand_left_keypoints_2d']
            arr_x = pose[0::3] + r_hand[0::3] + l_hand[0::3]
            arr_y = pose[1::3] + r_hand[1::3] + l_hand[1::3]
            if not any(arr_x) or not any(arr_y):
                return np.ones((sample_size, sample_size))
            h, w, c = img.shape
            mp = np.zeros((h, w))
            for x, y in zip(arr_x, arr_y):
                x, y = round(x), round(y)
                if x < w and x > 0 and y < h and y > 0:
                    mp[y, x] = 1
            mp = gaussian_filter(mp, sigma=20)
            mp = (mp - np.min(mp)) / (np.max(mp) - np.min(mp) + 0.0000001)
            mp = Image.fromarray(self.rotate(np.asarray(mp), rotated))
            mp = mp.resize(resize).crop(crop_rect).resize((sample_size, sample_size))
            if is_flip:
                mp = mp.transpose(Image.FLIP_LEFT_RIGHT)

            mp = np.array(mp)
            # plt.matshow(mp)
            # plt.imsave('vis.png', img, cmap=mycmap, vmin=.1)
            # plt.savefig('res.png')
            return mp

        def getskg_tensor(skgmap):
            a = torch.from_numpy(skgmap.astype(np.float32))
            return a.view(1, 1, sample_size, sample_size)
        def Sample_Image(imgs_path, sl):
            frams = []
            skgmaparr = []
            for a in sl:
                # img = Image.open(os.path.join(imgs_path, "%06d.jpg" % a))
                img = image_to_np(accimage.Image(os.path.join(imgs_path, "%06d.jpg" % a)))
                imgs = transform(img)
                frams.append(self.transform(imgs).view(3, sample_size, sample_size, 1))
                kp_path = imgs_path.replace('Image', 'keypoints')
                gmap = get_gmap(kp_path, np.asarray(img), a)
                skgmaparr.append(gmap)
            return torch.cat(frams, dim=3).type(torch.FloatTensor), torch.cat([getskg_tensor(a) for a in skgmaparr], dim=1)
        return Sample_Image(data_path, sl)
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sl = self.get_sl(self.inputs[index][1])
        self.data_path = os.path.join(self.dataset_root, self.inputs[index][0])
        self.clip, skgmaparr = self.image_propose(self.data_path, sl)

        return self.clip.permute(0, 3, 1, 2), self.inputs[index][2], skgmaparr

    def __len__(self):
        return len(self.inputs)
