from torch.utils.data import Dataset
import glob
from PIL import Image
import os
from .base_dataset import BaseDataset
from .augmentations import resize_sample, to_tensor_sample, ha_augment_sample
from skimage import io
import torch
import numpy as np
from .line_utils import WireframeHuangKun, CropAugmentation
from ..settings import DATA_PATH


class Wireframe_line(BaseDataset):

    default_conf = {
        'shape': [512, 512],
        'jittering': [0.5, 0.5, 0.2, 0.05],
        'train_path': os.path.join(DATA_PATH, 'wireframe1_datarota_3w/train'),
        #'valid_path': os.path.join(DATA_PATH, 'wireframe1_datarota_3w/valid'),
        'valid_path': '../assets/line_match_data',
        'crop': False
    }

    def _init(self, config):

        self._paths = {}
        self._config = config
        # Train split
        self._paths['train'] = self._config.train_path
        # Valid split
        self._paths['val'] = self._config.valid_path

    def get_dataset(self, split):
        return _Dataset(self._paths[split], self._config, split)


def image_transforms(shape, jittering):

    def train_transforms(sample):
        sample = resize_sample(sample, image_shape=shape)
        sample = to_tensor_sample(sample)
        sample = ha_augment_sample(sample, jitter_paramters=jittering)
        return sample

    return {'train': train_transforms}


class _Dataset(Dataset):

    def __init__(self, paths, config, split):

        self._paths = paths
        data_transforms = image_transforms(shape=config.shape, jittering=config.jittering)
        self.data_transform = data_transforms['train']
        self.files = []

        for filename in glob.glob(self._paths + '/*.png'):
            self.files.append(filename)

        self.files.sort()

        self.conf = config
        self.split = split

    def _get_im_name(self, idx):

        iname = self.files[idx]

        return iname

    def __getitem__(self, idx):
        filename = self.files[idx]

        image = Image.open(filename)
        sample = {'image': image, 'idx': idx}
        if self.data_transform:
            sample = self.data_transform(sample)

        #image_ = io.imread(filename).astype(float)[:, :, :3]

        target = {}

        # step 1 load npz
        lcmap, lcoff, lleng, angle = WireframeHuangKun.fclip_parsing(
            filename[:-4] + "_line.npz", 'radian'
        )
        #with np.load(filename[:-4] + '_label.npz') as npz:
        #    lpos = npz["lpos"][:, :, :2]

        '''
        # step 2 crop augment
        if self.split == "train":
            if self.conf.crop:
                s = np.random.choice(np.arange(0.9, 1.6, 0.1))
                image_t, lcmap, lcoff, lleng, angle, cropped_lines, cropped_region \
                    = CropAugmentation.random_crop_augmentation(image_, lpos, s)
                image_ = image_t
                lpos = cropped_lines
        '''

        target["lcmap"] = torch.from_numpy(lcmap).float()
        target["lcoff"] = torch.from_numpy(lcoff).float()
        target["lleng"] = torch.from_numpy(lleng).float()
        target["angle"] = torch.from_numpy(angle).float()

        #mean = [109.730, 103.832, 98.681]
        #stddev = [22.275, 22.124, 23.229]
        #image = (image_ - mean) / stddev
        #image = np.rollaxis(image, 2).copy()

        #sample['image_line'] = torch.from_numpy(image).float()
        sample['target'] = target

        return sample

    def __len__(self):
        return len(self.files)
