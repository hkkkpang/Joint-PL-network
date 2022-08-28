from torch.utils.data import Dataset
import glob
from PIL import Image
from .base_dataset import BaseDataset
from .augmentations import resize_sample, to_tensor_sample, ha_augment_sample
import torch
from .line_utils import WireframeHuangKun, CropAugmentation
import os
from ..settings import DATA_PATH


class york(BaseDataset):

    default_conf = {
        'shape': [512, 512],
        'jittering': [0.5, 0.5, 0.2, 0.05],
        'valid_path': os.path.join(DATA_PATH, 'york/valid'),
        'crop': False
    }

    def _init(self, config):

        self._paths = {}
        self._config = config

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

        target = {}

        # step 1 load npz
        lcmap, lcoff, lleng, angle = WireframeHuangKun.fclip_parsing(
            filename[:-4] + "_line.npz", 'radian'
        )

        target["lcmap"] = torch.from_numpy(lcmap).float()
        target["lcoff"] = torch.from_numpy(lcoff).float()
        target["lleng"] = torch.from_numpy(lleng).float()
        target["angle"] = torch.from_numpy(angle).float()
        sample['target'] = target

        return sample

    def __len__(self):
        return len(self.files)
