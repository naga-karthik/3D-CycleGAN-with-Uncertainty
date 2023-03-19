import glob, os, sys
import random
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', one_side=False):
        """
        Custom class for loading the data
        :param root_dir: path of the root directory from where the data is loaded
        :param transform: defines the composition of transformations to be applied
        :param mode: training mode or testing mode
        :param one_side: if True, only loads data from one domain - used during testing
        """
        self.transform = transforms.Compose(transform)
        self.mode = mode
        self.one_side = one_side
        self.files_A = sorted(glob.glob(os.path.join(root_dir, '%s/A' % self.mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root_dir, '%s/B' % self.mode) + '/*.*'))

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        # NOTE: the data is transformed into a tensor in [z, x, y] format. Because PyTorch puts the z-axis first.
        # Therefore, when shape is printed, the format is [batch_size x Z x X x Y] or, [batch_size x 48 x 256 x 128]
        if (self.mode == 'test' or self.mode == "test_1") and self.one_side:
            item_A = self.transform(
                torch.FloatTensor(np.load(self.files_A[idx % len(self.files_A)])).permute((2, 0, 1)))
            return {'A': item_A}
        else:
            item_A = self.transform(
                torch.FloatTensor(np.load(self.files_A[idx % len(self.files_A)])).permute((2, 0, 1)))
            item_B = self.transform(
                torch.FloatTensor(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)])).permute((2, 0, 1)))

            # unsqueezing to add the channel dimension
            item_A = item_A.unsqueeze(0)
            item_B = item_B.unsqueeze(0)

            return {'A': item_A, 'B': item_B}



