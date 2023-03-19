import argparse
import os, sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, '/home/karthik7/projects/def-laporte1/karthik7/new_env/cycleGAN_lightning/code_scripts')

from datasets3D_pl import ImageDataset
from models3D_dropout_pl import ResnetGenerator3D, LighterUnetGenerator3D, PatchGANDiscriminatorwithSpectralNorm

# These path names refer to the paths on the Cedar Cluster of Compute Canada. They must be changed accordingly to your username.
path = '/home/karthik7/projects/def-laporte1/karthik7/cycleGAN3D/datasets/'
dataroot = path+'mr2ct_latest/'

log_path = '/home/karthik7/projects/def-laporte1/karthik7/new_env/cycleGAN_lightning/tensorboard_logs/'

version_name = "v8_withGC_withUnc_lgc=1.0"    # should match with what is used in train3D_pl.py file
epoch_num = 199  
exp_name = version_name + "_" + str(epoch_num) + "/"

# paths for saving the images generated during testing after training ends
testgen_path_A2B = '/home/karthik7/projects/def-laporte1/karthik7/new_env/cycleGAN_lightning/testing_generated/'+exp_name
if not os.path.exists(testgen_path_A2B):
    os.makedirs(testgen_path_A2B)
exp_name_1 = version_name + "_" + str(epoch_num) + "_images" "/"
testgen_path_images = '/home/karthik7/projects/def-laporte1/karthik7/new_env/cycleGAN_lightning/testing_generated/'+exp_name_1
if not os.path.exists(testgen_path_images):
    os.makedirs(testgen_path_images)

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default=dataroot, help='root directory of the dataset')

parser.add_argument('--epoch', type=int, default=1, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs for training')
parser.add_argument('--decay_epoch', type=int, default=100,
                    help='epoch from where the learning rate starts linearly decaying to zero')

parser.add_argument('--test_batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--mcd_samples', type=int, default=20, help='number of MC samples')
parser.add_argument('--use_dropout', type=bool, default=True, help='enabling dropout by default')

parser.add_argument('--size_z', type=int, default=48, help='default z dimension of the image data')
parser.add_argument('--size_x', type=int, default=256, help='default x dimension of the image data')
parser.add_argument('--size_y', type=int, default=128, help='default y dimension of the image data')

parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
parser.add_argument('--output_nc', type=int, default=2, help='number of output channels')
args = parser.parse_args()
print(args)


# Explicitly enabling (only) dropout - will be used during test_step (for MC Dropout)
def enable_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


class MRCTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=args.dataroot, test_batch_size=args.test_batch_size,
                 num_workers=8, shuffle=False):
        super().__init__()
        self.data_dir = data_dir
        # self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.transform_ = [transforms.Normalize((0.0,), (1.0,))]

    def prepare_data(self):
        # ImageDataset(root_dir=self.data_dir, transform=self.transform_, mode='train', one_side=False)
        ImageDataset(root_dir=self.data_dir, transform=self.transform_, mode='test_1', one_side=True)

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.mrct_test = ImageDataset(root_dir=self.data_dir, transform=self.transform_,
                                          mode='test_1', one_side=True)

    def test_dataloader(self):
        return DataLoader(self.mrct_test, batch_size=self.test_batch_size,
                          num_workers=self.num_workers, shuffle=False, pin_memory=True)


class LightningCycleGAN3D(pl.LightningModule):
    def __init__(self, test_batch_size=args.test_batch_size, lr=args.lr, b1=0.5, b2=0.999,
                 num_feat_maps=[35, 70, 140, 280], num_residual_blocks=6, use_dropout=args.use_dropout,
                 mcd_samples=args.mcd_samples, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        torch.manual_seed(0)

        # instatiating the generators and the discriminators
        self.netG_A2B = LighterUnetGenerator3D(args.input_nc, args.output_nc, self.hparams.num_feat_maps)
        self.netG_B2A = LighterUnetGenerator3D(args.input_nc, args.output_nc, self.hparams.num_feat_maps)
        # for using Resnet Generator
        # self.netG_A2B = ResnetGenerator3D(args.input_nc, args.output_nc, self.hparams.num_feat_maps[0],
        #                                   self.hparams.num_residual_blocks, self.hparams.use_dropout)
        # self.netG_B2A = ResnetGenerator3D(args.input_nc, args.output_nc, self.hparams.num_feat_maps[0],
        #                                   self.hparams.num_residual_blocks, self.hparams.use_dropout)

        self.netD_A = PatchGANDiscriminatorwithSpectralNorm(args.input_nc)
        self.netD_B = PatchGANDiscriminatorwithSpectralNorm(args.input_nc)

        self.input_A = torch.Tensor(args.test_batch_size, args.input_nc, args.size_z, args.size_x, args.size_y).to(self.device)
        self.input_B = torch.Tensor(args.test_batch_size, args.input_nc, args.size_z, args.size_x, args.size_y).to(self.device)

    def forward(self, x):
        pass

    def test_step(self, batch, batch_idx):
        # -------- Aleatoric and Epistemic Uncertainty ---------
        # MC Dropout testing
        num_mc_samples = self.hparams.mcd_samples
        mcd_preds_B = torch.zeros((num_mc_samples, args.size_z, args.size_x, args.size_y))  # hard-coded the shape of the image
        aleatoric_maps_B = torch.zeros((num_mc_samples, args.size_z, args.size_x, args.size_y))

        # enabling dropout manually using the function defined above
        self.netG_A2B.apply(enable_dropout)
        # self.netG_B2A.apply(enable_dropout)

        real_A = self.input_A.copy_(batch['A']).to(self.device)
        # Get MC Samples
        for i in range(num_mc_samples):
            # real_A = self.input_A.copy_(batch['A']).to(self.device)

            # generating A2B volumes
            fake_B, logSigma_B = self.netG_A2B(real_A)
            # print("Before", fake_B.shape, logSigma_B.shape)
            mcd_preds_B[i] = (fake_B.data).squeeze()
            aleatoric_maps_B[i] = (logSigma_B.data).squeeze()
            # print("data shape", fake_B.data.shape, logSigma_B.data.shape)


        # Epistemic Uncertainty
        mean_mcd_pred_B = (torch.mean(mcd_preds_B, dim=0)).cpu().numpy()
        epistemic_mcd_pred_B = (torch.var(mcd_preds_B, dim=0)).cpu().numpy()
        np.save(testgen_path_A2B + "epistemic_mean_%02d" % (batch_idx + 1), mean_mcd_pred_B)

        np.save(testgen_path_A2B + "epistemic_var_%02d" % (batch_idx + 1), epistemic_mcd_pred_B)
        plt.imsave(testgen_path_images + "epistemic_sag_var_%02d.png" % (batch_idx+1),
                   np.flip(epistemic_mcd_pred_B[24, :, :]),      # .reshape(args.size, args.size, 3),
                   cmap='jet', format='png')

        # Mean Aleatoric Uncertainty for A2B
        aleatoric_pred_B = (torch.mean(aleatoric_maps_B, dim=0)).cpu().numpy()
        np.save(testgen_path_A2B + "aleatoric_mean_%02d" % (batch_idx + 1), aleatoric_pred_B)
        plt.imsave(testgen_path_images + "aleatoric_sag_%02d.png" % (batch_idx+1),
                   np.flip(aleatoric_pred_B[24, :, :]),
                   cmap='jet', format='png')

