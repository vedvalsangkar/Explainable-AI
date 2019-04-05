# import torch
import os
# from torch.utils import data

import pandas as pd
from torch import nn        # , optim
# from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image


class AndImageDataSet(Dataset):

    def __init__(self, img_folder, pair_file=None, transform=transforms.ToTensor()):

        self.img_transforms = transform

        self.img_folder = img_folder

        if img_folder is None:
            raise ValueError("Provide image folder name!")

        # if img_folder is not None:
        #     self.training_mode = True
        #     self.main_data = os.listdir(img_folder)
        #     self.data_len = len(self.main_data)

        if pair_file is not None:

            self.training_mode = False
            self.main_data = pd.read_csv(filepath_or_buffer=pair_file,
                                         header=0,
                                         index_col=0)
            self.data_len = self.main_data.shape[0]

        else:

            self.training_mode = True
            self.main_data = os.listdir(img_folder)
            self.data_len = len(self.main_data)

    def __getitem__(self, index):

        if self.training_mode:

            image = Image.open(self.img_folder + self.main_data[index])

            return self.img_transforms(image)

        else:

            l_name = self.main_data.iloc[index, 0]
            r_name = self.main_data.iloc[index, 1]

            image1 = Image.open(self.img_folder + l_name)
            image2 = Image.open(self.img_folder + r_name)
            label = self.main_data.iloc[index, 2]

            left_im = self.img_transforms(image1)
            right_im = self.img_transforms(image2)

            return left_im, right_im, label, l_name, r_name

    def __len__(self):
        """
        Returns number of data entries.
        :return: Data size.
        """
        return self.data_len


class AutoEncoder(nn.Module):

    def __init__(self, bias=False, temperature=1.0):

        super(AutoEncoder, self).__init__()

        self.temperature = temperature

        self.k_size = 3
        self.padding = self.k_size // 2
        self.st = 1
        self.bias = bias

        self.filter_size = 16

        self.upsample_mode = 'nearest'
        r"""
        Upsampling algorithm: one of ``'nearest'``, ``'linear'``, ``'bilinear'``, ``'bicubic'``x and ``'trilinear'``.
        """

        # -------------------------------- Encoding segment --------------------------------

        self.encoding_1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                                  out_channels=self.filter_size,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        # nn.ReLU(),
                                        # nn.Conv2d(in_channels=self.filter_size,
                                        #           out_channels=self.filter_size,
                                        #           kernel_size=self.k_size,
                                        #           stride=self.st,
                                        #           padding=self.padding,
                                        #           bias=self.bias
                                        #           ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size),
                                        nn.MaxPool2d(kernel_size=2,
                                                     stride=2)
                                        )

        self.encoding_2 = nn.Sequential(nn.Conv2d(in_channels=self.filter_size,
                                                  out_channels=self.filter_size * 2,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        # nn.ReLU(),
                                        # nn.Conv2d(in_channels=self.filter_size * 2,
                                        #           out_channels=self.filter_size * 2,
                                        #           kernel_size=self.k_size,
                                        #           stride=self.st,
                                        #           padding=self.padding,
                                        #           bias=self.bias
                                        #           ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size * 2),
                                        nn.MaxPool2d(kernel_size=2,
                                                     stride=2)
                                        )

        self.encoding_3 = nn.Sequential(nn.Conv2d(in_channels=self.filter_size * 2,
                                                  out_channels=self.filter_size * 4,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        # nn.ReLU(),
                                        # nn.Conv2d(in_channels=self.filter_size * 4,
                                        #           out_channels=self.filter_size * 4,
                                        #           kernel_size=self.k_size,
                                        #           stride=self.st,
                                        #           padding=self.padding,
                                        #           bias=self.bias
                                        #           ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size * 4),
                                        nn.MaxPool2d(kernel_size=2,
                                                     stride=2)
                                        )

        self.encoding_4 = nn.Sequential(nn.Conv2d(in_channels=self.filter_size * 4,
                                                  out_channels=self.filter_size * 8,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        # nn.ReLU(),
                                        # nn.Conv2d(in_channels=self.filter_size * 8,
                                        #           out_channels=self.filter_size * 8,
                                        #           kernel_size=self.k_size,
                                        #           stride=self.st,
                                        #           padding=self.padding,
                                        #           bias=self.bias
                                        #           ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size * 8),
                                        nn.MaxPool2d(kernel_size=2,
                                                     stride=2)
                                        )

        self.encoding_5 = nn.Sequential(nn.Conv2d(in_channels=self.filter_size * 8,
                                                  out_channels=self.filter_size * 16,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        # nn.ReLU(),
                                        # nn.Conv2d(in_channels=self.filter_size * 16,
                                        #           out_channels=self.filter_size * 16,
                                        #           kernel_size=self.k_size,
                                        #           stride=self.st,
                                        #           padding=self.padding,
                                        #           bias=self.bias
                                        #           ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size * 16),
                                        nn.MaxPool2d(kernel_size=2,
                                                     stride=2)
                                        )

        self.encoding_6 = nn.Sequential(nn.Conv2d(in_channels=self.filter_size * 16,
                                                  out_channels=self.filter_size * 32,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        # nn.ReLU(),
                                        # nn.Conv2d(in_channels=self.filter_size * 32,
                                        #           out_channels=self.filter_size * 32,
                                        #           kernel_size=self.k_size,
                                        #           stride=self.st,
                                        #           padding=self.padding,
                                        #           bias=self.bias
                                        #           ),
                                        # nn.ReLU(),
                                        # nn.Sigmoid(),
                                        nn.BatchNorm2d(self.filter_size * 32),
                                        nn.MaxPool2d(kernel_size=2,
                                                     stride=2)
                                        )

        # -------------------------------- Decoding segment --------------------------------

        self.decoding_6 = nn.Sequential(nn.Upsample(scale_factor=2,
                                                    mode=self.upsample_mode),
                                        nn.Conv2d(in_channels=self.filter_size * 32,
                                                  out_channels=self.filter_size * 16,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size * 16),
                                        )

        self.decoding_5 = nn.Sequential(nn.Upsample(scale_factor=2,
                                                    mode=self.upsample_mode),
                                        nn.Conv2d(in_channels=self.filter_size * 16,
                                                  out_channels=self.filter_size * 8,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size * 8),
                                        )

        self.decoding_4 = nn.Sequential(nn.Upsample(scale_factor=2,
                                                    mode=self.upsample_mode),
                                        nn.Conv2d(in_channels=self.filter_size * 8,
                                                  out_channels=self.filter_size * 4,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size * 4),
                                        )

        self.decoding_3 = nn.Sequential(nn.Upsample(scale_factor=2,
                                                    mode=self.upsample_mode),
                                        nn.Conv2d(in_channels=self.filter_size * 4,
                                                  out_channels=self.filter_size * 2,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size * 2),
                                        )

        self.decoding_2 = nn.Sequential(nn.Upsample(scale_factor=2,
                                                    mode=self.upsample_mode),
                                        nn.Conv2d(in_channels=self.filter_size * 2,
                                                  out_channels=self.filter_size,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(self.filter_size),
                                        )

        self.decoding_1 = nn.Sequential(nn.Upsample(scale_factor=2,
                                                    mode=self.upsample_mode),
                                        nn.Conv2d(in_channels=self.filter_size,
                                                  out_channels=3,
                                                  kernel_size=self.k_size,
                                                  stride=self.st,
                                                  padding=self.padding,
                                                  bias=self.bias
                                                  ),
                                        nn.ReLU(),
                                        # nn.BatchNorm2d(3),
                                        )

    def forward(self, x):

        out = self.encoding_1(x)
        out = self.encoding_2(out)
        out = self.encoding_3(out)
        out = self.encoding_4(out)
        out = self.encoding_5(out)
        out = self.encoding_6(out)

        if self.training:
            out = self.decoding_6(out)
            out = self.decoding_5(out)
            out = self.decoding_4(out)
            out = self.decoding_3(out)
            out = self.decoding_2(out)
            out = self.decoding_1(out)
        else:
            out = out.reshape(out.size(0), -1)

        return out

    def set_temperature(self, temperature):
        self.temperature = temperature
