# import torch
import os
# from torch.utils import data

import pandas as pd
# import numpy as np
from torch import nn  # , optim
# from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


class FeatureExtractorDataSet(Dataset):

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

            return left_im, right_im, label

    def __len__(self):
        """
        Returns number of data entries.
        :return: Data size.
        """
        return self.data_len


class FeatureClassifierDataSet(Dataset):

    def __init__(self, img_folder, pair_file=None, feature_csv="15features.csv", transform=transforms.ToTensor()):

        self.img_transforms = transform

        self.img_folder = img_folder

        if img_folder is None:
            raise ValueError("Provide image folder name!")

        self.features_df = pd.read_csv(filepath_or_buffer=feature_csv,
                                       header=0,
                                       index_col=0
                                       ) - 1

        if pair_file is not None:

            self.training_mode = False

            # self.__clean_list_2__(pair_file, img_folder)

            self.main_data = pd.read_csv(filepath_or_buffer=pair_file,
                                         header=0,
                                         index_col=0
                                         )
            self.data_len = self.main_data.shape[0]

        else:

            self.training_mode = True
            # self.main_data = os.listdir(img_folder)
            self.main_data = self.__clean_list__(os.listdir(img_folder))
            self.data_len = len(self.main_data)

    def __clean_list__(self, input_list):

        remove_list = []

        for ele in input_list:
            try:
                _ = self.features_df.loc[ele]
            except KeyError:
                remove_list.append(ele)

        return [e for e in input_list if e not in remove_list]

    def __clean_list_2__(self, filename, folder):

        df = pd.read_csv(filepath_or_buffer=filename, header=0, index_col=0)
        del_list = []

        for i, row in df.iterrows():
            # print(row.values)
            try:
                f = open(folder+row.iloc[0])
                f.close()
            except FileNotFoundError:
                del_list.append(i)
                print("Deleting row: ", row.values.tolist())
                continue
            try:
                f = open(folder+row.iloc[1])
                f.close()
            except FileNotFoundError:
                del_list.append(i)
                print("Deleting row: ", row.values.tolist())
                continue

        # print(del_list)

        if len(del_list) != 0:
            df.drop(labels=del_list, inplace=True)
            df.to_csv(path_or_buf=filename)

    def __getitem__(self, index):

        if self.training_mode:

            image = Image.open(self.img_folder + self.main_data[index])
            val = 0

            try:
                val = self.features_df.loc[self.main_data[index]].values
            except KeyError:
                print("getitem error:", index, self.main_data[index])

            return self.img_transforms(image), val

        else:

            l_name = self.main_data.iloc[index, 0]
            r_name = self.main_data.iloc[index, 1]

            image1 = Image.open(self.img_folder + l_name)
            image2 = Image.open(self.img_folder + r_name)
            label = self.main_data.iloc[index, 2]

            left_im = self.img_transforms(image1)
            right_im = self.img_transforms(image2)

            # left_feat = self.features_df.loc[l_name].values
            # right_feat = self.features_df.loc[r_name].values

            # return left_im, right_im, label, left_feat, right_feat, l_name, r_name
            return left_im, right_im, label, l_name, r_name

    def __len__(self):
        """
        Returns number of data entries.
        :return: Data size.
        """
        return self.data_len


class AutoEncoder(nn.Module):

    def __init__(self, bias=False, kernel_size=3):

        super(AutoEncoder, self).__init__()

        self.k_size = kernel_size
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


class FeatureClassifier(nn.Module):

    def __init__(self, num_classes, input_feats=512, temprature=1.0, bias=False):
        super(FeatureClassifier, self).__init__()

        self.temp = temprature

        self.hidden = 128

        self.layer_1 = nn.Sequential(nn.Linear(in_features=input_feats,
                                               out_features=self.hidden,
                                               bias=bias
                                               ),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.4)
                                     )

        self.layer_2 = nn.Sequential(nn.Linear(in_features=self.hidden,
                                               out_features=num_classes,
                                               bias=bias
                                               ),
                                     nn.Softmax(dim=1)
                                     )

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out / self.temp)

        return out


def plot_roc(y_true, y_score, tag, tstmp):
    r"""
    Code referred from official scikit-learns examples:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    <p>
    :param y_true:
    :param y_score:
    :param tag:
    :param tstmp:
    :return:
    """

    sets = {"SN": "Seen", "UN": "Unseen", "SH": "Shuffled"}

    fpr, tpr, thresh = roc_curve(y_true=y_true, y_score=y_score)
    area = auc(fpr, tpr)

    # print("\nThresholds: ", thresh)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label="ROC curve (area = {0:.2f})".format(area))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic for {} data".format(sets[tag]))
    plt.legend(loc="lower right")
    plt.savefig(fname="results/ROC_{0}_{1}.jpg".format(tag, tstmp))
    # plt.show()
