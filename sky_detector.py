'''
This code is to run simple version of inference model of SOTA semantic segmentation with pytorch
'''

import os
import numpy as np
import pdb
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import csv
from PIL import Image
import matplotlib.pyplot as plt
from scipy.io import loadmat
# MIT libraries
from semantic_segmentation.mit_semseg.models import ModelBuilder, SegmentationModule
from semantic_segmentation.mit_semseg.utils import colorEncode
from semantic_segmentation.mit_semseg.lib.utils import as_numpy
from semantic_segmentation.mit_semseg.lib.nn import async_copy_to

class SkyDetector:
    def __init__(self):
        # arguments for model HRNETV2:
        self.arch_encoder = 'hrnetv2'
        self.arch_decoder = 'c1'
        self.fc_dim = 720
        self.encoder_weights_path = './ade20k-hrnetv2-c1/encoder_epoch_30.pth'
        self.decoder_weights_path = './ade20k-hrnetv2-c1/decoder_epoch_30.pth'
        self.num_class = 150
        self.imgSizes = (300, 375, 450, 525, 600) # multi-scale prediction
        self.padding_constant = 32
        self.imgMaxSize = 1000
        self.gpu=0

        # read color table:
        colors = loadmat('./data/color150.mat')['colors']
        names = {}
        with open('./data/object150_info.csv') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                names[int(row[0])] = row[5].split(";")[0]

        # Build Models:
        self.net_encoder = self.build_encoder(arch_encoder=self.arch_encoder, fc_dim=self.fc_dim, weights=self.encoder_weights_path)
        self.net_decoder = self.build_decoder(arch_decoder=self.arch_decoder, fc_dim=self.fc_dim, num_class=self.num_class, weights=self.decoder_weights_path)

        # Negative likelihood loss:
        self.crit = nn.NLLLoss(ignore_index=-1)

        self.segmentation_module = SegmentationModule(self.net_encoder, self.net_decoder, self.crit)

        # turn on cuda:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(self.device)
        self.segmentation_module.to(self.device)
        self.segmentation_module.eval()
    
    def build_encoder(self, arch_encoder,fc_dim,weights):
        # Network Builders
        net_encoder = ModelBuilder.build_encoder(
            arch=arch_encoder,
            fc_dim=fc_dim,
            weights=weights)

        return net_encoder

    def build_decoder(self, arch_decoder,fc_dim,num_class,weights):
        # Network Builders
        net_decoder = ModelBuilder.build_decoder(
            arch=arch_decoder,
            fc_dim=fc_dim,
            num_class=num_class,
            weights=weights,
            use_softmax=True)

        return net_decoder

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def imresize(self, im, size, interp='bilinear'):
        if interp == 'nearest':
            resample = Image.NEAREST
        elif interp == 'bilinear':
            resample = Image.BILINEAR
        elif interp == 'bicubic':
            resample = Image.BICUBIC
        else:
            raise Exception('resample method undefined!')

        return im.resize(size, resample)

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        # mean and std
        normalize = transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        img = normalize(torch.from_numpy(img.copy()))
        return img

    def datapreprocess(self, img,imgSizes,imgMaxSize,padding_constant):
        ori_width, ori_height = img.size
        # Check if image is larger than 2048 x 2048
        # if too large than resize
        if ori_width > 2048 or ori_height > 2048:
            img = self.imresize(img, (2048, 2048), interp='bilinear')
            ori_width, ori_height = img.size

        img_resized_list = []

        for this_short_size in imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, padding_constant)
            target_height = self.round2nearest_multiple(target_height, padding_constant)

            # resize images
            img_resized = self.imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        data = dict()
        data['img_ori'] = np.array(img)
        data['img_data'] = [x.contiguous() for x in img_resized_list]
        
        segSize = (data['img_ori'].shape[0],
                data['img_ori'].shape[1])
        img_resized_list = data['img_data']

        return img_resized_list, segSize

    def scene_filter(self, img_ori,prob,scene_index,W_global,H_global):
        prob_threshold = 0.99
        img_ori = np.array(img_ori)
        if img_ori.shape[0] > 2048 or img_ori.shape[1] > 2048:
            img_ori = cv2.resize(img_ori,(2048,2048))
        mask = np.zeros((img_ori.shape[0],img_ori.shape[1]))
        mask[prob[scene_index] > prob_threshold] = 1
        mask.astype('uint8')

        # resize to global if necessary:
        if img_ori.shape[0] != H_global and img_ori.shape[1] != W_global:
            img_ori = cv2.resize(img_ori,(W_global,H_global))
            mask = cv2.resize(mask,(W_global,H_global))

        return img_ori, mask

    def predict(self, img):
        # record the global image size:
        img = Image.fromarray(img)
        img = img.convert('RGB')
        W_global, H_global = img.size

        img_resized_list, segSize = self.datapreprocess(img, self.imgSizes, self.imgMaxSize, self.padding_constant)
        
        with torch.no_grad():
            scores = torch.zeros(1, self.num_class, segSize[0], segSize[1])
            if self.gpu >= 0: # use GPU memory to handle data
                scores = async_copy_to(scores, self.gpu)

            for img in img_resized_list:
                feed_dict = {}
                feed_dict['img_data'] = img
                if self.gpu >= 0: # use GPU memory to handle data
                    feed_dict = async_copy_to(feed_dict, self.gpu)
                # forward pass
                pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(self.imgSizes)
            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

            # sky filtering:
            scene_index = 2 # sky index = 2
            prob = as_numpy(scores.squeeze(0).cpu())
            img_ori, sky_mask = self.scene_filter(img ,prob,scene_index,W_global,H_global)
        
        return sky_mask


if __name__ == "__main__":
    sky_predictor = SkyDetector()
    img_path = r'example/0.jpg'
    img = cv2.imread(img_path)
    sky_mask = sky_predictor.predict(img)
    cv2.imshow("",sky_mask)
    cv2.waitKey(0)