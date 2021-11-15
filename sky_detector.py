'''
This code is to run simple version of inference model of SOTA semantic segmentation with pytorch
'''
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image

# MIT libraries
from semantic_segmentation.mit_semseg.models import ModelBuilder, SegmentationModule
from semantic_segmentation.mit_semseg.lib.utils import as_numpy
from semantic_segmentation.mit_semseg.lib.nn import async_copy_to


class SkyDetector:
    def __init__(self):
        # arguments for model HRNETV2:
        self.arch_encoder = 'hrnetv2'
        self.arch_decoder = 'c1'
        self.fc_dim = 720
        self.encoder_weights_path = './semantic_segmentation/ade20k-hrnetv2-c1/encoder_epoch_30.pth'
        self.decoder_weights_path = './semantic_segmentation/ade20k-hrnetv2-c1/decoder_epoch_30.pth'
        self.num_class = 150
        self.imgSizes = (300, 375, 450, 525, 600) # multi-scale prediction
        self.padding_constant = 32
        self.imgMaxSize = 1000
        self.prob_threshold = 0.99

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

    def scene_filter(self, img_ori, prob, W_global, H_global):
        scene_index = 0 # sky
        img_ori = np.array(img_ori)
        if img_ori.shape[0] > 2048 or img_ori.shape[1] > 2048:
            img_ori = cv2.resize(img_ori,(2048,2048))
        mask = np.zeros((img_ori.shape[0],img_ori.shape[1]))
        mask[prob[scene_index] > self.prob_threshold] = 1
        mask.astype('uint8')

        # resize to global if necessary:
        if img_ori.shape[0] != H_global or img_ori.shape[1] != W_global:
            img_ori = cv2.resize(img_ori,(W_global,H_global))
            mask = cv2.resize(mask,(W_global,H_global))

        return img_ori, mask

    def predict(self, img_input):
        # record the global image size:
        img_input = Image.fromarray(img_input)
        W_global, H_global = img_input.size
        scene_index = 2
        img_resized_list, segSize = self.datapreprocess(img_input, self.imgSizes, self.imgMaxSize, self.padding_constant)
        
        with torch.no_grad():
            scores = torch.zeros(1, 1, segSize[0], segSize[1])

            for img in img_resized_list:
                feed_dict = {}
                feed_dict['img_data'] = img
                feed_dict = async_copy_to(feed_dict, self.device)
                # forward pass
                pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                pred_tmp = pred_tmp[:, scene_index, :, :]
                pred_tmp = pred_tmp.cpu()
                scores = scores + pred_tmp / len(self.imgSizes)

            # sky filtering:
            prob = as_numpy(scores.squeeze(0).cpu())
            img_ori, sky_mask = self.scene_filter(img_input ,prob ,W_global,H_global)
        
        return sky_mask

    def get_ground_mask(self, sky_mask):
        ground_mask = np.copy(sky_mask)
        ground_mask = ground_mask - 1
        ground_mask *= -1
        return ground_mask

if __name__ == "__main__":
    cv2.namedWindow("sky", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ground", cv2.WINDOW_NORMAL)
    sky_predictor = SkyDetector()
    img_path = r'example/0.jpg'
    img = cv2.imread(img_path)
    sky_mask = sky_predictor.predict(img)
    ground_mask = sky_predictor.get_ground_mask(sky_mask)
    cv2.imshow("sky",sky_mask * 255)
    cv2.imshow("ground",sky_mask * 255)
    cv2.waitKey(0)