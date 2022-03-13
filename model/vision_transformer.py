# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
# from .cswin_transformer_unet_skip_expand_decoder_sys import CSWinTransformerSys
# from .cswin_transformer import CSWinTransformerSys
# from .Res_Unet import Res_Unet
# from .cswin_transformer_unet import CSWinTransformerSys 
# from .cswin_transformer_unet_cmt import CSWinTransformerSys
from .cswin_transformer_cmt import CSWinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


class CSwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(CSwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.cswin_unet = CSWinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.CSWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.CSWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.CSWIN.EMBED_DIM,
                                            depth=config.MODEL.CSWIN.DEPTHS,
                                            split_size=config.MODEL.CSWIN.SPLIT_SIZE,
                                            num_heads=config.MODEL.CSWIN.NUM_HEADS,
                                            mlp_ratio=config.MODEL.CSWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.CSWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.CSWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE)
        # self.Partition = Res_Unet(in_ch=self.num_classes, out_ch=self.num_classes)

    def forward(self, x):
        # print('model.shape:', x.shape)
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # print('model.shape_1:', x.shape)
        # result, d4, d3, d2, d1 = self.cswin_unet(x)
        result, side = self.cswin_unet(x)
        # print('result:', result.shape, 'd4:', d4.shape, 'd3:', d3.shape, 'd2', d2.shape, 'd1:', d1.shape)
        # logits = self.Partition(result)
        # 224*224 7*7 14*14 28*28 56*56
        # return result, d4, d3, d2, d1
        return result, side

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "state_dict_ema" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.cswin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['state_dict_ema']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.cswin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "stage1." in k:
                    current_k = "stage1_up." + k[7:]
                    current_k_1 = "cmt_1.tran_block." + k[7:]
                    full_dict.update({current_k: v})
                    full_dict.update({current_k_1: v})
                if "stage2." in k:
                    current_k = "stage2_up." + k[7:]
                    current_k_1 = "cmt_2.tran_block." + k[7:]
                    full_dict.update({current_k: v})
                    full_dict.update({current_k_1: v})
                if "stage3." in k:
                    current_k = "stage3_up." + k[7:]
                    current_k_1 = "cmt_3.tran_block." + k[7:]
                    full_dict.update({current_k: v})
                    full_dict.update({current_k_1: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.cswin_unet.load_state_dict(full_dict, strict=False)
            # for k in full_dict:
            #     print(k)
        else:
            print("none pretrain")