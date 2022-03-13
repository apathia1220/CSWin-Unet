import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
from .transformer import Transformer
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange
import time


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    output:B*(H//H_sp)*(W//W_sp) H_sp*W_sp C
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    output:B H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)  # 尺寸缩小一倍
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, dim * 2, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
         x:B, L, C
        """
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = self.expand(x)
        C = C * 2
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPathExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim * 16, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = self.expand(x)
        C = C * 16
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x.view(B, -1, self.output_dim)
        x = self.norm(x)
        return x


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        """
        将图片切分出可以计算attention的部分
        """
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)  # x:B*(H//H_sp)*(W//W_sp) H_sp*W_sp C
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  # B', C, H', W'

        lepe = func(x)  # B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        # Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class CNN_Downsample(nn.Module):
    """downsample cnn feature
    """

    def __init__(self, in_ch, embed_dim=64):
        super().__init__()
        filters = [embed_dim, embed_dim * 2, embed_dim * 4]

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])

        self.Maxpool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        e1 = self.Conv1(x)  # H W C

        e2 = self.Maxpool1(e1)  # H/4 W/4 C
        e2_1 = self.Conv2(e2)  # H/4 W/4  2C

        e3 = self.Maxpool2(e2_1)  # H/8 W/8 2C
        e3_1 = self.Conv3(e3)  # H/8 W/8 4C

        e4 = self.Maxpool3(e3_1)  # H/16 W/16 4C

        return e2, e3, e4


class CMT(nn.Module):
    def __init__(self, in_channel, img_size, split_size):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)
        # self.tran_block = Transformer(in_channel, 1, 8, in_channel // 8, in_channel * 2)
        self.tran_block = CSWinBlock(
            dim=in_channel * 2, num_heads=4, reso=img_size, mlp_ratio=4.,
            qkv_bias=True, qk_scale=None, split_size=split_size,
            drop=0., attn_drop=0.,
            drop_path=0, norm_layer=nn.LayerNorm)
        self.patch_expand = PatchExpand(in_channel * 2)
        self.down_dim = nn.Linear(in_channel * 2, in_channel, bias=False)
        self.norm = nn.LayerNorm(in_channel)

    def forward(self, cnn_input, trans_input):
        """
        cnn_input:B, C, H, W
        trans_input: B, L, C
        """
        B, L, C = trans_input.shape
        cnn_x = cnn_input.view(B, C, -1)  # B, C, L
        cnn_x = cnn_x.permute(0, 2, 1)  # B, L, C
        trans_new = torch.cat((trans_input, cnn_x), dim=2)  # B, L, 2C
        result = self.down_dim(trans_new)
        result = self.norm(result)

        return result


class CSWinTransformerSys(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2, 2, 6, 2],
                 depths_decoder=[2, 2, 6, 2],
                 split_size=[3, 5, 7],
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads
        self.s_size = [1, 2, 7]

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )

        # build encoder
        curr_dim = embed_dim  # C
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2  # 2C
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])

        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2  # 4C
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2  # 8C

        # build bottlebeck
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])
        self.norm = norm_layer(curr_dim)

        # build decoder
        self.contact_back_dim = nn.ModuleList()
        self.expand3 = PatchExpand(curr_dim)
        curr_dim = curr_dim // 2  # 4C
        self.contact_linear = nn.Linear(curr_dim * 2, curr_dim)
        self.contact_back_dim.append(self.contact_linear)
        temp_stage3_up = []
        temp_stage3_up.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3_up = nn.ModuleList(temp_stage3_up)

        self.expand2 = PatchExpand(curr_dim)
        curr_dim = curr_dim // 2  # 2C
        self.contact_linear = nn.Linear(curr_dim * 2, curr_dim)
        self.contact_back_dim.append(self.contact_linear)
        self.stage2_up = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])

        self.expand1 = PatchExpand(curr_dim)
        curr_dim = curr_dim // 2  # C
        self.contact_linear = nn.Linear(curr_dim * 2, curr_dim)
        self.contact_back_dim.append(self.contact_linear)
        self.stage1_up = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])

        self.norm_up = norm_layer(curr_dim)

        #################cnn_block################
        self.cnn_unet_downsample = CNN_Downsample(3)

        #################CMT##################
        self.cmt_1 = CMT(self.embed_dim, img_size // 4, self.s_size[0])
        self.cmt_2 = CMT(self.embed_dim * 2, img_size // 8, self.s_size[1])
        self.cmt_3 = CMT(self.embed_dim * 4, img_size // 16, self.s_size[2])

        ##############deep supervision###################
        self.upscore5 = nn.Upsample(scale_factor=32, mode='bilinear')###
        self.upscore4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.outconv5 = nn.Conv2d(512, self.num_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(256, self.num_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(128, self.num_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(64, self.num_classes, 3, padding=1)

        self.up = FinalPathExpand_X4(curr_dim)

        # Classifier head
        self.output = nn.Conv2d(in_channels=curr_dim, out_channels=self.num_classes, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        B = x.shape[0]
        x_downsample = []
        x = self.stage1_conv_embed(x)
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x_downsample.append(x)
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3],
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            x_downsample.append(x)
        x = self.norm(x)
        return x, x_downsample

    def forward_up_feature(self, x, x_downsample):
        idx = 0
        x_upsample = []
        x_upsample.append(x)
        for pre, blocks in zip([self.expand3, self.expand2, self.expand1],
                               [self.stage3_up, self.stage2_up, self.stage1_up]):
            # print(x.shape)
            x = pre(x)
            x = torch.cat([x, x_downsample[2-idx]], -1)
            x = self.contact_back_dim[idx](x)
            idx += 1
            # print('d:',x.shape)
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            x_upsample.append(x)
        x = self.norm_up(x)
        side = []
        for item in x_upsample:
            # print(item.shape)
            B, L, C = item.shape
            H = W = int(np.sqrt(L))
            side.append(item.view(B, H, W, C).permute(0, 3, 1, 2))

        # upsample = self.x_upsample
        # return x, side[0], side[1], side[2], side[3]
        return x, side[0]

    def up_x4(self, x):
        B, L, C = x.shape
        H = W = int(np.sqrt(L))
        x = self.up(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)
        return x


    def forward(self, x):
        inputs = x
        x, x_downsample = self.forward_features(x)
        x_e1, x_e2, x_e3 = self.cnn_unet_downsample(inputs)
        # print(x_e1.shape, x_e2.shape, x_e3.shape)
        # print(x_downsample[0].shape, x_downsample[1].shape, x_downsample[2].shape)
        downsample_cmt = []
        downsample_cmt.append(self.cmt_1(x_e1, x_downsample[0]))
        downsample_cmt.append(self.cmt_2(x_e2, x_downsample[1]))
        downsample_cmt.append(self.cmt_3(x_e3, x_downsample[2]))
        # x = self.forward_up_feature(x, downsample_cmt)
        x, side = self.forward_up_feature(x, downsample_cmt)
        # print(side.shape)
        # x, d4, d3, d2, d1 = self.forward_up_feature(x, downsample_cmt)
        side_4 = self.upscore5(side)
        side_4 = self.outconv5(side_4)
        # side_3 = self.upscore4(d3)
        # side_3 = self.outconv4(side_3)
        # side_2 = self.upscore3(d2)
        # side_2 = self.outconv3(side_2)
        # side_1 = self.upscore2(d1)
        # side_1 = self.outconv2(side_1)
        
        x = self.up_x4(x)
        # print(x.shape)
        x = self.output(x)
        # return x, side_4, side_3, side_2, side_1
        return x, side_4