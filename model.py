import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchio as tio
import torch.nn.functional as F

class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet3D_Attention(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(UNet3D_Attention, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_rate),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_rate)
            )

        self.pool = nn.MaxPool3d(2)

        self.enc1 = CBR(1, 32)
        self.enc2 = CBR(32, 64)
        self.enc3 = CBR(64, 128)
        self.enc4 = CBR(128, 256)

        self.bottleneck = CBR(256, 512)

        # Attention gates
        self.att4 = AttentionBlock3D(F_g=512, F_l=256, F_int=128)
        self.att3 = AttentionBlock3D(F_g=256, F_l=128, F_int=64)
        self.att2 = AttentionBlock3D(F_g=128, F_l=64, F_int=32)
        self.att1 = AttentionBlock3D(F_g=64, F_l=32, F_int=16)

        # Decoder blocks
        self.dec4 = CBR(512 + 256, 256)
        self.dec3 = CBR(256 + 128, 128)
        self.dec2 = CBR(128 + 64, 64)
        self.dec1 = CBR(64 + 32, 32)

        self.final = nn.Conv3d(32, 1, kernel_size=1)

    def center_crop(self, enc_feat, target_size):
        _, _, d, h, w = enc_feat.size()
        td, th, tw = target_size
        d1 = (d - td) // 2
        h1 = (h - th) // 2
        w1 = (w - tw) // 2
        return enc_feat[:, :, d1:d1+td, h1:h1+th, w1:w1+tw]

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder + Attention
        up4 = F.interpolate(bottleneck, size=enc4.shape[2:], mode='trilinear', align_corners=True)
        att4 = self.att4(g=up4, x=self.center_crop(enc4, up4.shape[2:]))
        dec4 = self.dec4(torch.cat([up4, att4], dim=1))

        up3 = F.interpolate(dec4, size=enc3.shape[2:], mode='trilinear', align_corners=True)
        att3 = self.att3(g=up3, x=self.center_crop(enc3, up3.shape[2:]))
        dec3 = self.dec3(torch.cat([up3, att3], dim=1))

        up2 = F.interpolate(dec3, size=enc2.shape[2:], mode='trilinear', align_corners=True)
        att2 = self.att2(g=up2, x=self.center_crop(enc2, up2.shape[2:]))
        dec2 = self.dec2(torch.cat([up2, att2], dim=1))

        up1 = F.interpolate(dec2, size=enc1.shape[2:], mode='trilinear', align_corners=True)
        att1 = self.att1(g=up1, x=self.center_crop(enc1, up1.shape[2:]))
        dec1 = self.dec1(torch.cat([up1, att1], dim=1))

        output = self.final(dec1)
        return torch.sigmoid(output)
