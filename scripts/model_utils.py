import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Resize
from PIL import Image
import os
import numpy as np

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.conv1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = self.conv_block(512, 1024)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = self.conv_block(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = self.conv_block(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = self.conv_block(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = self.conv_block(128, 64)
        self.conv10 = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        conv5 = self.conv5(pool4)
        
        up6 = self.up6(conv5)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(conv6)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(conv7)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(conv8)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        
        return conv10

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )