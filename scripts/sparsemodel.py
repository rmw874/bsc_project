import torch
import torch.nn as nn
from torch.nn.functional import relu

class SparseUNET(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.dropout = nn.Dropout(0.5)  # Dropout to help with regularization
        
        # Encoder
        self.e11 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Halved from 32 to 16
        self.bn11 = nn.BatchNorm2d(16)
        self.e12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Halved from 64 to 32
        self.bn21 = nn.BatchNorm2d(32)
        self.e22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Halved from 128 to 64
        self.bn31 = nn.BatchNorm2d(64)
        self.e32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Halved from 256 to 128
        self.bn41 = nn.BatchNorm2d(128)
        self.e42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # Halved from 512 to 256
        self.bn51 = nn.BatchNorm2d(256)
        self.e52 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(256)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Adjusted channels
        self.d11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dbn11 = nn.BatchNorm2d(128)
        self.d12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dbn12 = nn.BatchNorm2d(128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # Adjusted channels
        self.d21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dbn21 = nn.BatchNorm2d(64)
        self.d22 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dbn22 = nn.BatchNorm2d(64)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # Adjusted channels
        self.d31 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dbn31 = nn.BatchNorm2d(32)
        self.d32 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dbn32 = nn.BatchNorm2d(32)

        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # Adjusted channels
        self.d41 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dbn41 = nn.BatchNorm2d(16)
        self.d42 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dbn42 = nn.BatchNorm2d(16)

        self.outconv = nn.Conv2d(16, n_class, kernel_size=1)  # Adjusted final channels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.bn11(self.e11(x)))
        xe12 = relu(self.bn12(self.e12(xe11)))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.bn21(self.e21(xp1)))
        xe22 = relu(self.bn22(self.e22(xe21)))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.bn31(self.e31(xp2)))
        xe32 = relu(self.bn32(self.e32(xe31)))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.bn41(self.e41(xp3)))
        xe42 = relu(self.bn42(self.e42(xe41)))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.bn51(self.e51(xp4)))
        xe51 = self.dropout(xe51)  # Dropout in bottleneck
        xe52 = relu(self.bn52(self.e52(xe51)))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.dbn11(self.d11(xu11)))
        xd11 = self.dropout(xd11)  # Dropout after skip connection
        xd12 = relu(self.dbn12(self.d12(xd11)))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.dbn21(self.d21(xu22)))
        xd21 = self.dropout(xd21)  # Dropout after skip connection
        xd22 = relu(self.dbn22(self.d22(xd21)))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.dbn31(self.d31(xu33)))
        xd31 = self.dropout(xd31)  # Dropout after skip connection
        xd32 = relu(self.dbn32(self.d32(xd31)))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.dbn41(self.d41(xu44)))
        xd41 = self.dropout(xd41)  # Dropout after skip connection
        xd42 = relu(self.dbn42(self.d42(xd41)))

        out = self.outconv(xd42)
        
        return out