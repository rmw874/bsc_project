import torch
import torch.nn as nn

class StridedUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StridedUNet, self).__init__()

        # Encoder with strided convolutions for downsampling
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128, stride=2)
        self.encoder3 = self.conv_block(128, 256, stride=2)
        self.encoder4 = self.conv_block(256, 512, stride=2)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, stride=2)

        # Decoder
        self.upconv4 = self.up_conv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = self.up_conv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = self.up_conv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = self.up_conv(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, stride=1):
        """2-layer Conv2d block with optional strided convolutions for downsampling"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        """ConvTranspose2d layer for upsampling"""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Padding if needed
        x = nn.functional.pad(x, (2, 3, 0, 1))  # Padding width (2, 3), height (0, 1)

        # Encoder path (Downsampling using strided convolutions)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)  # Strided convolution downsamples
        e3 = self.encoder3(e2)  # Strided convolution downsamples
        e4 = self.encoder4(e3)  # Strided convolution downsamples

        # Bottleneck
        b = self.bottleneck(e4)  # Strided convolution downsamples

        # Decoder path (Upsampling)
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        # Final output layer
        out = self.final_conv(d1)

        # Remove padding
        out = out[:, :, :-1, :-5]  # Remove extra height and width padding

        return out
