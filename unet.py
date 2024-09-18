import torch
import torch.nn as nn

from parts import DoubleConv, DownSample, UpSample

# Constructing The U-Net
class UNet(nn.Module):

    def __init__(self, in_channels, num_classes):

        super().__init__()

        # downsampling block 1
        self.down1 = DownSample(in_channels, 64) # For RGB: 3 channel image applying 64 filters so 64 output channels
        # downsampling block 2
        self.down2 = DownSample(64, 128) # 64 -> 128
        # downsampling block 3
        self.down3 = DownSample(128, 256) # 128 -> 256
        # downsampling block 4
        self.down4 = DownSample(256, 512) #256 -> 512 

        # The flat / bottleneck part of the UNET is just a double convolution that takes the output number of channels of the last downsampling block
        self.flat = DoubleConv(512, 1024) # 512 -> 1024

        # upsampling block 1
        self.up1 = UpSample(1024, 512)
        # upsampling block 2
        self.up2 = UpSample(512, 256)
        # upsampling block 3
        self.up3 = UpSample(256, 128)
        # upsampling block 4
        self.up4 = UpSample(128, 64)

        # OUTPUT LAYER OF THE NEURAL NETWORK has number of input channels equal to the output channels of the final upsampling block (64) and 
        # output channels equal to the number of classes we have to segment the image into:
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1), # FINAL OUTPUT OF NN
            nn.Sigmoid() # <--- NEWLY ADDED
        )

    def forward(self, x):
        down_1, p1 = self.down1(x)
        down_2, p2 = self.down2(p1)
        down_3, p3 = self.down3(p2)
        down_4, p4 = self.down4(p3)

        b = self.flat(p4) # output of flat block will be input for the first upsampling layer

        # concatenate with corresponding conv block output in downsampling layer
        u1 = self.up1(b, down_4) 
        u2 = self.up2(u1, down_3)
        u3 = self.up3(u2, down_2)
        u4 = self.up4(u3, down_1)

        out = self.out(u4) # final output

        return out
        
