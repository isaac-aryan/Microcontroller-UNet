import torch
import torch.nn as nn

# Double Convolution Block
class DoubleConv(nn.Module):

    # number of output channels of a convolution is equal to the number of filters used in the convolution

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.double_conv = nn.Sequential(
            # convolution 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # convolution 2
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
        
# Downsampling -> Performing a Double Convolution and Max Pooling it
class DownSample(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels) # double convolution operation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # max pooling operation

    def forward(self, x):
        down = self.conv(x)
        pool = self.pool(down)

        return down, pool # Returning the output of the double convolution and the max pooling
    
# Upsampling -> Transposed Convolution, increasing dimensions and concatenating with the downsampling convolution blocks, then performing double convolution
class UpSample(nn.Module):

    # in_channels -> input for the upsampling block i.e output channels from the previous downsampling/upsampling block
    def __init__(self, in_channels, out_channels):

        super().__init__()

        # First you UP Convolve, then you concatenate, 
        # in_channels for the convolution will be equal to the output channels from the last convolution block so 1024, we want 512 output channels
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2) # with every up convolution the size will double
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # The Upsampling Operation -> Up Convolution on x1 and x2 is the output from the corresponding downsampling block. We concatenate these and perform a double convolution
        x1 = self.up(x1) # up convolution
        x = torch.cat([x1, x2], 1) # concatenating
        return self.conv(x) # double convolution

             


