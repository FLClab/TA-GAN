
import numpy
import torch

from torch import nn


class DoubleConvolver(nn.Module):
    """
    Class for the double convolution in the contracting path. The kernel size is
    set to 3x3 and a padding of 1 is enforced to avoid lost of pixels. The convolution
    is followed by a batch normalization and relu.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Contracter(nn.Module):
    """
    Class for the contraction path. Max pooling of the input tensor is 
    followed by the double convolution. 

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Contracter, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvolver(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Expander(nn.Module):
    """
    Class for the expansion path. Upsampling with a kernel size of 2 and stride 2
    is performed and followed by a double convolution following the concatenation 
    of the skipping link information from higher layers.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Expander, self).__init__()
        self.expand = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolver(in_channels=in_channels, out_channels=out_channels)

    def center_crop(self, links, target_size):
        _, _, links_height, links_width = links.size()
        diff_x = (links_height - target_size[0]) // 2
        diff_y = (links_width - target_size[1]) // 2
        return links[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        x = self.expand(x)
        crop = self.center_crop(bridge, x.size()[2 : ])
        concat = torch.cat([x, crop], 1)
        x = self.conv(concat)
        return x


class UNet(nn.Module):
    """
    Class for creating the UNet architecture. A first double convolution is performed
    on the input tensor then the contracting path is created with a given depth and
    a preset number of filters. The number of filter is doubled at every step.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of output channels (i.e. number of classes)
    :param number_filter: Number of filters in the first layer (2 ** number_filter)
    :param depth: Depth of the network
    :param size: The size of the crops that are fed to the network
    """
    def __init__(self, in_channels, out_channels, number_filter=4, depth=4, size=128):
        super(UNet, self).__init__()
        self.size = size
        self.out_channels = out_channels

        self.input_conv = DoubleConvolver(in_channels=in_channels, out_channels=2**number_filter)

        self.contracting_path = nn.ModuleList()
        for i in range(depth - 1):
            self.contracting_path.append(
                Contracter(in_channels=2**(number_filter + i), out_channels=2**(number_filter + i + 1))
            )

        self.expanding_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.expanding_path.append(
                Expander(in_channels=2**(number_filter + i + 1), out_channels=2**(number_filter + i))
            )

        self.output_conv = nn.Conv2d(in_channels=2**number_filter, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        links = [] # keeps track of the links
        x = self.input_conv(x)
        links.append(x)

        # Contracting path
        for i, contracting in enumerate(self.contracting_path):
            x = contracting(x)
            if i != len(self.contracting_path) - 1:
                links.append(x)

        # Expanding path
        for i, expanding in enumerate(self.expanding_path):
            x = expanding(x, links[- i - 1])
        x = self.output_conv(x)

        # Sigmoid layer
        x = torch.sigmoid(x)
        return x
