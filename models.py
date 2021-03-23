import torch
import torch.nn as nn
import torch.nn.functional as F

class DilateBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, dilate=1, pad=1, normalize=None):
        super(DilateBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, dilation=dilate, padding=pad),
            nn.Conv2d(out_channel, out_channel, 1, 1, bias=False),
        ]
        if normalize=='instance':
            layers.append(nn.InstanceNorm2d(out_channel))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)      
    def forward(self, x):
        x = self.model(x)
        return x   
    
    
class UpBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, up_factor):
        super(UpBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channel, out_channel*(up_factor**2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(up_factor),
        ]
        self.model = nn.Sequential(*layers)       
    def forward(self, x):
        x = self.model(x)
        return x  
    
class ESPBlock(torch.nn.Module):
    def __init__(self, in_channel, normalize=None):
        super(ESPBlock, self).__init__() 
        self.dilate_1 = DilateBlock(in_channel, in_channel, dilate=1, pad=1, normalize=normalize)
        self.dilate_2 = DilateBlock(in_channel, in_channel, dilate=4, pad=4, normalize=normalize)
        self.dilate_3 = DilateBlock(in_channel, in_channel, dilate=8, pad=8, normalize=normalize)
        self.dilate_4 = DilateBlock(in_channel, in_channel, dilate=12, pad=12, normalize=normalize)
    def forward(self, x):
        x1 = self.dilate_1(x)
        x2 = self.dilate_2(x)
        x3 = self.dilate_3(x)
        x4 = self.dilate_4(x)
        out = torch.cat((x+x1, x1+x2, x1+x2+x3, x1+x2+x3+x4), 1)
        return out
    
class Net(nn.Module):
    def __init__(self, in_channel, upscale_factor=2):
        super(Net, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, in_channel * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.pixel_shuffle(self.conv3(x)) + F.interpolate(input, scale_factor=self.upscale_factor, mode='bilinear', align_corners=None)
        return x
    
class Generator(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, base_channel=16, up_factor=2, normalize=None):
        super(Generator, self).__init__()
        self.up_factor = up_factor
        self.conv0 = nn.Conv2d(in_channel, base_channel, 5, 1, 2)
        self.esp_1 = ESPBlock(base_channel, normalize)
        self.esp_2 = ESPBlock(base_channel*4, normalize)
        self.up = UpBlock(base_channel*16, out_channel, up_factor)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.esp_1(x)
        x = self.esp_2(x)
        x = self.up(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, in_channel=1, norm=None):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, norm):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if norm == 'instance':
                layers.append(nn.InstanceNorm2d(out_filters))
            if norm == 'batch':
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channel * 2, 32, norm),
            *discriminator_block(32, 64, norm),
            *discriminator_block(64, 128, norm),
            *discriminator_block(128, 256, norm),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(256, 1, 1, 1, bias=False)
        )

    def forward(self, img_A, img_B, upscale_factor=2):
        # Concatenate image and condition image by channels to produce input
        img_B = F.interpolate(img_B, scale_factor=upscale_factor, mode='bilinear', align_corners=None)
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)