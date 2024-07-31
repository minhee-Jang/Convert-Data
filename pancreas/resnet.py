import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes, init_weights=True):
        super().__init__()

        self.in_channels=64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes):
    return ResNet(BottleNeck, [3,4,6,3], num_classes)

def resnet101(num_classes):
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes):
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )
            
    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

# class IdentityBlock(nn.Module):
#   def __init__(self, in_channels, out_channels , downsample=False):
#     super(IdentityBlock, self).__init__()



#     super(IdentityBlock, self).__init__()
#     self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride =1, padding=1) 
#     self.bn1 = nn.BatchNorm2d(out_channels) #batchnorm 실행 
#     self.act1 = nn.ReLU()

#     self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride =1, padding=1)
#     self.bn2 = nn.BatchNorm2d(out_channels)
#     self.act2 = nn.ReLU()

#     self.add = nn.Sequential()

#     self.downsample = downsample
#     if self.downsample:
#       self.ds = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2, padding=1)
#       self.dsbn = nn.BatchNorm2d(in_channels)
    
#   def forward(self, inputs):
#     ds= inputs
#     if self.downsample:
#       ds = self.ds(inputs)
#       ds = self.dsbn(ds)

#     x = self.conv1(ds)
#     x = self.bn1(x)
#     x = self.act1(x)
    
#     x = self.conv2(x)
#     x = self.bn2(x)

#     x += self.add(x)
#     out = self.act2(x)

#     return out
# # ResNet 모델 정의
# class ResNet18(nn.Module):
#   def __init__(self, num_classes):
#     super(ResNet18, self).__init__()

#     self.conv1 = nn.Conv2d(3, 64, kernel_size=7,stride=2, padding=1)
#     self.bn1 = nn.BatchNorm2d(64)
#     self.act1 = nn.ReLU()
#     self.maxp = nn.MaxPool2d(kernel_size=(3,3), stride=2)

#     self.conv2x1 = IdentityBlock(64, 64, downsample=False)
#     self.conv2x2 = IdentityBlock(64,64, downsample=False)

#     self.conv3x1 = IdentityBlock(64, 128, downsample=True)
#     self.conv3x2 = IdentityBlock(128, 128, downsample=False)

#     self.conv4x1 = IdentityBlock(128, 256, downsample=True)
#     self.conv4x2 = IdentityBlock(256, 256, downsample=False)

#     self.conv5x1 = IdentityBlock(256, 512, downsample=True)
#     self.conv5x2 = IdentityBlock(512, 512, downsample=False)

#     self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#     self.flat = nn.Flatten()
#     self.dropout = nn.Dropout(0.5)
#     self.fc = nn.Linear(512, num_classes)

#   def forward(self, inputs):
#     x = self.conv1(inputs)
#     x = self.bn1(x)
#     x = self.act1(x)
#     x = self.maxp(x)

#     for idblock in [self.conv2x1, self.conv2x2, self.conv3x1, self.conv3x2, \
#                     self.conv4x1, self.conv4x2, self.conv5x1, self.conv5x2]:

    
#       x = idblock(x)

#     x = self.avg_pool(x)
#     x = self.flat(x)
#     x = self.dropout(x)    #fc에서 dropout을 사용했음 
#     out = self.fc(x)

#     return out


# # ResNet-18 모델 생성
# def ResNet18(num_classes=3):
#     return ResNet(ResidualBlock, [2, 2, 2, 2], num_classes)