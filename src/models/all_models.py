import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =========================
# SE Block
# =========================
class SE_Block(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b,c,_,_ = x.size()
        y = self.pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x * y


# =========================
# Residual Block
# =========================
class Residual_Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c,out_c,3,stride,1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c,out_c,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.dropout = nn.Dropout2d(0.2)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_c!=out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c,out_c,1,stride),
                nn.BatchNorm2d(out_c)
            )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        return F.relu(out)


# =========================
# Residual + Attention
# =========================
class Residual_SE_Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c,out_c,3,stride,1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c,out_c,3,1,1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.se = SE_Block(out_c)
        self.dropout = nn.Dropout2d(0.2)

        self.shortcut = nn.Sequential()
        if stride!=1 or in_c!=out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c,out_c,1,stride),
                nn.BatchNorm2d(out_c)
            )

    def forward(self,x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out += self.shortcut(x)
        return F.relu(out)


# =========================
# Simple CNN
# =========================
class Simple_CNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(2),
        )

        self.fc_dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128*16*16, num_classes)

    def forward(self,x):
        x = self.net(x)
        x = x.view(x.size(0),-1)
        x = self.fc_dropout(x)
        return self.fc(x)


# =========================
# ResNet
# =========================
class ResNet_Custom(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.conv = nn.Conv2d(1,64,7,2,3)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3,2,1)

        self.layer1 = self._make_layer(64,64,2,1)
        self.layer2 = self._make_layer(64,128,2,2)
        self.layer3 = self._make_layer(128,256,2,2)
        self.layer4 = self._make_layer(256,512,2,2)

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512,num_classes)

    def _make_layer(self,in_c,out_c,blocks,stride):
        layers = [Residual_Block(in_c,out_c,stride)]
        for _ in range(1,blocks):
            layers.append(Residual_Block(out_c,out_c))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        return self.fc(x)
    

# =========================
# ResNet + Attention
# =========================
class ResNet_Attention(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.conv = nn.Conv2d(1,64,7,2,3)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3,2,1)

        self.layer1 = self._make_layer(64,64,2,1)
        self.layer2 = self._make_layer(64,128,2,2)
        self.layer3 = self._make_layer(128,256,2,2)

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256,num_classes)

    def _make_layer(self,in_c,out_c,blocks,stride):
        layers = [Residual_SE_Block(in_c,out_c,stride)]
        for _ in range(1,blocks):
            layers.append(Residual_SE_Block(out_c,out_c))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        return self.fc(x)


# =========================
# MobileNet
# =========================
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_c, in_c, kernel_size=3, stride=stride,
            padding=1, groups=in_c
        )

        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dropout = nn.Dropout2d(0.2)   # 🔥 added

    def forward(self,x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.dropout(x)
        return F.relu(x)


class MobileNetCustom(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            DepthwiseSeparableConv(32,64,1),

            DepthwiseSeparableConv(64,128,2),
            DepthwiseSeparableConv(128,128,1),

            DepthwiseSeparableConv(128,256,2),
            DepthwiseSeparableConv(256,256,1),

            DepthwiseSeparableConv(256,512,2),

            DepthwiseSeparableConv(512,512,1),
            DepthwiseSeparableConv(512,512,1),
            DepthwiseSeparableConv(512,512,1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

    def forward(self,x):
        x = self.model(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)
    
# =========================
# CNN + Attention
# =========================
class CNN_Attention(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        self.conv1 = nn.Conv2d(1,64,3,1,1)
        self.bn1 = nn.BatchNorm2d(64)
        self.se1 = SE_Block(64)

        self.conv2 = nn.Conv2d(64,128,3,1,1)
        self.bn2 = nn.BatchNorm2d(128)
        self.se2 = SE_Block(128)

        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(0.2)   # balanced
        self.fc_dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(128*32*32, num_classes)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = self.dropout2d(x)
        x = self.pool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.dropout2d(x)
        x = self.pool(x)

        x = x.view(x.size(0),-1)
        x = self.fc_dropout(x)
        return self.fc(x)


# =========================
# VGG16
# =========================
class VGG16_Custom(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        self.features = vgg.features

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

