import torch.nn as nn
import torch.nn.functional as F
import torch
def conv5x5(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 5 x 5 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding, dilation=dilation, bias=False)
def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 3 x 3 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=False)

def conv1x1(in_planes, out_planes, stride=1, dilation=1, padding=1):
    " 1 x 1 conv"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, dilation=dilation, bias=False)
class LargeFOVModified(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=5):
        super(LargeFOVModified, self).__init__()
        self.embed_dim = 512
        self.dilation = dilation

        self.conv3x3_1 = conv3x3(in_planes=in_planes, out_planes=self.embed_dim, padding=self.dilation, dilation=self.dilation)
        self.relu3x3_1 = nn.ReLU(inplace=True)

        self.conv3x3_2 = conv3x3(in_planes=self.embed_dim, out_planes=self.embed_dim, padding=self.dilation, dilation=self.dilation)
        self.relu3x3_2 = nn.ReLU(inplace=True)

        self.conv5x5 = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.embed_dim, kernel_size=5, padding=2, dilation=self.dilation)
        self.relu5x5 = nn.ReLU(inplace=True)

        self.conv1x1 = conv1x1(in_planes=self.embed_dim * 3, out_planes=out_planes, padding=0)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_3x3_1 = self.conv3x3_1(x)
        x_3x3_1 = self.relu3x3_1(x_3x3_1)

        x_3x3_2 = self.conv3x3_2(x_3x3_1)
        x_3x3_2 = self.relu3x3_2(x_3x3_2)

        x_5x5 = self.conv5x5(x_3x3_2)
        x_5x5 = self.relu5x5(x_5x5)

        # Resize x_5x5 to have the same spatial dimensions as x_3x3_2
        x_5x5_resized = F.interpolate(x_5x5, size=x_3x3_2.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate features from different scales
        x_concat = torch.cat([x_3x3_1, x_3x3_2, x_5x5_resized], dim=1)

        # 1x1 convolution for dimensionality reduction
        out = self.conv1x1(x_concat)

        return out

class MultiScaleDecoder(nn.Module):
    def __init__(self, in_planes, out_planes, aspp_dilations=[6, 12, 18, 24], largefov_dilation=5):
        super(MultiScaleDecoder, self).__init__()
        self.embed_dim = 512

        # ASPP layers
        self.aspp = ASPP(in_planes=in_planes, out_planes=self.embed_dim, atrous_rates=aspp_dilations)

        # LargeFOV layers
        self.largefov = LargeFOV(in_planes=self.embed_dim, out_planes=self.embed_dim, dilation=largefov_dilation)

        # 1x1 convolution to reduce the number of channels
        self.reduce_channels = nn.Conv2d(in_channels=self.embed_dim * 2, out_channels=self.embed_dim, kernel_size=1)

        # Final convolution layer
        self.conv_out = nn.Conv2d(in_channels=self.embed_dim, out_channels=out_planes, kernel_size=1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # ASPP path
        aspp_out = self.aspp(x)

        # LargeFOV path
        largefov_out = self.largefov(aspp_out)

        # Concatenate features from ASPP and LargeFOV
        concat_features = torch.cat([aspp_out, largefov_out], dim=1)

        # Reduce the number of channels
        reduced_features = self.reduce_channels(concat_features)

        # Final convolution layer
        out = self.conv_out(reduced_features)

        return out


class LargeFOV(nn.Module):
    def __init__(self, in_planes, out_planes, dilation=5):
        super(LargeFOV, self).__init__()
        self.embed_dim = 512
        self.dilation = dilation
        # self.conv5 = conv5x5(in_planes=in_planes, out_planes=self.embed_dim, padding=self.dilation,
        #                      dilation=self.dilation)
        # self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = conv3x3(in_planes=in_planes, out_planes=self.embed_dim, padding=self.dilation, dilation=self.dilation)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = conv3x3(in_planes=self.embed_dim, out_planes=self.embed_dim, padding=self.dilation, dilation=self.dilation)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = conv1x1(in_planes=self.embed_dim, out_planes=out_planes, padding=0)

    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)
        return None

    def forward(self, x):
        # x = self.conv5(x)
        # x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)

        out = self.conv8(x)

        return out

class LargeFOVWithDilation(nn.Module):
    def __init__(self, in_planes, out_planes, dilations=[3, 5]):
        super(LargeFOVWithDilation, self).__init__()
        self.embed_dim = 512

        # Original LargeFOV layers
        self.conv6 = conv3x3(in_planes=in_planes, out_planes=self.embed_dim, padding=3, dilation=3)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = conv3x3(in_planes=self.embed_dim, out_planes=self.embed_dim, padding=3, dilation=3)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = conv1x1(in_planes=self.embed_dim, out_planes=out_planes, padding=0)

        # Additional dilated convolutions
        self.dilated_conv1 = conv3x3(in_planes=self.embed_dim, out_planes=self.embed_dim, padding=dilations[0], dilation=dilations[0])
        self.dilated_conv2 = conv3x3(in_planes=self.embed_dim, out_planes=self.embed_dim, padding=dilations[1], dilation=dilations[1])

    def forward(self, x):
        # Original LargeFOV path
        x_largefov = self.conv6(x)
        x_largefov = self.relu6(x_largefov)
        x_largefov = self.conv7(x_largefov)
        x_largefov = self.relu7(x_largefov)
        out_largefov = self.conv8(x_largefov)

        # Additional dilated convolutions
        x_dilated1 = self.dilated_conv1(x_largefov)
        x_dilated2 = self.dilated_conv2(x_dilated1)

        # Concatenate features from different paths
        out = torch.cat([out_largefov, x_dilated1, x_dilated2], dim=1)

        return out

class ASPP(nn.Module):
    def __init__(self, in_planes, out_planes, atrous_rates=[6, 12, 18, 24]):
        super(ASPP, self).__init__()
        for i, rate in enumerate(atrous_rates):
            self.add_module("c%d"%(i), nn.Conv2d(in_planes, out_planes, 3, 1, padding=rate, dilation=rate, bias=True))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
        return None
    def forward(self, x):
        return sum([stage(x) for stage in self.children()])