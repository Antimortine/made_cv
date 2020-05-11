import torch
from torch import nn
from torch.nn import functional as fnn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import torchvision.models.resnet as resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock

NUM_PTS = 971


class LastLevelAvgPool(ExtraFPNBlock):
    """
    Applies a adaptive_avg_pool2d on top of the last feature map
    """
    def forward(self, x, y, names):
        # type: (List[Tensor], List[Tensor], List[str]) -> Tuple[List[Tensor], List[str]]
        names.append("pool")
        x.append(fnn.adaptive_avg_pool2d(x[-1], (1, 1)))
        return x, names


class CustomBackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list, out_channels):
        super(CustomBackboneWithFPN, self).__init__()
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=None,
        )
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


def custom_resnet_fpn_backbone(backbone_name, out_channels=1024, pretrained=True, freeze_first_layers=True, norm_layer=nn.BatchNorm2d):
    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained,
        norm_layer=norm_layer)
    if freeze_first_layers:
	    for name, parameter in backbone.named_parameters():
	        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
	            parameter.requires_grad_(False)

    return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    return CustomBackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)


class LandmarksResNext101FPN(nn.Module):
    def __init__(self, num_pts=NUM_PTS, out_fpn_channels=1024, out_conv_channels=1024, pretrained=True, freeze_first_layers=True):
        super(LandmarksResNext101FPN, self).__init__()
        self.fpn = custom_resnet_fpn_backbone('resnext101_32x8d', out_fpn_channels, pretrained, freeze_first_layers)
        
#         self.bn = nn.BatchNorm2d(out_fpn_channels)
        self.conv = nn.Conv2d(out_fpn_channels, out_conv_channels, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(out_conv_channels, num_pts * 2, bias=True)

    def forward(self, x):
        x = self.fpn(x)['0']
        
        x = self.relu(x)
#         x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        
        x = fnn.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
