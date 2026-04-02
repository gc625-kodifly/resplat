import torch.nn as nn

from torchvision.models import resnet18, resnet34, resnet50


class ResNetFeatureWarpper(nn.Module):
    def __init__(self, shallow_resnet_feature=False,
                 resnet_layers=18,
                 ):
        super(ResNetFeatureWarpper, self).__init__()

        self.shallow_resnet_feature = shallow_resnet_feature

        if resnet_layers == 18:
            resnet = resnet18(pretrained=True)
        elif resnet_layers == 34:
            resnet = resnet34(pretrained=True)
        elif resnet_layers == 50:
            resnet = resnet50(pretrained=True)
        else:
            raise NotImplementedError

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        if not shallow_resnet_feature:
            self.layer2 = resnet.layer2

    def forward(self, x):
        out = []
        x = self.conv1(x)
        out.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        out.append(x)

        if not self.shallow_resnet_feature:
            x = self.layer2(x)
            out.append(x)

        return out

