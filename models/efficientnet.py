import torch
import torch.nn as nn
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights, \
                            efficientnet_v2_l, EfficientNet_V2_L_Weights, \
                            efficientnet_v2_s, EfficientNet_V2_S_Weights


class EfficientNet(nn.Module):
    def __init__(self, num_classes, network=efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT)):
        super().__init__()
        self.net = network
        for param in self.net.parameters():
            param.requires_grad = False
        
        # Modify the final classifier
        in_features = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_features, num_classes)


    def forward(self, x):
        x = self.net(x)
        return x


def tf_efficientnet_b7(num_classes) -> EfficientNet:
    weights = EfficientNet_B7_Weights.DEFAULT
    net = EfficientNet(num_classes, efficientnet_b7(weights=weights)) 
    return net


def tf_efficientnet_v2_s(num_classes) -> EfficientNet:
    weights = EfficientNet_V2_S_Weights.DEFAULT
    net = EfficientNet(num_classes, efficientnet_v2_s(weights=weights)) 
    return net


def tf_efficientnet_v2_l(num_classes) -> EfficientNet:
    weights = EfficientNet_V2_L_Weights.DEFAULT
    net = EfficientNet(num_classes, efficientnet_v2_l(weights=weights)) 
    return net