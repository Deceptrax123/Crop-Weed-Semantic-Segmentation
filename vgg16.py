import torch 
from torchvision.models import vgg16,VGG16_Weights
from torch import nn 


def extractor():
    feature_extractor=vgg16(weights=VGG16_Weights.DEFAULT)
    classifier_1=nn.Sequential(*list(feature_extractor.classifier.children())[:-7])

    feature_extractor.classifier=classifier_1

    classifier_2=nn.Sequential(*list(feature_extractor.features._modules.values())[:-1])
    feature_extractor.classifier=classifier_2

    return feature_extractor
