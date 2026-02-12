from torchvision.models import resnet18
from torchvision.models import resnet50
import torch


def return_resnet18(cfg):
    return resnet18()


def return_resnet50(cfg):
    model = resnet50()
    if cfg['experiment']['analysis_type'] == 'representation_output':
        model.fc = torch.nn.Identity()
        if 'cifar' in cfg['dataset']['name'] or 'mnist' in cfg['dataset']['name']:
                model.conv1 = torch.nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                model.maxpool = torch.nn.Identity()
    return model