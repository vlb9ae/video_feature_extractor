import sys
import torch as th
import torchvision.models as models
from video_feature_extractor.videocnn.models import resnext
from torch import nn
from argparse import Namespace

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return th.mean(x, dim=[-2, -1])


def get_model(args):
    assert args.type in ['2d', '3d']
    if args.type == '2d':
        print('Loading 2D-ResNet-152 ...')
        model = models.resnet152(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2], GlobalAvgPool())
        model = model.cuda()
    else:
        print('Loading 3D-ResneXt-101 ...')
        model = resnext.resnet101(
            num_classes=400,
            shortcut_type='B',
            cardinality=32,
            sample_size=112,
            sample_duration=16,
            last_fc=False)
        model = model.cuda()
        model_data = th.load(args.resnext101_model_path)
        model.load_state_dict(model_data)

    model.eval()
    print('loaded')
    return model

def load_extractor2d(k):
    m = get_model(Namespace(type='2d'))
    m2 = nn.Sequential(*list(m.children())[:-1]) # removes final pooling layer   

    # TODO: Set requires_grad=False for all but the top k layers
    print('# of 2d blocks', len(list(m.children())))
    print('# of 2d modules', len(list(m.modules())))

    total_blocks = len(list(m.children()))
    total_modules = len(list(m.modules()))

    blocks_to_ignore = list(range(total_blocks-k, total_blocks))
    modules_to_ignore = list(range(total_modules-k, total_modules))

    print('ignoring 2d blocks', blocks_to_ignore)
    print('ignoring 2d modules', modules_to_ignore)

    for (index, child) in enumerate(m.children()):
        if index in blocks_to_ignore:
            print('leaving block', index, 'alone')
        else:
            for p in child.parameters():
                p.requires_grad = False
    return m
