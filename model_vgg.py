import torch.nn as nn
import torch



class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512*27, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for parm in self.modules():
            if isinstance(parm, nn.Conv1d):
                nn.init.xavier_uniform_(parm.weight)
                if parm.bias is not None:
                    nn.init.constant_(parm.bias, 0)
            elif isinstance(parm, nn.Linear):
                nn.init.xavier_uniform_(parm.weight)
                nn.init.constant_(parm.bias, 0)


def make_features(cfg: list):
    layers = []
    in_channels = 1
    for pool in cfg:
        if pool == "Max":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, pool, kernel_size=3, padding=1)
            layers += [conv1d, nn.ReLU(True)]
            in_channels = pool
    return nn.Sequential(*layers)


cfgs = {
    'vgg16': [64, 64, 'Max', 128, 128, 'Max', 256, 256, 256, 'Max', 512, 512, 512, 'Max', 512, 512, 512, 'Max'],
}


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs !".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model