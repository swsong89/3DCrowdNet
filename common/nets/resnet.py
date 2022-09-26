import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls

class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type):
	
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
		       34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
		       50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
		       101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
		       152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # 3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 6
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 3

        for m in self.modules():  # 这里的初始化是定义网络之后就进行的，没有使用预训练好的参数, 应该是递归调用，第一个m是ResNetBackbone,就是本身，第二个m就是conv1,第三个m是bn1
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):  #  Bottleneck 64 3
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # block.expansion=4,bottleneck 4倍维度放大
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )  # self.inplanes=256 上一层的输入，planes=128,这一层开始卷积维度, 128经过 1x1conv, 3x3conv. 1x1conv, 变成512维, block.expansion=4

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 256, 128, 1, downsample, 在第一个bottleneck后面加上downsample
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))  # bottleneck是通用的，只是层不同维度不同，都是1x1conv降低维度信息,3x3conv信息提取，1x1升高维度信息到输入的4倍

        return nn.Sequential(*layers)

    def forward(self, x, skip_early=False):  # x [1,3,256,256]
        if not skip_early:
            x = self.conv1(x)  # x[1,64,128,128]
            x = self.bn1(x)  # x[1,64,128,128]
            x = self.relu(x)  # x[1,64,128,128]
            x = self.maxpool(x)  # x[1,64,64,64]

            return x

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4

    def init_weights(self):  # 使用训练好的resnet50参数
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])  # 'https://download.pytorch.org/models/resnet50-0676ba61.pth'
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)  # [1000, 2048] 最后conv5_x结果是2048,7,7,一般就是先接一个averagepool变成2048,1,1,再接一个全连接层然后进行分类得到[1000]
        org_resnet.pop('fc.bias', None)  # [1000]

        self.load_state_dict(org_resnet)  # 自定义的conv1,bn1,layer1与resnet50对应，所以可以通过去除fc.weight, fc.bias与之对应
        print("Initialize resnet from model zoo")

    """
    conv1.weight 64,3,7,7
    bn1.running_mean 64
    bn1.running_var 64
    bn1.weight 64
    bn1.bias 64
    layer1.0.conv1.weight 64,64,1,1
    ....
    """
