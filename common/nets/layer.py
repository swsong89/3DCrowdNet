import torch
import torch.nn as nn
from torch.nn import functional as F

from config import cfg

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):  # feat_dims=[64+joint_num,64], [in_channels, out_channels]
    layers = []
    for i in range(len(feat_dims)-1): # 如果是[A,B,C]则表明有2层，第一层卷积A,B 第二次B,C
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation  Pose2Feat使用bnrelu
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))  # inplace原地操作，即是否该边输入的数据，True即输入A改变从B，顺便就赋值了，不需要申请新的空间

    return nn.Sequential(*layers) # *号加在了是实参上，代表的是将输入迭代器拆成一个个元素 (*[nn,conv2d, nn.BatchNorm2d, nn.relu]) -> (nn,conv2d, nn.BatchNorm2d, nn.relu)

def make_conv1d_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv1d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_deconv_layers(feat_dims, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class GraphConvBlock(nn.Module):
    def __init__(self, adj, dim_in, dim_out):  # adj=15, dim_in=2052, dim_out=128
        super(GraphConvBlock, self).__init__()
        self.adj = adj  # [15,15]
        self.vertex_num = adj.shape[0]  # [15]
        self.fcbn_list = nn.ModuleList([nn.Sequential(*[nn.Linear(dim_in, dim_out), nn.BatchNorm1d(dim_out)]) for _ in range(self.vertex_num)])

    def forward(self, feat):  # [1,15,2052]
        batch_size = feat.shape[0]  # 1

        # apply kernel for each vertex, 相当于取每个关节点[1,2048]进行一个卷积得到[1,128]
        feat = torch.stack([fcbn(feat[:,i,:]) for i,fcbn in enumerate(self.fcbn_list)],1)  # [1,15,128]= 15个[1,1,128] <- [1,1,2052] 卷积和[128,5012]，一个节点不同维度特征之间的信息加权和，然后需要多少维度做多少次。一层神经元数等于一层特征通道数，每个神经元特征就是HW，

        # apply adj
        adj = self.adj.cuda()[None, :, :].repeat(batch_size, 1, 1)  # TODO repeat函数是复制，【15，15】 1，1 ->【15，15】 1,2 -> [15,30]
        feat = torch.bmm(adj, feat)  # 【1，15，128】 <- [15,15], [1,15,128] 不同节点同纬度特征进行信息交换

        # apply activation function
        out = F.relu(feat)
        return out  # 【1，15，1288】


class GraphResBlock(nn.Module):
    def __init__(self, adj, dim):
        super(GraphResBlock, self).__init__()
        self.adj = adj  # 15
        self.graph_block1 = GraphConvBlock(adj, dim, dim)  # 15，128，128
        self.graph_block2 = GraphConvBlock(adj, dim, dim)

    def forward(self, feat):
        feat_out = self.graph_block1(feat)
        feat_out = self.graph_block2(feat_out)
        out = feat_out + feat
        return out
