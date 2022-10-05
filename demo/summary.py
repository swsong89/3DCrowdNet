import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import colorsys
import json
import random
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torchsummary import summary

from thop import profile
from thop import clever_format

# 添加path
root_dir = '/home/ssw/code/3DCrowdNet'
sys.path.insert(0, osp.join(root_dir, 'main'))
sys.path.insert(0, osp.join(root_dir, 'data'))
sys.path.insert(0, osp.join(root_dir, 'common'))

from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image, get_bbox
from utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
from utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton
sys.path.insert(0, cfg.smpl_path)
from utils.smpl import SMPL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--model_path', type=str, default='demo_checkpoint.pth.tar')
    parser.add_argument('--img_idx', type=str, default='101570')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, is_summary=True)
cfg.render = True
cudnn.benchmark = True


vertex_num = 6890
joint_num = 30

# snapshot load
model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

# model = DataParallel(model).cuda()
ckpt = torch.load(model_path)  # model_path=demo_checkpint.pth.tar
model.load_state_dict(ckpt['network'], strict=False)
model.eval()
model.cuda()

# 统计模型每层输入输出情况
summary(model, [(3, 256, 256), (30, 3), (30, 1), (4,)], batch_size=1)  # Total params: 30,208,169 30.208M


# 默认方法 统计参数量
# total = sum([param.nelement() for param in model.parameters()])
# print("Number of parameter: %.2fM" % (total / 1e6))   # 30.54M  不准,里面有重复项


# thop 统计参数量和运算量
# input1 = torch.randn(1, 3, 256, 256).cuda()
# input2 = torch.randn(1, 30, 3).cuda()
# input3 = torch.randn(1, 30, 1).cuda()
# input4 = torch.randn(1, 4).cuda()
#
# flops, params = profile(model, inputs=(input1, input2, input3, input4))
# print(flops, params)  # 5641332352.0 30208169.0
#
# flops, params = clever_format([flops, params], '%.3f')
# print(flops, params)  # 5.641G 30.208M


""" torchsummary需要修改的地方
- 模型有的输出项可能为None, 需要将如果是None的话改成-1. 19,20注释掉
/home/ssw/.local/lib/python3.8/site-packages/torchsummary/torchsummary.py 27行
summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:]  for o in output
                ]
summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] if o is not None else [-1] for o in output
                ]

- 对于模型对输入的情况，输入参数量统计， *4是int占4个字节 B
/home/ssw/.local/lib/python3.8/site-packages/torchsummary/torchsummary.py
total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
total_input_size = abs(np.sum([np.prod(in_tuple) for in_tuple in input_size]) * batch_size * 4. / (1024 ** 2.))
"""