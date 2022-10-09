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


sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image, get_bbox
from utils.transforms import pixel2cam, cam2pixel, transform_joint_to_other_db
from utils.vis import vis_mesh, save_obj, render_mesh, vis_coco_skeleton
sys.path.insert(0, cfg.smpl_path)
from utils.smpl import SMPL



def add_pelvis(joint_coord, joints_name):
    lhip_idx = joints_name.index('L_Hip')
    rhip_idx = joints_name.index('R_Hip')
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5  # [112.48779297 223.70556641   0.35623145] + [ 94.90966797 221.14208984   0.44777694]*0.5
    pelvis[2] = joint_coord[lhip_idx, 2] * joint_coord[rhip_idx, 2]  # confidence for openpose, 前两个是坐标点，计算中点是pelvis, 第三个是置信度所以相乘
    pelvis = pelvis.reshape(1, 3)

    joint_coord = np.concatenate((joint_coord, pelvis))

    return joint_coord

def add_neck(joint_coord, joints_name):
    lshoulder_idx = joints_name.index('L_Shoulder')
    rshoulder_idx = joints_name.index('R_Shoulder')
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck[2] = joint_coord[lshoulder_idx, 2] * joint_coord[rshoulder_idx, 2]
    neck = neck.reshape(1,3)

    joint_coord = np.concatenate((joint_coord, neck))

    return joint_coord

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
cfg.set_args(args.gpu_ids, is_test=True, no_log=True)
cfg.render = True
cudnn.benchmark = True

# SMPL joint set
joint_num = 30  # original: 24. manually add nose, L/R eye, L/R ear, head top
joints_name = (
'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
'Neck', 'L_Thorax', 'R_Thorax',
'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 
'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')  # SMPL本身没有这6个关节点
flip_pairs = (
(1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
skeleton = (
(0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17), (17, 19),
(19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25), (24, 26),
(25, 27), (26, 28), (24, 29))

# SMPl mesh
vertex_num = 6890
smpl = SMPL()
face = smpl.face

# other joint set 19个，coco本身只有17个
coco_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear','L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle',
'Pelvis', 'Neck') # coco本身是没有pelvis盆骨, neck脖子
coco_skeleton = (
(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 6),
(11, 17), (12,17), (17,18))

vis_joints_name = ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle',
'Thorax', 'Pelvis')
vis_skeleton = ((0, 1), (0, 2), (2, 4), (1, 3), (5, 7), (7, 9), (12, 14), (14, 16), (11, 13), (13, 15), (5, 17), (6, 17), (11, 18), (12, 18), (17, 18), (17, 0), (6, 8), (8, 10),)

# snapshot load
model_path = args.model_path
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model).cuda()
ckpt = torch.load(model_path)  # model_path=demo_checkpint.pth.tar
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()
pose2d_result_path = './input/2d_pose_result.json'  # demo的标注坐标，原点是左上角，x是从左到右，y是从上到下
with open(pose2d_result_path) as f:
    pose2d_result = json.load(f)
# pose2d_result [1086] {'100023.jpg':list}, 这个list维度为[9, 17, 5] 9个人。
# 1086张图片,每张图片数据为[person_nums, 17, 5] 17为coco数据关节点坐标, 5分别为中心坐标,上下宽度，左右长度，以及置信度
img_dir = './input/images'
for img_name in sorted(pose2d_result.keys()):
    img_path = osp.join(img_dir, img_name)
    original_img = cv2.imread(img_path)  # [346, 500, 3]  h w c
    input = original_img.copy()
    input2 = original_img.copy()  # cv2.imread会将图片读成B\G\R,
    original_img_height, original_img_width = original_img.shape[:2]  # 346, 500
    coco_joint_list = pose2d_result[img_name]  # pose2d_result是一个图片对应坐标的字典，所以是取该图片对应的2d pose, 坐标 x,y,confidence,不知道，不知道

    if args.img_idx not in img_name:
        continue

    drawn_joints = []
    c = coco_joint_list  # [person_nums, 17, 5]  17为coco数据集关节点数量, 5中分别为关节点[x,y,confidence,不知道，不知道]
    # manually assign the order of output meshes
    # coco_joint_list = [c[2], c[0], c[1], c[4], c[3]]

    for idx in range(len(coco_joint_list)):
        """ 2D pose input setting & hard-coding for filtering """
        pose_thr = 0.1
        coco_joint_img = np.asarray(coco_joint_list[idx])[:, :3]  # [17, 3], coco5个值的前3个 [346, 500, 3] 93.81104, 182.68994,0.85222
        coco_joint_img = add_pelvis(coco_joint_img, coco_joints_name)  # 取l_hip和r_hip的中点为pevis
        coco_joint_img = add_neck(coco_joint_img, coco_joints_name)  # 取l_shoulder, r_shouder中点为neck  # 【19，3】
        coco_joint_valid = (coco_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32) # coco_joint_img[:, 2] [19] reshape后变成[19, 1]  第5个 0 R_ear 置信度 0.02976


        # filter inaccurate inputs 过滤某个人的关节点太少的情况
        det_score = sum(coco_joint_img[:, 2])  # 10.866020886755301 如果这个人所有关节点的置信度和小于1的话放弃这个人
        if det_score < 1.0:
            continue
        if len(coco_joint_img[:, 2:].nonzero()[0]) < 1:  #TODO 弄懂.nonzeros 如果这个人的关节点置信度都是0的话跳过。 coco_joint_img[:, 2:] [19,1] coco_joint_img[:, 2] [19,]
            continue

        # filter the same targets drawn_joints如果有给定的话，需要过滤一下相同位置的关节点
        tmp_joint_img = coco_joint_img.copy()  # [19,3]
        continue_check = False
        for ddx in range(len(drawn_joints)):
            drawn_joint_img = drawn_joints[ddx]
            drawn_joint_val = (drawn_joint_img[:, 2].copy().reshape(-1, 1) > pose_thr).astype(np.float32)
            diff = np.abs(tmp_joint_img[:, :2] - drawn_joint_img[:, :2]) * coco_joint_valid * drawn_joint_val
            diff = diff[diff != 0]
            if diff.size == 0:
                continue_check = True
            elif diff.mean() < 20:
                continue_check = True
        if continue_check:
            continue
        drawn_joints.append(tmp_joint_img)  # drawn_joints 【1，19，3】

        """ Prepare model input """
        # prepare bbox get_bbox
        bbox = get_bbox(coco_joint_img, coco_joint_valid[:, 0]) # xmin, ymin, width, height 先根据关节点坐标和置信度得到最小bbox, bbox= [ 74.18213  170.1289    46.58203   89.208984]
        bbox = process_bbox(bbox, original_img_width, original_img_height)  # 1.根据原始图像h,w处理bbox不要越界，2.按照cfg.input_img_shape缩小或放大bbox高宽比 3.增加容错能力bbox_h,w变成1.25倍 [ 41.84253 159.10278 110.26123 110.26123]
        if bbox is None:
            continue
        img, img2bb_trans, bb2img_trans = generate_patch_image(input2[:,:,::-1], bbox, 1.0, 0.0, False, cfg.input_img_shape)  # B\G\R->R\G\B img [346, 500, 3] -> [256,256,3], img2bb_trans是图像直接转网络输入bbox  cfg.input_img_shape [256,2556,3]
        img = transform(img.astype(np.float32))/255  # img [3,256,256]
        img = img.cuda()[None,:,:,:]  # img [1,3,256,256]
        # 下面将coco坐标由原始图像转到hm热力图图像中  img2bb_trans 原始图片[346,500] 到[256,256] hw
        #  img2bb_trans 2.32176,0.00000,-97.14827 仿射变换矩阵，矩阵是高度宽度变换的，所以坐标需要transpose到1,0
        #               0.00000,2.32176,-369.39832
        coco_joint_img_xy1 = np.concatenate((coco_joint_img[:, :2], np.ones_like(coco_joint_img[:, :1])), 1)  # coco_joint_img_xy1 [19,3] 将每个关节点坐标的置信度都变成1 93.81104, 182.68994,1
        coco_joint_img[:, :2] = np.dot(img2bb_trans, coco_joint_img_xy1.transpose(1, 0)).transpose(1, 0)  # 坐标由[346,500]空间变到[256,256]img2bb_trans [2,3] coco_joint_img_xy1.transpose(1, 0) [3,19]因为trans是根据hwc旋转的，所以需要xy旋转变成yx才可以 120.65834,54.76370,0.85222
        coco_joint_img[:, 0] = coco_joint_img[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]  # coco_joint_img[:, 0] x  cfg.input_img_shape[1] w  output_hm_shape[2] w
        coco_joint_img[:, 1] = coco_joint_img[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1] #  30.16459,13.69092,0.85222    之前的 93.81104, 182.68994,0.85222
        # 坐标由input_img_shape 256,256转到Output_hm_shape 64,64
        coco_joint_img = transform_joint_to_other_db(coco_joint_img, coco_joints_name, joints_name)  # 将coco关节点转到定义的超关节点网络中 93.81104,182.68994,0.85222
        coco_joint_valid = transform_joint_to_other_db(coco_joint_valid, coco_joints_name, joints_name)
        coco_joint_valid[coco_joint_img[:, 2] <= pose_thr] = 0

        # check truncation
        coco_joint_trunc = coco_joint_valid * ((coco_joint_img[:, 0] >= 0) * (coco_joint_img[:, 0] < cfg.output_hm_shape[2]) * (coco_joint_img[:, 1] >= 0) * (coco_joint_img[:, 1] < cfg.output_hm_shape[1])).reshape(
            -1, 1).astype(np.float32)  # 该关节点置信度*坐标xy是否在热力图范围内
        coco_joint_img, coco_joint_trunc, bbox = torch.from_numpy(coco_joint_img).cuda()[None, :, :], torch.from_numpy(coco_joint_trunc).cuda()[None, :, :], torch.from_numpy(bbox).cuda()[None, :]
        # coco_joint_img [1,30,3]  coco_joint_trunc[1,30,1] bbox变成符合input_img_shape的bbox [[ 41.8425, 159.1028, 110.2612, 110.2612]]
        """ Model forward """
        inputs = {'img': img, 'joints': coco_joint_img, 'joints_mask': coco_joint_trunc}  # [1,3,256,256] [1,30,3] [1,30,1] 256，256真实坐标点
        targets = {}  # coco_joint_img[0,0]= [35.9038, 36.7541,  0.1595]
        meta_info = {'bbox': bbox}

        # 统计模型的参数
        # total = sum([param.nelement() for param in model.parameters()])
        # print("Number of parameter: %.2fM" % (total / 1e6))   # 30.54M
        with torch.no_grad():
            out = model(inputs, targets, meta_info, 'test')
        '''
        {'cam_param': tensor维度为[1,3], 'joint_img': [1,15,3], 'joint_proj': [1,30,2], 'joint_score': [1,15,1], 
        'smpl_mesh_cam': [1,6890,3], 'smpl_pose': [1,72], 'smpl_shape':[1,10], 'mesh_cam_render': [1,6890,3], 
        'bbox': tensor([[ 41.8425, 159.1028, 110.2612, 110.2612]], device='cuda:0')}  'bbox'为input_img即网络输入图片的bbox
        '''
        # draw output mesh
        mesh_cam_render = out['mesh_cam_render'][0].cpu().numpy()   # [6890,3] <- [1,6890,3]
        bbox = out['bbox'][0].cpu().numpy()  # [ 41.8425, 159.1028, 110.2612, 110.2612] input_img的bbox中点
        princpt = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)  # input_img的bbox中点 (c_x,c_y) (96.97314453125, 214.2333984375)
        # original_img = vis_bbox(original_img, bbox, alpha=1)  # for debug

        # generate random color
        color = colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0)
        original_img = render_mesh(original_img, mesh_cam_render, face, {'focal': cfg.focal, 'princpt': princpt}, color=color)  #  [375,500,3] h,w,c [6890,3] [13776,3]  {(5000,5000), (96.97314453125, 214.2333984375)}

        # Save output mesh
        output_dir = 'output'
        file_name = f'{output_dir}/{img_path.split("/")[-1][:-4]}_{idx}.jpg'  #  img_path= './input/images/101570.jpg'
        print("file name: ", file_name)
        save_obj(mesh_cam_render, face, file_name=f'{output_dir}/{img_path.split("/")[-1][:-4]}_{idx}.obj')
        cv2.imwrite(file_name, original_img)  # 将该人的mesh画出

        # Draw input 2d pose
        tmp_joint_img[-1], tmp_joint_img[-2] = tmp_joint_img[-2].copy(), tmp_joint_img[-1].copy()  # coco本来只有17，上面操作添加了pelvis, neck, 可视化的是thorax,pelvis,所以后面两个关节点需要调换下
        input = vis_coco_skeleton(input, tmp_joint_img.T, vis_skeleton)  # [375,500,3] [3,19] [18] 19个关节点18条线
        cv2.imwrite(file_name[:-4] + '_2dpose.jpg', input)


