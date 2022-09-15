import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from nets.module import Pose2Feat, PositionNet, RotationNet, Vposer
from nets.loss import CoordLoss, ParamLoss, NormalVectorLoss, EdgeLengthLoss
from utils.smpl import SMPL
from utils.mano import MANO
from config import cfg
from contextlib import nullcontext
import math

from utils.transforms import rot6d_to_axis_angle


class Model(nn.Module):
    def __init__(self, backbone, pose2feat, position_net, rotation_net, vposer):
        super(Model, self).__init__()
        self.backbone = backbone  # 特征提取部分的ResNet50
        self.pose2feat = pose2feat
        self.position_net = position_net
        self.rotation_net = rotation_net
        self.vposer = vposer

        if 'FreiHAND' in cfg.trainset_3d + cfg.trainset_2d + [cfg.testset]:
            self.human_model = MANO()
            self.human_model_layer = self.human_model.layer.cuda()
        else:
            self.human_model = SMPL()
            self.human_model_layer = self.human_model.layer['neutral'].cuda()
        self.root_joint_idx = self.human_model.root_joint_idx  # 0
        self.mesh_face = self.human_model.face  # [13776, 3]
        self.joint_regressor = self.human_model.joint_regressor  # [30, 6890]

        self.coord_loss = CoordLoss()
        self.param_loss = ParamLoss()

    def get_camera_trans(self, cam_param, meta_info, is_render):  # [1,3] meta_info={'bbox': tensor([[ 41.8425, 159.1028, 110.2612, 110.2612]], device='cuda:0')} True
        # camera translation
        t_xy = cam_param[:, :2]  # [1,2]
        gamma = torch.sigmoid(cam_param[:, 2]) # [1] apply sigmoid to make it positive
        k_value = torch.FloatTensor([math.sqrt(cfg.focal[0]*cfg.focal[1]*cfg.camera_3d_size*cfg.camera_3d_size/(cfg.input_img_shape[0]*cfg.input_img_shape[1]))]).cuda().view(-1)  # [1] 值为48.8281
        if is_render:
            bbox = meta_info['bbox']  # [[ 41.8425, 159.1028, 110.2612, 110.2612]]
            k_value = k_value * math.sqrt(cfg.input_img_shape[0]*cfg.input_img_shape[1]) / (bbox[:, 2]*bbox[:, 3]).sqrt()
        t_z = k_value * gamma  # tensor([61.6558], device='cuda:0') [1]
        cam_trans = torch.cat((t_xy, t_z[:, None]), 1)  # [1,3] [[ 0.0899,  0.2569, 61.6558]]
        return cam_trans

    def make_2d_gaussian_heatmap(self, joint_coord_img):  # 利用2d坐标生成热力图  joint_coord_img [1,30,3]
        x = torch.arange(cfg.output_hm_shape[2])  # x轴从左到右 [64]
        y = torch.arange(cfg.output_hm_shape[1])  # y轴从上到下 [64]
        yy, xx = torch.meshgrid(y, x)  ## [64,64] yy是每行相同的,即64行,0行全是0,  xx是每列相同的,每行都是0-63,即左上角是原点，往右是x轴正方向，往下是y轴正方向
        xx = xx[None, None, :, :].cuda().float();  # [1,1,64,64]
        yy = yy[None, None, :, :].cuda().float();  # [1,1,64,64]

        x = joint_coord_img[:, :, 0, None, None];  # [1,30,1,1] [:, :, 0]变成[1,30] None,None再加2维度
        y = joint_coord_img[:, :, 1, None, None];
        heatmap = torch.exp(
            -(((xx - x) / cfg.sigma) ** 2) / 2 - (((yy - y) / cfg.sigma) ** 2) / 2)
        return heatmap  # [1,30,64,64] NxJsxYxX

    def get_coord(self, smpl_pose, smpl_shape, smpl_trans):  # pose [1,72] shape [1,10] camera [1,3]
        batch_size = smpl_pose.shape[0]
        mesh_cam, _ = self.human_model_layer(smpl_pose, smpl_shape, smpl_trans)  # [1,6890,3
        # camera-centered 3D coordinate
        joint_cam = torch.bmm(torch.from_numpy(self.joint_regressor).cuda()[None,:,:].repeat(batch_size,1,1), mesh_cam)  # [1,30,3] <- [1,30,6890], [1,6890,3]
        root_joint_idx = self.human_model.root_joint_idx  # 0

        # project 3D coordinates to 2D space
        x = joint_cam[:, :, 0] / (joint_cam[:, :, 2] + 1e-4) * cfg.focal[0] + cfg.princpt[0]  # [1,30]此时关节点位置是在input_img上
        y = joint_cam[:, :, 1] / (joint_cam[:, :, 2] + 1e-4) * cfg.focal[1] + cfg.princpt[1]  # [1,30]
        x = x / cfg.input_img_shape[1] * cfg.output_hm_shape[2]  # cfg.input_img_shape[1]=256   cfg.output_hm_shape[2]=64
        y = y / cfg.input_img_shape[0] * cfg.output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)  # [1,30,2]

        mesh_cam_render = mesh_cam.clone()  # [1,6890,3]
        # root-relative 3D coordinates  相对于根节点的3d坐标
        root_cam = joint_cam[:, root_joint_idx, None, :]  # [1,1,3]  [[[8.7456e-02, 3.8836e-02, 6.1679e+01]]]
        joint_cam = joint_cam - root_cam  # [1,30,3]
        mesh_cam = mesh_cam - root_cam  # [1,6890,3]  #  mesh_cam =  mesh_cam_render - root_cam
        return joint_proj, joint_cam, mesh_cam, mesh_cam_render  # [1,30,2]  [1,30,3] [1,6890,3]  [1,6890,3]

    def forward(self, inputs, targets, meta_info, mode):  # inputs = {'img':[1,3,256,256], 'joints':[1,30,3],'joints_mask':[1,30,1]} NCHW
        early_img_feat = self.backbone(inputs['img'])  #pose_guided_img_feat  输入img经过ResNet50 early-stage提取的特征F [1,64,64,64]

        # get pose gauided image feature
        joint_coord_img = inputs['joints']  # 2d pose
        with torch.no_grad():
            joint_heatmap = self.make_2d_gaussian_heatmap(joint_coord_img.detach())  # 通过2d pose生成2d pose heatmap  joint_heatmap [1,30,64,64] .detach()返回一个和原始tensor共同内存的变量，但是永远不需要梯度计算
            # remove blob centered at (0,0) == invalid ones
            joint_heatmap = joint_heatmap * inputs['joints_mask'][:,:,:,None]  # inputs['joints_mask'][:,:,:,None] [1,30,1,1] inputs['joints_mask']是这个数据集关节点置信度小于阈值或者超集中的这个关节点数据集没有, 所以省略
        pose_img_feat = self.pose2feat(early_img_feat, joint_heatmap)  # [1,64,64,64] = conv(concat([1,64,64,64] [1,30,64,64]))
        pose_guided_img_feat = self.backbone(pose_img_feat, skip_early=True)  # F' [1,2048,8,8] 输出 2048 x 8 x 8 C'2048

        joint_img, joint_score = self.position_net(pose_guided_img_feat)  # refined 2D pose or 3D pose  joint_img [1,15,3] joint_score [1,15,1] pose_guided_img_feat [1,2048,8,8]

        # estimate model parameters  [1，6] [1，32] [1,10] [1，3]
        root_pose_6d, z, shape_param, cam_param = self.rotation_net(pose_guided_img_feat, joint_img.detach(), joint_score.detach())
        # change root pose 6d + latent code -> axis angles
        root_pose = rot6d_to_axis_angle(root_pose_6d)  # [1,3]
        pose_param = self.vposer(z)   # [1,69]
        cam_trans = self.get_camera_trans(cam_param, meta_info, is_render=(cfg.render and (mode == 'test')))  # cam_trans [1,3] [[ 0.0899,  0.2569, 61.6558]] cam_param [1,3] [[ 0.0899,  0.2569, 0.1759]
        pose_param = pose_param.view(-1, self.human_model.orig_joint_num - 1, 3)  # [1,23,3]因为vpose没有l_hand,r_hand, 所以最后的是0,0,0,0,0,0
        pose_param = torch.cat((root_pose[:, None, :], pose_param), 1).view(-1, self.human_model.orig_joint_num * 3)  # 将root节点也就是pelvis放在第1个变成[1,24,3] -> [1,72]
        joint_proj, joint_cam, mesh_cam, mesh_cam_render = self.get_coord(pose_param, shape_param, cam_trans)  # [1,30,2] [1,30,3] [1,6890,3] [1,6890,3] <- [1,72] [1,10] [1,3]

        if mode == 'train':
            # loss functions
            loss = {}
            # joint_img: 0~8, joint_proj: 0~64, target: 0~64
            loss['body_joint_img'] = (1/8) * self.coord_loss(joint_img*8, self.human_model.reduce_joint_set(targets['orig_joint_img']), self.human_model.reduce_joint_set(meta_info['orig_joint_trunc']), meta_info['is_3D'])
            loss['smpl_joint_img'] = (1/8) * self.coord_loss(joint_img*8, self.human_model.reduce_joint_set(targets['fit_joint_img']),
                                                     self.human_model.reduce_joint_set(meta_info['fit_joint_trunc']) * meta_info['is_valid_fit'][:, None, None])
            loss['smpl_pose'] = self.param_loss(pose_param, targets['pose_param'], meta_info['fit_param_valid'] * meta_info['is_valid_fit'][:, None])
            loss['smpl_shape'] = self.param_loss(shape_param, targets['shape_param'], meta_info['is_valid_fit'][:, None])
            loss['body_joint_proj'] = (1/8) * self.coord_loss(joint_proj, targets['orig_joint_img'][:, :, :2], meta_info['orig_joint_trunc'])
            loss['body_joint_cam'] = self.coord_loss(joint_cam, targets['orig_joint_cam'], meta_info['orig_joint_valid'] * meta_info['is_3D'][:, None, None])
            loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['fit_joint_cam'], meta_info['is_valid_fit'][:, None, None])

            return loss

        else:
            # test output
            out = {'cam_param': cam_param}  # [1,3] [[0.0899, 0.2569, 0.1759]]
            # out['input_joints'] = joint_coord_img
            out['joint_img'] = joint_img * 8  # [1,15,3] TODO 为什么*8
            out['joint_proj'] = joint_proj  # [1,15,2]
            out['joint_score'] = joint_score  # [1,15,1]
            out['smpl_mesh_cam'] = mesh_cam  # [1,6890,3]  mesh_cam =  mesh_cam_render - root_cam
            out['smpl_pose'] = pose_param  # [1,72]
            out['smpl_shape'] = shape_param  # [1,10]

            out['mesh_cam_render'] = mesh_cam_render  # [1,6890,3]

            if 'smpl_mesh_cam' in targets:
                out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
            if 'bb2img_trans' in meta_info:
                out['bb2img_trans'] = meta_info['bb2img_trans']
            if 'img2bb_trans' in meta_info:
                out['img2bb_trans'] = meta_info['img2bb_trans']
            if 'bbox' in meta_info:
                out['bbox'] = meta_info['bbox']
            if 'tight_bbox' in meta_info:
                out['tight_bbox'] = meta_info['tight_bbox']
            if 'aid' in meta_info:
                out['aid'] = meta_info['aid']

            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(vertex_num, joint_num, mode):

    backbone = ResNetBackbone(cfg.resnet_type)
    pose2feat = Pose2Feat(joint_num)
    position_net = PositionNet()
    rotation_net = RotationNet()
    vposer = Vposer()

    if mode == 'train':
        backbone.init_weights()
        pose2feat.apply(init_weights)
        position_net.apply(init_weights)
        rotation_net.apply(init_weights)
   
    model = Model(backbone, pose2feat, position_net, rotation_net, vposer)
    return model

