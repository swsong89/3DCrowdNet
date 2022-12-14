import numpy as np
import torch
import os.path as osp
import json
from config import cfg

import sys
sys.path.insert(0, cfg.smpl_path)
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from utils.transforms import  build_adj, normalize_adj, transform_joint_to_other_db


class SMPL(object):
    def __init__(self):
        self.layer = {'neutral': self.get_layer(), 'male': self.get_layer('male'), 'female': self.get_layer('female')}
        self.vertex_num = 6890
        self.face = self.layer['neutral'].th_faces.numpy()  # [13776,3]
        self.joint_regressor = self.layer['neutral'].th_J_regressor.numpy()  # 预先训练好的，默认，[24,6890] joint [24,3] = joint_regresoor [24,5890] * vertex [6890,3]
        self.shape_param_dim = 10
        self.vposer_code_dim = 32

        # add nose, L/R eye, L/R ear, 下面比如nose节点就joint_regressor只需要对应该节点是1，其余是0，是因为6890定点种331号节点就是nose节点，所以不需要回归
        self.face_kps_vertex = (331, 2802, 6262, 3489, 3990) # mesh vertex idx
        nose_onehot = np.array([1 if i == 331 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)  # [1,6890] 331对应nose的idx是1,别的都是0,下面同理
        left_eye_onehot = np.array([1 if i == 2802 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        right_eye_onehot = np.array([1 if i == 6262 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        left_ear_onehot = np.array([1 if i == 3489 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        right_ear_onehot = np.array([1 if i == 3990 else 0 for i in range(self.joint_regressor.shape[1])], dtype=np.float32).reshape(1,-1)
        self.joint_regressor = np.concatenate((self.joint_regressor, nose_onehot, left_eye_onehot, right_eye_onehot, left_ear_onehot, right_ear_onehot))  # [29, 6890]
        # add head top
        self.joint_regressor_extra = np.load(osp.join('..', 'data', 'J_regressor_extra.npy'))  # [9,6890]
        self.joint_regressor = np.concatenate((self.joint_regressor, self.joint_regressor_extra[3:4, :])).astype(np.float32)  # 将第3行数据放进  [30，6890]

        self.orig_joint_num = 24
        self.joint_num = 30 # original: 24. manually add nose, L/R eye, L/R ear, head top
        self.joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax',
                            'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')
        self.flip_pairs = ( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), (22,23) , (25,26), (27,28) )  # 人体对称的关节，比如l_hip, r_hip
        self.skeleton = ( (0,1), (1,4), (4,7), (7,10), (0,2), (2,5), (5,8), (8,11), (0,3), (3,6), (6,9), (9,14), (14,17), (17,19), (19, 21), (21,23), (9,13), (13,16), (16,18), (18,20), (20,22), (9,12), (12,24), (24,15), (24,25), (24,26), (25,27), (26,28), (24,29) )  # 人体关节连接而成的骨架图,30个点，29条线
        self.root_joint_idx = self.joints_name.index('Pelvis')

        # joint set for PositionNet prediction
        self.graph_joint_num = 15
        self.graph_joints_name = ('Pelvis', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Neck', 'Head_top', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist')
        self.graph_flip_pairs = ((1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14))
        self.graph_skeleton = ((0, 1), (1, 3), (3, 5), (0, 2), (2, 4), (4, 6), (0, 7), (7, 8), (7, 9), (9, 11), (11, 13), (7, 10), (10, 12), (12, 14))
        # construct graph adj
        self.graph_adj = self.get_graph_adj()  # 构建节点的邻接矩阵

    def reduce_joint_set(self, joint):  # output_hm_shape
        new_joint = []
        for name in self.graph_joints_name:
            idx = self.joints_name.index(name)
            new_joint.append(joint[:,idx,:])
        new_joint = torch.stack(new_joint,1)
        return new_joint

    def get_graph_adj(self):
        """以skeleton节点连线，flip_pairs对称关节点构建图"""
        adj_mat = build_adj(self.graph_joint_num, self.graph_skeleton, self.graph_flip_pairs)
        normalized_adj = normalize_adj(adj_mat)
        return normalized_adj

    def get_layer(self, gender='neutral'):
        return SMPL_Layer(gender=gender, model_root=cfg.smpl_path + '/smplpytorch/native/models')

"""
self.j_names = {
    0: 'Pelvis',
    1: 'L_Hip',  2: 'R_Hip',
    3:  'Spine1',
    4: 'L_Knee',  5: 'R_Knee',
    6:  'Spine2',
    7:  'L_Ankle', 8: 'R_Ankle',
    9:  'Spine3',
    10: 'L_Foot', 11: 'R_Foot',
    12: 'Neck',
    13: 'L_Collar', 14: 'R_Collar',
    15: 'Head',
    16: 'L_Shoulder', 17: 'R_Shoulder',
    18: 'L_Elbow',  19: 'R_Elbow',
    20: 'L_Wrist',  21: 'R_Wrist',
    22: 'L_Hand', 23: 'R_Hand'
    }
SMPL   24个关节点  23个身体关节点 + 1个根关节点pevis
SMPLH  51个关节点  51个身体关节点 21个除了左右手关节点 + 2*15手关节点, 用15个复杂手关节点替代简单的手关节点
SMPLX  55个关节点  54个身体关节点 21个除了左右手关节点 + 2*15手关节点 + 3脸关节点, 猜脸关节点是jaw, eyeballs， 下巴，左右眼球

'Pelvis', 
'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax',
                            'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'Head_top')

"""