import torch
import numpy as np
from config import cfg
import torchgeometry as tgm
from torch.nn import functional as F


def denorm_joints(pose_out_img, body_bb2img_trans):
    pose_out_img[:, 0] = pose_out_img[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    pose_out_img[:, 1] = pose_out_img[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    pose_out_img_xy1 = np.concatenate((pose_out_img[:, :2], np.ones_like(pose_out_img[:, :1])), 1)
    pose_out_img[:, :2] = np.dot(body_bb2img_trans, pose_out_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]

    return pose_out_img

def cam2pixel(cam_coord, f, c):  # [17,3] f焦距 [2] c像中点 [2]
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)  # x,y转到像素坐标,z是相机坐标z

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):  # world_coord [17,3] R [3,3] t [3]   TODO搞懂坐标转换
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord  # [17,3]

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s) 

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t

def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2

def transform_joint_to_other_db(src_joint, src_name, dst_name):  # src_joint [19,3]
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)  # (30,) + (3,) = (30, 3)  new_joint [30,3]
    for src_idx in range(len(src_name)):  # 根据关节点name将src转到dst中
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def build_adj(vertex_num, skeleton, flip_pairs):  """构建Jc15关节点交集的邻接矩阵,skeleton有连线的，flip_pairs对称的 vertex_num=15, skeleton [14,] flip_pair [6,]"""
    adj_matrix = np.zeros((vertex_num, vertex_num))  # [15,15]
    for line in skeleton:
        adj_matrix[line] = 1
        adj_matrix[line[1], line[0]] = 1
    for pair in flip_pairs:
        adj_matrix[pair] = 1
        adj_matrix[pair[1], pair[0]] = 1
    return adj_matrix

def normalize_adj(adj):
    vertex_num = adj.shape[0]
    adj_self = adj + np.eye(vertex_num)  # 加一个单位阵，对角阵
    D = np.diag(adj_self.sum(0)) + np.spacing(np.array(0))
    _D = 1 / np.sqrt(D)
    _D = _D * np.eye(vertex_num) # make diagonal matrix
    normalized_adj = np.dot(np.dot(_D, adj_self), _D)
    return normalized_adj

def rot6d_to_axis_angle(x):  # [1, 6]
    batch_size = x.shape[0]

    x = x.view(-1, 3, 2)  # [1,3,2]
    a1 = x[:, :, 0]  # [1,3]
    a2 = x[:, :, 1]  # [1,3]
    b1 = F.normalize(a1)  # 标准化，该数除以这行数的平方和开方
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)  # b2 [1,3] torch.einsum('bi,bi->b', b1, a2) [1] 加.queeze(-1) ->[1,1]
    b3 = torch.cross(b1, b2)  # [1,3]
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix  [1,3,3]相当于一列一列

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix [1,3,4]
    axis_angle = tgm.rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle [1,3] [N,axis]
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle # axis-angle [1,3] [N,axis]


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

