import numpy as np
import cv2
import random
from config import cfg
import math



def load_img(path, order='RGB'):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def get_bbox(joint_img, joint_valid):
    """
    1.先是根据joint_valid阈值去除置信度较低的点，然后计算得到包含所有关节点的最小bbox
    2. 将bbox框放大一点，增加冗余
    """
    # 计算得到目前关节点位置的xmin xmax ymin ymax,即左下角和右上角的点
    x_img, y_img = joint_img[:,0], joint_img[:,1]
    x_img = x_img[joint_valid==1]; y_img = y_img[joint_valid==1];  # 有一个关节点置信度不够，所以验证后变成[18,3]
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    # 将目前的bbox框放大一点，增加冗余
    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2  # x_center - 0.5*width刚好是xmin,然后减成0.5*width*1.2,相当于把关节点附近区域也包括
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox  # [ 74.18213  170.1289    46.58203   89.208984]

def compute_iou(src_roi, dst_roi):
    # IoU calculate with GTs
    xmin = np.maximum(dst_roi[:, 0], src_roi[:, 0])
    ymin = np.maximum(dst_roi[:, 1], src_roi[:, 1])
    xmax = np.minimum(dst_roi[:, 0] + dst_roi[:, 2], src_roi[:, 0] + src_roi[:, 2])
    ymax = np.minimum(dst_roi[:, 1] + dst_roi[:, 3], src_roi[:, 1] + src_roi[:, 3])

    interArea = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    boxAArea = dst_roi[:, 2] * dst_roi[:, 3]
    boxBArea = np.tile(src_roi[:, 2] * src_roi[:, 3], (len(dst_roi), 1))
    sumArea = boxAArea + boxBArea

    iou = interArea / (sumArea - interArea + 1e-5)

    return iou

# def trunc_bbox(bbox):
#     if False and random.random() >= 0.3:
#         return bbox
#     else:
#         x, y, w, h = bbox
#         x_aug_range, y_aug_range = w/2, h/2
#         x_aug, y_aug = random.random() * x_aug_range, random.random() * y_aug_range
#
#         if random.random() <= 0.5:
#             x, y = x+x_aug, y+y_aug
#         else: # good
#             w, h = w-x_aug, h-y_aug
#
#     return [x,y,w,h]

def process_bbox(bbox, img_width, img_height, is_3dpw_test=False):  # 函数作用1.先比较图片和最小关节点图，缩小bbox, 正常境况下这步骤没什么用。img_width=500, img_height=375,
    # sanitize bboxes
    x, y, w, h = bbox  # [ 74.18213  170.1289    46.58203   89.208984]
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))  # 取原始图像的右上角x坐标和关节点图像坐标右上角的最小值
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))  # 即如果这个人只占一部分图像，那么就裁减一下只包含到这个人就行
    if is_3dpw_test:
        bbox = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)
    else:
        if w*h > 0 and x2 >= x1 and y2 >= y1:
            bbox = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)  # [ 74.18213  170.1289    45.58203   88.208984]
        else:
            return None

   # aspect ratio preserving bbox 2.根据输入要求的尺寸，按比例缩放关节点坐标bbox比例为网络输入尺寸比例，即w=h,中心点位置不变
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = cfg.input_img_shape[1]/cfg.input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:  # w=45.58203  aspect_ratio * h=88.208984375
        w = h * aspect_ratio
    bbox[2] = w*1.25  # 这一步感觉bbox没有必要再放大了
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.
    # 原始图像的coco_joint bbox [ 74.18213  170.1289    46.58203   89.208984]
    return bbox  #            [ 41.84253 159.10278 110.26123 110.26123]

def get_aug_config(exclude_flip):
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
    if exclude_flip:
        do_flip = False
    else:
        do_flip = random.random() <= 0.5

    return scale, rot, color_scale, do_flip


def augmentation(img, bbox, data_split, exclude_flip=False):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config(exclude_flip,)
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False
    
    img, trans, inv_trans = generate_patch_image(img, bbox, scale, rot, do_flip, cfg.input_img_shape)

    # cv2.imshow('input', img/255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img, trans, inv_trans, rot, do_flip


def generate_patch_image(cvimg, bbox, scale, rot, do_flip, out_shape):  # rot旋转 do_flip翻转, 函数作用是处理输入图像，根据do_flip, scale, rotation处理bbox,然后转到网络输入图像尺寸
    # cvimg [346, 500, 3] scale=1, rot=0, do_flip=False, out_shape=[256,256,3]
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
   
    bb_c_x = float(bbox[0] + 0.5*bbox[2])  # bb_c_x = 96.97314453125 = 41.84253 + 0.5*110.26123
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:  # 左右翻转 y flip
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1  # 和process_bbox函数中img_width-1一样

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)  # 将原图转到输入图片的旋转角，trans = [[ 2.3217590e+00  0.0000000e+00 -9.7148270e+01], [ 3.7371475e-16  2.3217590e+00 -3.6939832e+02]]
    # 利用trans仿射变换矩阵直接将img转成input_img，trans是由bbox大小和input_img得到的
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)  # 直接将原图img根据旋转trans处理得到网络输入  TODO 看明白函数cv2.warpAffine
    img_patch = img_patch.astype(np.float32)  # [256， 256，3】
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)  #将input_img转成orig_img [346, 500, 3]

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):  # pt_2d= [ 0, 55.130615] rot_rad=0
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False): # 根据bbox信息，目标图像的w,h,以及bbox需要放大缩小的尺寸scale,需要旋转的角度rot得到旋转处理信息
    # augment size with scale scale的作用是应该将bbox放大多少 TODO 需要搞明白是如何处理得到trans，然后原始图像如何根据trans据可以得到目标图像
    # 根据src三个点，dst三个点，分别为中心点，右边的点，下边的点，三个对应点求出的仿射变换矩阵，利用这个矩阵可以直接将bbox
    src_w = src_width * scale  # src_w=110.26123046875
    src_h = src_height * scale # src_h=110.26123046875
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)  # [ 0.       55.130615]
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)  # [55.130615  0.      ]

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)  # [128. 128.]
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)  # [  0. 128.]
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)  # [128.   0.]

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center  # [ 96.973145 214.2334  ] 框中心
    src[1, :] = src_center + src_downdir  # [ 96.973145 269.364   ] =  [ 96.973145 214.2334  ] +  [ 0.       55.130615] 框右边宽度
    src[2, :] = src_center + src_rightdir  # [152.10376 214.2334 ] 框坐标宽度

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center  # [128,128] 框中心
    dst[1, :] = dst_center + dst_downdir  # [128,256] 框右边宽度
    dst[2, :] = dst_center + dst_rightdir  # [256,128] 框坐标宽度
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans   # [[ 2.3217590e+00  0.0000000e+00 -9.7148270e+01], [ 3.7371475e-16  2.3217590e+00 -3.6939832e+02]]

