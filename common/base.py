import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from config import cfg
from model import get_model
from dataset import MultipleDatasets
dataset_list = ['CrowdPose', 'Human36M', 'MPII', 'MSCOCO', 'MuCo', 'PW3D']
for i in range(len(dataset_list)):
    exec('from ' + dataset_list[i] + ' import ' + dataset_list[i])


class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, log_name='logs.txt'):
        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam([
            {'params': model.module.backbone.parameters(), 'lr': cfg.lr_backbone},
            {'params': model.module.pose2feat.parameters()},
            {'params': model.module.position_net.parameters()},
            {'params': model.module.rotation_net.parameters()},
        ],
        lr=cfg.lr)
        print('The parameters of backbone, pose2feat, position_net, rotation_net, are added to the optimizer.')

        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))  # osp.join = '/home/ssw/code/3DCrowdNet/main/../output/exp_09-26_18:18/checkpoint/*.pth.tar'  model_file_list是个列表，包含0 1 2 3...
        cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])  # int里面是截取版本数字，比如0,1,2,13迭代版本
        ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path) 
        start_epoch = ckpt['epoch'] + 1

        model.load_state_dict(ckpt['network'], strict=False)
        #optimizer.load_state_dict(ckpt['optimizer'])

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < cfg.lr_dec_epoch[-1]:
            idx = cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        trainset3d_loader = []
        for i in range(len(cfg.trainset_3d)):
            trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
        trainset2d_loader = []
        for i in range(len(cfg.trainset_2d)):
            trainset2d_loader.append(eval(cfg.trainset_2d[i])(transforms.ToTensor(), "train"))

        if len(trainset3d_loader) > 0 and len(trainset2d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset3d_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
            trainset2d_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
            trainset_loader = MultipleDatasets([trainset3d_loader, trainset2d_loader], make_same_len=True)
        elif len(trainset3d_loader) > 0:
            self.vertex_num = trainset3d_loader[0].vertex_num
            self.joint_num = trainset3d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset3d_loader, make_same_len=False)
        elif len(trainset2d_loader) > 0:
            self.vertex_num = trainset2d_loader[0].vertex_num
            self.joint_num = trainset2d_loader[0].joint_num
            trainset_loader = MultipleDatasets(trainset2d_loader, make_same_len=False)
        else:
            assert 0, "Both 3D training set and 2D training set have zero length."
            
        self.itr_per_epoch = math.ceil(len(trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=cfg.num_gpus*cfg.train_batch_size, shuffle=True, num_workers=cfg.num_thread, pin_memory=True)

    def _make_model(self):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model(self.vertex_num, self.joint_num, 'train')
        model = DataParallel(model).cuda()
        optimizer = self.get_optimizer(model)
        if cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer)
        else:
            start_epoch = 0
        model.train()  # 和model.eval相对应，即BN是否计算新的方差，eval,不会计算新的方差，直接用训练的方法数据进行计算，with_no_grad是不保留梯度数据，在测试的时候可以节省显存使用

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer


class Tester(Base):
    def __init__(self, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(log_name='test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset... ")
        testset_loader = eval(cfg.testset)(transforms.ToTensor(), "test")  # pw_3d_test  1923
        batch_generator = DataLoader(dataset=testset_loader, batch_size=cfg.num_gpus*cfg.test_batch_size, shuffle=False, num_workers=cfg.num_thread, pin_memory=True)
        
        self.testset = testset_loader
        self.vertex_num = testset_loader.vertex_num
        self.joint_num = testset_loader.joint_num
        self.batch_generator = batch_generator

    def _make_model(self):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(self.vertex_num, self.joint_num, 'test')
        model = DataParallel(model).cuda()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model

    def _evaluate(self, outs, cur_sample_idx):
        eval_result = self.testset.evaluate(outs, cur_sample_idx)
        return eval_result

    def _print_eval_result(self, eval_result):
        result_str = self.testset.print_eval_result(eval_result)
        self.logger.info(result_str)

"""
{'ann_id': 0, 'img_path': '../data/PW3D/data/imageFiles/courtyard_hug_00/image_00000.jpg', 'img_shape': (1920, 1080), 'bbox': array([-186.2363 ,  394.99792, 1056.894  , 1056.894  ], dtype=float32), 'tight_bbox': array([241.65889, 500.68732, 202.10365, 846.51526], dtype=float32), 'smpl_param': {'shape': [-0.7517868876457214, 1.404891848564148, -0.26399368047714233, -0.13454507291316986, 0.09165917336940765, -0.2704026401042938, 0.06477046757936478, -0.009567486122250557, 0.022963471710681915, 0.007497868500649929], 'pose': [-2.8965206146240234, -0.06445767730474472, 0.9081289172172546, 0.198399618268013, -0.14874643087387085, -0.11773217469453812, 0.17083318531513214, -0.06860806792974472, -0.10743704438209534, -0.039876170456409454, 0.03960094973444939, -0.019487325102090836, -0.0628167986869812, -0.06455118209123611, -0.01725228875875473, -0.010671827010810375, 0.057830289006233215, 0.05218059569597244, 0.07821841537952423, 0.05109081789851189, -0.01917252130806446, 0.0005913575878366828, 0.003509542904794216, -0.004536268766969442, 0.0006974991993047297, -0.010056748986244202, 0.025151880457997322, 0.023476898670196533, 0.02864404208958149, -0.004718175623565912, 0.01340175699442625, 0.0025029636453837156, 0.014611868187785149, -0.00806124322116375, 0.01947561651468277, -0.057278119027614594, -0.15240664780139923, 0.056679148226976395, 0.025872226804494858, 0.20038199424743652, -0.29672712087631226, -0.27397456765174866, 0.18001273274421692, 0.28831517696380615, 0.23997445404529572, -0.06519249081611633, -0.010126576758921146, 0.0740201398730278, 0.4362682104110718, -0.4186422824859619, -1.167391061782837, 0.23026473820209503, 0.1840885579586029, 1.1890023946762085, 0.11472294479608536, -0.15904028713703156, -0.06394972652196884, 0.08475533872842789, 0.15266747772693634, -0.03921102359890938, 0.10672637075185776, -0.09504203498363495, 0.15452663600444794, 0.03437865152955055, 0.05165733024477959, -0.1167544573545456, -0.22219766676425934, -0.03540075942873955, -0.18191535770893097, -0.13806882500648499, 0.10286311060190201, 0.20317676663398743], 'trans': [-0.3982424470374603, -0.008470407046992895, 3.8558891825074637], 'gender': 'female'}, 'cam_param': {'focal': array([1961.8529, 1969.2307], dtype=float32), 'princpt': array([540., 960.], dtype=float32)}, 'root_joint_depth': None, 'pose_score_thr': 0.05,
'openpose':     array([[3.1210400e+02, 5.2688800e+02, 9.4062698e-01],
                       [3.4848401e+02, 6.4672198e+02, 9.8120397e-01],
                       [2.7075601e+02, 6.4171698e+02, 9.6052003e-01],
                       [2.7043100e+02, 7.6688599e+02, 8.1998098e-01],
                       [2.6526801e+02, 8.7129303e+02, 7.6673502e-01],
                       [4.1098999e+02, 6.4694897e+02, 9.0446001e-01],
                       [4.3178101e+02, 7.8256097e+02, 8.3774799e-01],
                       [4.2154001e+02, 8.9721301e+02, 8.4732401e-01],
                       [3.0168201e+02, 8.9231500e+02, 8.1067997e-01],
                       [3.1217499e+02, 1.0645500e+03, 9.0843201e-01],
                       [3.4334698e+02, 1.2262500e+03, 9.2748302e-01],
                       [3.8493100e+02, 8.9722498e+02, 8.3085197e-01],
                       [3.8503400e+02, 1.0695900e+03, 7.9622900e-01],
                       [3.9014700e+02, 1.2364500e+03, 8.9548999e-01],
                       [3.0695401e+02, 5.1653699e+02, 1.9056840e-01],
                       [3.3302301e+02, 5.1664203e+02, 1.7827000e-01],
                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                       [3.7974399e+02, 5.3217798e+02, 1.7221460e-01],
                       [3.4330652e+02, 8.9477002e+02, 6.7355508e-01]], dtype=float32),
       
'hhrnetpose':   array([[3.10605469e+02, 5.35488281e+02, 8.02343786e-01],
                       [3.30644531e+02, 5.23886719e+02, 7.49305248e-01],
                       [3.04277344e+02, 5.19667969e+02, 8.32207918e-01],
                       [3.75996094e+02, 5.40761719e+02, 7.39133596e-01],
                       [2.84238281e+02, 5.22832031e+02, 1.45413745e-02],
                       [4.16074219e+02, 6.55722656e+02, 6.44148648e-01],
                       [2.78964844e+02, 6.44121094e+02, 7.50938118e-01],
                       [4.30839844e+02, 7.91777344e+02, 6.66251540e-01],
                       [2.74746094e+02, 7.69628906e+02, 6.61850035e-01],
                       [4.27675781e+02, 9.01464844e+02, 7.12419152e-01],
                       [2.64199219e+02, 8.78261719e+02, 5.05439758e-01],
                       [3.79160156e+02, 9.00410156e+02, 5.05289793e-01],
                       [2.92675781e+02, 8.97246094e+02, 5.15974581e-01],
                       [3.82324219e+02, 1.07232422e+03, 6.11415625e-01],
                       [3.12714844e+02, 1.06599609e+03, 6.26912475e-01],
                       [3.92871094e+02, 1.24740234e+03, 6.09808922e-01],
                       [3.43300781e+02, 1.23263672e+03, 6.07394457e-01],
                       [3.35917969e+02, 8.98828125e+02, 2.60716689e-01],
                       [3.47519531e+02, 6.49921875e+02, 4.83715773e-01]])}


{'ann_id': 1, 'img_path': '../data/PW3D/data/imageFiles/courtyard_hug_00/image_00000.jpg', 'img_shape': (1920, 1080), 'bbox': array([ -10.036423,  245.35367 , 1370.0154  , 1370.0154  ], dtype=float32), 'tight_bbox': array([ 521.8819 ,  382.3552 ,  307.17874, 1097.0123 ], dtype=float32), 'smpl_param': {'shape': [-0.004723833408206701, -0.002423676662147045, -0.00831096526235342, 0.00782056525349617, 0.045562613755464554, -0.06737242639064789, -0.04167397320270538, -0.018426623195409775, -0.027335019782185555, 0.00555424764752388], 'pose': [2.9028878211975098, 0.16799458861351013, -0.8112385869026184, 0.40913110971450806, 0.19835568964481354, 0.0036739937495440245, 0.39756980538368225, -0.10714941471815109, -0.015975769609212875, 0.1609826385974884, -0.01945224590599537, 0.01819852739572525, 0.005878095515072346, 0.025371888652443886, 0.0316896066069603, 0.024100802838802338, -0.06779655069112778, -0.06255612522363663, 0.1620536595582962, -0.011207498610019684, -0.006707859691232443, -0.0012546939542517066, -0.00031642301473766565, -0.003094764892011881, -7.275662937900051e-05, -0.006446676794439554, 0.018738966435194016, 0.05708162114024162, -0.008427012711763382, -0.007939046248793602, 0.01041944045573473, 0.004709208384156227, 0.010014984756708145, -0.008094320073723793, 0.0359039232134819, -0.0375797338783741, 0.051074303686618805, 0.16111576557159424, -0.003222576342523098, 0.10844939202070236, -0.30555298924446106, -0.23313948512077332, 0.020630905404686928, 0.22054332494735718, 0.17337018251419067, 0.13008414208889008, 0.09226376563310623, 0.05706222355365753, 0.07718143612146378, -0.16870859265327454, -1.1182408332824707, -0.012317612767219543, 0.2824184000492096, 1.265819787979126, -0.5238991975784302, -0.1626581996679306, -0.25562211871147156, -0.1316584348678589, 0.2678449749946594, 0.1213688850402832, 0.05897695943713188, -0.0967855453491211, 0.20978476107120514, -0.02406904846429825, 0.06468063592910767, -0.14646147191524506, -0.20514197647571564, -0.036338597536087036, -0.18421059846878052, -0.13453218340873718, 0.10293830931186676, 0.20540176331996918], 'trans': [0.194703169617251, -0.0024393532345551583, 3.3020711865733787], 'gender': 'male'}, 'cam_param': {'focal': array([1961.8529, 1969.2307], dtype=float32), 'princpt': array([540., 960.], dtype=float32)}, 'root_joint_depth': None, 'pose_score_thr': 0.05,
'openpose':     array([[6.3486603e+02, 4.3856100e+02, 9.0925002e-01],
                       [6.7646503e+02, 5.5816901e+02, 9.1025102e-01],
                       [5.7234399e+02, 5.5815601e+02, 8.8398701e-01],
                       [5.5656299e+02, 7.2524799e+02, 8.4706801e-01],
                       [5.5654602e+02, 8.7649103e+02, 7.9782200e-01],
                       [7.6509601e+02, 5.5811798e+02, 9.1666901e-01],
                       [8.0654102e+02, 7.3043597e+02, 9.1313499e-01],
                       [7.7550500e+02, 8.9211700e+02, 8.8914001e-01],
                       [5.9832599e+02, 8.9225098e+02, 8.0990499e-01],
                       [6.1927197e+02, 1.1114000e+03, 8.6155403e-01],
                       [6.3972198e+02, 1.2991400e+03, 8.5377002e-01],
                       [7.1801300e+02, 8.9217902e+02, 8.0217302e-01],
                       [7.1805402e+02, 1.1166400e+03, 8.4371698e-01],
                       [7.1807098e+02, 1.3149600e+03, 8.8884503e-01],
                       [6.1391400e+02, 4.2783801e+02, 1.7972720e-01],
                       [6.5046997e+02, 4.2779700e+02, 1.8081920e-01],
                       [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
                       [7.1787799e+02, 4.2789899e+02, 1.7900360e-01],
                       [6.5816949e+02, 8.9221497e+02, 6.4968395e-01]], dtype=float32),
'hhrnetpose':   array([[6.36503906e+02, 4.49003906e+02, 8.07437360e-01],
                       [6.58652344e+02, 4.27910156e+02, 8.20578098e-01],
                       [6.21738281e+02, 4.26855469e+02, 8.52737188e-01],
                       [7.14550781e+02, 4.28964844e+02, 7.88198709e-01],
                       [6.06972656e+02, 4.37402344e+02, 1.32608861e-02],
                       [7.75722656e+02, 5.58691406e+02, 5.50836504e-01],
                       [5.75332031e+02, 5.53417969e+02, 6.12422705e-01],
                       [8.04199219e+02, 7.35878906e+02, 5.54246068e-01],
                       [5.54238281e+02, 7.15839844e+02, 5.43373644e-01],
                       [7.90488281e+02, 8.94082031e+02, 6.32665157e-01],
                       [5.57402344e+02, 8.68769531e+02, 4.61777747e-01],
                       [7.17714844e+02, 8.91972656e+02, 3.69385839e-01],
                       [5.91152344e+02, 8.83535156e+02, 3.51835668e-01],
                       [7.14550781e+02, 1.12294922e+03, 5.60445786e-01],
                       [6.23847656e+02, 1.11240234e+03, 5.92104256e-01],
                       [7.14550781e+02, 1.32017578e+03, 6.06627941e-01],
                       [6.37558594e+02, 1.30119141e+03, 5.89775145e-01],
                       [6.54433594e+02, 8.87753906e+02, 1.29963113e-01],
                       [6.75527344e+02, 5.56054688e+02, 3.37344781e-01]])}

"""