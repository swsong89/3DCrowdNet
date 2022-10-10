import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', dest='gpu_ids')

    # test 3dpw_crowd
    # parser.add_argument('--test_epoch', type=str, dest='test_epoch', default='6')
    # parser.add_argument('--exp_dir', type=str, default='/home/ssw/code/3DCrowdNet/output/exp_03-28_18:26')
    # parser.add_argument('--cfg', type=str, default='/home/ssw/code/3DCrowdNet//assets/yaml/3dpw_crowd.yml', help='experiment configure file name')

    # test 3dpw
    parser.add_argument('--test_epoch', type=str, dest='test_epoch', default='10')
    parser.add_argument('--exp_dir', type=str, default='/home/ssw/code/3DCrowdNet/output/exp_04-06_23:43')
    parser.add_argument('--cfg', type=str, default='/home/ssw/code/3DCrowdNet//assets/yaml/3dpw_crowd.yml',
                        help='experiment configure file name')

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids, is_test=True, exp_dir=args.exp_dir)
    cudnn.benchmark = True
    if args.cfg:
        cfg.update(args.cfg)

    tester = Tester(args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    
    eval_result = {}
    cur_sample_idx = 0
    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):  # tester.batch_generator DataLoader 31 batch_size
        # inpuits = {'img': [64,3,256,256], 'joints': [64,30,3], 'joints_mask':[64,30,1]} targets = {'smpl_mesh_cam': [64,6890,3]}
        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'test')
       
        # save output
        out = {k: v.cpu().numpy() for k,v in out.items()}  # 仅仅是将上边的tensor转成ndarray
        for k,v in out.items(): batch_size = out[k].shape[0]  # batch_size 64
        out = [{k: v[bid] for k,v in out.items()} for bid in range(batch_size)]  # 变成 64[14{}],64的列表，每个列表是一帧，一帧里面14字典，

        # evaluate
        cur_eval_result = tester._evaluate(out, cur_sample_idx)
        for k,v in cur_eval_result.items():
            if k in eval_result: eval_result[k] += v
            else: eval_result[k] = v
        cur_sample_idx += len(out)
    
    tester._print_eval_result(eval_result)  # 1923

if __name__ == "__main__":
    main()
