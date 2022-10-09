import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='0')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--exp_dir', type=str, default='', help='for resuming train')
    parser.add_argument('--amp', dest='use_mixed_precision', action='store_true', help='use automatic mixed precision training')
    parser.add_argument('--init_scale', type=float, default=1024., help='initial loss scale')
    parser.add_argument('--cfg', type=str, default='/home/ssw/code/3DCrowdNet/assets/yaml/3dpw.yml', help='experiment configure file name')
    # parser.add_argument('--cfg', type=str, default='/data2/2020/ssw/3DCrowdNet/assets/yaml/3dpw.yml',help='experiment configure file name')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"
 
    if '-' in args.gpu_ids:   # 0-3, 使用3块gpu训练,0 ,1, 2
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args


def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.gpu_ids, args.continue_train, exp_dir=args.exp_dir)  # 先设置gpu_id和是否继续训练以及exp_dir
    cudnn.benchmark = True
    if args.cfg:
        cfg.update(args.cfg)

    trainer = Trainer()
    trainer._make_batch_generator()  # 加载数据
    trainer._make_model()  # 加载模型

    scaler = amp.GradScaler(init_scale=args.init_scale, enabled=args.use_mixed_precision)

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):  # inputs = {'img': [16,3,256,256], 'joints':[16,30,2], 'joints_mask':[16,30,1]} 虽然30个点，但是coco数据只有17个点有数据，其余都是0，targets={'orig_joint_img', 'fit_joint_img','origin_joint_cam', 'fit_joint_cam', 'pose_param', 'shape_param'} meta_info {'orig_joint_vald': [1,30,1], 'origin_joint_truc':[1,30,1], 'fit_param_valid':[1,72], 'fit_joint_truc':[1,30,1], 'is_valid_fit':[1], 'is_3D':[1]}
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()
            with amp.autocast(args.use_mixed_precision):
                loss = trainer.model(inputs, targets, meta_info, 'train')
                loss = {k: loss[k].mean() for k in loss}  #
                _loss = sum(loss[k] for k in loss)

            # backward
            with amp.autocast(False):
                _loss = scaler.scale(_loss)
                _loss.backward()
                scaler.step(trainer.optimizer)

            scaler.update(args.init_scale)

            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        
        trainer.save_model({
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, epoch)
        

if __name__ == "__main__":
    main()


## model
"""
python train.py --amp --continue --gpu 0 --cfg ../assets/yaml/3dpw_crowd.yml --exp_dir ../output/exp_09-17_11:15
python test.py --gpu 0-3 --cfg ../assets/yaml/3dpw_crowd.yml --exp_dir ../output/exp_03-28_18:26 --test_epoch 6 


python test.py --gpu 2 --cfg ../assets/yaml/3dpw_crowd.yml --exp_dir ../output/exp_03-28_18:26 --test_epoch 6 

"""