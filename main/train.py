import argparse
from config import cfg
import torch
from base import Trainer
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids', default='2')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--exp_dir', type=str, default='', help='for resuming train')
    parser.add_argument('--amp', dest='use_mixed_precision', action='store_true', help='use automatic mixed precision training')
    parser.add_argument('--init_scale', type=float, default=1024., help='initial loss scale')
    parser.add_argument('--cfg', type=str, default='/home/ssw/code/3DCrowdNet/assets/yaml/3dpw.yml', help='experiment configure file name')
    # parser.add_argument('--cfg', type=str, default='/data2/2020/ssw/3DCrowdNet/assets/yaml/3dpw.yml',help='experiment configure file name')
    print('hello')
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
    trainer._make_batch_generator()
    trainer._make_model()

    scaler = amp.GradScaler(init_scale=args.init_scale, enabled=args.use_mixed_precision)

    # train
    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):  # inputs = {'img': [16,3,256,256], 'joints':[16,30,2], 'joints_mask':[16,30,1]} 虽然30个点，但是coco数据只有17个点有数据，其余都是0
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
"""Model(
  (backbone): ResNetBackbone(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (5): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
  )
  (pose2feat): Pose2Feat(
    (conv): Sequential(
      (0): Conv2d(94, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
    )
  )
  (position_net): PositionNet(
    (conv): Sequential(
      (0): Conv2d(2048, 120, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (rotation_net): RotationNet(
    (graph_block): Sequential(
      (0): GraphConvBlock(
        (fcbn_list): ModuleList(
          (0): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (2): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (3): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (4): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (5): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (6): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (7): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (8): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (9): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (10): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (11): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (12): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (13): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (14): Sequential(
            (0): Linear(in_features=2052, out_features=128, bias=True)
            (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (1): GraphResBlock(
        (graph_block1): GraphConvBlock(
          (fcbn_list): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (4): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (5): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (6): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (7): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (8): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (9): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (10): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (11): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (12): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (13): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (14): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (graph_block2): GraphConvBlock(
          (fcbn_list): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (4): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (5): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (6): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (7): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (8): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (9): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (10): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (11): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (12): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (13): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (14): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
      (2): GraphResBlock(
        (graph_block1): GraphConvBlock(
          (fcbn_list): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (4): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (5): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (6): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (7): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (8): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (9): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (10): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (11): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (12): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (13): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (14): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (graph_block2): GraphConvBlock(
          (fcbn_list): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (4): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (5): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (6): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (7): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (8): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (9): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (10): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (11): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (12): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (13): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (14): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
      (3): GraphResBlock(
        (graph_block1): GraphConvBlock(
          (fcbn_list): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (4): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (5): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (6): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (7): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (8): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (9): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (10): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (11): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (12): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (13): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (14): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (graph_block2): GraphConvBlock(
          (fcbn_list): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (4): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (5): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (6): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (7): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (8): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (9): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (10): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (11): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (12): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (13): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (14): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
      (4): GraphResBlock(
        (graph_block1): GraphConvBlock(
          (fcbn_list): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (4): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (5): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (6): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (7): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (8): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (9): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (10): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (11): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (12): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (13): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (14): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
        (graph_block2): GraphConvBlock(
          (fcbn_list): ModuleList(
            (0): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (1): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (2): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (3): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (4): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (5): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (6): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (7): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (8): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (9): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (10): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (11): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (12): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (13): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (14): Sequential(
              (0): Linear(in_features=128, out_features=128, bias=True)
              (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
        )
      )
    )
    (root_pose_out): Sequential(
      (0): Linear(in_features=1920, out_features=6, bias=True)
    )
    (pose_out): Sequential(
      (0): Linear(in_features=1920, out_features=32, bias=True)
    )
    (shape_out): Sequential(
      (0): Linear(in_features=1920, out_features=10, bias=True)
    )
    (cam_out): Sequential(
      (0): Linear(in_features=1920, out_features=3, bias=True)
    )
  )
  (vposer): Vposer(
    (vposer): VPoser(
      (bodyprior_enc_bn1): BatchNorm1d(63, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bodyprior_enc_fc1): Linear(in_features=63, out_features=512, bias=True)
      (bodyprior_enc_bn2): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (bodyprior_enc_fc2): Linear(in_features=512, out_features=512, bias=True)
      (bodyprior_enc_mu): Linear(in_features=512, out_features=32, bias=True)
      (bodyprior_enc_logvar): Linear(in_features=512, out_features=32, bias=True)
      (dropout): Dropout(p=0.1, inplace=False)
      (bodyprior_dec_fc1): Linear(in_features=32, out_features=512, bias=True)
      (bodyprior_dec_fc2): Linear(in_features=512, out_features=512, bias=True)
      (rot_decoder): ContinousRotReprDecoder()
      (bodyprior_dec_out): Linear(in_features=512, out_features=126, bias=True)
    )
  )
  (human_model_layer): SMPL_Layer()
  (coord_loss): CoordLoss()
  (param_loss): ParamLoss()
)"""