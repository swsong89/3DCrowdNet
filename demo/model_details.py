"""
ResNetBackbone(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  上面的属于early-stage


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
)"""

"""
Pose2Feat(
  (conv): Sequential(
    (0): Conv2d(94, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
  )
)"""

"""RotationNet(
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
)"""

"""Vposer(
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
  (human_model_layer): SMPL_Layer()
  (coord_loss): CoordLoss()
  (param_loss): ParamLoss()
)"""


"""
torchsummary
1-5      ResNetBackbone-5       输入 input_img [1, 3, 256, 256]   输出 early_img_feat [1, 64, 64, 64] early-stage对input_img特征提取
6-9      Pose2Feat-9            输入 early_img_feat [1, 64, 64, 64], joint_heatmap [1, 30, 64, 64],  输出 pose_img_feat [1, 64, 64, 64]
10-178   ResNetBackbone-178     输入 pose_img_feat  [1, 64, 64, 64],   输出 pose_guided_img_feat [1, 2048, 8, 8] ,   进行resnet后面提取得到  pose_guided_img_feat
179-180  PositionNet-180        输入 pose_guided_img_feat [1, 2048, 8, 8],   输出[[-1, 15, 3], [-1, 15, 1]]得到P3d 和置信度
181-468  RotationNet-468        输入 pose_guided_img_feat [1, 2048, 8, 8], P3D [1, 15,3] 置信度 [1,15,1], 输出 root_pose_6d [[-1, 6], z[-1, 32], shape_param[-1, 10], cam_param[-1, 3]]               0
469-474  Vposer-474             输入 z[-1, 32],  输出 pose_param [1, 69], 利用vposer的decoder部分将z转成pose_aram, [1,69] -> [1,23,3]
475      SMPL_Layer-475         输入pose_param [1, 23, 3], shape_param [1, 10], cam_trans [1,3],  输出 mesh_cam [[-1, 6890, 3], joint_cam [-1, 24, 3]]
                                
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [1, 64, 128, 128]           9,408
       BatchNorm2d-2          [1, 64, 128, 128]             128
              ReLU-3          [1, 64, 128, 128]               0
         MaxPool2d-4            [1, 64, 64, 64]               0
    ResNetBackbone-5            [1, 64, 64, 64]               0
            Conv2d-6            [1, 64, 64, 64]          54,208
       BatchNorm2d-7            [1, 64, 64, 64]             128
              ReLU-8            [1, 64, 64, 64]               0
         Pose2Feat-9            [1, 64, 64, 64]               0
           Conv2d-10            [1, 64, 64, 64]           4,096
      BatchNorm2d-11            [1, 64, 64, 64]             128
             ReLU-12            [1, 64, 64, 64]               0
           Conv2d-13            [1, 64, 64, 64]          36,864
      BatchNorm2d-14            [1, 64, 64, 64]             128
             ReLU-15            [1, 64, 64, 64]               0
           Conv2d-16           [1, 256, 64, 64]          16,384
      BatchNorm2d-17           [1, 256, 64, 64]             512
           Conv2d-18           [1, 256, 64, 64]          16,384
      BatchNorm2d-19           [1, 256, 64, 64]             512
             ReLU-20           [1, 256, 64, 64]               0
       Bottleneck-21           [1, 256, 64, 64]               0
           Conv2d-22            [1, 64, 64, 64]          16,384
      BatchNorm2d-23            [1, 64, 64, 64]             128
             ReLU-24            [1, 64, 64, 64]               0
           Conv2d-25            [1, 64, 64, 64]          36,864
      BatchNorm2d-26            [1, 64, 64, 64]             128
             ReLU-27            [1, 64, 64, 64]               0
           Conv2d-28           [1, 256, 64, 64]          16,384
      BatchNorm2d-29           [1, 256, 64, 64]             512
             ReLU-30           [1, 256, 64, 64]               0
       Bottleneck-31           [1, 256, 64, 64]               0
           Conv2d-32            [1, 64, 64, 64]          16,384
      BatchNorm2d-33            [1, 64, 64, 64]             128
             ReLU-34            [1, 64, 64, 64]               0
           Conv2d-35            [1, 64, 64, 64]          36,864
      BatchNorm2d-36            [1, 64, 64, 64]             128
             ReLU-37            [1, 64, 64, 64]               0
           Conv2d-38           [1, 256, 64, 64]          16,384
      BatchNorm2d-39           [1, 256, 64, 64]             512
             ReLU-40           [1, 256, 64, 64]               0
       Bottleneck-41           [1, 256, 64, 64]               0
           Conv2d-42           [1, 128, 64, 64]          32,768
      BatchNorm2d-43           [1, 128, 64, 64]             256
             ReLU-44           [1, 128, 64, 64]               0
           Conv2d-45           [1, 128, 32, 32]         147,456
      BatchNorm2d-46           [1, 128, 32, 32]             256
             ReLU-47           [1, 128, 32, 32]               0
           Conv2d-48           [1, 512, 32, 32]          65,536
      BatchNorm2d-49           [1, 512, 32, 32]           1,024
           Conv2d-50           [1, 512, 32, 32]         131,072
      BatchNorm2d-51           [1, 512, 32, 32]           1,024
             ReLU-52           [1, 512, 32, 32]               0
       Bottleneck-53           [1, 512, 32, 32]               0
           Conv2d-54           [1, 128, 32, 32]          65,536
      BatchNorm2d-55           [1, 128, 32, 32]             256
             ReLU-56           [1, 128, 32, 32]               0
           Conv2d-57           [1, 128, 32, 32]         147,456
      BatchNorm2d-58           [1, 128, 32, 32]             256
             ReLU-59           [1, 128, 32, 32]               0
           Conv2d-60           [1, 512, 32, 32]          65,536
      BatchNorm2d-61           [1, 512, 32, 32]           1,024
             ReLU-62           [1, 512, 32, 32]               0
       Bottleneck-63           [1, 512, 32, 32]               0
           Conv2d-64           [1, 128, 32, 32]          65,536
      BatchNorm2d-65           [1, 128, 32, 32]             256
             ReLU-66           [1, 128, 32, 32]               0
           Conv2d-67           [1, 128, 32, 32]         147,456
      BatchNorm2d-68           [1, 128, 32, 32]             256
             ReLU-69           [1, 128, 32, 32]               0
           Conv2d-70           [1, 512, 32, 32]          65,536
      BatchNorm2d-71           [1, 512, 32, 32]           1,024
             ReLU-72           [1, 512, 32, 32]               0
       Bottleneck-73           [1, 512, 32, 32]               0
           Conv2d-74           [1, 128, 32, 32]          65,536
      BatchNorm2d-75           [1, 128, 32, 32]             256
             ReLU-76           [1, 128, 32, 32]               0
           Conv2d-77           [1, 128, 32, 32]         147,456
      BatchNorm2d-78           [1, 128, 32, 32]             256
             ReLU-79           [1, 128, 32, 32]               0
           Conv2d-80           [1, 512, 32, 32]          65,536
      BatchNorm2d-81           [1, 512, 32, 32]           1,024
             ReLU-82           [1, 512, 32, 32]               0
       Bottleneck-83           [1, 512, 32, 32]               0
           Conv2d-84           [1, 256, 32, 32]         131,072
      BatchNorm2d-85           [1, 256, 32, 32]             512
             ReLU-86           [1, 256, 32, 32]               0
           Conv2d-87           [1, 256, 16, 16]         589,824
      BatchNorm2d-88           [1, 256, 16, 16]             512
             ReLU-89           [1, 256, 16, 16]               0
           Conv2d-90          [1, 1024, 16, 16]         262,144
      BatchNorm2d-91          [1, 1024, 16, 16]           2,048
           Conv2d-92          [1, 1024, 16, 16]         524,288
      BatchNorm2d-93          [1, 1024, 16, 16]           2,048
             ReLU-94          [1, 1024, 16, 16]               0
       Bottleneck-95          [1, 1024, 16, 16]               0
           Conv2d-96           [1, 256, 16, 16]         262,144
      BatchNorm2d-97           [1, 256, 16, 16]             512
             ReLU-98           [1, 256, 16, 16]               0
           Conv2d-99           [1, 256, 16, 16]         589,824
     BatchNorm2d-100           [1, 256, 16, 16]             512
            ReLU-101           [1, 256, 16, 16]               0
          Conv2d-102          [1, 1024, 16, 16]         262,144
     BatchNorm2d-103          [1, 1024, 16, 16]           2,048
            ReLU-104          [1, 1024, 16, 16]               0
      Bottleneck-105          [1, 1024, 16, 16]               0
          Conv2d-106           [1, 256, 16, 16]         262,144
     BatchNorm2d-107           [1, 256, 16, 16]             512
            ReLU-108           [1, 256, 16, 16]               0
          Conv2d-109           [1, 256, 16, 16]         589,824
     BatchNorm2d-110           [1, 256, 16, 16]             512
            ReLU-111           [1, 256, 16, 16]               0
          Conv2d-112          [1, 1024, 16, 16]         262,144
     BatchNorm2d-113          [1, 1024, 16, 16]           2,048
            ReLU-114          [1, 1024, 16, 16]               0
      Bottleneck-115          [1, 1024, 16, 16]               0
          Conv2d-116           [1, 256, 16, 16]         262,144
     BatchNorm2d-117           [1, 256, 16, 16]             512
            ReLU-118           [1, 256, 16, 16]               0
          Conv2d-119           [1, 256, 16, 16]         589,824
     BatchNorm2d-120           [1, 256, 16, 16]             512
            ReLU-121           [1, 256, 16, 16]               0
          Conv2d-122          [1, 1024, 16, 16]         262,144
     BatchNorm2d-123          [1, 1024, 16, 16]           2,048
            ReLU-124          [1, 1024, 16, 16]               0
      Bottleneck-125          [1, 1024, 16, 16]               0
          Conv2d-126           [1, 256, 16, 16]         262,144
     BatchNorm2d-127           [1, 256, 16, 16]             512
            ReLU-128           [1, 256, 16, 16]               0
          Conv2d-129           [1, 256, 16, 16]         589,824
     BatchNorm2d-130           [1, 256, 16, 16]             512
            ReLU-131           [1, 256, 16, 16]               0
          Conv2d-132          [1, 1024, 16, 16]         262,144
     BatchNorm2d-133          [1, 1024, 16, 16]           2,048
            ReLU-134          [1, 1024, 16, 16]               0
      Bottleneck-135          [1, 1024, 16, 16]               0
          Conv2d-136           [1, 256, 16, 16]         262,144
     BatchNorm2d-137           [1, 256, 16, 16]             512
            ReLU-138           [1, 256, 16, 16]               0
          Conv2d-139           [1, 256, 16, 16]         589,824
     BatchNorm2d-140           [1, 256, 16, 16]             512
            ReLU-141           [1, 256, 16, 16]               0
          Conv2d-142          [1, 1024, 16, 16]         262,144
     BatchNorm2d-143          [1, 1024, 16, 16]           2,048
            ReLU-144          [1, 1024, 16, 16]               0
      Bottleneck-145          [1, 1024, 16, 16]               0
          Conv2d-146           [1, 512, 16, 16]         524,288
     BatchNorm2d-147           [1, 512, 16, 16]           1,024
            ReLU-148           [1, 512, 16, 16]               0
          Conv2d-149             [1, 512, 8, 8]       2,359,296
     BatchNorm2d-150             [1, 512, 8, 8]           1,024
            ReLU-151             [1, 512, 8, 8]               0
          Conv2d-152            [1, 2048, 8, 8]       1,048,576
     BatchNorm2d-153            [1, 2048, 8, 8]           4,096
          Conv2d-154            [1, 2048, 8, 8]       2,097,152
     BatchNorm2d-155            [1, 2048, 8, 8]           4,096
            ReLU-156            [1, 2048, 8, 8]               0
      Bottleneck-157            [1, 2048, 8, 8]               0
          Conv2d-158             [1, 512, 8, 8]       1,048,576
     BatchNorm2d-159             [1, 512, 8, 8]           1,024
            ReLU-160             [1, 512, 8, 8]               0
          Conv2d-161             [1, 512, 8, 8]       2,359,296
     BatchNorm2d-162             [1, 512, 8, 8]           1,024
            ReLU-163             [1, 512, 8, 8]               0
          Conv2d-164            [1, 2048, 8, 8]       1,048,576
     BatchNorm2d-165            [1, 2048, 8, 8]           4,096
            ReLU-166            [1, 2048, 8, 8]               0
      Bottleneck-167            [1, 2048, 8, 8]               0
          Conv2d-168             [1, 512, 8, 8]       1,048,576
     BatchNorm2d-169             [1, 512, 8, 8]           1,024
            ReLU-170             [1, 512, 8, 8]               0
          Conv2d-171             [1, 512, 8, 8]       2,359,296
     BatchNorm2d-172             [1, 512, 8, 8]           1,024
            ReLU-173             [1, 512, 8, 8]               0
          Conv2d-174            [1, 2048, 8, 8]       1,048,576
     BatchNorm2d-175            [1, 2048, 8, 8]           4,096
            ReLU-176            [1, 2048, 8, 8]               0
      Bottleneck-177            [1, 2048, 8, 8]               0
  ResNetBackbone-178            [1, 2048, 8, 8]               0
          Conv2d-179             [1, 120, 8, 8]         245,880
     PositionNet-180  [[-1, 15, 3], [-1, 15, 1]]               0
          Linear-181                   [1, 128]         262,784
     BatchNorm1d-182                   [1, 128]             256
          Linear-183                   [1, 128]         262,784
     BatchNorm1d-184                   [1, 128]             256
          Linear-185                   [1, 128]         262,784
     BatchNorm1d-186                   [1, 128]             256
          Linear-187                   [1, 128]         262,784
     BatchNorm1d-188                   [1, 128]             256
          Linear-189                   [1, 128]         262,784
     BatchNorm1d-190                   [1, 128]             256
          Linear-191                   [1, 128]         262,784
     BatchNorm1d-192                   [1, 128]             256
          Linear-193                   [1, 128]         262,784
     BatchNorm1d-194                   [1, 128]             256
          Linear-195                   [1, 128]         262,784
     BatchNorm1d-196                   [1, 128]             256
          Linear-197                   [1, 128]         262,784
     BatchNorm1d-198                   [1, 128]             256
          Linear-199                   [1, 128]         262,784
     BatchNorm1d-200                   [1, 128]             256
          Linear-201                   [1, 128]         262,784
     BatchNorm1d-202                   [1, 128]             256
          Linear-203                   [1, 128]         262,784
     BatchNorm1d-204                   [1, 128]             256
          Linear-205                   [1, 128]         262,784
     BatchNorm1d-206                   [1, 128]             256
          Linear-207                   [1, 128]         262,784
     BatchNorm1d-208                   [1, 128]             256
          Linear-209                   [1, 128]         262,784
     BatchNorm1d-210                   [1, 128]             256
  GraphConvBlock-211               [1, 15, 128]               0
          Linear-212                   [1, 128]          16,512
     BatchNorm1d-213                   [1, 128]             256
          Linear-214                   [1, 128]          16,512
     BatchNorm1d-215                   [1, 128]             256
          Linear-216                   [1, 128]          16,512
     BatchNorm1d-217                   [1, 128]             256
          Linear-218                   [1, 128]          16,512
     BatchNorm1d-219                   [1, 128]             256
          Linear-220                   [1, 128]          16,512
     BatchNorm1d-221                   [1, 128]             256
          Linear-222                   [1, 128]          16,512
     BatchNorm1d-223                   [1, 128]             256
          Linear-224                   [1, 128]          16,512
     BatchNorm1d-225                   [1, 128]             256
          Linear-226                   [1, 128]          16,512
     BatchNorm1d-227                   [1, 128]             256
          Linear-228                   [1, 128]          16,512
     BatchNorm1d-229                   [1, 128]             256
          Linear-230                   [1, 128]          16,512
     BatchNorm1d-231                   [1, 128]             256
          Linear-232                   [1, 128]          16,512
     BatchNorm1d-233                   [1, 128]             256
          Linear-234                   [1, 128]          16,512
     BatchNorm1d-235                   [1, 128]             256
          Linear-236                   [1, 128]          16,512
     BatchNorm1d-237                   [1, 128]             256
          Linear-238                   [1, 128]          16,512
     BatchNorm1d-239                   [1, 128]             256
          Linear-240                   [1, 128]          16,512
     BatchNorm1d-241                   [1, 128]             256
  GraphConvBlock-242               [1, 15, 128]               0
          Linear-243                   [1, 128]          16,512
     BatchNorm1d-244                   [1, 128]             256
          Linear-245                   [1, 128]          16,512
     BatchNorm1d-246                   [1, 128]             256
          Linear-247                   [1, 128]          16,512
     BatchNorm1d-248                   [1, 128]             256
          Linear-249                   [1, 128]          16,512
     BatchNorm1d-250                   [1, 128]             256
          Linear-251                   [1, 128]          16,512
     BatchNorm1d-252                   [1, 128]             256
          Linear-253                   [1, 128]          16,512
     BatchNorm1d-254                   [1, 128]             256
          Linear-255                   [1, 128]          16,512
     BatchNorm1d-256                   [1, 128]             256
          Linear-257                   [1, 128]          16,512
     BatchNorm1d-258                   [1, 128]             256
          Linear-259                   [1, 128]          16,512
     BatchNorm1d-260                   [1, 128]             256
          Linear-261                   [1, 128]          16,512
     BatchNorm1d-262                   [1, 128]             256
          Linear-263                   [1, 128]          16,512
     BatchNorm1d-264                   [1, 128]             256
          Linear-265                   [1, 128]          16,512
     BatchNorm1d-266                   [1, 128]             256
          Linear-267                   [1, 128]          16,512
     BatchNorm1d-268                   [1, 128]             256
          Linear-269                   [1, 128]          16,512
     BatchNorm1d-270                   [1, 128]             256
          Linear-271                   [1, 128]          16,512
     BatchNorm1d-272                   [1, 128]             256
  GraphConvBlock-273               [1, 15, 128]               0
   GraphResBlock-274               [1, 15, 128]               0
          Linear-275                   [1, 128]          16,512
     BatchNorm1d-276                   [1, 128]             256
          Linear-277                   [1, 128]          16,512
     BatchNorm1d-278                   [1, 128]             256
          Linear-279                   [1, 128]          16,512
     BatchNorm1d-280                   [1, 128]             256
          Linear-281                   [1, 128]          16,512
     BatchNorm1d-282                   [1, 128]             256
          Linear-283                   [1, 128]          16,512
     BatchNorm1d-284                   [1, 128]             256
          Linear-285                   [1, 128]          16,512
     BatchNorm1d-286                   [1, 128]             256
          Linear-287                   [1, 128]          16,512
     BatchNorm1d-288                   [1, 128]             256
          Linear-289                   [1, 128]          16,512
     BatchNorm1d-290                   [1, 128]             256
          Linear-291                   [1, 128]          16,512
     BatchNorm1d-292                   [1, 128]             256
          Linear-293                   [1, 128]          16,512
     BatchNorm1d-294                   [1, 128]             256
          Linear-295                   [1, 128]          16,512
     BatchNorm1d-296                   [1, 128]             256
          Linear-297                   [1, 128]          16,512
     BatchNorm1d-298                   [1, 128]             256
          Linear-299                   [1, 128]          16,512
     BatchNorm1d-300                   [1, 128]             256
          Linear-301                   [1, 128]          16,512
     BatchNorm1d-302                   [1, 128]             256
          Linear-303                   [1, 128]          16,512
     BatchNorm1d-304                   [1, 128]             256
  GraphConvBlock-305               [1, 15, 128]               0
          Linear-306                   [1, 128]          16,512
     BatchNorm1d-307                   [1, 128]             256
          Linear-308                   [1, 128]          16,512
     BatchNorm1d-309                   [1, 128]             256
          Linear-310                   [1, 128]          16,512
     BatchNorm1d-311                   [1, 128]             256
          Linear-312                   [1, 128]          16,512
     BatchNorm1d-313                   [1, 128]             256
          Linear-314                   [1, 128]          16,512
     BatchNorm1d-315                   [1, 128]             256
          Linear-316                   [1, 128]          16,512
     BatchNorm1d-317                   [1, 128]             256
          Linear-318                   [1, 128]          16,512
     BatchNorm1d-319                   [1, 128]             256
          Linear-320                   [1, 128]          16,512
     BatchNorm1d-321                   [1, 128]             256
          Linear-322                   [1, 128]          16,512
     BatchNorm1d-323                   [1, 128]             256
          Linear-324                   [1, 128]          16,512
     BatchNorm1d-325                   [1, 128]             256
          Linear-326                   [1, 128]          16,512
     BatchNorm1d-327                   [1, 128]             256
          Linear-328                   [1, 128]          16,512
     BatchNorm1d-329                   [1, 128]             256
          Linear-330                   [1, 128]          16,512
     BatchNorm1d-331                   [1, 128]             256
          Linear-332                   [1, 128]          16,512
     BatchNorm1d-333                   [1, 128]             256
          Linear-334                   [1, 128]          16,512
     BatchNorm1d-335                   [1, 128]             256
  GraphConvBlock-336               [1, 15, 128]               0
   GraphResBlock-337               [1, 15, 128]               0
          Linear-338                   [1, 128]          16,512
     BatchNorm1d-339                   [1, 128]             256
          Linear-340                   [1, 128]          16,512
     BatchNorm1d-341                   [1, 128]             256
          Linear-342                   [1, 128]          16,512
     BatchNorm1d-343                   [1, 128]             256
          Linear-344                   [1, 128]          16,512
     BatchNorm1d-345                   [1, 128]             256
          Linear-346                   [1, 128]          16,512
     BatchNorm1d-347                   [1, 128]             256
          Linear-348                   [1, 128]          16,512
     BatchNorm1d-349                   [1, 128]             256
          Linear-350                   [1, 128]          16,512
     BatchNorm1d-351                   [1, 128]             256
          Linear-352                   [1, 128]          16,512
     BatchNorm1d-353                   [1, 128]             256
          Linear-354                   [1, 128]          16,512
     BatchNorm1d-355                   [1, 128]             256
          Linear-356                   [1, 128]          16,512
     BatchNorm1d-357                   [1, 128]             256
          Linear-358                   [1, 128]          16,512
     BatchNorm1d-359                   [1, 128]             256
          Linear-360                   [1, 128]          16,512
     BatchNorm1d-361                   [1, 128]             256
          Linear-362                   [1, 128]          16,512
     BatchNorm1d-363                   [1, 128]             256
          Linear-364                   [1, 128]          16,512
     BatchNorm1d-365                   [1, 128]             256
          Linear-366                   [1, 128]          16,512
     BatchNorm1d-367                   [1, 128]             256
  GraphConvBlock-368               [1, 15, 128]               0
          Linear-369                   [1, 128]          16,512
     BatchNorm1d-370                   [1, 128]             256
          Linear-371                   [1, 128]          16,512
     BatchNorm1d-372                   [1, 128]             256
          Linear-373                   [1, 128]          16,512
     BatchNorm1d-374                   [1, 128]             256
          Linear-375                   [1, 128]          16,512
     BatchNorm1d-376                   [1, 128]             256
          Linear-377                   [1, 128]          16,512
     BatchNorm1d-378                   [1, 128]             256
          Linear-379                   [1, 128]          16,512
     BatchNorm1d-380                   [1, 128]             256
          Linear-381                   [1, 128]          16,512
     BatchNorm1d-382                   [1, 128]             256
          Linear-383                   [1, 128]          16,512
     BatchNorm1d-384                   [1, 128]             256
          Linear-385                   [1, 128]          16,512
     BatchNorm1d-386                   [1, 128]             256
          Linear-387                   [1, 128]          16,512
     BatchNorm1d-388                   [1, 128]             256
          Linear-389                   [1, 128]          16,512
     BatchNorm1d-390                   [1, 128]             256
          Linear-391                   [1, 128]          16,512
     BatchNorm1d-392                   [1, 128]             256
          Linear-393                   [1, 128]          16,512
     BatchNorm1d-394                   [1, 128]             256
          Linear-395                   [1, 128]          16,512
     BatchNorm1d-396                   [1, 128]             256
          Linear-397                   [1, 128]          16,512
     BatchNorm1d-398                   [1, 128]             256
  GraphConvBlock-399               [1, 15, 128]               0
   GraphResBlock-400               [1, 15, 128]               0
          Linear-401                   [1, 128]          16,512
     BatchNorm1d-402                   [1, 128]             256
          Linear-403                   [1, 128]          16,512
     BatchNorm1d-404                   [1, 128]             256
          Linear-405                   [1, 128]          16,512
     BatchNorm1d-406                   [1, 128]             256
          Linear-407                   [1, 128]          16,512
     BatchNorm1d-408                   [1, 128]             256
          Linear-409                   [1, 128]          16,512
     BatchNorm1d-410                   [1, 128]             256
          Linear-411                   [1, 128]          16,512
     BatchNorm1d-412                   [1, 128]             256
          Linear-413                   [1, 128]          16,512
     BatchNorm1d-414                   [1, 128]             256
          Linear-415                   [1, 128]          16,512
     BatchNorm1d-416                   [1, 128]             256
          Linear-417                   [1, 128]          16,512
     BatchNorm1d-418                   [1, 128]             256
          Linear-419                   [1, 128]          16,512
     BatchNorm1d-420                   [1, 128]             256
          Linear-421                   [1, 128]          16,512
     BatchNorm1d-422                   [1, 128]             256
          Linear-423                   [1, 128]          16,512
     BatchNorm1d-424                   [1, 128]             256
          Linear-425                   [1, 128]          16,512
     BatchNorm1d-426                   [1, 128]             256
          Linear-427                   [1, 128]          16,512
     BatchNorm1d-428                   [1, 128]             256
          Linear-429                   [1, 128]          16,512
     BatchNorm1d-430                   [1, 128]             256
  GraphConvBlock-431               [1, 15, 128]               0
          Linear-432                   [1, 128]          16,512
     BatchNorm1d-433                   [1, 128]             256
          Linear-434                   [1, 128]          16,512
     BatchNorm1d-435                   [1, 128]             256
          Linear-436                   [1, 128]          16,512
     BatchNorm1d-437                   [1, 128]             256
          Linear-438                   [1, 128]          16,512
     BatchNorm1d-439                   [1, 128]             256
          Linear-440                   [1, 128]          16,512
     BatchNorm1d-441                   [1, 128]             256
          Linear-442                   [1, 128]          16,512
     BatchNorm1d-443                   [1, 128]             256
          Linear-444                   [1, 128]          16,512
     BatchNorm1d-445                   [1, 128]             256
          Linear-446                   [1, 128]          16,512
     BatchNorm1d-447                   [1, 128]             256
          Linear-448                   [1, 128]          16,512
     BatchNorm1d-449                   [1, 128]             256
          Linear-450                   [1, 128]          16,512
     BatchNorm1d-451                   [1, 128]             256
          Linear-452                   [1, 128]          16,512
     BatchNorm1d-453                   [1, 128]             256
          Linear-454                   [1, 128]          16,512
     BatchNorm1d-455                   [1, 128]             256
          Linear-456                   [1, 128]          16,512
     BatchNorm1d-457                   [1, 128]             256
          Linear-458                   [1, 128]          16,512
     BatchNorm1d-459                   [1, 128]             256
          Linear-460                   [1, 128]          16,512
     BatchNorm1d-461                   [1, 128]             256
  GraphConvBlock-462               [1, 15, 128]               0
   GraphResBlock-463               [1, 15, 128]               0
          Linear-464                     [1, 6]          11,526
          Linear-465                    [1, 32]          61,472
          Linear-466                    [1, 10]          19,210
          Linear-467                     [1, 3]           5,763
     RotationNet-468  [[-1, 6], [-1, 32], [-1, 10], [-1, 3]]               0
          Linear-469                   [1, 512]          16,896
         Dropout-470                   [1, 512]               0
          Linear-471                   [1, 512]         262,656
          Linear-472                   [1, 126]          64,638
ContinousRotReprDecoder-473                  [1, 3, 3]               0
          Vposer-474                    [1, 69]               0
      SMPL_Layer-475  [[-1, 6890, 3], [-1, 24, 3]]               0
================================================================
Total params: 30,208,169
Trainable params: 30,208,169
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.75
Forward/backward pass size (MB): 397.18
Params size (MB): 115.24
Estimated Total Size (MB): 513.17
----------------------------------------------------------------


"""