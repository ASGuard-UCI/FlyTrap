# Model settings

cfg = dict(
        CKPT='ckpts/siamrpn_r50_l234_dwxcorr.pth',
        META_ARC='siamrpn_r50_l234_dwxcorr',
        BACKBONE=dict(
            TYPE='resnet50',
            KWARGS=dict(
                used_layers=[2, 3, 4]
            )
        ),
        ADJUST=dict(
            ADJUST=True,
            TYPE='AdjustAllLayer',
            KWARGS=dict(
                in_channels=[512, 1024, 2048],
                out_channels=[256, 256, 256]
            )
        ),
        RPN=dict(
            TYPE='MultiRPN',
            KWARGS=dict(
                anchor_num=5,
                in_channels=[256, 256, 256],
                weighted=True
            )
        ),
        MASK=dict(
            MASK=False
        ),
        ANCHOR=dict(
            STRIDE=8,
            RATIOS=[0.33, 0.5, 1, 2, 3],
            SCALES=[8],
            ANCHOR_NUM=5
        ),
        # seems not used
        TRACK=dict(
            TYPE='SiamRPNTracker',
            PENALTY_K=0.05,
            WINDOW_INFLUENCE=0.42,
            LR=0.38,
            EXEMPLAR_SIZE=127,
            INSTANCE_SIZE=255,
            BASE_SIZE=8,
            CONTEXT_AMOUNT=0.5
        ))

model = dict(
    type='ModelBuilder',
    cfg=cfg)

tracker = dict(
    type='SiamRPNTracker',
    cfg=model
)
