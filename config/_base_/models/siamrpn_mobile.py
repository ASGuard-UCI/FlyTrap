# Model settings

cfg = dict(
    CKPT='ckpts/siamrpn_mobilev2_l234_dwxcorr.pth',
    META_ARC='siamrpn_mobilev2_l234_dwxcorr',
    BACKBONE=dict(
        TYPE='mobilenetv2',
        KWARGS=dict(
            used_layers=[3, 5, 7],
            width_mult=1.4
        )
    ),
    ADJUST=dict(
        ADJUST=True,
        TYPE='AdjustAllLayer',
        KWARGS=dict(
            in_channels=[44, 134, 448],
            out_channels=[256, 256, 256]
        )
    ),
    RPN=dict(
        TYPE='MultiRPN',
        KWARGS=dict(
            anchor_num=5,
            in_channels=[256, 256, 256],
            weighted=False
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
    TRACK=dict(
        TYPE='SiamRPNTracker',
        PENALTY_K=0.04,
        WINDOW_INFLUENCE=0.4,
        LR=0.5,
        EXEMPLAR_SIZE=127,
        INSTANCE_SIZE=255,
        BASE_SIZE=8,
        CONTEXT_AMOUNT=0.5
    )
)

model = dict(
    type='ModelBuilder',
    cfg=cfg)

tracker = dict(
    type='SiamRPNTracker',
    cfg=model
)
