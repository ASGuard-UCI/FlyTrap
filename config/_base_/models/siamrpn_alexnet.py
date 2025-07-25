# Model settings

cfg = dict(
    CKPT='ckpts/siamrpn_alex_dwxcorr.pth',
    META_ARC="siamrpn_alex_dwxcorr",
    BACKBONE=dict(
        TYPE="alexnetlegacy",
        KWARGS=dict(
            width_mult=1.0
        )
    ),
    ADJUST=dict(
        ADJUST=False
    ),
    RPN=dict(
        TYPE='DepthwiseRPN',
        KWARGS=dict(
            anchor_num=5,
            in_channels=256,
            out_channels=256
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
        PENALTY_K=0.16,
        WINDOW_INFLUENCE=0.40,
        LR=0.30,
        EXEMPLAR_SIZE=127,
        INSTANCE_SIZE=287,
        BASE_SIZE=0,
        CONTEXT_AMOUNT=0.5
    )
)

model = dict(
    type='ModelBuilder',
    cfg=cfg
)

tracker = dict(
    type='SiamRPNTracker',
    cfg=model
)