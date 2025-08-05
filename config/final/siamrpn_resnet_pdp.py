# based on physical engine, decrease the evolve_step
# remove detector NDS
# decrease the detector confidence to 0.01

# (v3) fix the overwrite problem in engine_v6_fix_v2
# (v4) add the pose estimation model
# (v5) change the pose regression loss to mse loss

_base_ = [
    '../_base_/metrics/masr.py'
]

debug = False
eval = True

# used for patch generation
img_config = {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]}
# used for model input normalization
model_img_config = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}

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


# color transformation range: [0, 1]
# imporve physical robustness
patch_transform = [
    dict(type='Brightness', factor=0.1),
    dict(type='Contrast', factor=0.1),
    dict(type='Saturation', factor=0.1),
    dict(type='Hue', factor=0.1),
]

# patch training setting
epochs = 80
patch_size = 300
eval_interval = 20
patch_path = 'patches/siamrpn_resnet_flytrap_pdp.png'

# log settings
log = False
log_cfg = dict(
    project='adcover-final',
    entity='shaoyux',
    save_code=True,
    group='siamrpn',
    notes=''
)

optimizer = dict(
    type='Adam',
    params=dict(
        lr=10
    )
)

# dataset settings
img_keys = ['search', 'template', 'online_template']
train_pipeline = [
    dict(type='TemplateSample', same_video=True, online_template=True, num_online=1),
    dict(type='CustomLoadImageFromFile', img_keys=img_keys, to_float32=True, color_type='color'),
    dict(type='CropTargetObjectTemplate', template_factor=2.0, template_size=127),
    dict(type='GaussianNoise', img_keys=img_keys, mean=0, var=5),
    # dict(type='CustomNormalize', img_keys=img_keys, **img_config),
    dict(type='CustomCollect',
         keys=['search', 'template', 'online_template', 'template_bbox', 'umbrella_bbox'],
         meta_keys=['video', 'img_shape_template'])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    shuffle=True,
    drop_last=True,
    collate_fn=dict(type='custom_collate_fn'),
    dataset=dict(
        type='CustomDataset',
        meta_file='data/dataset_v4.0/collect_train_final.json',
        pipeline=train_pipeline
    )
)

scheduler = dict(type='WarmupCosineDecayLR',
                 params=dict(
                     warmup_steps=0,
                     total_steps=epochs,
                     eta_min=0.1
                 ))

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='NormalizeCoordinates'),
    dict(type='CustomCollect',
         keys=['img', 'coords', 'init_bbox', 'apply_attack', 'umbrella_bbox'],
         meta_keys=['img_shape'])
]

test_dataset = dict(
    type='CustomEvalDataset',
    meta_file='./data/dataset_v4.0/collect_eval.json',
    # filter out video with specific name
    sub_string='',
    pipeline=test_pipeline
)

# loss settings
loss = [
    # dict(type='TVLoss', weight=0.000005),
    # should put this before LocalizationLoss
    # LocalizationLoss will change `gt_bbox` value
    dict(type='AlignClassificationLoss', top_k=40, weight=1e-2),
    dict(type='AlignLocalizationLoss', top_k=40, weight=10),
    # dict(type='TargetLocalizationLossDet', weight=1e-2),
    # dict(type='SmoothDynamicClassificationLossDet', weight=1),
    # dict(type='PoseMSELoss', weight=1e-3),
]

engine = dict(
    type='PhysicalEngineV4',
    evolve_step=2,
    interval_distance=4,
    out_size=255,
    render = dict(
        type='Renderer',
        device='cuda',
        mesh_file='data/render_obj/umbrella/umbrella.obj',
        image_size=300,
        camera_args=dict(dist=[55, 60], elev=[-5, 5], azim=[-5, 5]),
        camera_up=((0.3827, -0.9239, 0),),
        lights_args=dict(ambient_color=[0.5, 1.0])
    ),
    applyer = dict(
                type='PatchApplyer',
                rotate_mean=-3.14159 * 3 / 4,
                rotate_var=3.14159 / 9,
                fix_orien=False,
                distribution='normal')
)

# post normalization
# ugly workaround here since apply the patch require range [0, 255]
# convert into model input distribution
post_transform = [
    dict(type='CustomNormalize', img_keys=img_keys, normalize_factor=1.0, **model_img_config),
]


# used for evaluation
applyer = dict(
    type='PatchApplyer',
    rotate_mean=-3.14159 * 3 / 4,
    rotate_var=3.14159 / 9,
    fix_orien=False,
    distribution='normal'
)

# mesh renderer
renderer = dict(
    type='Renderer',
    device='cuda',
    mesh_file='data/render_obj/umbrella/umbrella.obj',
    image_size=300,
    camera_args=dict(dist=[55, 60], elev=[-5, 5], azim=[-5, 5]),
    camera_up=((0.3827, -0.9239, 0),),
    lights_args=dict(ambient_color=[0.5, 1.0])
)