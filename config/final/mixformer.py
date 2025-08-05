# based on engine_v3, 
#   - delete distance based rendered (v4)
#       - Increase batch size to 16 (v5)
#   - random crop the first frame of images using random scale factor (v6)
#       - Fix the `target_shrink_bbox` center position, previously mistakenly overwrite it (v6_fix)
#       - Fix the `target_shrink_bbox` center position, previously mistakenly overwrite it in `stage` (v6_fix_v2)

# final version, we use v6_fix_v2: test if the performance match v6_fix_v2

_base_ = [
    '../_base_/metrics/masr.py'
]

debug = False
eval = True

# used for patch generation
img_config = {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]}
# used for model input normalization
model_img_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# model settings
model = dict(type='mixformer_cvt_online_scores',
             checkpoint='./models/MixFormer/models/mixformer_online_22k.pth.tar')

# tracker settings, used for evaluation
tracker = dict(type='mixformer_cvt_online_scores_tracker',
               params=dict(
                   model='mixformer_online_22k.pth.tar',
                   update_interval=10000, online_sizes=1, search_area_scale=4.5,
                   max_score_decay=1.0, vis_attn=0))

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
patch_path = 'patches/mixformer_flytrap.png'

# log settings
log = False
log_cfg = dict(
    project='adcover-final',
    entity='shaoyux',
    save_code=True,
    group='mixformer',
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
    dict(type='CropTargetObjectTemplate', template_factor=2.0, template_size=128),
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
    dict(type='TVLoss', weight=0.000005),
    # should put this before LocalizationLoss
    # LocalizationLoss will change `gt_bbox` value
    dict(type='SmoothDynamicClassificationLoss', weight=1),
    dict(type='TargetLocalizationLoss', weight=10),
]

engine = dict(
    type='PhysicalEngineV4',
    evolve_step=2,
    interval_distance=4,
    render = dict(
        type='Renderer',
        device='cuda',
        mesh_file='mesh/umbrella/umbrella.obj',
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
    dict(type='CustomNormalize', img_keys=img_keys, **model_img_config),
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
    mesh_file='mesh/umbrella/umbrella.obj',
    image_size=300,
    camera_args=dict(dist=[55, 60], elev=[-5, 5], azim=[-5, 5]),
    camera_up=((0.3827, -0.9239, 0),),
    lights_args=dict(ambient_color=[0.5, 1.0])
)