# used for patch generation
img_config = {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]}
# used for model input normalization
model_img_config = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

# color transformation range: [0, 1]
# imporve physical robustness
patch_transform = [
    dict(type='Brightness', factor=0.1),
    dict(type='Contrast', factor=0.1),
    dict(type='Saturation', factor=0.1),
    dict(type='Hue', factor=0.1),
]

# dataset settings
img_keys = ['search', 'template', 'online_template']
train_pipeline = [
    dict(type='TemplateSample', same_video=False, online_template=True, num_online=1),
    dict(type='CustomLoadImageFromFile', img_keys=img_keys, to_float32=True, color_type='color'),
    dict(type='CropTargetObject', template_factor=2.0, search_factor=[3, 6],
                                  template_size=128, search_size=320,
                                  randomize=True),
    dict(type='NormalizeCoordinates'),
    dict(type='GaussianNoise', img_keys=img_keys, mean=0, var=5),
    dict(type='CustomCollect',
         keys=['search', 'template', 'online_template', 'normalize_bbox', 'coords'],
         meta_keys=['video', 'img_shape_template'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='color'),
    dict(type='NormalizeCoordinates'),
    dict(type='CustomCollect',
         keys=['img', 'coords', 'init_bbox', 'apply_attack', 'umbrella_bbox'],
         meta_keys=['img_shape'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    shuffle=True,
    collate_fn=dict(type='custom_collate_fn'),
    dataset=dict(
        type='CustomDataset',
        meta_file='./data/dataset_v3.0/collect_train.json',
        pipeline=train_pipeline
    )
)
test_dataset = dict(
    type='CustomEvalDataset',
    meta_file='./data/dataset_v4.0/collect_eval.json',
    # filter out video with specific name
    sub_string='',
    pipeline=test_pipeline
)

# post normalization
# ugly workaround here since apply the patch require range [0, 255]
# convert into model input distribution
post_transform = [
    dict(type='CustomNormalize', img_keys=img_keys, **model_img_config),
]

eval_interval = 50