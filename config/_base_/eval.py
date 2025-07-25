# Put all unnecessary configs in evaluation here

debug = False
eval = True

# patch training setting
epochs = 300

# log settings
log = False
log_cfg = dict(
    project='adcover-final',
    entity='shaoyux',
    save_code=True,
    group='mixformer',
    name='eval',
    notes=''
)

optimizer = dict(
    type='Adam',
    params=dict(
        lr=10
    )
)

scheduler = dict(type='WarmupCosineDecayLR',
                 params=dict(
                     warmup_steps=0,
                     total_steps=epochs,
                     eta_min=0.1
                 ))

# loss settings
loss = [
    dict(type='TVLoss', weight=0.000005),
    # should put this before LocalizationLoss
    # LocalizationLoss will change `gt_bbox` value
    dict(type='SmoothClassificationLossUnfollowV2', weight=1),
    dict(type='LocalizationLossUnfollow', weight=10),]
