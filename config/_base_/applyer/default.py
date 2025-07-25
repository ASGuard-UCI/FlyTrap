# patch applyer
applyer = dict(
    type='PatchApplyer',
    rotate_mean=-3.14159 * 3 / 4,
    rotate_var=3.14159 / 9,
    fix_orien=False,
    distribution='normal'
)

# mesh renderer
renderer = dict(
    type='NoRenderer'
)