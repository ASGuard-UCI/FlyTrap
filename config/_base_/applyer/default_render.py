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
    type='Renderer',
    device='cuda',
    mesh_file='data/render_obj/umbrella/umbrella.obj',
    image_size=300,
    camera_args=dict(dist=[55, 60], elev=[-5, 5], azim=[-5, 5]),
    camera_up=((0.3827, -0.9239, 0),),
    lights_args=dict(ambient_color=[0.5, 1.0])
)