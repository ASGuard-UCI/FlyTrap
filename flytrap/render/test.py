import torch
import numpy as np
import cv2
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams
)
from pytorch3d.structures import Meshes
from typing import List, Optional

from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesAtlas, TexturesUV
from pytorch3d.structures import join_meshes_as_batch, Meshes
import matplotlib.pyplot as plt

def load_obj_as_mesh(
    file: list,
    device: Optional[Device] = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
):
    """
    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_obj function for more
    details. material_colors and normals are not stored.

    Args:
        files: A list of file-like objects (with methods read, readline, tell,
            and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        New Meshes object.
    """

    verts, faces, aux = load_obj(
        file,
        load_textures=load_textures,
        create_texture_atlas=create_texture_atlas,
        texture_atlas_size=texture_atlas_size,
        texture_wrap=texture_wrap,
        path_manager=path_manager,
    )
    tex = None
    if create_texture_atlas:
        # TexturesAtlas type
        tex = TexturesAtlas(atlas=[aux.texture_atlas.to(device)])
    else:
        # TexturesUV type
        tex_maps = aux.texture_images
        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
            faces_uvs = faces.textures_idx.to(device)  # (F, 3)
            image = list(tex_maps.values())[0].to(device)[None]
            tex = TexturesUV(
                verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
            )

    mesh = Meshes(
        verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
    )

    return mesh, verts, faces, aux

# Load the mesh from an .obj file
device = torch.device("cuda:0")

# Load your mesh
mesh, verts, faces, aux = load_obj_as_mesh("data/render_obj/umbrella/umbrella.obj", device=device)

# Create a texture with gradients
texture_image = cv2.imread('work_dirs/mixformer_cvt_position_baseline_epoch80_two_stage_shrink/patch_epoch20.png')
texture_image = torch.tensor(texture_image / 255.0, device=device, dtype=torch.float32).unsqueeze(0)
texture_image.requires_grad = True
# verts_uvs = torch.rand((mesh.verts_packed().shape[0], 2), device=device)
verts_uvs = aux.verts_uvs.to(device)
# faces_uvs = mesh.faces_packed()
faces_uvs = faces.textures_idx.to(device)

# Create a Textures UV object
texture = TexturesUV(maps=texture_image, verts_uvs=verts_uvs.unsqueeze(0), faces_uvs=faces_uvs.unsqueeze(0))
mesh.textures = texture

# Initialize a camera
# R, T = look_at_view_transform(60.0, 60, 0, up=((0.3827, -0.9239, 0),)) 
R, T = look_at_view_transform(60.0, 5, 0, up=((0.3827, -0.9239, 0),)) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading
raster_settings = RasterizationSettings(
    image_size=300, 
    blur_radius=0.0,
    bin_size=0,
    # faces_per_pixel=5,
    # max_faces_per_bin=1e10,
)

# Place a point light in front of the object
lights = PointLights(device=device, 
                     ambient_color=[[1.0, 1.0, 1.0]],
                    #  diffuse_color=[[1.0, 1.0, 1.0]],
                    #  specular_color=[[1.0, 1.0, 1.0]],
                     location=[[0, -50, 0]])

# Create a renderer
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=[0.0, 0.0, 0.0])
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights,
        blend_params=blend_params
    )
)

# Render the image
images = renderer(mesh)
images = torch.clamp(images, 0.0, 1.0)

# Convert the image to a numpy array and then to grayscale to create a mask
image_np = images[0, ..., :3].cpu().detach().numpy()
plt.imsave("render.png", (image_np * 255).astype(np.uint8))
gray_image = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

# Create a binary mask where the non-black pixels are white (255)
_, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
kernel_2 = np.ones((4, 4), dtype=np.uint8)
binary_mask = cv2.dilate(binary_mask, kernel_2, iterations=1)
plt.imsave("render_mask.png", binary_mask)

# Find contours (bounding boxes) of the white areas
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the bounding box of the largest contour (assuming it's the umbrella)
x, y, w, h = cv2.boundingRect(contours[0])

# Convert the bounding box coordinates to tensor indices
x1, y1, x2, y2 = x, y, x + w, y + h

# Crop the image tensor based on the bounding box
cropped_image = images[:, y1:y2, x1:x2, :]

a = 1
# Visualize the rendered image

plt.imsave("render_tight.png", cropped_image[0, ..., :3].cpu().detach().numpy())
# plt.imshow(images[0, ..., :3].cpu().detach().numpy())
# plt.show()

# # Example of loss computation and backpropagation
# loss = images.mean()
# loss.backward()

# # The gradients for the texture can now be accessed via `texture_image.grad`
# print(texture_image.grad)