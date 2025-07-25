import torch
import numpy as np
import cv2
import wandb
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
from typing import List, Optional, Dict

from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesAtlas, TexturesUV
from pytorch3d.structures import join_meshes_as_batch, Meshes

from ..builder import APPLYER


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


@APPLYER.register_module()
class NoRenderer:
    """Render the mesh"""
    
    def __init__(self):
        pass
    
    def __call__(self, texture_image, train=True):
        return texture_image


@APPLYER.register_module()
class Renderer:
    """Render the mesh"""
    
    def __init__(self, 
                 device: str,
                 mesh_file: str,
                 image_size: int,
                 camera_args: Dict,
                 camera_up: List,
                 lights_args: Dict,
                 camera_args_eval: Dict = dict(dist=55, elev=0, azim=0)):
        # TODO: [ ] hard code the device to cuda here
        self.device = device
        mesh, verts, faces, aux = load_obj_as_mesh(mesh_file, device=self.device)
        self.verts_uvs = aux.verts_uvs.to(device)
        self.faces_uvs = faces.textures_idx.to(device)
        self.rasterization_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            bin_size=0 # TODO: [ ] this might influence the speed
        )
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0))
        self.mesh = mesh
        self.verts = verts
        self.faces = faces
        self.aux = aux
        self.camera_args = camera_args
        self.camera_args_eval = camera_args_eval
        self.camera_up = camera_up
        self.lights_args = lights_args
        
    def _sample_distribution(self, args: Dict):
        new_args = {}
        for key, value in args.items():
            if isinstance(value, list):
                assert len(value) == 2, f"Expect a list of length 2, got {value}"
                new_args[key] = np.random.uniform(value[0], value[1])
            else:
                new_args[key] = value
        return new_args
        
    def _sample_light_params(self, args: Dict):
        new_args = {}
        for key, value in args.items():
            if isinstance(value, list):
                assert len(value) == 2, f"Expect a list of length 2, got {value}"
                new_args[key] = np.random.uniform(value[0], value[1], size=(1, 3))
            else:
                new_args[key] = value
        return new_args
        
    def __call__(self, texture_image, train=True):
        texture_image = texture_image.permute(1, 2, 0).unsqueeze(0) / 255.0
        texture = TexturesUV(maps=texture_image, 
                             verts_uvs=self.verts_uvs.unsqueeze(0),
                             faces_uvs=self.faces_uvs.unsqueeze(0))
        self.mesh.textures = texture
        
        if train:
            # sample distribution for camera and light for EOT
            R, T = look_at_view_transform(**self._sample_distribution(self.camera_args), up=self.camera_up)
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            lights = PointLights(device=self.device, 
                                location=[[0, -50, 0]],
                                **self._sample_light_params(self.lights_args))
        else:
            # fix the camera and light for evaluation
            # TODO: [ ] hard code position here, vary depend on the 3D mesh
            R, T = look_at_view_transform(**self.camera_args_eval, up=self.camera_up)
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            lights = PointLights(device=self.device, 
                                location=[[0, -50, 0]],
                                ambient_color=[[1.0, 1.0, 1.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.rasterization_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=self.blend_params
            )
        )
        # ignore the alpha channel
        image = renderer(self.mesh)[..., :3]
        image = torch.clamp(image, 0.0, 1.0)
        image = self._tight_layout(image) * 255.0
        return image.squeeze().permute(2, 0, 1)
        
    def _tight_layout(self, images):
        """Crop out the black background of the rendered image"""
        image_np = images[0, ..., :3].cpu().detach().numpy()
        gray_image = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # Use dilation to fill in the holes within the umbrella areas
        # This step can help better capture the umbrella contour
        kernel_2 = np.ones((4, 4), dtype=np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel_2, iterations=1)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the largest contour (assuming it's the umbrella)
        x, y, w, h = cv2.boundingRect(contours[0])
        if w < 10 and h < 10:
            print("[Warning] Tight layout too small, check the tight layout method")
            return images
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Crop the image tensor based on the bounding box
        cropped_image = images[:, y1:y2, x1:x2, :]
        return cropped_image
    
    
@APPLYER.register_module()
class PhysicalRenderer:
    """Render the mesh, based on the physical engine to determine the distance"""
    
    def __init__(self, 
                 device: str,
                 mesh_file: str,
                 image_size: int,
                 camera_up: List,
                 lights_args: Dict,
                 mesh_size=37.3, # meter, TODO: hard code use the umbrella mesh we have
                 actual_size=0.4, # meter
                 camera_args_eval: Dict = dict(dist=55, elev=0, azim=0)):
        # TODO: [ ] hard code the device to cuda here
        self.device = device
        mesh, verts, faces, aux = load_obj_as_mesh(mesh_file, device=self.device)
        self.verts_uvs = aux.verts_uvs.to(device)
        self.faces_uvs = faces.textures_idx.to(device)
        self.rasterization_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            bin_size=0 # TODO: [ ] this might influence the speed
        )
        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0))
        self.mesh = mesh
        self.verts = verts
        self.faces = faces
        self.aux = aux
        self.camera_args_eval = camera_args_eval
        self.camera_up = camera_up
        self.lights_args = lights_args
        self.scale = actual_size / mesh_size
        
    def _sample_distribution(self, args: Dict):
        new_args = {}
        for key, value in args.items():
            if isinstance(value, list):
                assert len(value) == 2, f"Expect a list of length 2, got {value}"
                new_args[key] = np.random.uniform(value[0], value[1])
            else:
                new_args[key] = value
        return new_args

    def distance_mapping(self, distance):
        """
        Softly maps x in the range (0.5, ∞) to y in the range [40, 110].

        Args:
            x (float or numpy array): Input value(s), must be > 0.5.

        Returns:
            float or numpy array: Mapped value(s) in the range [40, 110].
        """
        # TODO: we hard code the mapping function here from observation and personal experience
        # Parameters derived from observed points
        x_min, y_min = 0.5, 40  # Lower bound of x and corresponding y
        x_max, y_max = 14, 110  # Upper bound of x and corresponding y

        # Scale the observed points for smooth mapping
        scale_factor = (y_max - y_min) / (np.log(x_max) - np.log(x_min))

        # Mapping function using logarithmic scaling
        y = y_min + scale_factor * (np.log(np.clip(distance, x_min, None)) - np.log(x_min))
        
        # Clip to ensure it remains within [y_min, y_max]
        y = np.clip(y, y_min, y_max)

        return y
 
    def __call__(self, texture_image, camera_args_eval):
        """Args
            texture_image: torch.Tensor, (3, H, W)
            distance: float, the distance between the camera and the object
            size: int, the size of the physical objects"""
        texture_image = texture_image.permute(1, 2, 0).unsqueeze(0) / 255.0
        texture = TexturesUV(maps=texture_image, 
                             verts_uvs=self.verts_uvs.unsqueeze(0),
                             faces_uvs=self.faces_uvs.unsqueeze(0))
        self.mesh.textures = texture
        
        # camera_args_eval['dist'] = camera_args_eval['dist'] / self.scale
        
        # TODO: manually set the distance to the object
        camera_args_eval['dist'] = self.distance_mapping(camera_args_eval['dist'])
        R, T = look_at_view_transform(**camera_args_eval, up=self.camera_up)
        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        lights = PointLights(device=self.device, 
                            location=[[0, -50, 0]],
                            ambient_color=[[1.0, 1.0, 1.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.rasterization_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=self.blend_params
            )
        )
        # ignore the alpha channel
        image = renderer(self.mesh)[..., :3]
        image = torch.clamp(image, 0.0, 1.0)
        
        ## debug
        # cv2.imwrite('debug.png', image.squeeze().cpu().detach().numpy() * 255)
        
        image = self._tight_layout(image) * 255.0
        return image.squeeze().permute(2, 0, 1)
        
    def _tight_layout(self, images):
        """Crop out the black background of the rendered image"""
        image_np = images[0, ..., :3].cpu().detach().numpy()
        gray_image = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        # Use dilation to fill in the holes within the umbrella areas
        # This step can help better capture the umbrella contour
        kernel_2 = np.ones((4, 4), dtype=np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel_2, iterations=1)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the bounding box of the largest contour (assuming it's the umbrella)
        x, y, w, h = cv2.boundingRect(contours[0])
        if w < 10 and h < 10:
            print("[Warning] Tight layout too small, check the tight layout method")
            return images
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Crop the image tensor based on the bounding box
        cropped_image = images[:, y1:y2, x1:x2, :]
        return cropped_image