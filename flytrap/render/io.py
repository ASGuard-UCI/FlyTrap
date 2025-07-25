# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""This module implements utility functions for loading and saving meshes."""
from typing import List, Optional

from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesAtlas, TexturesUV
from pytorch3d.structures import join_meshes_as_batch, Meshes


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

    return mesh, aux