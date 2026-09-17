"""Blender script that renders one 3D object into multiple views.

Reference copy, kept for documentation purposes (see the repo README for the actual dataset
config). Forked from AI2's objaverse-rendering template; the render path actually used to
produce this dataset is `save_images_exr`, invoked from `__main__`. It renders `--num_images`
views uniformly spaced in azimuth around the object, all at one fixed elevation, and writes:
  - `rgba_<i>.png`: the RGBA color render
  - `nd_<i>.exr`: a combined normal (RGB) + depth (alpha) EXR, 32-bit float

The elevation is hard-coded below (`phi = math.radians(90 - 60)`, i.e. elevation=60 degrees) and
was edited in place between rendering passes to produce the different elevation levels used
across the dataset ({0, 30, 60} degrees for base views, ~90 for top-down views) -- there was no
`--elevation` CLI flag in the version that actually ran. Camera distance/FOV are hard-coded to
match the original GObjaverse per-view camera JSON convention rather than computed from
`--camera_dist`.

Example invocation (one elevation pass):
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --num_images 24
"""

import argparse
import math
import os
import sys
import time
import urllib.request
from typing import Tuple

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=float, default=1.5)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

# Camera distance matches the origin vector used by the GObjaverse per-view camera JSONs,
# rather than the `--camera_dist` CLI arg above.
args.camera_dist = Vector([1.52213764, 0.8788066, 0.552170455]).length

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    import random

    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a glb/fbx model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    # Target max bounding-box edge length = 0.9, matching the GObjaverse convention:
    # https://github.com/modelscope/richdreamer/issues/53#issuecomment-2366150631
    scale = 0.9 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def setup_camera():
    cam = scene.objects["Camera"]
    # FOV matches the GObjaverse per-view camera JSON convention.
    cam.data.angle = 0.691150367
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint


def save_images_exr(object_file: str) -> None:
    """Renders `args.num_images` views (uniform azimuth, fixed elevation) as RGBA PNG +
    normal/depth EXR pairs into `<output_dir>/<object_uid>/`."""
    reset_scene()
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()
    add_lighting()
    cam, cam_constraint = setup_camera()
    empty = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty)
    cam_constraint.target = empty

    # Enable normal + depth render passes and wire up a compositor to write:
    #   - RGBA render -> PNG (`rgba_<i>.png`)
    #   - normal (RGB) + depth (alpha) -> combined 32-bit EXR (`nd_<i>.exr`)
    scene = bpy.context.scene
    scene.view_layers[0].use_pass_normal = True
    scene.view_layers[0].use_pass_z = True

    scene.use_nodes = True
    tree = scene.node_tree
    for node in tree.nodes:
        tree.nodes.remove(node)

    render_layers = tree.nodes.new("CompositorNodeRLayers")

    output_rgb = tree.nodes.new("CompositorNodeOutputFile")
    output_rgb.label = "RGB Output"
    output_rgb.format.file_format = "PNG"
    output_rgb.format.color_mode = "RGBA"

    output_nd = tree.nodes.new("CompositorNodeOutputFile")
    output_nd.label = "Normal & Depth Output"
    output_nd.format.file_format = "OPEN_EXR"
    output_nd.format.color_depth = "32"
    output_nd.format.color_mode = "RGBA"

    separate_normal = tree.nodes.new("CompositorNodeSeparateXYZ")
    combine_rgba = tree.nodes.new("CompositorNodeCombRGBA")

    tree.links.new(render_layers.outputs["Image"], output_rgb.inputs[0])
    tree.links.new(render_layers.outputs["Normal"], separate_normal.inputs["Vector"])
    tree.links.new(separate_normal.outputs["X"], combine_rgba.inputs["R"])
    tree.links.new(separate_normal.outputs["Y"], combine_rgba.inputs["G"])
    tree.links.new(separate_normal.outputs["Z"], combine_rgba.inputs["B"])
    tree.links.new(render_layers.outputs["Depth"], combine_rgba.inputs["A"])
    tree.links.new(combine_rgba.outputs["Image"], output_nd.inputs[0])

    for i in range(args.num_images):
        theta = (i / args.num_images) * math.pi * 2  # azimuth, uniform sweep over 360 deg
        phi = math.radians(90 - 60)  # elevation = 60 deg; edited per rendering pass (0/30/60/~90)
        point = (
            args.camera_dist * math.sin(phi) * math.cos(theta),
            args.camera_dist * math.sin(phi) * math.sin(theta),
            args.camera_dist * math.cos(phi),
        )
        cam.location = point

        render_output_dir = os.path.join(args.output_dir, object_uid)
        os.makedirs(render_output_dir, exist_ok=True)
        output_rgb.base_path = render_output_dir
        output_rgb.file_slots[0].path = "rgba_"
        output_nd.base_path = render_output_dir
        output_nd.file_slots[0].path = "nd_"

        scene.frame_current = i  # drives the file-slot frame-number suffix
        bpy.ops.render.render(write_still=True)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    return os.path.abspath(local_path)


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images_exr(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
