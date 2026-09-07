"""`o_voxel.postprocess.to_glb` split into a shared prefix and a swappable bake.

Vendored from o-voxel (MIT, (c) Microsoft / Jianfeng Xiang) at TRELLIS.2 commit
75fbf01, restructured only so the rasterizer can be swapped. The cleaning,
remeshing and UV-unwrapping prefix is expensive and identical for both backends,
so it runs once and both bakes consume the same `out_vertices/out_faces/out_uvs`.
That makes any measured difference attributable to the rasterizer alone rather
than to a re-run of cuMesh's unwrapper.

Kept deliberately close to the original line-for-line so it stays diffable
against upstream.
"""
from typing import *

import cv2
import numpy as np
import torch
import trimesh
import trimesh.visual
from PIL import Image
from flex_gemm.ops.grid_sample import grid_sample_3d
import cumesh


def prefix_cached(cache_path, **kwargs) -> dict:
    """`prefix` memoised to disk.

    cuMesh's remesher and unwrapper are non-deterministic (atomics), so repeated
    calls give slightly different atlases — 999737 vs 998522 faces on the same
    input, observed. Caching makes every backend and tie-break variant read the
    exact same atlas, which is the only way the comparison isolates the
    rasterizer. Rebuilds the BVH on load since it holds CUDA state.
    """
    import os

    if os.path.exists(cache_path):
        blob = torch.load(cache_path, weights_only=False)
        state = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in blob.items()}
        state["bvh"] = cumesh.cuBVH(state["vertices"], state["faces"])
        return state

    state = prefix(**kwargs)
    torch.save({k: (v.cpu() if torch.is_tensor(v) else v)
                for k, v in state.items() if k != "bvh"}, cache_path)
    return state


def prefix(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor,
    coords: torch.Tensor,
    attr_layout: Dict[str, slice],
    aabb,
    voxel_size=None,
    grid_size=None,
    decimation_target: int = 1000000,
    remesh: bool = False,
    remesh_band: float = 1,
    remesh_project: float = 0.9,
    mesh_cluster_threshold_cone_half_angle_rad=np.radians(90.0),
    mesh_cluster_refine_iterations=0,
    mesh_cluster_global_iterations=1,
    mesh_cluster_smooth_strength=1,
    verbose: bool = False,
) -> dict:
    """to_glb up to and including UV unwrapping (postprocess.py:60-221)."""
    # Captures are stored on CPU; upstream is always called with CUDA tensors and
    # derives `aabb`'s device from `coords`, so hoist before any of that runs.
    coords = coords.cuda()
    attr_volume = attr_volume.cuda()

    if isinstance(aabb, (list, tuple)):
        aabb = np.array(aabb)
    if isinstance(aabb, np.ndarray):
        aabb = torch.tensor(aabb, dtype=torch.float32, device=coords.device)

    if voxel_size is not None:
        if isinstance(voxel_size, float):
            voxel_size = [voxel_size, voxel_size, voxel_size]
        if isinstance(voxel_size, (list, tuple)):
            voxel_size = np.array(voxel_size)
        if isinstance(voxel_size, np.ndarray):
            voxel_size = torch.tensor(voxel_size, dtype=torch.float32, device=coords.device)
        grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    else:
        assert grid_size is not None, "Either voxel_size or grid_size must be provided"
        if isinstance(grid_size, int):
            grid_size = [grid_size, grid_size, grid_size]
        if isinstance(grid_size, (list, tuple)):
            grid_size = np.array(grid_size)
        if isinstance(grid_size, np.ndarray):
            grid_size = torch.tensor(grid_size, dtype=torch.int32, device=coords.device)
        voxel_size = (aabb[1] - aabb[0]) / grid_size

    vertices = vertices.cuda()
    faces = faces.cuda()

    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.fill_holes(max_hole_perimeter=3e-2)
    vertices, faces = mesh.read()

    bvh = cumesh.cuBVH(vertices, faces)

    if not remesh:
        mesh.simplify(decimation_target * 3, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.simplify(decimation_target, verbose=verbose)
        mesh.remove_duplicate_faces()
        mesh.repair_non_manifold_edges()
        mesh.remove_small_connected_components(1e-5)
        mesh.fill_holes(max_hole_perimeter=3e-2)
        mesh.unify_face_orientations()
    else:
        center = aabb.mean(dim=0)
        scale = (aabb[1] - aabb[0]).max().item()
        resolution = grid_size.max().item()
        mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
            vertices, faces,
            center=center,
            scale=(resolution + 3 * remesh_band) / resolution * scale,
            resolution=resolution,
            band=remesh_band,
            project_back=remesh_project,
            verbose=verbose,
            bvh=bvh,
        ))
        mesh.simplify(decimation_target, verbose=verbose)

    out_vertices, out_faces, out_uvs, out_vmaps = mesh.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": mesh_cluster_threshold_cone_half_angle_rad,
            "refine_iterations": mesh_cluster_refine_iterations,
            "global_iterations": mesh_cluster_global_iterations,
            "smooth_strength": mesh_cluster_smooth_strength,
        },
        return_vmaps=True,
        verbose=verbose,
    )
    out_vertices = out_vertices.cuda()
    out_faces = out_faces.cuda()
    out_uvs = out_uvs.cuda()
    out_vmaps = out_vmaps.cuda()
    mesh.compute_vertex_normals()
    out_normals = mesh.read_vertex_normals()[out_vmaps]

    return dict(
        out_vertices=out_vertices, out_faces=out_faces, out_uvs=out_uvs,
        out_normals=out_normals, bvh=bvh, vertices=vertices, faces=faces,
        attr_volume=attr_volume, coords=coords,
        attr_layout=attr_layout, aabb=aabb, voxel_size=voxel_size,
        grid_size=grid_size, remesh=remesh,
    )


def rasterize_nvdiffrast(out_uvs, out_faces, out_vertices, texture_size):
    """postprocess.py:229-249 verbatim."""
    import nvdiffrast.torch as dr

    ctx = dr.RasterizeCudaContext()
    uvs_rast = torch.cat([
        out_uvs * 2 - 1,
        torch.zeros_like(out_uvs[:, :1]),
        torch.ones_like(out_uvs[:, :1]),
    ], dim=-1).unsqueeze(0)
    rast = torch.zeros((1, texture_size, texture_size, 4), device='cuda', dtype=torch.float32)

    for i in range(0, out_faces.shape[0], 100000):
        rast_chunk, _ = dr.rasterize(
            ctx, uvs_rast, out_faces[i:i + 100000],
            resolution=[texture_size, texture_size],
        )
        mask_chunk = rast_chunk[..., 3:4] > 0
        rast_chunk[..., 3:4] += i
        rast = torch.where(mask_chunk, rast_chunk, rast)

    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    return mask, pos


def rasterize_torch(out_uvs, out_faces, out_vertices, texture_size, tie_break="amax"):
    """Same contract, backed by the dependency-free rasterizer."""
    import uv_raster_torch as urt

    rast = urt.rasterize_uv(out_uvs, out_faces, texture_size, tie_break=tie_break)
    mask = rast[0, ..., 3] > 0
    pos = urt.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    return mask, pos


def rasterize_torch_amin(out_uvs, out_faces, out_vertices, texture_size):
    return rasterize_torch(out_uvs, out_faces, out_vertices, texture_size, tie_break="amin")


BACKENDS = {
    "nvdiffrast": rasterize_nvdiffrast,
    "torch": rasterize_torch,
    "torch_amin": rasterize_torch_amin,
}


def bake(state: dict, texture_size: int, backend: str = "torch"):
    """to_glb from rasterization to the finished trimesh (postprocess.py:223-331)."""
    out_vertices = state["out_vertices"]
    out_faces = state["out_faces"]
    out_uvs = state["out_uvs"]
    out_normals = state["out_normals"]
    bvh = state["bvh"]
    vertices, faces = state["vertices"], state["faces"]
    attr_volume, coords = state["attr_volume"], state["coords"]
    attr_layout, aabb = state["attr_layout"], state["aabb"]
    voxel_size, grid_size = state["voxel_size"], state["grid_size"]
    remesh = state["remesh"]

    mask, pos = BACKENDS[backend](out_uvs, out_faces, out_vertices, texture_size)

    valid_pos = pos[mask]
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    orig_tri_verts = vertices[faces[face_id.long()]]
    valid_pos = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    attrs = torch.zeros(texture_size, texture_size, attr_volume.shape[1], device='cuda')
    attrs[mask] = grid_sample_3d(
        attr_volume,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attr_volume.shape[1], *grid_size.tolist()]),
        grid=((valid_pos - aabb[0]) / voxel_size).reshape(1, -1, 3),
        mode='trilinear',
    )

    mask_np = mask.cpu().numpy()

    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

    mask_inv = (~mask_np).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    base_color_tex = np.concatenate([base_color, alpha], axis=-1)
    mr_tex = np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(base_color_tex),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(mr_tex),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=True if not remesh else False,
    )

    vertices_np = out_vertices.cpu().numpy()
    faces_np = out_faces.cpu().numpy()
    uvs_np = out_uvs.cpu().numpy()
    normals_np = out_normals.cpu().numpy()

    vertices_np[:, 1], vertices_np[:, 2] = vertices_np[:, 2], -vertices_np[:, 1]
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1]
    uvs_np[:, 1] = 1 - uvs_np[:, 1]

    textured_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_normals=normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )

    return dict(
        mesh=textured_mesh,
        mask=mask_np,
        pos=pos,
        attrs=attrs,
        base_color=base_color_tex,
        metallic_roughness=mr_tex,
    )
