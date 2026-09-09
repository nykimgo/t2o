"""GLB → geometry.usda 네이티브 변환 (trimesh + pxr, 외부 바이너리 불필요).

usd_from_gltf 를 대체한다. TRELLIS.2 가 뱉는 GLB 는 항상
**단일 메시 + baseColor / metallicRoughness 텍스처 + per-vertex UV** 형태로
고정적이라, 범용 glTF 변환기의 복잡함(씬 그래프·애니메이션·다중 머티리얼)이
필요 없다. 그 고정 형태만 정확히 커버한다.

출력 USD 계층은 usd_from_gltf 출력과 **구조적으로 동일**하게 맞춘다
(merge_glb_to_usd.py 의 postprocess_geometry_usd / inject 가 그대로 동작하도록):

    Xform <object_name> (kind=component)
      Scope Materials
        Material material0
          Shader pbr_shader  (UsdPreviewSurface): diffuseColor ← tex_base.rgb
          Shader uvset0      (UsdPrimvarReader_float2): varname = "st"
          Shader tex_base    (UsdUVTexture): file = ./bin/texture_<obj>_0.jpg
      Xform Meshes (xformOp:scale = (100,100,100))   # GLB(미터) → USD 관례 크기
        Xform world
          Xform geometry_0
            Mesh geometry_0  (points / faceVertexCounts / faceVertexIndices /
                              normals[vertex] / primvars:st[varying])

기준 파일(usd_from_gltf 산출물)과 실측으로 대조해 맞춘 값:
  - USD world = GLB local × 100 (scale 을 Meshes 에 건다)
  - normals interp = vertex, st interp = varying
  - UV v 축 flip 불필요 (trimesh 로드 결과가 이미 USD 와 동일한 0~1)

의존성: trimesh, pxr(usd-core) — 둘 다 trellis2 env 에 이미 있다. bpy/Blender,
usd_from_gltf 빌드 모두 불필요.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt

# GLB 는 미터, usd_from_gltf 는 이 스케일로 USD 를 뽑았다(기준 파일 실측: ×100).
_MESH_SCALE = 100.0


def _load_single_mesh(glb_path: str) -> trimesh.Trimesh:
    """GLB 를 단일 Trimesh 로 로드. TRELLIS.2 출력은 항상 메시 1개다."""
    mesh = trimesh.load(glb_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"GLB 를 단일 메시로 로드하지 못함: {glb_path} ({type(mesh)})")
    return mesh


def _extract_base_texture(mesh: trimesh.Trimesh, bin_dir: Path,
                          object_name: str) -> Optional[str]:
    """baseColor 텍스처를 bin/texture_<obj>_0.jpg 로 저장하고 상대경로 반환.

    반환 형식은 postprocess_geometry_usd 가 기대하는 './bin/...' 형태다.
    """
    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return None
    img = getattr(mat, "baseColorTexture", None)
    if img is None:
        # PBRMaterial 이 아니면 image 속성으로 폴백
        img = getattr(mat, "image", None)
    if img is None:
        return None

    bin_dir.mkdir(parents=True, exist_ok=True)
    tex_name = f"texture_{object_name}_0.jpg"
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img))
    img.convert("RGB").save(bin_dir / tex_name, quality=95)
    return f"./bin/{tex_name}"


def _scalar_factor(mesh: trimesh.Trimesh, attr: str, default: float) -> float:
    mat = getattr(mesh.visual, "material", None)
    val = getattr(mat, attr, None) if mat is not None else None
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def convert(glb_path: str, out_usd: str, root_prim_name: str,
            texture_object_name: Optional[str] = None) -> Tuple[str, str]:
    """GLB → geometry.usda. (출력 경로, 루트 prim 이름) 반환.

    Args:
        root_prim_name: USD 루트 prim 이름. inject 가
            ``AddReference(asset, f"/{prim_name}")`` 로 참조하는 값이다.
        texture_object_name: 텍스처 파일명(texture_<name>_0.jpg)용. 미지정 시
            root_prim_name 을 쓴다. main() 은 루트에 '<obj>_geometry',
            텍스처에 '<obj>' 를 쓰므로 둘을 분리한다.
    """
    tex_name = texture_object_name or root_prim_name
    object_name = root_prim_name
    mesh = _load_single_mesh(glb_path)
    out_usd_path = Path(out_usd)
    out_usd_path.parent.mkdir(parents=True, exist_ok=True)
    bin_dir = out_usd_path.parent / "bin"
    tex_rel = _extract_base_texture(mesh, bin_dir, tex_name)

    # 배열은 Vt.*Array.FromNumpy 로 numpy 를 통째로 벌크 복사한다(C++ 경로).
    # .tolist() 후 요소마다 Gf.Vec3f(*p) 로 감싸면 81만 정점 기준 수백만 개
    # boost.python 객체를 만들어 ~9초가 든다(FromNumpy 는 0.04초).
    # 단, 스칼라 factor 만은 pxr 이 numpy.float32 를 Gf 로 못 받으므로
    # _scalar_factor 에서 float() 로 내려 처리한다.
    verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32).reshape(-1)
    normals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)
    uv = mesh.visual.uv
    uv = None if uv is None else np.ascontiguousarray(uv, dtype=np.float32)
    n_faces = len(faces) // 3

    stage = Usd.Stage.CreateNew(str(out_usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)

    root = UsdGeom.Xform.Define(stage, f"/{object_name}")
    root.GetPrim().SetMetadata("kind", "component")
    stage.SetDefaultPrim(root.GetPrim())

    # -- Materials --------------------------------------------------------
    UsdGeom.Scope.Define(stage, f"/{object_name}/Materials")  # scope 정의 부수효과
    material = UsdShade.Material.Define(
        stage, f"/{object_name}/Materials/material0")

    pbr = UsdShade.Shader.Define(
        stage, f"/{object_name}/Materials/material0/pbr_shader")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(
        _scalar_factor(mesh, "metallicFactor", 1.0))
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(
        _scalar_factor(mesh, "roughnessFactor", 1.0))
    pbr.CreateInput("occlusion", Sdf.ValueTypeNames.Float).Set(1.0)
    surf_out = pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surf_out)

    uvreader = UsdShade.Shader.Define(
        stage, f"/{object_name}/Materials/material0/uvset0")
    uvreader.CreateIdAttr("UsdPrimvarReader_float2")
    uvreader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_out = uvreader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    if tex_rel is not None:
        tex = UsdShade.Shader.Define(
            stage, f"/{object_name}/Materials/material0/tex_base")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_rel)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_out)
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set((1, 0, 1, 1))
        rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    else:
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.8, 0.8, 0.8))

    # -- Mesh 계층 (scale 을 Meshes 에 건다) ------------------------------
    meshes_xform = UsdGeom.Xform.Define(stage, f"/{object_name}/Meshes")
    meshes_xform.AddScaleOp().Set(Gf.Vec3f(_MESH_SCALE, _MESH_SCALE, _MESH_SCALE))
    UsdGeom.Xform.Define(stage, f"/{object_name}/Meshes/world")
    UsdGeom.Xform.Define(stage, f"/{object_name}/Meshes/world/geometry_0")

    gm = UsdGeom.Mesh.Define(
        stage, f"/{object_name}/Meshes/world/geometry_0/geometry_0")
    gm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts))
    gm.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(n_faces, 3, dtype=np.int32)))
    gm.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
    gm.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
    gm.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    gm.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    gm.CreateExtentAttr(UsdGeom.PointBased(gm).ComputeExtent(gm.GetPointsAttr().Get()))

    if uv is not None:
        st_pv = UsdGeom.PrimvarsAPI(gm.GetPrim()).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
        st_pv.Set(Vt.Vec2fArray.FromNumpy(uv))

    UsdShade.MaterialBindingAPI.Apply(gm.GetPrim())
    UsdShade.MaterialBindingAPI(gm.GetPrim()).Bind(material)

    stage.GetRootLayer().Save()
    return str(out_usd_path), object_name


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="GLB → geometry.usda (trimesh+pxr)")
    ap.add_argument("glb")
    ap.add_argument("out_usd")
    ap.add_argument("root_prim_name")
    ap.add_argument("--texture-name", default=None)
    args = ap.parse_args()
    path, name = convert(args.glb, args.out_usd, args.root_prim_name,
                         args.texture_name)
    print(f"NATIVE_USD_OK root=/{name} -> {path}")
