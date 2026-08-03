---
name: glb-to-usd-native-converter
description: t2o_pipeline 의 GLB→USD 변환을 usd_from_gltf(레거시 바이너리) 대신 trimesh+pxr 네이티브로 교체함 — 추가 의존성 0
metadata: 
  node_type: memory
  type: project
  originSessionId: 29995152-3fc1-44b5-b411-af8c45740532
  modified: 2026-07-22T01:07:03.576Z
---

2026-07-22. `previz_pipeline/merge_glb_to_usd.py` 의 3단계(GLB→geometry.usda)를
`usd_from_gltf` 실행 파일 대신 순수 파이썬 변환기로 교체했다:
`previz_pipeline/glb_to_usd_native.py` (trimesh + pxr, 둘 다 trellis2 env 에 이미 있음).

**왜:** `usd_from_gltf` 는 (a) 업데이트가 끊긴 레거시라 최신 OpenUSD 와 호환이 깨져
소스를 마개조해야 했고, (b) 새 서버마다 별도 빌드가 필요했고, (c) 이 서버엔 아예 없어서
acceptance 3단계가 막혔다. 동료 공용 서버에 무거운 의존(bpy/Blender 등)을 늘리기도 곤란했다.

**핵심 통찰:** `usd_from_gltf` 는 파이프라인에서 딱 한 단계(GLB→geometry.usda)만 담당한다.
텍스처 정리·머티리얼 후처리·원본 USD 참조 주입(`inject_geometry_reference_into_original`)은
전부 pxr 기반이고, geometry.usda 가 **어떻게** 만들어졌는지 신경 안 쓴다(유효한
UsdPreviewSurface USD + 루트 prim 이름만 있으면 됨). TRELLIS.2 출력은 항상 단일 메시 +
baseColor/metallicRoughness 텍스처 + per-vertex UV 로 고정적이라 범용 변환기가 불필요.

**기준 파일(usd_from_gltf 산출물)과 실측 대조해 맞춘 값:**
- USD world = GLB local × 100 (scale 을 `Meshes` Xform 에 건다). metersPerUnit=0.01, upAxis=Y.
- normals interp = vertex, st(UV) interp = varying. **UV v축 flip 불필요**
  (trimesh 로드 결과가 이미 USD 와 같은 0~1).
- 계층: `<root>/Materials/{material0/pbr_shader,uvset0,tex_base}` + `<root>/Meshes/world/geometry_0/geometry_0`.
- 텍스처는 `bin/texture_<object>_0.jpg`, USD 참조는 `./bin/...` (postprocess 가 '불필요'로 넘어감).
- pxr(boost.python)은 numpy.float32 스칼라를 Gf.Vec* 로 못 받는다 → `.tolist()` 로 파이썬 float 로 내려야 함.

검증: verts/world-bbox/normal·UV interp 가 원본과 완전 일치, inject+composed stage 에서
mesh(5260 pts)+텍스처 정상 해석. merge CLI 전체 완주. → [[t2o-pipeline-server-layout]]

주의: `movie_usd/hidden_time` 의 object_N.usda·geometry.usda 는 전부 2026-07-21 14:15
전송본 = **원 서버 완성본(정답 레퍼런스)**. 전체 acceptance 재실행하면 덮어쓰므로 주의.
