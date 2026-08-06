"""Phase 2b — §4a 구조 편집 실측: coords 집합을 실제로 바꾸는 편집 (제거 / 추가).

Phase 2 게이트는 coords 고정이라 sub-voxel 디테일만 변했다(실루엣 무변). 이번엔 구조를 바꾼다:

  R(제거): 날개끝 토큰(축0 상위 25%, Phase2 와 동일 영역)을 SLat 에서 **삭제**하고,
           절단 경계밴드(축0 ∈ [thr-3, thr])만 재생성해 절단면을 치유한다.
  A(추가): 날개끝 바깥으로 bbox 를 복셀로 채워(coords 합집합) 날개를 연장한다.
           신규 토큰은 자동 편집 영역, 기존 접경밴드(축0 ∈ [max-2, max])도 블렌드로 재생성.

판별:
  - R: 제거 bbox 의 디코드 점유 ≈ 0 (구조가 실제로 사라짐) + 비편집 영역 불변 + 실루엣 diff 빨강
  - A: 추가 bbox 의 디코드 점유 > 0 (stage-2 가 실제 형상을 채움 — flexible dual grid 는 빈 채로
       둘 수도 있으므로 이것 자체가 실측 대상) + 실루엣 diff 초록

실행: conda activate trellis2; CUDA_VISIBLE_DEVICES=0 python edit3d/phase2b_struct.py
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PREVIS = os.path.dirname(REPO)
sys.path.insert(0, os.path.join(REPO, "trellis2_src"))
sys.path.insert(0, HERE)

import torch
import trimesh
from PIL import Image
import eval as E
import coords as C
import region as R

ROOT = os.path.join(REPO, "hf_models", "TRELLIS.2-4B")
ENC_CKPT = os.path.join(ROOT, "ckpts", "shape_enc_next_dc_f16c32_fp16")
ASSET = os.path.join(PREVIS, "movie_usd/hidden_time/scene_1/objects/assets/object_1/scene_1_seagull_209626.glb")
REFIMG = os.path.join(REPO, "t2o_results/TRELLIS.2-4B/20260804/run_134025_en/previews/scene_1/object_1/scene_1_seagull_209626_ref.png")
SHAPE_RES = 1024
GRID = SHAPE_RES // 16
SEED = 42
OUTDIR = os.path.join(HERE, "phase2b_out")


def load():
    from trellis2 import models as M
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from trellis2.pipelines.trellis2_texturing import Trellis2TexturingPipeline
    Trellis2ImageTo3DPipeline.model_names_to_load = [
        'shape_slat_flow_model_1024', 'shape_slat_decoder']
    print("[load] pipelines ...", flush=True)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(ROOT)
    pipe._device = torch.device("cuda")
    enc = M.from_pretrained(ENC_CKPT).eval()
    tex = Trellis2TexturingPipeline({"shape_slat_encoder": enc})
    tex.low_vram = True; tex._device = pipe._device
    print("[load] done", flush=True)
    return pipe, tex


def decode_mesh(pipe, slat):
    meshes, _ = pipe.decode_shape_slat(slat, SHAPE_RES)
    m = meshes[0]
    return (m.vertices.detach().cpu().numpy().astype(np.float64),
            m.faces.detach().cpu().numpy().astype(np.int64))


def bbox_occupancy(vi, lo, hi):
    inside = np.all((vi >= lo) & (vi <= hi), axis=1)
    return int(inside.sum())


def subset_slat(slat, keep_rows):
    from trellis2.modules.sparse import SparseTensor
    return SparseTensor(feats=slat.feats[keep_rows], coords=slat.coords[keep_rows])


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pipe, tex = load()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    image = Image.open(REFIMG).convert("RGB")

    raw = trimesh.load(ASSET, force="mesh", process=False)
    pm = tex.preprocess_mesh(trimesh.Trimesh(
        vertices=np.asarray(raw.vertices), faces=np.asarray(raw.faces), process=False))
    with torch.no_grad():
        slat = tex.encode_shape_slat(pm, resolution=SHAPE_RES)
    xyz = slat.coords[:, 1:].float()
    ext = xyz.max(0).values - xyz.min(0).values
    axis = int(ext.argmax())
    thr = float(torch.quantile(xyz[:, axis], 0.75))
    xmax = float(xyz[:, axis].max())
    print(f"[encode] {slat.coords.shape[0]} tokens  axis={axis} thr={thr:.0f} max={xmax:.0f}", flush=True)

    results = {"asset": os.path.basename(ASSET), "seed": SEED, "grid": GRID,
               "axis": axis, "thr": thr, "tokens": int(slat.coords.shape[0])}

    # ---------- R: 제거 + 절단면 치유 ----------
    ax_vals = slat.coords[:, 1 + axis].float()
    keep_rows = ax_vals <= thr                       # 날개끝(>thr) 삭제
    slat_R = subset_slat(slat, keep_rows)
    heal = (slat_R.coords[:, 1 + axis].float() >= thr - 3)   # 경계밴드 재생성
    print(f"[R] removed={int((~keep_rows).sum())} kept={int(keep_rows.sum())} heal_band={int(heal.sum())}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        edited_R = R.edit_region(pipe, image, slat_R, heal, SEED)
    print(f"[R] edit done ({time.time()-t0:.0f}s)", flush=True)

    # ---------- A: 추가 (돌출부 생성) ----------
    # ⚠️ 정규화가 최장축을 큐브에 꽉 채우므로(날개끝 max=63) 그 방향으론 빈 공간이 없다.
    # → 그리드 여유(clearance)가 가장 큰 축·방향을 자동 선택해 그쪽으로 6복셀 돌출부를 만든다.
    cands = []
    for a in range(3):
        amin, amax = float(xyz[:, a].min()), float(xyz[:, a].max())
        cands.append((GRID - 1 - amax, a, +1, amax))   # max 쪽 여유
        cands.append((amin, a, -1, amin))              # min 쪽 여유
    clearance, add_axis, sign, extreme = max(cands)
    depth = min(6, int(clearance))
    assert depth >= 3, f"여유 부족: clearance={clearance}"
    band = xyz[(np.abs if False else torch.abs)(xyz[:, add_axis] - extreme) <= 2]  # 접경 단면
    other = [a for a in range(3) if a != add_axis]
    lo_add = np.zeros(3); hi_add = np.zeros(3)
    if sign > 0:
        lo_add[add_axis], hi_add[add_axis] = extreme + 1, extreme + depth
    else:
        lo_add[add_axis], hi_add[add_axis] = extreme - depth, extreme - 1
    for a in other:
        # 단면 중앙 60% 만 사용 — 가장자리 얇은 부위가 아니라 몸통에 붙게
        q20, q80 = torch.quantile(band[:, a], 0.2), torch.quantile(band[:, a], 0.8)
        lo_add[a], hi_add[a] = float(q20), float(q80)
    extra = R.fill_bbox_coords(lo_add, hi_add, device=slat.coords.device)
    blend = (torch.abs(slat.coords[:, 1 + add_axis].float() - extreme) <= 2)  # 접경 블렌드
    print(f"[A] extra={extra.shape[0]} voxels bbox={lo_add.tolist()}~{hi_add.tolist()} blend={int(blend.sum())}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        edited_A = R.edit_region(pipe, image, slat, blend, SEED, extra_coords=extra)
    print(f"[A] edit done ({time.time()-t0:.0f}s)", flush=True)

    # ---------- 디코드 + 판별 ----------
    with torch.no_grad():
        v0, f0 = decode_mesh(pipe, slat)
        vR, fR = decode_mesh(pipe, edited_R)
        vA, fA = decode_mesh(pipe, edited_A)
    vi0 = C.mesh_to_voxel_indices(v0, f0, GRID)
    viR = C.mesh_to_voxel_indices(vR, fR, GRID)
    viA = C.mesh_to_voxel_indices(vA, fA, GRID)

    # R 판별: 제거 bbox(축>thr) 점유
    lo_rm = np.zeros(3); hi_rm = np.full(3, GRID - 1, float); lo_rm[axis] = thr + 1
    occ_rm_before = bbox_occupancy(vi0, lo_rm, hi_rm)
    occ_rm_after = bbox_occupancy(viR, lo_rm, hi_rm)
    # R 비편집 영역(축 < thr-3) 불변성
    lo_keep = np.zeros(3); hi_keep = np.full(3, GRID - 1, float); hi_keep[axis] = thr - 4
    def sub(vi, lo, hi):
        return vi[np.all((vi >= lo) & (vi <= hi), axis=1)]
    keep_iou = E.coords_iou(sub(vi0, lo_keep, hi_keep), sub(viR, lo_keep, hi_keep))
    results["removal"] = {
        "removed_tokens": int((~keep_rows).sum()), "heal_band": int(heal.sum()),
        "bbox_occ_before": occ_rm_before, "bbox_occ_after": occ_rm_after,
        "unedited_iou": round(keep_iou, 3)}
    print(f"[verify-R] bbox occ {occ_rm_before}->{occ_rm_after}  unedited_iou={keep_iou:.3f}", flush=True)

    # A 판별: 추가 bbox 점유 (before는 정의상 0)
    occ_add_before = bbox_occupancy(vi0, lo_add, hi_add)
    occ_add_after = bbox_occupancy(viA, lo_add, hi_add)
    keep_iou_A = E.coords_iou(sub(vi0, lo_keep, hi_keep), sub(viA, lo_keep, hi_keep))
    results["addition"] = {
        "extra_voxels": int(extra.shape[0]), "blend_band": int(blend.sum()),
        "bbox_occ_before": occ_add_before, "bbox_occ_after": occ_add_after,
        "unedited_iou": round(keep_iou_A, 3)}
    print(f"[verify-A] bbox occ {occ_add_before}->{occ_add_after}  unedited_iou={keep_iou_A:.3f}", flush=True)

    for name, (vv, ff) in {"orig": (v0, f0), "removed": (vR, fR), "added": (vA, fA)}.items():
        trimesh.Trimesh(vertices=vv, faces=ff, process=False).export(f"{OUTDIR}/{name}.glb")
    results["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    with open(os.path.join(HERE, "phase2b_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[done]", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
