"""Phase 2 실측 — C: Region Editing 게이트 검증 (§4c).

시나리오: seagull 에셋을 encode → 날개끝(가장 넓은 축의 상위 25% 분위) 영역만 재생성.
검증:
  1) 마스크 밖 latent 변화량 == 정확히 0  (0 아니면 구현 버그)
  2) 마스크 안은 실제로 변했는가          (latent/점유 변화)
  3) 디코드 후 마스크 밖 점유 IoU ≈ 1     (디코더 수용영역에 의한 경계 근방 변화는 실측해 보고)
  4) 정성: 원본/편집 실루엣 몽타주

실행: conda activate trellis2; CUDA_VISIBLE_DEVICES=0 python edit3d/phase2_region.py
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
GRID = SHAPE_RES // 16          # SLat 그리드 = 64
SEED = 42
OUTDIR = os.path.join(HERE, "phase2_out")


def load():
    from trellis2 import models as M
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    from trellis2.pipelines.trellis2_texturing import Trellis2TexturingPipeline
    Trellis2ImageTo3DPipeline.model_names_to_load = [
        'shape_slat_flow_model_1024', 'shape_slat_decoder']
    print("[load] ImageTo3D (flow1024+decoder) ...", flush=True)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(ROOT)
    pipe._device = torch.device("cuda")
    print("[load] shape_slat_encoder ...", flush=True)
    enc = M.from_pretrained(ENC_CKPT).eval()
    tex = Trellis2TexturingPipeline({"shape_slat_encoder": enc})
    tex.low_vram = True; tex._device = pipe._device
    return pipe, tex


def decode_mesh(pipe, slat):
    meshes, _ = pipe.decode_shape_slat(slat, SHAPE_RES)
    m = meshes[0]
    v = m.vertices.detach().cpu().numpy().astype(np.float64)
    f = m.faces.detach().cpu().numpy().astype(np.int64)
    return v, f


def occupancy64(v, f):
    return C.mesh_to_voxel_indices(v, f, GRID)          # 이미 canonical 프레임


def region_split_iou(vi_a, vi_b, edit_lo, edit_hi):
    """복셀 집합을 편집 bbox 안/밖으로 나눠 각각 IoU."""
    def split(vi):
        inside = np.all((vi >= edit_lo) & (vi <= edit_hi), axis=1)
        return vi[inside], vi[~inside]
    ai, ao = split(vi_a); bi, bo = split(vi_b)
    return {"inside_iou": round(E.coords_iou(ai, bi), 3),
            "outside_iou": round(E.coords_iou(ao, bo), 3),
            "inside_n": [int(len(ai)), int(len(bi))],
            "outside_n": [int(len(ao)), int(len(bo))]}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pipe, tex = load()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    image = Image.open(REFIMG).convert("RGB")

    # 1) 원본 에셋 → SLat
    raw = trimesh.load(ASSET, force="mesh", process=False)
    pm = tex.preprocess_mesh(trimesh.Trimesh(
        vertices=np.asarray(raw.vertices), faces=np.asarray(raw.faces), process=False))
    t0 = time.time()
    with torch.no_grad():
        slat = tex.encode_shape_slat(pm, resolution=SHAPE_RES)
    print(f"[encode] {slat.coords.shape[0]} tokens ({time.time()-t0:.0f}s)", flush=True)

    # 2) 편집 마스크: 가장 넓은 축의 상위 25% (날개끝 쪽)
    xyz = slat.coords[:, 1:].float()
    ext = (xyz.max(0).values - xyz.min(0).values)
    axis = int(ext.argmax())
    edit_mask = R.quantile_halfspace_mask(slat.coords, axis, 0.75, "greater")
    thr = float(torch.quantile(xyz[:, axis], 0.75))
    n_edit = int(edit_mask.sum())
    print(f"[mask] axis={axis} thr={thr:.1f} edit={n_edit}/{len(edit_mask)} tokens", flush=True)
    # bbox (검증용 분할 기준)
    lo = np.array([0, 0, 0]); hi = np.array([GRID] * 3, float)
    lo_e = lo.copy(); lo_e[axis] = thr

    # 3) 영역 재생성
    t0 = time.time()
    with torch.no_grad():
        edited = R.edit_region(pipe, image, slat, edit_mask, SEED)
    print(f"[edit] done ({time.time()-t0:.0f}s)", flush=True)

    # 4) 검증 ① — 마스크 밖 latent 변화량 (정확히 0 이어야)
    keep = (~edit_mask).to(edited.feats.device)
    outside_delta = (edited.feats[keep] - slat.feats.to(edited.feats.device)[keep]).abs().max().item()
    inside_delta = (edited.feats[edit_mask.to(edited.feats.device)]
                    - slat.feats.to(edited.feats.device)[edit_mask.to(edited.feats.device)]).abs().mean().item()
    print(f"[verify-latent] outside max|Δ|={outside_delta}  inside mean|Δ|={inside_delta:.4f}", flush=True)

    # 5) 검증 ②③ — 디코드 후 점유 비교 + 형상 지표
    with torch.no_grad():
        v0, f0 = decode_mesh(pipe, slat)
        v1, f1 = decode_mesh(pipe, edited)
    vi0, vi1 = occupancy64(v0, f0), occupancy64(v1, f1)
    split = region_split_iou(vi0, vi1, lo_e, hi)
    ch = E.chamfer_distance(v0, f0, v1, f1, n=50000)
    si = E.silhouette_iou(v0, f0, v1, f1)
    print(f"[verify-decode] outside_iou={split['outside_iou']} inside_iou={split['inside_iou']} "
          f"chamfer={ch['chamfer']:.5f} silh={si['silhouette_iou']:.3f}", flush=True)

    # 6) 산출물
    for name, (vv, ff) in {"orig_decoded": (v0, f0), "edited_decoded": (v1, f1)}.items():
        trimesh.Trimesh(vertices=vv, faces=ff, process=False).export(f"{OUTDIR}/{name}.glb")
    results = {
        "asset": os.path.basename(ASSET), "seed": SEED, "grid": GRID,
        "tokens": int(slat.coords.shape[0]), "edit_tokens": n_edit, "edit_axis": axis,
        "latent": {"outside_max_abs_delta": outside_delta,
                   "inside_mean_abs_delta": round(inside_delta, 4)},
        "occupancy64": split,
        "mesh": {"chamfer": round(ch["chamfer"], 5),
                 "silhouette_iou": round(si["silhouette_iou"], 3)},
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    with open(os.path.join(HERE, "phase2_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[done]", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
