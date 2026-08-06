"""Phase 1 실측 — B: Detail Variation 의 형상 고정도.

질문: v2(1024_cascade, ss_res=32)에서 구조(32³ coords)를 고정하고 stage-2 를 재실행하면
      형상이 얼마나 흔들리는가? (PLAN §3 — v1 64³ 대비 더 흔들릴 것이라는 가설의 실측)

두 경로:
  A(캡처): 정상 sample_sparse_structure 로 32³ coords 를 뽑아 고정 → seed 만 바꿔 재실행.
           '구조 고정 looseness' 의 순수 측정.
  B(제품): 기존 에셋 GLB → coords.py.mesh_to_coords(32) 로 coords 주입(실제 편집 경로).

지표: 실루엣 IoU / Chamfer / bbox 종횡비 (원본·baseline 대비, seed 간).
실행: conda activate trellis2; CUDA_VISIBLE_DEVICES=0 python edit3d/phase1_variant.py
"""
import os, sys, json, time
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "trellis2_src"))
sys.path.insert(0, HERE)

import torch
import trimesh
from PIL import Image
import eval as E
import coords as C
import variant as V

PREVIS = os.path.dirname(REPO)          # /home/sr/previs_proj (movie_usd 는 여기 아래)
ROOT = os.path.join(REPO, "hf_models", "TRELLIS.2-4B")
ASSET = os.path.join(PREVIS, "movie_usd/hidden_time/scene_1/objects/assets/object_1/scene_1_seagull_209626.glb")
REFIMG = os.path.join(REPO, "t2o_results/TRELLIS.2-4B/20260804/run_134025_en/previews/scene_1/object_1/scene_1_seagull_209626_ref.png")
SS_RES = 32
SEEDS = [42, 123, 777]          # S0=42 = baseline
OUTDIR = os.path.join(HERE, "phase1_out")


def load_pipeline():
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline
    # tex flow/decoder 제외 → VRAM 절약 (형상만 측정)
    Trellis2ImageTo3DPipeline.model_names_to_load = [
        'sparse_structure_flow_model', 'sparse_structure_decoder',
        'shape_slat_flow_model_512', 'shape_slat_flow_model_1024', 'shape_slat_decoder',
    ]
    print("[load] Trellis2ImageTo3DPipeline (SS+shape, no tex) ...", flush=True)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(ROOT)
    pipe._device = torch.device("cuda")
    print(f"[load] done. low_vram={pipe.low_vram}", flush=True)
    return pipe


def bbox_extents(v):
    v = np.asarray(v); e = v.max(0) - v.min(0)
    return e / e.max()          # 최대축=1 로 정규화한 종횡비 벡터


def compare(a, b):
    ch = E.chamfer_distance(a["vertices"], a["faces"], b["vertices"], b["faces"], n=50000)
    si = E.silhouette_iou(a["vertices"], a["faces"], b["vertices"], b["faces"])
    ea, eb = bbox_extents(a["vertices"]), bbox_extents(b["vertices"])
    return {"chamfer": round(ch["chamfer"], 5),
            "silhouette_iou": round(si["silhouette_iou"], 3),
            "aspect_l1": round(float(np.abs(ea - eb).sum()), 4)}


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pipe = load_pipeline()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    image = Image.open(REFIMG).convert("RGB")

    # 원본 에셋 (preprocess 후, canonical 프레임) + 제품경로 coords(B)
    raw = trimesh.load(ASSET, force="mesh", process=False)
    from trellis2.pipelines.trellis2_texturing import Trellis2TexturingPipeline  # preprocess_mesh 규약
    pm = Trellis2TexturingPipeline.preprocess_mesh(pipe, trimesh.Trimesh(
        vertices=np.asarray(raw.vertices), faces=np.asarray(raw.faces), process=False))
    orig = {"vertices": np.asarray(pm.vertices, np.float64), "faces": np.asarray(pm.faces, np.int64)}
    viB = C.mesh_to_voxel_indices(orig["vertices"], orig["faces"], SS_RES)
    coordsB = C.indices_to_coords(viB)

    # 캡처경로 coords(A): 정상 SS, seed S0
    with torch.no_grad():
        coordsA_t = V.sample_structure_coords(pipe, image, SEEDS[0], ss_res=SS_RES)
    viA = coordsA_t[:, 1:].detach().cpu().numpy().astype(np.int64)
    coordsAB_iou = E.coords_iou(viA, viB)
    print(f"[coords] A(SS)={len(viA)}  B(voxelized)={len(viB)}  IoU(A,B)={coordsAB_iou:.3f}", flush=True)

    results = {"asset": os.path.basename(ASSET), "ss_res": SS_RES, "seeds": SEEDS,
               "coords": {"A_ss_n": int(len(viA)), "B_voxel_n": int(len(viB)),
                          "iou_A_B": round(coordsAB_iou, 3)},
               "A_captured": {}, "B_product": {}}

    def gen(tag, coords, seed):
        t0 = time.time()
        with torch.no_grad():
            out = V.run_variant_shape(pipe, image, coords, seed)
        out["seconds"] = round(time.time() - t0, 1)
        print(f"  [{tag}] seed={seed} res={out['res']} n_in={out['n_input']} "
              f"n_hr={out['n_hr']} faces={len(out['faces'])} ({out['seconds']}s)", flush=True)
        return out

    # 경로 A: 캡처 coords 고정, seed 변주
    A = {s: gen("A", coordsA_t, s) for s in SEEDS}
    baseA = A[SEEDS[0]]
    # 경로 B: 제품 coords 고정, seed 변주
    B = {s: gen("B", coordsB, s) for s in SEEDS}

    # 비교
    results["A_captured"]["baseline_vs_orig"] = compare(baseA, orig)
    results["A_captured"]["variant_vs_baseline"] = {
        str(s): compare(A[s], baseA) for s in SEEDS[1:]}
    # A seed 간 드리프트 (pairwise)
    pair = []
    ks = SEEDS
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            pair.append(compare(A[ks[i]], A[ks[j]]))
    results["A_captured"]["interseed_pairwise_mean"] = {
        "chamfer": round(float(np.mean([p["chamfer"] for p in pair])), 5),
        "silhouette_iou": round(float(np.mean([p["silhouette_iou"] for p in pair])), 3),
    }
    results["B_product"]["variant_vs_orig"] = {str(s): compare(B[s], orig) for s in SEEDS}

    # 정성용 GLB (형상만) 저장
    exports = {"orig": orig, "A_baseline": baseA,
               f"A_seed{SEEDS[1]}": A[SEEDS[1]], f"B_seed{SEEDS[0]}": B[SEEDS[0]]}
    saved = []
    for name, m in exports.items():
        p = os.path.join(OUTDIR, f"{name}.glb")
        trimesh.Trimesh(vertices=m["vertices"], faces=m["faces"], process=False).export(p)
        saved.append(p)
    results["exports"] = [os.path.relpath(p, REPO) for p in saved]

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    results["peak_vram_gb"] = round(peak, 2) if peak else None
    with open(os.path.join(HERE, "phase1_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] peak_vram={results['peak_vram_gb']}GB", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
