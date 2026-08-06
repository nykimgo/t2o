"""Phase 0-1 / 0-2 측정 러너.

0-1: GLB → preprocess_mesh → encode_shape_slat → decode_shape_slat → mesh.
     원본(preprocess 후)과 디코드 메시의 Chamfer distance + 실루엣 IoU.
0-2: 원본/디코드 메시를 ss_res 로 복셀화 → coords 집합 IoU.

부분 로드: shape_slat_encoder / shape_slat_decoder 두 모델만 (flow model 제외).
low_vram=True 로 배포와 동일하게 모델을 스텝마다 GPU↔CPU 이동 → 24GB 안전.

실행: conda activate trellis2; CUDA_VISIBLE_DEVICES=0 python edit3d/phase0_roundtrip.py
"""
import os, sys, json, time, traceback
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                       # t2o_pipeline
sys.path.insert(0, os.path.join(REPO, "trellis2_src"))
sys.path.insert(0, HERE)

import torch
import trimesh
import eval as E
import coords as C

ROOT = os.path.join(REPO, "hf_models", "TRELLIS.2-4B")
ENC_CKPT = os.path.join(ROOT, "ckpts", "shape_enc_next_dc_f16c32_fp16")
DEC_CKPT = os.path.join(ROOT, "ckpts", "shape_dec_next_dc_f16c32_fp16")
SHAPE_RES = 1024        # encode/decode 해상도
SS_RES = 32             # 배포 pipeline_type=1024_cascade → ss_res=32 (0-3 확정)
DEVICE = torch.device("cuda")


def load_minimal_pipelines():
    from trellis2 import models
    from trellis2.pipelines.trellis2_texturing import Trellis2TexturingPipeline
    from trellis2.pipelines.trellis2_image_to_3d import Trellis2ImageTo3DPipeline

    print("[load] shape_slat_encoder ...", flush=True)
    enc = models.from_pretrained(ENC_CKPT).eval()
    print("[load] shape_slat_decoder ...", flush=True)
    dec = models.from_pretrained(DEC_CKPT).eval()

    tex = Trellis2TexturingPipeline({"shape_slat_encoder": enc})
    tex.low_vram = True                 # 메서드가 스텝마다 GPU↔CPU
    tex._device = DEVICE
    i2o = Trellis2ImageTo3DPipeline({"shape_slat_decoder": dec})
    i2o.low_vram = True
    i2o._device = DEVICE
    return tex, i2o


def mesh_np(m):
    """trellis Mesh / trimesh → (vertices, faces) numpy."""
    v = m.vertices.detach().cpu().numpy() if torch.is_tensor(m.vertices) else np.asarray(m.vertices)
    f = m.faces.detach().cpu().numpy() if torch.is_tensor(m.faces) else np.asarray(m.faces)
    return v.astype(np.float64), f.astype(np.int64)


def run_item(tex, i2o, item, root):
    path = os.path.join(root, item["path"])
    rec = {"id": item["id"], "category": item["category"], "geometry": item["geometry"],
           "quality": item["quality"], "path": item["path"]}
    t0 = time.time()

    scene_or_mesh = trimesh.load(path, force="mesh", process=False)
    raw_v = np.asarray(scene_or_mesh.vertices, dtype=np.float64)
    raw_f = np.asarray(scene_or_mesh.faces, dtype=np.int64)
    rec["orig_verts"], rec["orig_faces"] = int(len(raw_v)), int(len(raw_f))

    pm = tex.preprocess_mesh(trimesh.Trimesh(vertices=raw_v, faces=raw_f, process=False))
    ov, of = np.asarray(pm.vertices, dtype=np.float64), np.asarray(pm.faces, dtype=np.int64)

    with torch.no_grad():
        slat = tex.encode_shape_slat(pm, resolution=SHAPE_RES)
        rec["slat_voxels"] = int(slat.coords.shape[0])
        meshes, _subs = i2o.decode_shape_slat(slat, SHAPE_RES)
    dv, df = mesh_np(meshes[0])
    rec["decoded_verts"], rec["decoded_faces"] = int(len(dv)), int(len(df))

    # 0-1: 형상 왕복 손실 (단위 큐브 정규화 후)
    rec["chamfer"] = E.chamfer_distance(ov, of, dv, df, n=50000)
    rec["silhouette"] = E.silhouette_iou(ov, of, dv, df, res=256, n=200000)

    # 0-2: 좌표 왕복 (ss_res 복셀 집합 IoU)
    try:
        vi_orig = C.mesh_to_voxel_indices(ov, of, SS_RES)
        vi_dec = C.mesh_to_voxel_indices(dv, df, SS_RES)
        rec["coords_iou_ss"] = E.coords_iou(vi_orig, vi_dec)
        rec["coords_orig_n"], rec["coords_dec_n"] = int(len(vi_orig)), int(len(vi_dec))
    except Exception as e:
        rec["coords_iou_ss"] = None
        rec["coords_error"] = f"{type(e).__name__}: {e}"

    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    with open(os.path.join(HERE, "eval_set.json")) as f:
        cfg = json.load(f)
    root, items = cfg["root"], cfg["items"]

    tex, i2o = load_minimal_pipelines()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results = []
    for it in items:
        try:
            rec = run_item(tex, i2o, it, root)
            ch = rec["chamfer"]["chamfer"]
            si = rec["silhouette"]["silhouette_iou"]
            print(f"[ok] {it['id']:<12} chamfer={ch:.5f}  silhouette_iou={si:.3f}  "
                  f"coords_iou_ss={rec['coords_iou_ss']}  ({rec['seconds']}s)", flush=True)
        except Exception as e:
            rec = {"id": it["id"], "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()}
            print(f"[FAIL] {it['id']}: {rec['error']}", flush=True)
        results.append(rec)

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    out = {"config": {"shape_res": SHAPE_RES, "ss_res": SS_RES,
                      "enc": os.path.basename(ENC_CKPT), "dec": os.path.basename(DEC_CKPT),
                      "peak_vram_gb": round(peak, 2) if peak else None},
           "results": results}
    outp = os.path.join(HERE, "phase0_results.json")
    with open(outp, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[done] peak_vram={out['config']['peak_vram_gb']}GB  →  {outp}", flush=True)


if __name__ == "__main__":
    main()
