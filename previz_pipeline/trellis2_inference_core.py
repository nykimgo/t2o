"""TRELLIS.2 inference core (Phase 3) — image-to-3D backend for the t2o pipeline.

Drop-in replacement for :class:`TrellisInferenceCore` (TRELLIS v1, text-to-3D).
TRELLIS.2 is image-conditioned, so this core inserts a Text→Image (FLUX) bridge
before the 3D stage:

    prompt --(FLUX)--> reference image --(TRELLIS.2)--> MeshWithVoxel (PBR)
           --> o_voxel.to_glb (PNG) --> GLB  --> [downstream glb_to_usd_native]

Design
------
* Subclasses ``TrellisInferenceCore`` to REUSE all path/naming/manifest/CSV/
  generation-json logic unchanged; only pipeline loading and the per-object
  generation+export+render are overridden.
* GLB is exported with ``extension_webp=False`` (PNG textures) so the downstream
  GLB->USD step gets plain .png/.jpg (no EXT_texture_webp). The converter is now
  the native trimesh+pxr path (glb_to_usd_native.py), which also extracts jpg
  textures — PNG/jpg keeps that simple and portable. PBR metallic/roughness maps
  ARE glTF-core and convert fine.
* HDRI .exr is read via the ``OpenEXR`` package (the installed
  opencv-python-headless is built with OpenEXR:NO). Render is best-effort.

E2E 검증: previz_pipeline/e2e_test.py 로 확인됨 (TRELLIS2_MIGRATION.md 참고).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def _detect_cuda_arch(default: str = "8.0") -> str:
    """Compute capability of GPU 0 as a TORCH_CUDA_ARCH_LIST value (e.g. '8.9').

    Queried via nvidia-smi because this runs before ``import torch`` — the arch
    list must be set before any JIT extension build. A100=8.0, RTX 4090=8.9,
    H100=9.0. Falls back to ``default`` when nvidia-smi is unavailable.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        cap = out.stdout.strip().splitlines()[0].strip()
        return cap if cap else default
    except Exception:
        return default


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _detect_cuda_arch())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image

import trellis_inference_core as _base_mod
from trellis_inference_core import TrellisInferenceCore  # base: reuse all helpers

# The base module guards on TRELLIS **v1** availability (its module-level import
# of trellis.pipelines). In the trellis2 env v1 is absent, so the base's
# __init__ guard would raise. We supply our own v2 pipeline, so neutralize it.
_base_mod.TRELLIS_AVAILABLE = True

# The base module sets ATTN_BACKEND=xformers (TRELLIS v1). TRELLIS.2 must use
# flash_attn (xformers isn't installed in the trellis2 env). Force it BEFORE
# importing trellis2, whose backend is selected at import time.
os.environ["ATTN_BACKEND"] = "flash_attn"
os.environ["SPCONV_ALGO"] = "native"

try:
    import imageio
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    import o_voxel
    TRELLIS2_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    print(f"❌ TRELLIS.2 import 실패: {e}")
    print("💡 trellis2_src 에서 실행하거나 PYTHONPATH를 설정하세요")
    TRELLIS2_AVAILABLE = False

# Repo root (= <repo>/t2o_pipeline), derived so the same checkout works on any
# server. Absolute paths here used to be pinned to the original host.
_REPO_ROOT = Path(__file__).resolve().parents[1]

# Default HDRI for PBR preview render (best-effort only).
_DEFAULT_HDRI = str(_REPO_ROOT / "trellis2_src/assets/hdri/forest.exr")
_AABB = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]


def _ensure_o_voxel_no_nvdiffrast() -> None:
    """설치본 o_voxel 이 nvdiffrast 버전이면 patches/ 를 자동 적용한다.

    setup.sh 로 env 를 재구축하면 site-packages 의 o_voxel 이 원본(= nvdiffrast,
    NVIDIA 비상업 라이선스)으로 되돌아간다. 사람이 ENV_REBUILD_GUIDE §함정 6 을
    기억해 patch 스크립트를 다시 돌려야 하는 구조는 언젠가 반드시 잊히고, 잊혀도
    아무 에러 없이 라이선스 위반 상태로 조용히 돌아간다. 그래서 배송 경로 코드가
    로드 시점에 직접 확인하고, 스크립트(멱등·원본 검증 포함)를 자동 실행한다.
    """
    import o_voxel.postprocess as _pp
    if "uv_raster" in Path(_pp.__file__).read_text(encoding="utf-8"):
        return  # 이미 패치됨 (정상 상태)

    script = _REPO_ROOT / "patches" / "apply_o_voxel_no_nvdiffrast.sh"
    logging.warning("⚠️ o_voxel 이 nvdiffrast 원본 상태 (env 재구축?) — 패치 자동 적용")
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "o_voxel nvdiffrast 제거 패치 자동 적용 실패 — nvdiffrast(비상업 전용)로는 "
            f"진행하지 않는다. 수동 실행: {script}\n{r.stdout}\n{r.stderr}")

    import importlib
    importlib.reload(_pp)
    if "uv_raster" not in Path(_pp.__file__).read_text(encoding="utf-8"):
        raise RuntimeError(f"패치 적용 후에도 o_voxel 이 nvdiffrast 상태: {_pp.__file__}")
    logging.info("✅ o_voxel 패치 자동 적용 완료 (nvdiffrast 제거)")


# CC BY-NC 라 상업 사용 불가 — MIT 인 원저자 모델로 강제한다. 두 모델은 같은
# BiRefNet 아키텍처이고, 현행 §9 프롬프트 산출물에서 IoU 0.99+/bbox 이동 ≤2px 로
# 실질 동일함을 검증했다 (experiments/rembg_swap/).
_REMBG_NC = "briaai/RMBG-2.0"
_REMBG_MIT = "ZhengPeng7/BiRefNet"


def _ensure_rembg_commercial(model_path: str) -> None:
    """pipeline.json 의 rembg 가 비상업(briaai) 모델이면 MIT 모델로 고쳐 쓴다.

    hf_models/ 는 gitignore 대상이라 모델을 재다운로드하면 업스트림 기본값
    (briaai/RMBG-2.0, CC BY-NC)으로 조용히 되돌아간다. o_voxel 가드와 같은 이유로
    로드 시점에 검사해 자동 교정한다. from_pretrained 가 rembg 를 즉시 인스턴스화
    하므로 반드시 로드 **전에** json 을 고쳐야 NC 가중치가 로드조차 되지 않는다.
    """
    fixed = []
    for fname in ("pipeline.json", "texturing_pipeline.json"):
        p = Path(model_path) / fname
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if _REMBG_NC in text:
            p.write_text(text.replace(_REMBG_NC, _REMBG_MIT), encoding="utf-8")
            fixed.append(fname)
    if fixed:
        logging.warning(
            "⚠️ rembg 가 비상업 모델(%s)로 되돌아가 있었음 (모델 재다운로드?) — "
            "%s 를 %s 로 자동 교정", _REMBG_NC, "/".join(fixed), _REMBG_MIT)

# FLUX runs as a subprocess to keep its VRAM lifecycle separate from TRELLIS.2,
# but in the same env as this process (the separate t2i env was folded into
# trellis2 — ENV_REBUILD_GUIDE.md §6). Override with T2I_PYTHON / T2I_SCRIPT.
_T2I_PYTHON = os.environ.get("T2I_PYTHON", sys.executable)
_T2I_SCRIPT = os.environ.get(
    "T2I_SCRIPT", str(_REPO_ROOT / "previz_pipeline" / "text_to_image.py"))


def _read_exr_rgb(path: str) -> np.ndarray:
    """Read an .exr HDRI to an (H,W,3) float32 RGB array via the OpenEXR pkg."""
    import OpenEXR
    f = OpenEXR.File(path)
    return np.asarray(f.channels()["RGB"].pixels, dtype=np.float32)


class Trellis2InferenceCore(TrellisInferenceCore):
    """Image-to-3D core using TRELLIS.2-4B, with a FLUX Text→Image front-end."""

    def __init__(
        self,
        model_path: str = str(_REPO_ROOT / "hf_models" / "TRELLIS.2-4B"),
        base_output_dir: str = os.environ.get(
            "TRELLIS_BASE_OUTPUT", str(_REPO_ROOT / "t2o_results")),
        t2i_model_path: str = str(_REPO_ROOT / "hf_models" / "FLUX.1-schnell"),
        hdri_path: str = _DEFAULT_HDRI,
        pipeline_type: Optional[str] = None,   # None -> model default '1024_cascade'
    ) -> None:
        if not TRELLIS2_AVAILABLE:
            raise ImportError("TRELLIS.2 modules are not available")
        super().__init__(model_path=model_path, base_output_dir=base_output_dir)
        self.t2i_model_path = t2i_model_path
        self.hdri_path = hdri_path
        self.pipeline_type = pipeline_type
        self.envmap = None

    # -- overridden: load TRELLIS.2 + FLUX + envmap (replaces TrellisTextTo3D) --
    def load_pipeline(self) -> None:
        _ensure_o_voxel_no_nvdiffrast()  # env 재구축 후 패치 유실 자동 복구
        logging.info(f"🔄 Loading TRELLIS.2 pipeline from: {self.model_path}")
        src = self.model_path if os.path.exists(self.model_path) else "microsoft/TRELLIS.2-4B"
        if os.path.isdir(src):
            _ensure_rembg_commercial(src)  # 모델 재다운로드 후 NC rembg 복귀 자동 교정
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(src)
        if not os.path.isdir(src):
            # 허브 폴백은 업스트림 pipeline.json(= briaai/RMBG-2.0, CC BY-NC)을
            # 그대로 쓰므로 로드 후 rembg 인스턴스를 MIT 모델로 갈아끼운다.
            from trellis2.pipelines.rembg import BiRefNet as _BiRefNet
            self.pipeline.rembg_model = _BiRefNet(model_name=_REMBG_MIT)
            logging.warning("⚠️ 허브 폴백 로드 — rembg 를 %s 로 강제 교체", _REMBG_MIT)
        if torch.cuda.is_available():
            self.pipeline.cuda()
            logging.info("✅ TRELLIS.2 pipeline on GPU")
        else:
            logging.warning("ℹ️ GPU not available")

        # HDRI for PBR preview render (optional).
        try:
            self.envmap = EnvMap(torch.tensor(
                _read_exr_rgb(self.hdri_path), dtype=torch.float32, device="cuda"))
            logging.info(f"✅ HDRI envmap loaded: {self.hdri_path}")
        except Exception as e:
            logging.warning(f"⚠️ HDRI load failed (render will be skipped): {e}")
            self.envmap = None

    def _t2i_generate(self, prompt: str, seed: int, out_path: str) -> Image.Image:
        """Run FLUX in a subprocess and load the PNG it wrote.

        Kept out-of-process for VRAM, not for env isolation: FLUX now runs in
        this same env, and ending the process hands its VRAM back in full. That
        is what makes the single-card "generate all images, then lift" batch
        order work.

        VRAM: schnell in bf16 barely fits a 24GB card even with CPU offload
        (measured ~23.4GB peak on an empty 4090). TRELLIS.2 is already resident
        here, so sharing the card OOMs. Set ``T2I_GPU`` to hand FLUX its own
        device; on a single-GPU box leave it unset and the two stages must be
        split into separate runs instead.
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            _T2I_PYTHON, _T2I_SCRIPT,
            "--prompt", prompt, "--out", out_path,
            "--seed", str(int(seed)), "--model", self.t2i_model_path,
        ]
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        t2i_gpu = os.environ.get("T2I_GPU")
        if t2i_gpu:
            # The subprocess then sees it as cuda:0 regardless of the index.
            env["CUDA_VISIBLE_DEVICES"] = t2i_gpu
            logging.info(f"🎨 T2I (subprocess, GPU {t2i_gpu}): {prompt[:60]}")
        else:
            logging.info(f"🎨 T2I (subprocess): {prompt[:60]}")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(f"T2I subprocess failed (rc={proc.returncode}):\n{proc.stderr[-1500:]}")
        return Image.open(out_path).convert("RGB")

    # -- overridden: per-object generation for TRELLIS.2 (T2I -> mesh -> GLB) --
    def _generate_single(
        self,
        prompt: str,
        predefined_name: Optional[str],
        config: Dict,
        formats: List[str],
        postprocessing_config: Dict,
        llm_model: Optional[str] = None,
        record_context: Optional[Dict[str, str]] = None,
    ) -> Dict:
        import random
        start_time = time.time()

        object_name = predefined_name or self.generate_unique_name(prompt)
        seed_val = config.get("seed", "random")
        seed = random.randint(0, 999999) if str(seed_val).lower() == "random" else int(seed_val)

        context = record_context or {}
        scene_name = self._sanitize_path_segment(context.get("scene"), "scene_unknown")
        shot_name = self._sanitize_path_segment(context.get("shot"), "shot_unknown")
        target_dir_name = self._sanitize_path_segment(
            context.get("target_dir_name") or object_name, object_name or "item_unknown")

        # Output dirs — identical rule to the base (assets next to source USD).
        usd_file_path = context.get("usd_file_path")
        preview_dir = self._preview_output_path(scene_name, shot_name, target_dir_name)
        if usd_file_path:
            object_dir = Path(usd_file_path).resolve().parent / "assets" / target_dir_name
        else:
            object_dir = preview_dir
        object_dir.mkdir(parents=True, exist_ok=True)
        preview_dir.mkdir(parents=True, exist_ok=True)

        asset_label_name = self._resolve_asset_label_name(context, context.get("target_type") or "item")
        file_prefix = self._build_file_prefix(scene_name, shot_name, asset_label_name, seed)

        # --- Stage A: Text -> Image (FLUX) ---
        gen_start = time.time()
        ref_path = preview_dir / f"{file_prefix}_ref.png"
        image = self._t2i_generate(prompt, seed, str(ref_path))
        t2i_time = time.time() - gen_start
        logging.info(f"⏱️  [T2I] {object_name}: {t2i_time:.1f}s")

        # --- Stage B: Image -> 3D (TRELLIS.2) + 빌보드(깊이붕괴) 게이트 ---
        # TRELLIS.2 가 가끔(EXP1 실측 5/200=2.5%) 메시를 종잇장으로 붕괴시킨다.
        # 감지는 bbox 두께비 계산이라 무비용(~1ms)이므로 항상 수행·로그하고,
        # 자동 재시도(lift 시드 재롤 → 그래도면 T2I 이미지 재생성)는 MESH_GATE=0 으로
        # 끌 수 있다(기본 ON). 임계 0.02 는 EXP1 전수 스캔에서 무오탐 — 진짜 붕괴는
        # 전부 0.002 근처, 뱀처럼 원래 얇은 지오메트리는 0.036 이상이었다.
        # 근거: EXP1_rigging/EXP1_ANALYSIS.md §9-3·§10-2, flatness.csv
        def _lift_once(img, lift_seed):
            m = self.pipeline.run(
                img,
                seed=lift_seed,
                preprocess_image=True,                     # internal rembg
                pipeline_type=self.pipeline_type,
                sparse_structure_sampler_params=config.get("sparse_structure_sampler_params", {}),
                shape_slat_sampler_params=config.get("shape_slat_sampler_params", {}),
                tex_slat_sampler_params=config.get("tex_slat_sampler_params", {}),
            )[0]
            m.simplify(int(postprocessing_config.get("simplify_target", 16777216)))
            return m

        def _thickness_ratio(m) -> float:
            v = m.vertices
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            elif hasattr(v, "cpu"):
                v = v.cpu().numpy()
            v = np.asarray(v)
            ext = v.max(axis=0) - v.min(axis=0)
            mx = float(ext.max())
            return float(ext.min()) / mx if mx > 0 else 0.0

        gate_retry = os.environ.get("MESH_GATE", "1") != "0"
        gate_thr = float(os.environ.get("MESH_GATE_THRESHOLD", "0.02"))
        i2o_start = time.time()
        try:
            mesh = _lift_once(image, seed)
            ratio = _thickness_ratio(mesh)
            logging.info(f"📐 [GATE] {object_name}: 두께비 {ratio:.4f}")
            if ratio < gate_thr and gate_retry:
                # 1) lift 시드 재롤 — 샘플링 요인 배제 (~25s)
                logging.warning(
                    f"⚠️ [GATE] 깊이붕괴 감지({ratio:.4f} < {gate_thr}) → lift 시드 재롤")
                mesh2 = _lift_once(image, seed + 7919)
                r2 = _thickness_ratio(mesh2)
                if r2 >= gate_thr:
                    mesh, ratio = mesh2, r2
                else:
                    # 2) 이미지 자체가 원인(예: 완전 정측면 → 깊이 단서 0)
                    #    → T2I 재생성 후 재-lift (~55s). ref 이미지도 교체 저장된다.
                    logging.warning(
                        f"⚠️ [GATE] 재롤 후에도 붕괴({r2:.4f}) → T2I 이미지 재생성")
                    image = self._t2i_generate(prompt, seed + 7919, str(ref_path))
                    mesh3 = _lift_once(image, seed + 7919)
                    r3 = _thickness_ratio(mesh3)
                    # 최선의 것을 채택 (전부 붕괴면 로그만 남기고 진행 — 파이프라인을 죽이지 않는다)
                    mesh, ratio = max(((mesh2, r2), (mesh3, r3), (mesh, ratio)),
                                      key=lambda t: t[1])
                    if ratio < gate_thr:
                        logging.error(
                            f"❌ [GATE] 재시도 2회 후에도 깊이붕괴 지속({ratio:.4f}) — 그대로 진행")
                    else:
                        logging.info(f"✅ [GATE] 이미지 재생성으로 회복 (두께비 {ratio:.4f})")
            elif ratio < gate_thr:
                logging.warning(
                    f"⚠️ [GATE] 깊이붕괴 감지({ratio:.4f} < {gate_thr}) — MESH_GATE=0, 재시도 생략")
        except Exception as e:
            logging.error(f"❌ TRELLIS.2 run failed: {e}")
            raise
        i2o_time = time.time() - i2o_start
        logging.info(f"⏱️  [I2O] {object_name}: {i2o_time:.1f}s")
        generation_time = time.time() - gen_start

        # --- Stage C: PBR preview render (best-effort) ---
        render_start = time.time()
        video = None
        # 프리뷰(mp4/jpg)를 실제로 요청했을 때만 렌더한다. 이 게이트가 없으면
        # --formats glb 로도 120프레임 턴테이블 렌더(객체당 ~41s, 실측)를 그대로
        # 지불한다. GLB 산출물은 이 렌더와 무관하게 생성되므로 건너뛰어도 안전하다.
        want_preview = ("mp4" in formats) or ("jpg" in formats)
        if want_preview and self.envmap is not None:
            try:
                video = render_utils.make_pbr_vis_frames(
                    render_utils.render_video(mesh, envmap=self.envmap))
            except Exception as e:
                logging.warning(f"⚠️ PBR render failed (skipping): {e}")
        render_time = time.time() - render_start

        # --- Stage D: export GLB (PNG textures for portable GLB->USD) + previews ---
        saved_files: List[str] = []
        glb_path_saved: Optional[Path] = None
        save_start = time.time()

        if "glb" in formats:
            glb_filename = self._get_unique_filename(object_dir, f"{file_prefix}.glb")
            glb_path = object_dir / glb_filename
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=_AABB,
                decimation_target=int(postprocessing_config.get("decimation_target", 1000000)),
                texture_size=int(postprocessing_config.get("texture_size", 2048)),
                remesh=bool(postprocessing_config.get("remesh", True)),
                remesh_band=postprocessing_config.get("remesh_band", 1),
                remesh_project=postprocessing_config.get("remesh_project", 0),
            )
            glb.export(str(glb_path), extension_webp=False)  # PNG, NOT webp
            saved_files.append(str(glb_path))
            glb_path_saved = glb_path
            logging.info(f"💾 GLB saved (PNG textures): {glb_filename}")
            self._write_glb_meta(glb_path, {
                "run_id": context.get("run_id") or self.run_id,
                "object_path": context.get("object_path") or context.get("file_identifier"),
                "prompt_used": prompt,
                "reference_image": str(ref_path),
                "t2i_prompt": context.get("t2i_prompt"),
                "description_en": context.get("description_en"),
                "seed": seed,
                "backend": "trellis2",
            })

        if "mp4" in formats and video is not None:
            mp4_name = self._get_unique_filename(preview_dir, f"{file_prefix}_pbr.mp4")
            mp4_path = preview_dir / mp4_name
            imageio.mimsave(str(mp4_path), video, fps=15)
            saved_files.append(str(mp4_path))
            logging.info(f"💾 PBR video saved: {mp4_name}")

        if "jpg" in formats and video is not None:
            try:
                for sec in (2, 4, 6):
                    idx = sec * 15
                    if len(video) > idx:
                        thumb = self._get_unique_filename(preview_dir, f"{file_prefix}_{sec:03d}s.jpg")
                        Image.fromarray(video[idx]).save(str(preview_dir / thumb), "JPEG", quality=90)
                        saved_files.append(str(preview_dir / thumb))
            except Exception as e:
                logging.warning(f"⚠️ Thumbnail generation failed: {e}")

        save_time = time.time() - save_start
        total_time = time.time() - start_time
        logging.info(
            f"⏱️  [객체 합계] {object_name}: {total_time:.1f}s "
            f"(T2I {t2i_time:.1f}s / I2O {i2o_time:.1f}s / "
            f"render {render_time:.1f}s / save {save_time:.1f}s)")

        preview_only = [Path(p).name for p in saved_files
                        if Path(p).parent.resolve() == preview_dir.resolve()]
        self._write_generation_json(
            preview_dir, prompt=prompt, context=context or {}, seed=seed,
            glb_path=str(glb_path_saved) if glb_path_saved else None,
            preview_files=preview_only,
        )

        return {
            "prompt": prompt,
            "object_name": object_name,
            "seed": seed,
            "model_name": self.model_name,
            "llm_model": llm_model,
            "run_id": context.get("run_id") or self.run_id,
            "object_path": context.get("object_path") or context.get("file_identifier"),
            "reference_image": str(ref_path),
            "generation_time": round(generation_time, 2),
            "t2i_time": round(t2i_time, 2),
            "i2o_time": round(i2o_time, 2),
            "render_time": round(render_time, 2),
            "save_time": round(save_time, 2),
            "total_time": round(total_time, 2),
            "success": True,
            "saved_files": saved_files,
            "save_path": str(object_dir),
            "preview_path": str(preview_dir) if preview_dir != object_dir else None,
            "timestamp": datetime.now().isoformat(),
        }
