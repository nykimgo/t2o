"""Phase 1 — B: Detail Variation (구조 고정, stage-2 재실행).

TRELLIS.2 배포는 `1024_cascade`(ss_res=32). run() 은 sample_sparse_structure 로 coords 를
만들지만, variant 은 그 자리에 **기존 에셋의 coords 를 주입**하고 나머지(stage-2 cascade)를 그대로 굴린다.

핵심(PLAN §0·Phase0): cascade 는 입력 32³ coords 로 LR 을 샘플한 뒤
`shape_slat_decoder.upsample(slat, ×4)` 로 hr_coords 를 slat 에서 **재유도**한다. 즉 입력 구조를
고정해도 고해상 구조는 seed 마다 다시 결정된다 → v2 variant 가 형상을 얼마나 흔드는지가 Phase 1 실측 대상.

trellis2_src 는 읽기 전용 — 여기서는 파이프라인 공개 메서드만 조합한다(원본 수정 없음).
"""
from __future__ import annotations
import numpy as np
import torch


def coords_to_tensor(coords, device) -> torch.Tensor:
    """(N,4) `[b,x,y,z]` numpy/tensor → int32 cuda 텐서 (SparseTensor.coords 규약)."""
    if torch.is_tensor(coords):
        return coords.to(device=device, dtype=torch.int32)
    return torch.as_tensor(np.asarray(coords), dtype=torch.int32, device=device)


def sample_structure_coords(pipe, image, seed: int, ss_res: int = 32,
                            preprocess_image: bool = True,
                            sparse_structure_sampler_params: dict = None) -> torch.Tensor:
    """정상 경로의 sparse structure coords 를 그대로 뽑아온다(캡처 경로 A: 정확한 원본 구조).

    run() 과 동일한 seed → 동일한 coords (torch.manual_seed 후 SS 샘플이 첫 RNG 소비)."""
    if preprocess_image:
        image = pipe.preprocess_image(image)
    torch.manual_seed(seed)
    cond_512 = pipe.get_cond([image], 512)
    return pipe.sample_sparse_structure(cond_512, ss_res, 1,
                                        sparse_structure_sampler_params or {})


def _cond(pipe, image, preprocess_image, seed):
    if preprocess_image:
        image = pipe.preprocess_image(image)
    torch.manual_seed(seed)
    cond_512 = pipe.get_cond([image], 512)
    cond_1024 = pipe.get_cond([image], 1024)
    return cond_512, cond_1024


def run_variant_shape(pipe, image, coords, seed: int,
                      preprocess_image: bool = True,
                      max_num_tokens: int = 49152,
                      shape_slat_sampler_params: dict = None):
    """구조 고정 shape variant (텍스처 생략, 형상만). 반환: dict(vertices, faces, res, n_input, n_hr).

    run() 의 1024_cascade 경로에서 sample_sparse_structure 만 주입 coords 로 대체.
    """
    device = pipe.device
    cond_512, cond_1024 = _cond(pipe, image, preprocess_image, seed)
    coords_t = coords_to_tensor(coords, device)
    n_input = int(coords_t.shape[0])

    shape_slat, res = pipe.sample_shape_slat_cascade(
        cond_512, cond_1024,
        pipe.models['shape_slat_flow_model_512'], pipe.models['shape_slat_flow_model_1024'],
        512, 1024,
        coords_t, shape_slat_sampler_params or {}, max_num_tokens,
    )
    n_hr = int(shape_slat.coords.shape[0])
    meshes, _subs = pipe.decode_shape_slat(shape_slat, res)
    m = meshes[0]
    v = m.vertices.detach().cpu().numpy() if torch.is_tensor(m.vertices) else np.asarray(m.vertices)
    f = m.faces.detach().cpu().numpy() if torch.is_tensor(m.faces) else np.asarray(m.faces)
    return {"vertices": v.astype(np.float64), "faces": f.astype(np.int64),
            "res": int(res), "n_input": n_input, "n_hr": n_hr}


def run_variant_full(pipe, image, coords, seed: int,
                     preprocess_image: bool = True,
                     max_num_tokens: int = 49152,
                     shape_slat_sampler_params: dict = None,
                     tex_slat_sampler_params: dict = None):
    """구조 고정 + 텍스처까지 (decode_latent → MeshWithVoxel). tex flow model 이 로드돼 있어야 함."""
    device = pipe.device
    cond_512, cond_1024 = _cond(pipe, image, preprocess_image, seed)
    coords_t = coords_to_tensor(coords, device)

    shape_slat, res = pipe.sample_shape_slat_cascade(
        cond_512, cond_1024,
        pipe.models['shape_slat_flow_model_512'], pipe.models['shape_slat_flow_model_1024'],
        512, 1024,
        coords_t, shape_slat_sampler_params or {}, max_num_tokens,
    )
    tex_slat = pipe.sample_tex_slat(
        cond_1024, pipe.models['tex_slat_flow_model_1024'], shape_slat,
        tex_slat_sampler_params or {},
    )
    torch.cuda.empty_cache()
    return pipe.decode_latent(shape_slat, tex_slat, res)  # List[MeshWithVoxel]
