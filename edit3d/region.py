"""Phase 2 — C: Region-Specific Editing (복셀 마스크 + 마스킹 샘플러).

논문(arXiv:2412.01506 §3.4) 의 영역편집을 v2 에 맞게 구현:
- stage-1 encoder 부재(PLAN §0-3) → coords 집합 직접 조작(§4a)으로 우회
- 개입 지점은 FlowEulerSampler.sample() 의 `sample = out.pred_x_prev` 한 곳(§0-4)
- 배포 샘플러는 FlowEulerGuidanceIntervalSampler(sigma_min=1e-5) — CFG 는 _inference_model
  오버라이드 체인이라 sample() 루프만 재구현하면 guidance 는 그대로 작동한다.

노이즈 규약(중요): flow_euler.py:30 실코드는
    x_t = (1-t)·x0 + (σ_min + (1-σ_min)·t)·ε
다. PLAN §4b 의 단순식 (1-t)x0 + t·ε 이 아니라 이 식으로 마스크 밖을 되돌린다.
마지막에 마스크 밖을 원본 x0 로 **정확히** 앵커해 §4c 의 '변화량 정확히 0' 을 보장한다.

trellis2_src 는 읽기 전용 — 서브클래싱만.
"""
from __future__ import annotations
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict


# ---- §4a: coords 집합 조작 --------------------------------------------------

def bbox_mask(coords: torch.Tensor, bbox_min, bbox_max) -> torch.Tensor:
    """coords (N,4) `[b,x,y,z]` → bool (N,). bbox 내부(=편집 영역)가 True.

    bbox 는 SLat 그리드 인덱스 기준(1024 해상도면 [0,64)). min/max 포함 범위."""
    xyz = coords[:, 1:]
    lo = torch.as_tensor(bbox_min, device=coords.device, dtype=xyz.dtype)
    hi = torch.as_tensor(bbox_max, device=coords.device, dtype=xyz.dtype)
    return ((xyz >= lo) & (xyz <= hi)).all(dim=1)


def quantile_halfspace_mask(coords: torch.Tensor, axis: int, q: float,
                            side: str = "greater") -> torch.Tensor:
    """축 분위수 기반 반공간 마스크 (예: 날개 끝 쪽 25%). axis ∈ {0,1,2}=x,y,z."""
    vals = coords[:, 1 + axis].float()
    thr = torch.quantile(vals, q)
    return vals > thr if side == "greater" else vals < thr


def fill_bbox_coords(bbox_min, bbox_max, batch: int = 0, device="cuda") -> torch.Tensor:
    """§4a '추가': bbox 내부를 복셀로 채운 coords (N,4). stage-2 가 내부 형상을 결정한다."""
    axes = [torch.arange(int(bbox_min[i]), int(bbox_max[i]) + 1, device=device)
            for i in range(3)]
    g = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3).int()
    b = torch.full((g.shape[0], 1), batch, device=device, dtype=g.dtype)
    return torch.cat([b, g], dim=1)


def merge_coords(base_coords: torch.Tensor, add_coords: torch.Tensor):
    """coords 합집합 + '새로 추가된 행' 마스크 반환. (추가 행은 x0 이 없으므로 항상 편집 영역.)"""
    merged = torch.cat([base_coords, add_coords], dim=0).unique(dim=0)
    # 원본에 있던 행 판별 (해시로)
    def key(c):  # (N,4) int → (N,) int64 해시 (그리드 ≤ 2^10 가정)
        c = c.long()
        return ((c[:, 0] << 30) | (c[:, 1] << 20) | (c[:, 2] << 10) | c[:, 3])
    base_keys = set(key(base_coords).tolist())
    is_new = torch.tensor([k not in base_keys for k in key(merged).tolist()],
                          device=merged.device)
    return merged, is_new


# ---- §4b: 마스킹 샘플러 -----------------------------------------------------

def make_region_masked_sampler_class(BaseSampler):
    """배포 샘플러 클래스(FlowEulerGuidanceIntervalSampler)를 받아 마스킹 버전을 만든다.

    클래스를 동적으로 만드는 이유: trellis2_src 를 import 경로에 넣기 전엔 base 를 참조할 수
    없고, edit3d 모듈은 trellis2 없이도 import 가능해야 하기 때문(경량 유틸과 분리)."""

    class RegionMaskedSampler(BaseSampler):
        """마스크 밖(`edit_mask=False`)을 매 스텝 노이즈된 원본으로 되돌리는 인페인팅 샘플러.

        모델은 전체 토큰을 본다(경계 연속성) — 결과만 마스크 밖이 원본으로 고정된다.
        """

        @torch.no_grad()
        def sample(
            self,
            model,
            noise,                      # SparseTensor(feats=randn, coords=전체 coords)
            cond,
            neg_cond,
            x0_feats: torch.Tensor,     # (N,C) 정규화된 원본 latent. 신규 coords 행은 값 무시됨
            edit_mask: torch.Tensor,    # (N,) bool. True=재생성, False=원본 고정
            steps: int = 50,
            rescale_t: float = 1.0,
            guidance_strength: float = 3.0,
            guidance_interval=(0.0, 1.0),
            verbose: bool = True,
            tqdm_desc: str = "Region-masked sampling",
            **kwargs,
        ):
            keep = ~edit_mask
            eps_feats = noise.feats            # 고정 ε (초기 노이즈 재사용)
            sample = noise
            t_seq = np.linspace(1, 0, steps + 1)
            t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
            t_seq = t_seq.tolist()   # 파이썬 float 필수 — np.float64*SparseTensor 는 numpy 경로로 새어 터진다
            t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(steps)]
            ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
            for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
                out = self.sample_once(
                    model, sample, t, t_prev, cond,
                    neg_cond=neg_cond, guidance_strength=guidance_strength,
                    guidance_interval=guidance_interval, **kwargs)
                feats = out.pred_x_prev.feats.clone()
                # 실코드 규약: x_t = (1-t)x0 + (σ_min+(1-σ_min)t)ε   (flow_euler.py:30)
                sigma = self.sigma_min + (1 - self.sigma_min) * t_prev
                renoised = (1 - t_prev) * x0_feats + sigma * eps_feats
                feats[keep] = renoised[keep]
                sample = out.pred_x_prev.replace(feats)
                ret.pred_x_t.append(sample)
                ret.pred_x_0.append(out.pred_x_0)
            # 정확 앵커: σ_min 잔차 제거 → 마스크 밖 latent 변화량 == 0 (§4c 판별식)
            feats = sample.feats.clone()
            feats[keep] = x0_feats[keep]
            ret.samples = sample.replace(feats)
            return ret

    RegionMaskedSampler.__name__ = f"RegionMasked{BaseSampler.__name__}"
    return RegionMaskedSampler


# ---- 상위 편의 함수 ---------------------------------------------------------

@torch.no_grad()
def edit_region(pipe, image, slat, edit_mask: torch.Tensor, seed: int,
                extra_coords: torch.Tensor = None,
                preprocess_image: bool = True,
                sampler_params: dict = None):
    """기존 SLat 의 마스크 영역만 재생성 (형상 latent 만; HR 단일 단계).

    Args:
        pipe: Trellis2ImageTo3DPipeline (shape_slat_flow_model_1024 + decoder 필요)
        slat: encode_shape_slat 결과 (un-normalized SparseTensor, coords [0,64) 그리드)
        edit_mask: slat.coords 행 기준 bool. extra_coords 를 주면 그 행들은 자동 True
        extra_coords: §4a '추가' — bbox 채움 coords (없으면 None)
    Returns:
        edited_slat (un-normalized, decode_shape_slat 에 바로 사용 가능)
    """
    from trellis2.modules.sparse import SparseTensor

    device = pipe.device
    mean = torch.tensor(pipe.shape_slat_normalization['mean'])[None].to(device)
    std = torch.tensor(pipe.shape_slat_normalization['std'])[None].to(device)

    coords = slat.coords
    orig_feats = slat.feats.to(device)
    x0 = ((orig_feats - mean) / std)                    # 정규화 공간으로
    if extra_coords is not None:
        coords, is_new = merge_coords(coords, extra_coords.to(coords.device))
        # 병합 후 행 정렬이 바뀌므로 x0/mask 를 재배열: 원본 행은 lookup, 신규 행은 0
        def key(c):
            c = c.long()
            return ((c[:, 0] << 30) | (c[:, 1] << 20) | (c[:, 2] << 10) | c[:, 3])
        old_map = {k: i for i, k in enumerate(key(slat.coords).tolist())}
        n, Cch = coords.shape[0], x0.shape[1]
        x0_full = torch.zeros(n, Cch, device=device, dtype=x0.dtype)
        orig_full = torch.zeros(n, Cch, device=device, dtype=orig_feats.dtype)
        mask_full = torch.ones(n, dtype=torch.bool, device=device)   # 신규 행 = 편집
        for i, k in enumerate(key(coords).tolist()):
            j = old_map.get(k)
            if j is not None:
                x0_full[i] = x0[j]
                orig_full[i] = orig_feats[j]
                mask_full[i] = bool(edit_mask[j])
        x0, orig_feats, edit_mask, slat_coords = x0_full, orig_full, mask_full, coords
    else:
        slat_coords = coords
        edit_mask = edit_mask.to(device)

    if preprocess_image:
        image = pipe.preprocess_image(image)
    torch.manual_seed(seed)
    cond = pipe.get_cond([image], 1024)

    flow = pipe.models['shape_slat_flow_model_1024']
    noise = SparseTensor(
        feats=torch.randn(slat_coords.shape[0], flow.in_channels).to(device),
        coords=slat_coords.int(),
    )

    Masked = make_region_masked_sampler_class(type(pipe.shape_slat_sampler))
    sampler = Masked(sigma_min=pipe.shape_slat_sampler.sigma_min)
    params = {**pipe.shape_slat_sampler_params, **(sampler_params or {})}

    if pipe.low_vram:
        flow.to(device)
    out = sampler.sample(flow, noise, **cond, x0_feats=x0, edit_mask=edit_mask, **params)
    if pipe.low_vram:
        flow.cpu()

    edited = out.samples * std + mean                    # de-normalize
    # 정확 앵커(비정규화 공간): 정규화 왕복 fp 잔차(~1e-6)까지 제거 → 마스크 밖 Δ == 정확히 0 (§4c)
    feats = edited.feats.clone()
    keep = ~edit_mask
    feats[keep] = orig_feats[keep].to(feats.dtype)
    return edited.replace(feats)
