"""briaai/RMBG-2.0 (CC BY-NC) vs ZhengPeng7/BiRefNet (MIT) — 교체 검증.

무엇이 하류로 가는가 (trellis2_image_to_3d.preprocess_image):
  1. alpha > 0.8*255 로 뽑은 bbox → 정사각 크롭
  2. 크롭된 RGB × alpha (프리멀티플라이) → DINOv3 컨디셔닝 입력

그래서 마스크 자체의 픽셀 일치보다 (a) bbox 가 얼마나 움직이는지,
(b) 최종 프리멀티플라이 이미지가 얼마나 달라지는지를 잰다.

입력은 배송 경로의 실물: FLUX.1-schnell 이 생성한 레퍼런스 이미지
(t2o_results/*_ref.png — 깨끗한 배경). 스트레스 테스트로 배경 있는
TRELLIS.2 예제 이미지도 하나 넣는다.

Run:
    conda activate trellis2
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=trellis2_src \
        python experiments/rembg_swap/compare_rembg.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]

# 기본: 현행 §9 프롬프트로 새로 뽑은 레퍼런스 (gen_fresh_refs.py).
# 구런(7~8월 초) 레퍼런스는 프롬프트 하드닝 이전이라 다중 객체 장면이 섞여 있고,
# 그런 입력에선 두 모델의 "무엇이 주 객체인가" 판단이 갈린다(측정으로 확인) —
# 어차피 image-to-3D 에 부적합한 입력이라 판정 기준으로 삼지 않는다.
IMAGES = sorted((Path(__file__).resolve().parent / "fresh_refs").glob("*.png")) or [
    _REPO / "trellis2_src/assets/example_image/T.png",
]
if len(sys.argv) > 1:  # 명시 경로가 오면 그것만
    IMAGES = [Path(p) for p in sys.argv[1:]]

MODELS = {
    "rmbg2 (NC)": "briaai/RMBG-2.0",
    "birefnet (MIT)": "ZhengPeng7/BiRefNet",
}


def bbox_of(alpha: np.ndarray) -> tuple:
    """preprocess_image 와 동일한 bbox 산출 (alpha > 0.8*255)."""
    pts = np.argwhere(alpha > 0.8 * 255)
    return (pts[:, 1].min(), pts[:, 0].min(), pts[:, 1].max(), pts[:, 0].max())


def premultiplied(img_rgba: Image.Image) -> np.ndarray:
    a = np.asarray(img_rgba).astype(np.float32) / 255
    return a[:, :, :3] * a[:, :, 3:4]


def main() -> None:
    from trellis2.pipelines.rembg import BiRefNet

    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모델을 순차 로드해 같은 입력에 돌린다.
    results = {}  # (model, image) -> RGBA
    for label, name in MODELS.items():
        print(f"🔄 loading {label}: {name}")
        model = BiRefNet(model_name=name)
        model.cuda()
        for img_path in IMAGES:
            img = Image.open(img_path).convert("RGB")
            # preprocess_image 와 동일하게 1024 상한 리사이즈
            scale = min(1, 1024 / max(img.size))
            if scale < 1:
                img = img.resize((int(img.width * scale), int(img.height * scale)),
                                 Image.Resampling.LANCZOS)
            results[(label, img_path.name)] = model(img.copy())
        model.cpu()
        del model
        torch.cuda.empty_cache()

    labels = list(MODELS)
    all_ok = True
    for img_path in IMAGES:
        n = img_path.name
        a = np.asarray(results[(labels[0], n)])[:, :, 3].astype(np.float32)
        b = np.asarray(results[(labels[1], n)])[:, :, 3].astype(np.float32)

        ta, tb = a > 204, b > 204              # bbox 임계값과 동일
        inter, union = (ta & tb).sum(), (ta | tb).sum()
        iou = inter / max(union, 1)

        ba, bb = bbox_of(a), bbox_of(b)
        bbox_shift = max(abs(x - y) for x, y in zip(ba, bb))
        diag = float(np.hypot(*a.shape))

        pa = premultiplied(results[(labels[0], n)])
        pb = premultiplied(results[(labels[1], n)])
        pm_diff = np.abs(pa - pb)
        mse = float((pm_diff ** 2).mean())
        psnr = float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)

        # 판정: 마스크 IoU 98%+, bbox 이동이 이미지 대각선의 1% 이하,
        # 프리멀티플라이 결과 PSNR 30dB+ (컨디셔닝 입력으로 사실상 동일).
        ok = iou >= 0.98 and bbox_shift <= 0.01 * diag and psnr >= 30
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {n}: IoU={iou:.4f}  "
              f"bbox_shift={bbox_shift}px (diag {diag:.0f})  "
              f"premult PSNR={psnr:.1f}dB  mean_diff={pm_diff.mean() * 255:.3f}/255")

        for label in labels:
            tag = "rmbg2" if "rmbg2" in label else "birefnet"
            results[(label, n)].save(out_dir / f"{Path(n).stem}_{tag}.png")

    print(f"\nRESULT: {'drop-in OK' if all_ok else 'DIFFERS — inspect out/'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
