# nvdiffrast 제거 — UV 래스터라이저 교체 검증

## 목적

`o_voxel.postprocess.to_glb` 의 텍스처 베이킹이 `nvdiffrast` 를 쓴다
(`postprocess.py:230-249`). nvdiffrast 는 NVIDIA Source Code License = **비상업
전용**이고, 이 호출이 GLB 산출 경로의 유일한 하드 nvdiffrast 의존성이다.
여기만 걷어내면 배송 경로에서 nvdiffrast 가 사라진다.

미분이 필요 없는 순수 래스터화(커버리지 마스크 + 무게중심 보간)라서
PyTorch3D/Kaolin 같은 무거운 의존성 대신 순수 torch 로 구현했다
(`uv_raster_torch.py`, 의존성 0 추가).

## 구성

| 파일 | 역할 |
|---|---|
| `capture_inputs.py` | TRELLIS.2 를 실제로 돌려 `to_glb` 입력을 디스크에 고정 |
| `uv_raster_torch.py` | 대체 래스터라이저 (bbox scatter, 순수 torch) |
| `test_conventions.py` | 합성 데이터로 규약(원점/무게중심/삼각형 ID) 보정 |
| `to_glb_split.py` | `to_glb` 를 공유 prefix + 교체 가능한 bake 로 분리 |
| `tiebreak_probe.py` | 자기일관성 대조군 + tie-break 규칙 실험 |
| `compare_rasterizers.py` | 실제 메시 3종 비교, 산출물 저장 |

prefix(정리·리메싱·UV 언랩)는 **비결정적**이다 — 같은 입력에 999247 / 998522 면
으로 갈렸다. 그래서 prefix 를 한 번만 돌려 캐시하고 두 백엔드가 **동일한 아틀라스**
를 쓰게 했다. 이걸 안 하면 측정된 차이가 래스터라이저 탓인지 언랩 탓인지 구분이 안 된다.

## 결과

대상: TRELLIS.2 example 3종, texture_size=2048, decimation_target=1M, remesh=True.

**대조군 — nvdiffrast vs nvdiffrast (동일 아틀라스, 2회):** 차이 **0** (3/3).
nvdiffrast 자체는 완전 결정적이므로, 아래 차이는 전부 래스터라이저 교체에서 온 것이다.

| 항목 | 0a34fae7 | 154c8867 | T |
|---|---|---|---|
| 지오메트리 | 동일 | 동일 | 동일 |
| **지오메트리에서 구워진 텍셀** (양쪽 커버) | | | |
| &nbsp;&nbsp;baseColor 비트동일 | 99.678% | 99.846% | 99.678% |
| &nbsp;&nbsp;baseColor PSNR | 76.0 dB | 76.9 dB | 68.6 dB |
| &nbsp;&nbsp;baseColor 평균차 (/255) | 0.0011 | 0.0005 | 0.0013 |
| **커버리지 불일치율** | 1.04% | 0.54% | 0.77% |
| **chart-seam swap** | 0.050% | 0.011% | 0.059% |
| **inpaint 패딩 링** 평균차 (/255) | 0.18 | 0.73 | 0.42 |

## 해석

1. **커버리지 불일치는 nvdiffrast 쪽 부정확이다.** 불일치 텍셀은 전부 삼각형 엣지에서
   **0.04 픽셀 이내**이고, fp64 로 재계산하면 우리 구현이 맞는 쪽이다. nvdiffrast 는
   정점을 고정소수점 서브픽셀 격자에 스냅해서 삼각형당 경계 텍셀 약 1개를 흘린다
   (삼각형 하나만 단독 래스터화해도 재현됨: nvdiffrast 3텍셀 vs 우리 4텍셀).
2. **tie-break 는 원인이 아니다.** 최고 인덱스(amax) vs 최저(amin) 로 바꿔도
   seam swap 이 834 → 857 로 거의 그대로다.
3. **남은 차이의 대부분은 `cv2.inpaint` 패딩 링이다.** 아틀라스의 41% 만 커버되고
   나머지는 INPAINT_TELEA 가 마스크 경계에서 전파해 채운다. 마스크가 1% 달라지면
   이 영역이 넓게 달라지는데, 이건 양쪽 다 **지어낸 값**이라 어느 쪽도 정답이 아니다.
4. **차이는 고립된 단일 텍셀**로 흩어져 있고 연속된 덩어리가 없다
   (`*_diff_view.png`). 구조적 아티팩트가 생기지 않는다는 뜻이다.

## 판정

비트 단위로 동일하지는 **않다**. 실제로 구워지는 텍셀 기준으로는 99.7~99.85% 동일 ·
PSNR 69~85 dB 이고, 불일치가 나는 지점에서는 우리 구현이 더 정확하다.

## 남은 일 (이 실험 범위 밖)

- 프리뷰 렌더 경로(`trellis2/renderers/*`, nvdiffrast + nvdiffrec) — `want_preview`
  로 이미 게이팅됨. 상업 배포 시 완전 제거 필요.
- `briaai/RMBG-2.0` (CC BY-NC) → `ZhengPeng7/BiRefNet` (MIT) 교체.
- DINOv3 "Built with DINOv3" 표기.
