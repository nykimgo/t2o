---
name: t2o-pipeline-server-layout
description: "이 서버(4090×4)에서 t2o_pipeline/TRELLIS.2 를 돌리는 실제 배치 — 기존 trellis2 env 재사용, t2i env 신규, NEW_SERVER_SETUP.md 와 어긋나는 지점들"
metadata: 
  node_type: memory
  type: project
  originSessionId: 29995152-3fc1-44b5-b411-af8c45740532
  modified: 2026-07-21T05:44:33.617Z
---

2026-07-21 기준. repo: `/home/sr/previs_object/t2o_pipeline` (github nykimgo/t2o, 브랜치 `feature/trellis2`).

`NEW_SERVER_SETUP.md` 는 "새 서버에 env 를 새로 구축하라"고 쓰여 있지만 **이 서버는 이미 구축돼 있다.**
`conda env trellis2` 를 그대로 쓰면 된다 — o_voxel / flex_gemm / cumesh / flash_attn 2.7.3 이
4090(sm_89)에서 실제 커널 실행까지 검증됨. 문서 §3 의 몇 시간짜리 setup.sh 빌드는 불필요하다.
`/home/sr/TRELLIS.2` 가 문서 지정 SHA `75fbf018...` 과 정확히 일치하며, repo 안 `trellis2_src/` 는 별도 clone.

문서와 어긋나는 지점 (문서가 기존 A100 서버 기준이라 그렇다):
- **transformers 는 4.56.2 를 유지한다.** 문서 §3.3 함정 5 는 4.57.6 을 지시하지만 이 서버의 검증된 값은
  4.56.2 다 → [[trellis2-env-pins]]. 문서 지시를 그대로 따르지 말 것.
- **GPU arch 는 8.9** (4090). 문서·코드가 A100 기준 8.0 으로 하드코딩돼 있었다.
- `huggingface-cli` 는 이 서버 huggingface_hub 1.21 에서 **폐기되어 동작하지 않는다** → `hf download` 를 쓴다.
- HF 계정은 `raengs` 로 게이트 3종(dinov3-vitl16 / RMBG-2.0 / FLUX.1-schnell) 승인 완료 상태.

`trellis2` env 에 추가로 넣어야 했던 것 (둘 다 `pip install --no-deps` 로 핀 보호):
- `usd-core` — **없으면 `usd_parser.py` 가 에러 없이 regex 폴백으로 떨어진다.** 파싱 결과가
  11개(canonical 3 + shot override 8) → 3개로 조용히 줄고 `[DEBUG USD]` 로그가 사라진다.
  무증상이라 "성공"으로 오인하기 쉽다. 원 서버 로그와 항목 수를 대조해서 검증할 것.
- `OpenEXR` — 없으면 HDRI envmap 로드가 실패하고 렌더가 조용히 skip 된다(GLB 생성은 계속됨).

**Why:** 문서를 곧이곧대로 따르면 동작 중인 env 를 깨뜨리거나(transformers 강등) 몇 시간을 재빌드에 날린다.
그리고 위 두 패키지 누락은 예외가 아니라 조용한 결과 축소로 나타나서 알아채기 어렵다.

**How to apply:** 이 repo 작업을 이어받을 때 env 구축 단계를 건너뛰고 바로 `conda activate trellis2` +
`export PYTHONPATH=<repo>/trellis2_src:<repo>/previz_pipeline` 로 시작하라.
