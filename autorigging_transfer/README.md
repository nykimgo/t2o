# AutoRigging 이관 패키지 — 시작 가이드

> 원 개발 서버: 4×RTX4090, Ubuntu (검증 수치는 이 환경 기준).
> 전체 설계·계약·테스트 계획은 `docs/DESIGN.md` — 이 문서는 "돌아가게 만드는 순서"만.

## 1. 패키지 구성

```
code/        rig_service.py            UniRig 상주 리깅 (배치+데몬) — 검증: 본 구조가
                                       프로세스 방식과 완전 동일, 소형 11s/대형 22s/에셋
             retarget.py, riglib.py    biped 리타게팅 (좌우반전 수정판, 제자리 처리 내장)
             motion_to_usd.py          모션 → USD 익스포트 (하드코딩 리스트 제거 필요 — DESIGN §6)
             movie_usd_pack.py         movie_usd 규칙 마감 (cm 단위·bin 텍스처·래퍼·검증)
             movie_char_render.py      align_and_transfer(웨이트 전이) 원본 — 발췌 대상
mocap/       16_15.bvh(walk 472f) 16_35.bvh(run 26f) 16_01.bvh(jump 323f)  — CMU
unirig/      UniRig 저장소 (로컬 패치 2건 적용 상태 — DESIGN §6 표 참조)
samples/     리깅 FBX 1종 + 규칙 usda(래퍼+레이어+bin) + usdz + 블렌더 뷰어 usda 2종
docs/        DESIGN.md (상세설계서)
```

## 2. 환경 구축 (unirig env)

```bash
conda create -n unirig python=3.11 -y && conda activate unirig
pip install torch==2.6.* --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124 lightning box python-box pyyaml tqdm numpy
pip install flash-attn --no-build-isolation   # 실패 시 프리빌드 휠 사용 (원 서버도 휠로 설치)
pip install torch_geometric torch_scatter torch_cluster  # PyG — cu124 휠 인덱스 사용
pip install bpy==4.2.*                        # 블렌더 파이썬 모듈 (USD 스켈레탈 익스포트는 4.1+)
pip install usd-core                          # pxr — env 통합의 핵심 (DESIGN §5)
```

스모크 테스트 (순서대로, 하나라도 실패하면 다음으로 넘어가지 말 것):

```bash
python -c "import torch, bpy; print(torch.cuda.is_available(), bpy.app.version_string)"
python -c "import bpy, pxr; print('pxr+bpy 공존 OK')"     # 실패 시 DESIGN §5 폴백(2-env)
UNIRIG_DIR=$PWD/unirig CUDA_VISIBLE_DEVICES=0 python code/rig_service.py \
    <아무 glb> --out-dir /tmp/rig_smoke                    # 첫 실행: HF 체크포인트 자동 다운로드
```

검증 기준: `/tmp/rig_smoke/*_skeleton.fbx` + `*_rigged.fbx` 생성, 로그에 로드 ~10s.
samples/rigs/ 의 FBX 와 같은 입력이면 본 수·head 좌표가 동일해야 한다(시드 12345).

## 3. 알아야 하는 함정 (전부 원 서버 실측)

1. **FLUX/TRELLIS 등 대형 모델과 동시 실행 금지** — RAM OOM 실사고. 단계는 직렬로.
2. bpy 는 파이썬 **종료 시** segfault 를 낼 수 있다(teardown) — 산출물 무해. 성공 판정은
   exit code 가 아니라 산출 로그로.
3. USD 익스포터는 armature 오브젝트 트랜스폼을 버린다 → 배치 보정은 USD 쪽에서
   (movie_usd_pack.py 방식). 이때 루트 prim 의 기존 Y-up 회전 op 를 덮지 말고 합성할 것.
4. movie_usd 는 **cm(metersPerUnit 0.01)** — 검증은 Mesh prim bbox 로 (SkelRoot extent
   힌트는 애니 전범위라 비교 금지).
5. run 모캡은 26프레임뿐 — 2초 채우려면 루프. BVH import 가 세운 frame_end 를 그대로
   쓰지 말 것(마지막 키프레임 클램프 — retarget.py 에 반영돼 있음).
6. UniRig 익명본 손 판정은 손가락 있는 52본에서 좌우분리 로직 필수 — riglib.py 반영분
   사용, 옛 방식 롤백 금지.

## 4. 개발 순서

DESIGN.md §10 마일스톤(M1 이식→M2 스캔→M3 biped→M4 비-biped→M5 단계화)을 따른다.
M1 완료 판정 = 위 스모크 + samples 와의 동등성 확인.
