# AutoRigging 모듈 — 실행 가이드·입출력 계약 (M5)

> 설계 정본: `../autorigging_transfer/docs/DESIGN.md` · 이관 가이드: `../autorigging_transfer/README.md`
> 이 서버(A100) 이식·검증: 2026-08-20 — 스모크 3종 + M1 동등성 + T1~T7 전부 통과.

## 실행

```bash
# 루트 런처 (env 활성화 불필요 — unirig env 파이썬을 직접 사용)
./autorig_generate.sh movie_usd/welcom2dmk/welcom2dmk.usda            # 영화 단위
./autorig_generate.sh movie_usd/welcom2dmk/scene_1/scene_1.usda       # 씬 단위
# 옵션: --dry-run(스캔만) --only <이름부분일치> --no-motion(리깅까지만)
```

- ⚠️ **FLUX/TRELLIS(object_generate.sh 등)와 동시 실행 금지** — RAM OOM 실사고. 단계 직렬.
- ⚠️ 이 서버 전역 `LD_LIBRARY_PATH=/root/previs_proj/usd/lib` 는 bpy 번들 MaterialX·libcuda 를
  가려 ImportError 를 낸다(실측) — 런처가 unset 처리. **수동 실행 시에도 반드시 unset**.
- bpy 는 종료 시 segfault 를 낼 수 있다(teardown, 산출물 무해) — 성공 판정은 리포트 기준.

## 입력 계약

| 대상 | 위치 규칙 | 판정 |
|---|---|---|
| 캐릭터 | `<영화>/characters/<char>.usda` + `assets/<char>/<char>.glb` | 무조건 수집 (biped) |
| 오브젝트 | `<영화>/scene_N/objects/object_M.usda` + `objects/assets/object_M/*.glb` | customData `rig_type` ∉ {static_object, 빈값} |

- rig_type enum(7클래스): biped/quadruped/hexapod/octopod/avian/serpentine/aquatic.
  구 enum 은 매핑+경고: bird→avian, insect→hexapod. 미지값·누락 → 스킵+리포트 경고.
- GLB 부재(빈 래퍼) → 스킵 + 사유 기록.

## 산출물 계약 (다음 파트 = action 구현 인수인계)

원본 트리에 **추가만** 한다 (기존 usda·glb·텍스처 무수정 — 실행마다 전 파일 해시로 검증).

```
assets/<이름>/rigs/<이름>_skeleton.fbx     ① 리깅 원본(본만) — 리그 재사용용
assets/<이름>/rigs/<이름>_rigged.fbx       ① 리깅+스킨 원본(모션 없음) — action 파트가 이 리그를 참조해 자체 모션 제작
assets/<이름>/<이름>.rt_default.usda       ② 기본모션 레이어 (cm, mpu 0.01, Y-up)
assets/<이름>/<이름>.rt_default.manifest.json  레이어 메타 (nframes/p95/정규화 파라미터)
<원본 usda 옆>/<이름>.rt_default.usda      ② 래퍼 — 원본 customData 무손실 복사 +
                                             geometry 참조 교체 + autorig customData 블록
autorig_out/<run_id>/{autorig_manifest.json, report.json, report.md, autorig.log}
```

- 래퍼 `autorig` customData: {rigger, bones, motion, motion_grade, frames, fps,
  stretch_p95, generated_at, source_fbx}. 비-biped 는 `motion_grade: "demo"`
  (관절 가동 시연 — 보행 품질 주장 아님, DESIGN §4-2-1).
- 기본모션: biped = 걷기(2s)→뛰기(26f 루프 2s)→점프 1회, 경계 0.25s 블렌드, 단일 액션,
  전부 제자리(in-place). 비-biped = 클래스별 절차 모션 (`default_motion.PROC_PARAMS` 테이블).
- UniRig 중간 npz(입력 glb 옆 `<이름>/` 하위폴더)는 UniRig 기본 동작 — 그대로 둔다.
- 멱등성: rigs FBX·rt_default 래퍼가 있으면 skip (재실행 안전).

## 모듈 구성

| 파일 | 역할 | 출처 |
|---|---|---|
| `scan.py` | usda 스캔 → autorig_manifest.json | 신규 (DESIGN §3) |
| `rig_service.py` | UniRig 상주 리깅 (배치+데몬) | 검증본 (UNIRIG_DIR 기본값만 교체) |
| `retarget.py`, `riglib.py` | biped 리타게팅 (좌우반전 수정판·제자리 내장) | 검증본 그대로 — **롤백 금지** |
| `weight_transfer.py` | align_and_transfer 웨이트 전이 | movie_char_render.py 발췌 |
| `default_motion.py` | biped 합성 클립 + 비-biped 절차 모션 | 신규 (DESIGN §4-2) |
| `usd_assemble.py` | 레이어 익스포트+단위보정+텍스처 재참조+래퍼+검증 | motion_to_usd+movie_usd_pack 통합 |
| `report.py` | 원본 불변 해시 검증 + 리포트 | 신규 |
| `autorig.py` | 오케스트레이터 (리깅은 subprocess 수명 = unload) | 신규 |
| `assets/mocap/` | CMU BVH 3종 (walk 16_15 / run 16_35 / jump 16_01) | 이관 패키지 |

UniRig 저장소: `../autorigging_transfer/unirig` (로컬 패치 2건 적용본 — 재clone 시 패치 재적용
필수). 체크포인트는 첫 실행 때 HF(VAST-AI/UniRig) 자동 다운로드 (~/.cache/huggingface).

## env (unirig — 이 서버 구축 완료)

`/root/miniconda3/envs/unirig` : py3.11 · torch 2.6.0+cu124 · flash-attn 2.7.4.post1(프리빌드
휠) · spconv-cu124 · torch_scatter/cluster(pt26cu124) · bpy 4.2.23 · usd-core 26.8 ·
transformers 4.51.3 등. 재구축 절차는 `../autorigging_transfer/README.md` §2 (+아래 함정).

이 서버 재구축 시 추가 함정 (원 README 에 없는 실측):
1. `pip install box` 금지 — python-box 의 `box` 모듈을 가린다. **python-box 만** 설치.
2. timm/open3d 가 torch 를 최신(2.13)으로 끌어올린다 → 마지막에
   `pip install torch==2.6.0 torchvision==0.21.0 --index-url .../cu124` 로 고정 복원.
3. UniRig requirements 의 transformers==4.51.3, omegaconf, addict, timm,
   fast-simplification, trimesh, open3d, pyrender, huggingface_hub 도 필요 (원 README 누락).

## 테스트 (이 서버 검증 로그: autorig_out/run_20260820_12*)

| # | 내용 | 결과 |
|---|---|---|
| M1 | samples 동등성 — 본 52/52·이름 전부 일치, 8본만 토큰 양자화 1스텝(2/256) 차이 (A100↔4090 AR 샘플링 차) | PASS |
| T1 | 스캔 dry-run (캐릭터 수집·static 제외·빈 래퍼 스킵·구 enum 경고) | PASS |
| T2 | E2E biped (welcom2dmk 캐릭터 2) — anim 217샘플·tex OK·bbox 1.00x | PASS |
| T3 | 원본 불변 (전 파일 sha256, 실행 전후) | PASS |
| T4 | 재실행 멱등성 — 전부 skip (exists) | PASS |
| T5 | 비-biped (avian 갈매기 실물 GLB + serpentine) — 절차 모션·demo 표기 | PASS |
| T6 | 블렌더 재생 (rt_default 임포트, 정점 이동 확인) | PASS |
| T7 | 실패 주입 (손상 glb) — 해당 에셋만 fail, 전체 계속 | PASS |

테스트는 반드시 사본에서: `cp -r movie_usd /tmp/autorig_test_usd` 후 실행 (DESIGN §8).
