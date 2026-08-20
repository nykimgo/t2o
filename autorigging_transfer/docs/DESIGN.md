# AutoRigging 모듈 — 개발 상세설계서

> 버전 v0.1 (2026-08-18) · 작성 기반: EXP1_rigging 검증 자산 (UniRig 상주 서비스,
> direction-copy 리타게팅, movie_usd 규칙 USD 익스포트 — 모두 이 서버에서 실측 검증 완료)
> 대상: 개발 서버 이식 + 파이프라인 단계화

---

## 1. 목적과 범위

**한 줄 정의**: T2O·T2Character 가 생성해 둔 에셋(USD+GLB) 중 **생명체만** 골라
자동 리깅하고, **클래스별 기본 제자리(in-place) 모션**을 입힌 규칙 USD 를 원본 옆에
추가로 저장하는 단계.

| 한다 (In) | 안 한다 (Out) |
|---|---|
| 씬/영화 usda 스캔 → 생명체·캐릭터 수집 | **action 별 실제 동작 구현** (다음 파트 — 본 모듈 산출물을 참조해 진행) |
| UniRig 자동 리깅 (FBX 산출) | 다른 모듈(의도분석·T2I·공간 등) 코드 수정 |
| 클래스별 기본 모션 1종 적용 (제자리) | 원본 USD/GLB **수정** (추가만 허용) |
| movie_usd 규칙 USD 재조립 + 검증 리포트 | 모션 품질의 연출적 완성 (검증·시연 목적) |

**후속 파트와의 계약**: 본 모듈 산출물 = ① 리깅 원본 FBX(모션 없음, 리그 재사용용)
② 기본모션 usda(제자리 동작 시연용). action 구현 파트는 ①의 리그를 참조해
자체 모션을 만든다. **생성 에셋 원본과 리깅+모션본은 별도 파일로 공존한다.**

---

## 2. 파이프라인 내 위치와 실행 형태

```
의도분석 → T2I → T2O(오브젝트) / T2Character(캐릭터) → ★AutoRigging → (다음: action 모션 파트)
                                    모두 USD 로 연결 —  이 모듈도 USD 만 읽고/추가한다
```

실행 (루트 셸 런처 — `object_generate.sh`/`space_generate.sh` 선례를 따름):

```bash
./autorig_generate.sh movie_usd/welcom2dmk/welcom2dmk.usda          # 영화 단위
./autorig_generate.sh movie_usd/welcom2dmk/scene_1/scene_1.usda     # 씬 단위
# 옵션: --dry-run(스캔 결과만 출력), --only <이름부분일치>, --no-motion(리깅까지만)
```

---

## 3. 입력 스캔 명세 (usd_parser.py 뼈대)

`t2o_pipeline/previz_pipeline/usd_parser.py`(T2O 소유 = 수정 가능)를 import 해
usda 를 파싱하되, AutoRigging 전용 스캐너는 별도 파일로 둔다 (`autorigging/scan.py`).
usd_parser 를 고치지 않고 쓰는 것을 우선하고, 부족하면 **추가 함수만** 덧붙인다.

### 3-1. 수집 대상과 판정 (모두 실측 확인된 스키마)

| 대상 | 위치 규칙 | 생명체 판정 |
|---|---|---|
| 캐릭터 | `<영화>/characters/<char>.usda` + `assets/<char>/{<char>.glb, bin/texture.jpg, *.geometry.usda}` | **무조건 수집** (T2Character 산출 = 전부 인물, rig_type=biped 로 취급) |
| 오브젝트 | `<영화>/scene_N/objects/object_M.usda` + `objects/assets/` (glb·bin·geometry.usda) | customData `rig_type` ∉ {`static_object`, 빈값} 일 때 수집 |

- rig_type enum = 7클래스(팀 확정): `biped/quadruped/hexapod/octopod/avian/serpentine/aquatic`
  (+`static_object`). 미지값·누락 → **스킵 + 리포트에 경고** (조용히 폴백하지 않는다 —
  t2i_prompt_builder 의 rig_type 계약 위반 경고와 동일 사상).
- **GLB 미생성(빈 래퍼) 케이스 존재 확인됨** (예: welcom2dmk object_2 — customData 만 있고
  지오메트리 참조 없음). glb 경로 부재 시 스킵 + 리포트.
- 스캔 산출: `autorig_manifest.json` — 대상별 {usda 경로, glb 경로, rig_type, 텍스처 경로,
  스킵 사유}. 이후 모든 단계가 이 매니페스트를 입력으로 받는다 (단계 재실행 가능).

---

## 4. 단계별 상세

### 4-1. 리깅 (상주 서비스, 완료 후 unload)

`autorigging/rig_service.py` (기존 `t2o_pipeline/rigging/rig_service.py` 를 이동) 사용.

- 매니페스트의 전 대상 GLB 를 **배치 모드 1회 실행**으로 처리 — 프로세스 시작 시 모델
  2종(스켈레톤·스킨) 로드(~10s), 전 에셋 처리 후 **프로세스 종료 = VRAM/RAM 자동 unload**
  (요구사항: 작업 완료 후 unload — 배치 프로세스 수명으로 보장, 별도 데몬 상주 금지).
- 실측 성능(4090): 에셋당 소형(≤1만면) ~11s, 대형(49만면) ~22s. 시간 지배는 extract
  데시메이션(면수 비례)이며 추론은 면수 무관.
- 출력: `assets/<이름>/rigs/<이름>_skeleton.fbx`, `<이름>_rigged.fbx` (원본 폴더 구조에
  **추가**). UniRig 중간 npz(입력 옆 하위폴더)는 UniRig 기본 동작 — 그대로 둔다.
- 실패 시: 해당 에셋 스킵 + 리포트 (파이프라인 전체는 계속. s3_rig 검증에서 하드 실패
  0/243 이었으나 방어는 유지).

### 4-2. 클래스별 기본 모션 (선택 로직 없음 — 클래스당 1종 고정)

**공통 원칙: 전부 제자리(in-place).** biped 리타게팅은 **이미 구현·검증돼 있다** —
`retarget.py` 가 루트의 전진 성분을 억제하고 상하(점프)·좌우 흔들림만 남긴다
(블렌더 재생으로 확인 완료). 절차적 모션도 같은 원칙을 따른다. 실제 이동·경로는
다음 파트 소관.

| rig_type | 기본 모션 | 소스/방법 |
|---|---|---|
| **biped** (캐릭터 + 이족 오브젝트) | **합성 클립 1개**: 걷기 3보(부족 시 2초) → 뛰기 3보(부족 시 2초) → 점프 1회 | CMU BVH 리타게팅(검증 완료): walk=16_15(472f, 3보 추출 여유), run=16_35(**26f뿐** → 2초가 되도록 루프 반복), jump=16_01. 클립 경계는 0.25s 포즈 블렌드로 연결, 전체를 단일 액션으로 베이크 |
| quadruped | 제자리 보행 사이클 근사 | 절차적(아래 §4-2-1) — 다리 체인 4개 위상차 스윙 |
| avian | 날개 플랩 + 몸 바운스 | 절차적 — 날개 체인 대칭 사인 |
| hexapod / octopod | 다리 파상(wave gait) | 절차적 — 다리 체인 위상차 |
| serpentine | 스파인 사행(sine wave) | 절차적 — 본 체인 순차 위상 |
| aquatic | 꼬리·지느러미 스윙 | 절차적 — 후방 체인 사인 |

#### 4-2-1. 절차적 기본 모션의 정직한 한계와 설계

- 비-biped 는 **모캡 소스가 없다**(CMU=사람용, `build_map`=휴머노이드 전제). 따라서
  1차 버전의 비-biped 모션은 **"리깅이 살아있음을 보여주는 관절 가동 시연"**이 목적이며
  보행 품질을 주장하지 않는다 (리포트·usda customData 에 `motion_grade: "demo"` 명시).
- 구현: UniRig 익명본(bone_N)에서 **본 체인 추출은 토폴로지만 사용** — 루트에서 리프까지
  체인 분해 후, 체인 길이·개수로 다리/스파인/날개를 휴리스틱 분류(EXP1 `riglib.detect` 는
  biped 전제라 재사용 불가 — 신규 소형 로직, 실패 시 전체 본 사인 웨이브로 폴백).
- 파라미터(진폭·주기·위상)는 클래스별 상수 테이블 하나로 관리 — 튜닝 여지를 코드 밖으로.
- biped 합성 클립은 EXP1 `retarget.py`(좌우반전 수정판)의 `build_animation()` 재사용.
  주의: **쇄골 매핑 제거 기본값(map_clavicle=False)·direction copy 방식 유지** — 검증된
  수치(p95 1.14~1.64)의 전제다.

### 4-3. USD 재조립 (movie_usd 규칙)

기존 검증 로직 일반화: `motion_to_usd.py`(하드코딩 리스트 제거 → 매니페스트 입력) +
`movie_usd_pack.py` 통합 → `autorigging/usd_assemble.py`.

**규칙 (전부 이 서버에서 검증 완료된 사항):**

1. 단위 = **cm, metersPerUnit 0.01** (movie_usd 전체 규칙). 익스포터가 armature 오브젝트
   트랜스폼을 버리므로(실측) UniRig 정규화 공간→원본 배치 복원 보정 행렬은 **USD 쪽**
   루트 prim 에 넣는다. 이때 **익스포터의 Y-up 회전 xformOp 를 덮어쓰지 말고 합성**할 것
   (M_existing * M_corr — 실측 시행착오 항목).
2. 텍스처 = 기존 `bin/texture.jpg` 재참조 (중복 저장 금지, 익스포터 textures/ 삭제).
3. 래퍼 = 원본 캐릭터/오브젝트 usda 루트 prim 을 `Sdf.CopySpec` 으로 복사(캐릭터 메타
   customData 무손실) + geometry 참조를 모션 레이어로 교체 + **`autorig` customData 블록**
   추가: {rigger: "UniRig", bones, motion: "default_biped_v1", motion_grade, frames, fps,
   stretch_p95, generated_at, source_fbx}.
4. 파일명 컨벤션 (원본과 절대 충돌하지 않는 접미사):
   - `<이름>.rig.usda` — (선택) 리깅만, 모션 없음 · `<이름>.rt_default.usda` — 기본모션
   - 래퍼는 원본 usda 와 같은 폴더, 레이어는 assets/<이름>/ 아래 (geometry.usda 위치 규칙)
5. 검증(자동, 실패 시 리포트 FAIL): SkelAnimation 타임샘플 존재 · 텍스처 resolve ·
   **Mesh prim 월드 bbox = 원본 geometry 대비 1.00x**(SkelRoot extent 힌트는 애니 전범위라
   비교에 쓰지 말 것 — 실측 함정) · 원본 파일 해시 불변(추가만 했는지).

### 4-4. 산출물 요약 (welcom2dmk 예)

```
movie_usd/welcom2dmk/characters/
├── char_동구_base.usda                     (원본 — 불변)
├── char_동구_base.rt_default.usda          ★ 래퍼 (기본모션)
└── assets/char_동구_base/
    ├── char_동구_base.glb, bin/, *.geometry.usda   (원본 — 불변)
    ├── rigs/char_동구_base_{skeleton,rigged}.fbx   ★ 리깅 원본 (다음 파트 참조용)
    └── char_동구_base.rt_default.usda              ★ 모션 레이어
movie_usd/welcom2dmk/scene_1/objects/       (생명체 오브젝트가 있다면 동일 패턴)
autorig_out/<run_id>/                        ★ 매니페스트·리포트·검증 결과
```

---

## 5. 실행 환경 — env 통합

현재 2-env 체인(unirig→previs)의 원인은 **pxr(usd-core)가 unirig env 에 없어서**다
(bpy 4.2 는 USD 를 내장하지만 파이썬 `pxr` 모듈로 노출하지 않음 — 실측).

- **통합안**: unirig env 에 `pip install usd-core` (순수 wheel, bpy 와 심볼 충돌 없음 —
  bpy 는 pxr 를 파이썬에 노출하지 않으므로 이름 충돌 원리상 없음).
  이식 후 스모크 테스트: `python -c "import bpy, pxr"` + 소형 usda round-trip.
- 실패 시 폴백: 현행 2-스텝을 셸 런처가 내부에서 순차 호출 (사용자에겐 단일 명령).
- GPU/RAM: 리깅 배치는 GPU 1장, FLUX·TRELLIS 와 **동시 실행 금지**
  (RAM 125GB OOM 실측 — EXP1 MANIFEST §4 교훈. 파이프라인 오케스트레이터가 단계 직렬 보장).

---

## 6. 이식 파일 목록 — **전부 이관 패키지(autorigging_transfer.tar.gz)에 포함됨**

패키지 구조 (압축 해제 후):

```
autorigging_transfer/
├── README.md              ← 시작 가이드: env 구축 → 스모크 → 실행 예 (여기부터 읽을 것)
├── docs/DESIGN.md         ← 이 문서
├── code/                  ← 검증된 코드 6종 (아래 표)
├── mocap/                 ← 기본모션 소스 BVH 3종 (walk/run/jump)
├── unirig/                ← UniRig 저장소 (로컬 패치 2건 적용된 상태 그대로)
└── samples/               ← 참고 산출물: 리깅 FBX·규칙 usda·usdz·뷰어 예시
```

| 패키지 내 파일 | 역할 | 이식 시 작업 |
|---|---|---|
| `code/rig_service.py` | UniRig 상주 리깅 (배치+데몬) | `UNIRIG_DIR` env 로 unirig/ 위치 주입 (하드코딩 기본값 교체) |
| `code/retarget.py`, `code/riglib.py` | biped 리타게팅 — 좌우반전 수정판·**제자리 처리 내장**(전진 억제, 상하·좌우 유지) | 경로 상수 제거, 모듈화. 좌우분리 손 판정 로직 롤백 금지 |
| `code/motion_to_usd.py`, `code/movie_usd_pack.py` | USD 익스포트 + movie_usd 규칙 마감(단위·텍스처·래퍼·검증) | CHARS/MOTIONS 하드코딩 제거 → 매니페스트 입력, 두 파일 통합 |
| `code/movie_char_render.py` | `align_and_transfer`(웨이트 전이 — motion_to_usd 가 import) | 웨이트 전이 함수만 발췌해 모듈화 권장 |
| `mocap/16_15·16_35·16_01.bvh` | 기본모션 소스 (walk 472f / run 26f / jump 323f) | 모듈 `assets/mocap/` 으로 |
| `unirig/` | UniRig 저장소 — **로컬 패치 2건 적용된 상태**: ① run.py torch2.6 safe_globals(Box) ② ar.py user_mode predict_skeleton.npz 저장 | 그대로 사용 (재clone 시엔 패치 재적용 필수). 체크포인트는 첫 실행 때 HF(VAST-AI/UniRig) 자동 다운로드 |
| `usd_parser.py` (패키지 미포함) | 스캔 뼈대 | **저쪽 서버의 t2o_pipeline 에 이미 존재** — 수정 없이 import 우선 |
| env 스펙 | unirig env: py3.11 · torch 2.6+cu124 · flash_attn(프리빌드 휠) · spconv-cu124 · PyG 확장 · bpy 4.2 · **+usd-core** | README.md 의 구축 절차 참조 |

**신규 작성**: `scan.py`, `default_motion.py`(합성 클립+절차 모션), `usd_assemble.py`(통합),
`autorig.py`(오케스트레이터), `autorig_generate.sh`(런처), `report.py`.

---

## 7. 불가침 원칙

1. **수정 금지**: 의도분석·T2I·T2Character·공간 등 타 모듈 코드, 그리고 movie_usd 의
   **기존 파일 전부** (usda·glb·텍스처). 본 모듈은 **새 파일 추가만** 한다.
2. 허용: T2O 소유 코드(usd_parser 등 previz_pipeline)와 AutoRigging 자신, USD 트리에의
   파일 **추가**.
3. 원본 불변 검증을 테스트 체크리스트에 포함 (실행 전후 기존 파일 해시 비교 — §8).

---

## 8. 테스트 계획

```bash
cp -r movie_usd /tmp/autorig_test_usd        # 반드시 사본에서 (원본 보호)
./autorig_generate.sh /tmp/autorig_test_usd/welcom2dmk/welcom2dmk.usda --dry-run  # T1
./autorig_generate.sh /tmp/autorig_test_usd/welcom2dmk/welcom2dmk.usda            # T2
```

| # | 테스트 | 합격 기준 |
|---|---|---|
| T1 | 스캔 dry-run (welcom2dmk: 캐릭터 2 + 오브젝트 2) | 캐릭터 2 수집, static_object 오브젝트 제외, 빈 래퍼(object_2 유형) 스킵+사유 기록 |
| T2 | E2E (biped) | 캐릭터당 rigs/*.fbx + rt_default.usda 생성, §4-3-5 검증 전부 통과 |
| T3 | 원본 불변 | 실행 전후 기존 파일 해시 100% 일치 |
| T4 | 재실행 멱등성 | 2회 실행 시 skip (exists) — 중복 생성 없음 |
| T5 | 비-biped (테스트 픽스처: E11 산출 quadruped·serpentine GLB 를 사본 씬에 주입) | 리깅 성공 + 절차 모션 usda 생성 + motion_grade=demo 표기 |
| T6 | 블렌더 확인 | rt_default.usda 임포트 → 재생 정상 (헤드리스 렌더 스크립트로 자동화 가능 — 검증 렌더 선례 있음) |
| T7 | 실패 주입 (glb 경로 오염) | 해당 에셋만 스킵, 전체 성공, 리포트에 FAIL 사유 |

---

## 9. 리스크와 완화

| 리스크 | 완화 |
|---|---|
| 비-biped 절차 모션이 어색함 | 목적을 "리그 가동 시연"으로 명시(motion_grade=demo), 파라미터 테이블로 튜닝 분리. 클래스별 모캡 확보는 후속 과제로 리포트에 기록 |
| usd-core ↔ bpy 충돌 | 스모크 테스트 선행, 실패 시 2-스텝 폴백 (§5) |
| UniRig 손 판정 휴리스틱 (손가락 스켈레톤에서 우연 의존 — 실측 버그 이력) | riglib 좌우분리 수정판 사용 + 회귀 테스트 케이스 포함, 옛 코드 롤백 금지 |
| run 모캡 26f 한계 | 루프 반복으로 2초 충족 (마지막 키프레임 클램프 수정판 사용 — 정지 중복 버그 이력) |
| bpy 종료 시 segfault (teardown, 산출물 무해 — 실측) | 익스포트 완료 로그를 성공 판정 기준으로, exit code 는 로그 기반 판정 |
| 한글 파일명/프림명 | 이 서버에서 전 구간 정상 실측 — 이식 서버 로케일(UTF-8)만 확인 |

---

## 10. 개발 순서 (마일스톤)

1. **M1 이식·환경**: §6 파일 이식, unirig env 재구축(+usd-core), UniRig 패치 재적용,
   rig_service 단독 스모크 (기존 FBX 와 본 수 동등성 — 검증 스크립트 재사용)
2. **M2 스캔**: scan.py + 매니페스트 + T1
3. **M3 biped 완주**: 합성 기본모션(제자리 처리는 retarget.py 내장) + usd_assemble 통합 + T2~T4, T6
4. **M4 비-biped**: 절차 모션 + T5
5. **M5 단계화**: autorig_generate.sh, 오케스트레이터 연결 문서(입출력 계약서), 다음
   파트(액션 구현)에의 인수인계 문서 — rigs FBX·rt_default.usda 참조 방법
