# T2I2O 실행 가이드 (A100 서버 / previs-prep 컨테이너 기준)

> 파이프라인: **text → image(FLUX.1-schnell) → 3D object(TRELLIS.2, image-to-3D) → GLB → USD 주입.**
> 이 문서는 **이 A100 서버에서 2026-07-28 실제로 검증된 실행 절차와 함정**만 담는다.
> 설계 배경은 `SERVER_4090_SETUP.md`(서버 차이·확정 T2I 프롬프트), `../ENV_REBUILD_GUIDE.md`(env 핀 근거), `../NEW_SERVER_SETUP.md`.

---

## 0. 어디서 도나 — 도커 컨테이너 안

파이프라인은 호스트가 아니라 **도커 컨테이너 `previs-prep`** 안 `/root/previs_proj` 에서 돈다.
호스트에서 들어가려면:

```bash
docker exec -it previs-prep bash
cd /root/previs_proj
```

### 프로젝트 구조 (2026-07-28 정리 후)
```
/root/previs_proj/
├── object_generate.sh          # ★ 객체 3D 생성 런처 (루트에 있어야 함 — 다른 스크립트가 참조)
├── movie_usd/                  # 입력 USD (+ 생성 결과가 주입되는 대상)
│   └── hidden_time/hidden_time.usda
└── t2o_pipeline/               # 파이프라인 repo (git, feature/trellis2)
    ├── previz_pipeline/        # 파싱·추론·병합·프롬프트 조립 스크립트
    ├── trellis2_src/           # TRELLIS.2 본체 (git 제외)
    ├── hf_models/              # TRELLIS.2-4B(16G), FLUX.1-schnell(54G) (git 제외)
    ├── prompt_lab/  prompt_lab.sh   # 프롬프트 최적화 실험 도구
    └── docs/                   # RUN_GUIDE.md(이 문서), SERVER_4090_SETUP.md, 등
```

---

## 1. 사전 준비 — `conda activate` 만 하면 된다

conda env `trellis2` 는 **이미 구축돼 있다(재구축 불필요).** FLUX 도 이 단일 env 에 통합돼 있어
**별도 t2i env 를 만들 필요 없다.** 런처가 `CUDA_HOME`·`PYTHONPATH`·`TRELLIS2_SRC`·`HF_HUB_DISABLE_XET`·
`T2I_PYTHON`·출력경로를 **전부 자동 설정**한다. 사용자가 할 일은 env 활성화뿐이다:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate trellis2
```

> 런처는 `conda activate` 를 스스로 하지 않는다. **반드시 먼저 활성화**할 것.

### 검증된 env 핀 (바꾸지 말 것 — 바꾸면 추론이 조용히 깨진다)
`transformers==4.56.2`, `opencv-python-headless==4.11.0.86`, `usd-core==25.8`, `OpenEXR`,
`torch 2.6.0+cu124`, `diffusers 0.39.0`. CUDA 확장(o_voxel/flex_gemm/cumesh/flash_attn)은
이 서버 GPU(A100, sm_80)로 빌드돼 있다. 근거·증상표는 `docs/ENV_REBUILD_GUIDE.md`.

### GPU
컨테이너에서 A100 80GB ×4(인덱스 0·1·2·4) + RTX3050(인덱스 3)이 보인다. A100 80GB 라
FLUX 가 **offload 없이 상주(fully resident)** 로 돌아 장당 ~2초로 빠르다(4090 offload 25~41초 대비).

---

## 2. 객체 3D 생성 — `object_generate.sh`

```bash
cd /root/previs_proj

# ★ 검증된 실행 (1개 객체, 빠른 확인) — 2026-07-28 e2e 통과
T2I_GPU=1 ./object_generate.sh movie_usd/hidden_time/hidden_time.usda --no-filter -- --max_items 1

# 전체 객체
T2I_GPU=1 ./object_generate.sh movie_usd/hidden_time/hidden_time.usda --no-filter
```

3단계로 돈다: **① USD 파싱/프롬프트 조립 → ② FLUX→TRELLIS.2 추론 → ③ GLB→USD 변환·주입.**

### ⚠️ 반드시 지킬 것
| 항목 | 이유 |
|---|---|
| `conda activate trellis2` 선(先)활성화 | 런처는 activate 안 함 |
| 입력은 **루트 USD**(`hidden_time.usda`) | shot 경로(`.../shot_1/objects/object_1.usda`)를 직접 주면 meta-only override 라 `필수 정보 0개` 로 끝난다. 루트에서 시작해야 scene canonical 이 잡힌다 |
| `--no-filter` (현재) | `--filter`/`--translate` 는 ollama 가 필요한데 **이 서버엔 아직 ollama 가 안 떠 있다**(§4). 안 붙이면 USD 의 en/description 을 직접 쓴다 |
| `T2I_GPU=1` | FLUX 를 별도 GPU(인덱스 1)에 올려 TRELLIS.2(인덱스 0)와 카드 분리. **4090(24GB): 필수** — TRELLIS.2 상주(~23GB)+FLUX(~23.4GB peak)가 한 카드를 공유하면 OOM(실측). A100(80GB): 공유해도 OOM 은 안 나지만 분리가 깔끔·빠름. ⚠️ TRELLIS.2 가 쓰는 인덱스 0 은 **주지 말 것**(같은 카드라 충돌) |

### 주요 환경변수 (전부 선택 — 기본값으로 동작)
| 변수 | 기본값 | 설명 |
|---|---|---|
| `T2I_GPU` | (미설정) | FLUX 를 올릴 GPU 인덱스 |
| `T2O_OUTPUT_DIR` | `t2o_pipeline/t2o_results` | 로그/run 산출물 루트 |
| `TRELLIS_BACKEND` | `trellis2` | v1 로 폴백하려면 `trellis` |
| `OLLAMA_FILTER_MODEL` | `gpt-oss:20b` | `--filter` 증강 모델(ollama 필요) |

---

## 2.5. ★ 지금 바로 돌리면 마주칠 함정 — 입력 데이터에 설명이 비어 있다

이 서버의 `movie_usd`(hidden_time/berlin/welcom2dmk)는 **경량본**이라 object USDA 의
`base_description`/`appearance` 가 **비어 있고 en 이름만** 있다. 그래서 기본 실행 시
stage1 이 **`필수 정보 추출 완료: 0개`** 로 멈추고 GLB 가 안 나온다(환경 문제 아님, 데이터 문제).

**실제로 생성해 보려면 둘 중 하나:**

**(A) 설명을 채운 사본으로 돌리기 (ollama 불필요 — 지금 바로 가능, 검증된 방법)**
```bash
# 원본 보존: 사본 만들고 객체 하나에 설명을 채운다
cp -r movie_usd/hidden_time /root/previs_proj/tmp/htest
# tmp/htest/scene_1/objects/object_1.usda 의 customData 에서 아래를 채운다:
#   string base_description = "a seagull, a white and grey seabird"
#   string appearance       = "standing with wings folded"
T2I_GPU=1 ./object_generate.sh tmp/htest/hidden_time.usda --no-filter -- --max_items 1
```

**(B) ollama 로 자동 증강(`--filter`)** — 아직 미설정. §4 참고. 설정되면:
```bash
T2I_GPU=1 ./object_generate.sh movie_usd/hidden_time/hidden_time.usda
```

---

## 3. T2I 프롬프트 — §9 확정 시스템 프롬프트가 default 적용됨 (2026-07-28)

stage1 이 객체마다 `t2i_prompt` 를 템플릿으로 조립한다(`previz_pipeline/t2i_prompt_builder.py`).
body-plan(rig_type/키워드 dict)으로 무생물/생명체를 라우팅한다:
- **무생물**: `{base_description}, {appearance}, {object}, single centered object, neutral background, full object visible in frame, unoccluded`
- **생명체**: `{object}, {body-plan 스캐폴딩}, {base_description}` (biped=A-pose / quadruped=자연기립 / bird=날개펼침 / insect=6족)
  - **base_description 은 생명체에도 포함**한다 — 정체성/외형(알비노 갈매기, 매 형상의 주인공 갈매기 등)을 잃지 않기 위해. 스캐폴딩이 이미 격리문구를 포함하므로 별도 격리문구는 안 붙인다.
  - `{base_description}` 슬롯 내용 = **filter ON 이면 정제 캡션, OFF 면 원본 en description**(§2.5·아래).

> ⚠️ 프롬프트 설계는 **잠정**이다. prompt_lab 실험 A(ablation)만 끝났고 B(LLM 최적화)·C(A→B 순차)가
> 남아 있어, 생명체 서술 필드 포함 방식은 실험 후 정련 예정. §9 원안(생명체=object만)은 서술의 *동작*이
> 강제 자세와 충돌한 ablation 관찰에서 나왔고(그건 filter 정제로 완화됨), 현재는 정체성 보존을 우선한다.

### rig_type 필드 (업스트림 권위)

> ⚠️ **경로 표기는 확정 예정.** 아래는 "최상위"로 적혀 있으나, 의도분석 담당자와 논의된 값은
> `object xform` > `etc` > `rig_type` 이다(`docs/CONTEXT.md` §4). 최종 합의 후 이 절을 갱신한다.
> 코드 `_resolve_rig_type()` 은 최상위·`etc` 중첩·자식 prim `etc` 를 **모두 읽으므로 어느 쪽이든 동작한다.**

object customData **최상위 `string rig_type`** 에 `biped|quadruped|bird|insect|static_object` 가
들어오면 그 값을 신뢰해 라우팅한다(`static_object`=무생물). **없으면 객체명 키워드 dict 로 폴백.**
```usda
def Xform "object_1" (
    customData = {
        string rig_type = "bird"
    }
)
```
**이 최상위 형태가 정본이다** — 의도분석 모듈이 실제로 내보내는 형태이고, 2026-07-28 산출물
(hidden_time / welcom2dmk / berlin)로 검증했다. 구 스펙의 중첩 dict `etc = { string rig_type = ... }`
와 자식 prim `etc` 형태도 `_resolve_rig_type()` 이 하위호환으로 계속 읽지만, 신규 작성은 최상위로 한다.

> `text_to_image.py` 의 옛 `_POSITIVE_SUFFIX`(studio product shot 등)는 §9 격리문구와 충돌해서
> 기본값을 비웠다. 독립 실험 시엔 `T2I_POSITIVE_SUFFIX` env 로 주입 가능.

---

## 4. ollama (`--filter`/`--translate`) — 셋업 완료 (2026-07-28)

`--filter`(프롬프트 증강)·`--translate`(ko→en 번역)는 ollama 를 쓴다. **호스트 `ollama` 컨테이너에
`gpt-oss:20b` 를 pull 해뒀다.** previs-prep 컨테이너에서는 **`http://172.17.0.1:11434`**(도커 브리지
게이트웨이 → host published 포트)로 도달한다. ⚠️ 컨테이너 네트워크가 달라 `ollama` 호스트명은 안 되고
`172.17.0.1` 이어야 한다.

`--filter` 로 돌리려면 `OLLAMA_HOST` 를 지정한다(파이프라인의 `import ollama` 가 이 env 를 읽는다):
```bash
OLLAMA_HOST=http://172.17.0.1:11434 \
  T2I_GPU=1 ./object_generate.sh movie_usd/hidden_time/hidden_time.usda -- --max_items 1
```
> ⚠️ gpt-oss 는 reasoning 모델이라 콜당 느리고(~99s), 토큰 예산이 작으면 출력이 빈다.
> prompt_lab 등에서 `llm.params.max_tokens` 를 넉넉히(4096).
> **단, `--filter` 는 설명이 있는 객체에만 의미** — 이 서버 경량 데이터(설명 빈 객체)는 여전히 §2.5 처럼
> 설명을 채워야 stage1 을 통과한다(filter 가 빈 설명을 만들어내진 않는다).

---

## 5. 알아둘 것 (함정)

1. **GLB→USD 는 usd_from_gltf 불필요.** trimesh+pxr 네이티브 변환기(`previz_pipeline/glb_to_usd_native.py`).
2. **`No module named 'rembg'` 경고는 무해.** 구 v1 코어(`trellis_inference_core.py`)의 폴백일 뿐,
   trellis2 는 `preprocess_image=True` 로 내부 RMBG-2.0(배경제거)을 쓴다. GLB 정상 생성됨.
3. **exit 0 ≠ 성공.** 마지막 요약(`✅ 성공 N개 / ❌ 오류 M개`)과 `results.csv` 를 확인할 것.
4. **usd 파싱 항목 수 대조.** usd-core 가 빠지면 파서가 조용히 regex 폴백(항목 축소)한다. `[DEBUG USD]`
   로그가 나오고 항목 수가 정상이면 OK.

---

## 6. 산출물 — 무엇이 어디에 생기나

⚠️ **산출물은 두 곳에 나뉜다.** run 폴더엔 **로그·프리뷰만**, **실제 3D 에셋(GLB·USD·텍스처)은
소스 USD 트리 옆**에 생긴다.

### (A) 로그·프리뷰 → `t2o_pipeline/t2o_results/{모델}/{YYYYMMDD}/run_{HHMMSS}_{flags}/`
| 파일 | 내용 |
|---|---|
| `run_manifest.json` | 실행 명령 전문·모델·옵션. **디버깅 1순위** |
| `usd_results.json` | stage1 파싱·프롬프트 결과(2단계 입력). `t2i_prompt`/`rig_type`/`body_plan` 확인 |
| `results.csv` | 객체별 prompt·seed·시간·**success/error** |
| `previews/{scene}/{object}/` | 성공한 객체만: `_ref.png`(FLUX 이미지), `_pbr.mp4`(턴테이블), `_00Ns.jpg` 썸네일, `generation.json` |

> run 폴더엔 GLB/USD 가 없다. `previews/.../{object}/` 가 비면 그 객체는 **실패** — `results.csv` 의 error 를 봐라.

### (B) 실제 3D 에셋 → 소스 USD 옆 `objects/assets/{object}/`
`movie_usd/{작품}/{scene}/objects/assets/{object}/` 에 `{object}.geometry.usda`(GLB→USD 변환본,
매 run 덮어씀) + `bin/` 텍스처 + `{prefix}.glb`(seed 별로 **누적** — 디스크 차면 오래된 `*.glb` 수동 정리).
그리고 **원본 `objects/{object}.usda` 자체에 geometry reference 가 주입**되어, 이 USD 나 이를 참조하는
shot/scene USD 를 열면 생성된 메시가 보인다.

### 확인 커맨드
```bash
# 최근 run 성공/실패 요약
cat "$(ls -dt t2o_pipeline/t2o_results/*/*/run_* | head -1)/results.csv"

# 주입된 원본 USD 에 mesh 가 보이는지 (composed stage)
python -c "from pxr import Usd; st=Usd.Stage.Open('tmp/htest/scene_1/objects/object_1.usda'); print('meshes:', sum(1 for p in st.Traverse() if p.GetTypeName()=='Mesh'))"
```
