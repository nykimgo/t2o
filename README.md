### 📁 Project Layout

`t2o_pipeline`은 **`/root/previs_proj`** 통합 프로젝트의 4단계(객체 3D 생성) 모듈입니다.

```text
/root/previs_proj/
├── run_previs_pipeline.sh          # intent → USD → space → t2o 전체 파이프라인
├── previs_pipeline.config          # 공통 설정 (MOVIE_TITLE, RUN_T2O 등)
├── movie_inputs/                   # 입력 (스케치·스토리보드 등)
├── movie_usd/                      # USD 통합 출력 (intent_analyzer + t2o 주입 대상)
│   └── hidden_time/
│       └── hidden_time.usda
├── space_generation/               # 공간(배경) 생성
├── intent_analyzer/                # 의도 분석 → USD 생성
├── usd_from_gltf/                  # GLB→USD 변환 (Stage 3 필수)
└── t2o_pipeline/                   # ← 이 저장소 (객체 3D 생성)
    ├── run_usd_to_3D_object.sh
    ├── previz_pipeline/
    ├── trellis/
    ├── hf_models/                  # TRELLIS 모델 캐시 (~12GB, git 제외)
    └── t2o_results/                # 중간 JSON 산출물 (git 제외)
```

-----

### 0\. 📖 User Manual (Quick Start)

**단독 실행 (t2o_pipeline만)**

```bash
cd /root/previs_proj/t2o_pipeline

# (LLM 번역/필터 없이) USD 파싱 → 3D 생성 → 원본 USD에 에셋 주입
./run_usd_to_3D_object.sh \
  /root/previs_proj/movie_usd/hidden_time/hidden_time.usda \
  /root/previs_proj/t2o_pipeline/t2o_results

# 한국어 description 번역 + 프롬프트 증강까지 사용
ollama pull gemma3:4b   # 최초 1회 (OLLAMA_MODEL 변경 시 해당 모델 pull)
./run_usd_to_3D_object.sh \
  /root/previs_proj/movie_usd/hidden_time/hidden_time.usda \
  /root/previs_proj/t2o_pipeline/t2o_results \
  --translate --filter
# 참고: ollama serve는 생략 가능 — --translate/--filter 사용 시 스크립트가 자동으로 시작/종료합니다.
```

**통합 파이프라인 (권장)**

```bash
cd /root/previs_proj

# previs_pipeline.config에서 RUN_T2O=true 로 설정 후
./run_previs_pipeline.sh

# t2o만 켜서 실행 (USD는 이미 생성된 경우)
./run_previs_pipeline.sh --skip-intent --skip-usd --skip-space --t2o-translate --t2o-filter
```

> 본 파이프라인은 결과 에셋을 **원본 USD 폴더 구조 안(`scene_n/objects/assets/`)** 에 저장하고, **원본 `object_n.usda` 파일에 직접 reference를 주입**합니다. 따라서 root/scene/shot/개별 object 어느 USD를 3D 툴(Blender, USD Viewer)에서 열어도 생성된 에셋이 함께 보입니다.

-----

### 1\. 🏛️ Pipeline Architecture Specification

**Data Flow:** **[USD Read] $\to$ [LLM Contextualization (optional)] $\to$ [TRELLIS Generation] $\to$ [USD Conversion & Patching] $\to$ [Reference Injection]**

이 파이프라인은 텍스트 묘사가 포함된 USD 씬으로부터 3D 에셋(.glb)을 생성하고, 이를 USD geometry로 변환하여 **원본 개별 오브젝트 USD에 직접 참조를 주입**하는 방식으로 동작합니다. 결과물은 원본 USD 옆 `assets` 폴더에 모입니다.

#### 🛠️ Stage-by-Stage Design

**1. Stage 1: USD Parse & Augment**

  * **Role:** 원본 USD를 파싱하여 `Object` 계층 구조와 description 메타데이터를 추출하고, (옵션) Ollama(LLM)로 프롬프트를 번역/증강합니다. (`Actor` 파싱은 현재 `ENABLE_ACTOR_PARSING=False`로 비활성)
      * **기본 (옵션 없음):** USD customData의 `en` 필드를 TRELLIS 프롬프트로 사용합니다 (LLM 불필요).
      * **Translation (`--translate`):** USD `ko` 필드를 LLM으로 영어 번역합니다.
      * **Filtering (`--filter`):** 영어 description에서 배경/행동을 제거하고 시각 요소만 남깁니다.
      * 폴더 구조(`scene_n/objects` 또는 `scene_n/shot_n/objects`)나 루트 폴더 이름에 무관하게, USD reference 그래프와 `scene_*/shot_*` prim 마커를 기반으로 경로를 추출합니다.
  * **Script:** `previz_pipeline/usd_parse_and_augment.py`
  * **Input:** `hidden_time.usda` (Source USD File)
  * **Output:** `{output_dir}/{model_name}/{YYYYMMDD}/run_{HHMMSS}_{flags}/`
      * `run_manifest.json` — 실행 명령·옵션·USD 경로
      * `usd_results.json` — 이번 run 입력 스냅샷
      * `results.csv` — TRELLIS 생성 결과 (prompt, seed, run_id)
      * `previews/scene_n/object_n/` — ply/mp4/jpg + `generation.json` (프롬프트 추적)
      * 각 항목에는 `object_path`, `usd_file_path`, `description`(`{ko,en}`), `description_ko`, `description_en`, `translated_description`, `aug_prompt` 등이 포함됩니다.

**USD object customData (bilingual 스키마)**

| 필드 | 형태 | 비고 |
|------|------|------|
| `name` | `dictionary { ko, en }` | 객체 이름 |
| `base_description` | `dictionary { ko, en }` | scene canonical 설명 |
| `appearance` | `dictionary { ko, en }` | shot override 시 샷별 외형 |
| `location`, `action` | `dictionary { ko, en }` | shot override 메타 |
| `category`, `object_id` | `string` | 단일 언어 |

**`--translate` 동작**

| 실행 | 소스 | LLM |
|------|------|-----|
| 기본 | USD `en` → `description_en` / `translated_description` | 없음 |
| `--translate` | USD `ko` → LLM 번역 → `translated_description` | 1단계 |
| `--filter` | 영어 프롬프트 증강 → `aug_prompt` | 2단계 |

**2. Stage 2: Asset Factory (TRELLIS)**

  * **Role:** 정제된 프롬프트를 TRELLIS 모델에 입력하여 3D 에셋(.glb 등)을 생성합니다.
  * **Script:** `previz_pipeline/json_parse_and_inference.py` (+ `trellis_inference_core.py`)
  * **Input:** `usd_results.json`
  * **Output (저장 위치):**
      * **DCC용 GLB:** `{usd_root_dir}/scene_n/objects/assets/{object_name}/*.glb` (+ `{name}.glb.meta.json`)
      * **미리보기:** `{output_dir}/{model}/{date}/run_*/previews/scene_n/object_n/`

**3. Stage 3: Format Converter & Reference Injector (Critical Step)**

  * **Role:** GLB를 USD geometry로 변환·수리(Patching)하고, **원본 `object_n.usda`에 geometry reference를 직접 주입**합니다.
  * **Script:** `previz_pipeline/merge_glb_to_usd.py`
      * **Conversion:** `usd_from_gltf`로 `.glb` $\to$ `.geometry.usda` 변환 (assets 폴더에 생성).
      * **Texture Renaming:** `bin/` 텍스처에 object_name을 부여하여 충돌 방지 (`bin/image0.jpg` $\to$ `bin/texture_{object_name}_0.jpg`).
      * **Path Fixing:** USD 내부 텍스처 경로를 상대 경로(`./bin/...`)로 통일.
      * **UV Fix:** PrimvarReader의 UV 이름을 `st0` $\to$ 표준 `st`로 변경 (Blender 호환성).
      * **Reference Injection:** 원본 `object_n.usda`의 defaultPrim 하위 `geometry` 프림에 `@./assets/{object_name}/{object_name}.geometry.usda@` reference를 직접 추가/갱신 후 저장. (기존 reference는 제거 후 교체 → 멱등)
  * **Input:** `assets/{object_name}/*.glb`, 원본 `object_n.usda`
  * **Output:** `assets/{object_name}/{object_name}.geometry.usda` (+ `bin/` 텍스처), reference가 주입된 `object_n.usda`

> **⚠️ 비파괴(Non-destructive) 워크플로우와의 차이:** 이전 버전은 별도 오버라이드 레이어(`*_generated.usda`)만 생성했지만, 현재 버전은 요구사항(어느 USD를 열어도 에셋이 보여야 함)에 맞춰 **원본 USD를 직접 수정**합니다.

-----

#### 📂 Directory Structure

**t2o_results (실험 추적용)**

```text
t2o_results/TRELLIS-text-base/20260624/
├── latest -> run_113439_en-filter/
└── run_112327_en-filter/
    ├── run_manifest.json
    ├── usd_results.json
    ├── results.csv
    └── previews/
        └── scene_1/
            └── object_1/
                ├── generation.json          # prompt, seed, run_id
                └── scene_1_seagull_708510_gs_005s.jpg
```

**movie_usd (DCC 주입용)**

생성 결과는 원본 USD 폴더 구조 안의 `assets` 디렉토리에 저장되며, 개별 오브젝트 USD에 직접 링크됩니다.

```text
/root/previs_proj/movie_usd/hidden_time/
├── hidden_time.usda                       # [Source/Root] scene 참조
└── scene_1/
    ├── scene_1.usda                       # objects/shots 참조
    ├── shot_1/
    │   └── shot_1.usda
    └── objects/
        ├── object_1.usda                  # [Modified] geometry reference가 주입됨
        ├── object_2.usda
        └── assets/                        # [Pipeline Output]
            ├── object_1/
            │   ├── scene_1_seagull_708510.glb
            │   ├── scene_1_seagull_708510.glb.meta.json
            │   ├── object_1.geometry.usda         # 변환된 geometry USD
            │   └── bin/
            │       └── texture_object_1_0.jpg     # Renamed Texture
            └── object_2/
                └── ...
```

> 참고: 객체가 `scene_n/objects/`(scene 직속) 또는 `scene_n/shot_n/objects/`(shot 하위) 어디에 있든, 에셋은 항상 해당 `object_n.usda` 옆의 `assets/{object_name}/`에 저장됩니다.

-----

### 2\. 📖 T2O Pipeline User Manual

이 문서는 텍스트 묘사가 포함된 USD 씬 파일로부터 3D 에셋을 자동 생성하고 원본 USD에 주입하는 파이프라인 사용법을 안내합니다.

#### ✅ 1. 사전 요구 사항 (Prerequisites)

  * **Python 환경:** `pxr` (USD), `torch`, `trellis`, `ollama`, `imageio`, `Pillow` 라이브러리가 설치되어 있어야 합니다.
  * **외부 툴:**
      * **usd\_from\_gltf:** 시스템 PATH 또는 일반 설치 경로에 있어야 합니다 (GLB→USD 변환에 필수).
      * **Ollama:** `--translate` 또는 `--filter` 사용 시에만 필요합니다. 스크립트가 자동으로 `ollama serve`를 시작/종료합니다.
  * **TRELLIS 모델:** `run_usd_to_3D_object.sh` 기본값은 `microsoft/TRELLIS-text-base`입니다. HF 모델은 프로젝트 루트 `hf_models/`에 캐시되며, 로컬에 없으면 자동 다운로드됩니다.

#### 🧠 모델 설정 (Model Configuration)

**TRELLIS (3D 생성)**

| 항목 | 설명 |
|------|------|
| 기본 모델 | `microsoft/TRELLIS-text-base` (`run_usd_to_3D_object.sh` 실행 시) |
| 선택지 | `microsoft/TRELLIS-text-base` · `-large` · `-xlarge` (속도 ↔ 품질/VRAM 트레이드오프) |
| CLI | `--model <hf_name_or_local_path>` (`TRELLIS_MODEL_PATH`보다 우선) |
| 환경변수 | `TRELLIS_MODEL_PATH`, `TRELLIS_BASE_OUTPUT` (기본: `<output_dir>`), `TRELLIS_CONFIG` (YAML, 미지정 시 내장 기본값 사용) |
| 캐시 | 프로젝트 루트 `hf_models/` — HF 이름(`microsoft/TRELLIS-text-large`) 또는 로컬 디렉터리 경로 모두 지원 |

> Stage 2를 스크립트 없이 직접 실행할 때(`json_parse_and_inference.py`) 기본 모델은 `microsoft/TRELLIS-text-xlarge`입니다. 쉘 스크립트와 동일하게 쓰려면 `--model_path microsoft/TRELLIS-text-base`를 명시하세요.

**Ollama (번역/필터, `--translate` / `--filter` 시에만)**

| 항목 | 설명 |
|------|------|
| 기본 모델 | `gemma3:4b` (`OLLAMA_MODEL`, `OLLAMA_FILTER_MODEL`) |
| 사전 설치 | `ollama pull gemma3:4b` (모델명 변경 시 해당 모델 pull) |
| 원격 서버 | `OLLAMA_BASE_URL=http://host:11434` |

**파싱 타입**

  * **`PARSE_TYPE`**: `object` (기본) · `actor` · `both` — 현재 `usd_parser.py`에서 `ENABLE_ACTOR_PARSING=False`이므로 **object만 실질적으로 동작**합니다.

#### 🚀 2. 실행 방법 (How to Run)

##### 전체 파이프라인 실행

```bash
./run_usd_to_3D_object.sh <usd_file> <output_dir> [옵션]
```

  * **`<usd_file>`**: 원본 USD 파일 경로 (root/scene/shot 어느 레벨이든 가능).
  * **`<output_dir>`**: 중간 산출물(JSON)이 저장될 디렉토리 (`{output_dir}/{model_name}/{YYYYMMDD}/usd_results.json`). 최종 에셋은 원본 USD 옆 `assets`에 저장됨.

**옵션:**

  * **`--model <model_path>`**: TRELLIS 모델 경로 또는 HF 모델명 (예: `microsoft/TRELLIS-text-large`).
  * **`--translate`**: 1단계 LLM 번역 활성화 (USD `ko` → 영어). 미지정 시 USD `en` 직접 사용.
  * **`--filter`**: 2단계 LLM 필터링/증강 활성화.
  * **`-- <extra_args>`**: `--` 이후 인자는 TRELLIS 추론 스크립트로 전달됩니다 (예: `-- --max_items 5`).

**주요 환경 변수:** (상세는 위 **모델 설정** 섹션 참고)

  * **`OLLAMA_MODEL`** / **`OLLAMA_FILTER_MODEL`**: 번역/필터용 Ollama 모델명 (기본: `gemma3:4b`).
  * **`OLLAMA_BASE_URL`**: 원격 Ollama 서버 URL.
  * **`PARSE_TYPE`**: 파싱 타입 (`object` | `actor` | `both`, 기본: `object`).
  * **`TRELLIS_MODEL_PATH`**: TRELLIS 모델 (`--model` 미지정 시 사용).
  * **`TRELLIS_BASE_OUTPUT`**: TRELLIS 폴백 출력 베이스 (기본: `<output_dir>`).
  * **`TRELLIS_CONFIG`**: TRELLIS YAML 설정 경로 (미지정 시 steps=12, cfg=7.5, simplify=0.95 등 내장 기본값).

**실행 예시:**

```bash
cd /root/previs_proj/t2o_pipeline

# 기본 실행 (LLM 없이 파싱 → 생성 → 주입)
./run_usd_to_3D_object.sh \
  /root/previs_proj/movie_usd/hidden_time/hidden_time.usda \
  /root/previs_proj/t2o_pipeline/t2o_results

# 번역 + 필터링 활성화, 특정 모델 사용
OLLAMA_MODEL=gemma3:12b \
  ./run_usd_to_3D_object.sh \
  /root/previs_proj/movie_usd/hidden_time/hidden_time.usda \
  /root/previs_proj/t2o_pipeline/t2o_results \
  --model microsoft/TRELLIS-text-large --translate --filter

# 처리 항목 수 제한 (TRELLIS 인자 전달)
./run_usd_to_3D_object.sh \
  /root/previs_proj/movie_usd/hidden_time/hidden_time.usda \
  /root/previs_proj/t2o_pipeline/t2o_results \
  -- --max_items 5
```

##### Stage별 개별 실행

```bash
# Stage 1: USD 파싱 및 증강 → usd_results.json (--type 기본: object)
python3 previz_pipeline/usd_parse_and_augment.py scene.usda --output usd_results.json
# (번역/필터 사용 시 — 여기서 --model은 Ollama 모델명)
python3 previz_pipeline/usd_parse_and_augment.py scene.usda --output usd_results.json --translate --filter --model gemma3:4b

# Stage 2: TRELLIS 3D 생성 (glb는 원본 USD 옆 assets에 저장됨)
# 쉘 스크립트와 동일한 모델을 쓰려면 --model_path를 명시 (--model_path 기본값은 text-xlarge)
python3 previz_pipeline/json_parse_and_inference.py --json usd_results.json --output /mnt/output --base_output /mnt/output --model_path microsoft/TRELLIS-text-base --usd_root /path/to/usd_root

# Stage 3: GLB → USD 변환 및 원본 USD에 reference 주입
python3 previz_pipeline/merge_glb_to_usd.py \
    /root/previs_proj/movie_usd/hidden_time/scene_1/objects/assets/object_1/scene_1_shot_1_NA_73259.glb \
    /root/previs_proj/movie_usd/hidden_time/scene_1/objects/object_1.usda
```

**참고사항:**

  * `merge_glb_to_usd.py`에 `--no-merge` 옵션을 주면 GLB→USD 변환만 수행하고 원본 USD는 수정하지 않습니다.
  * `merge_glb_to_usd.py`의 reference 주입은 멱등(idempotent)합니다 — 여러 번 실행해도 reference가 중복되지 않습니다.

#### 🖥️ 3. 결과 확인 (Output)

작업이 완료되면 별도의 통합 파일을 열 필요 없이, **원본 USD를 그대로** 열면 됩니다.

1.  **Blender** 또는 **USDView**를 실행합니다.
2.  다음 중 어느 파일을 열어도 생성된 에셋이 텍스처와 함께 로딩됩니다.
      * Root: `/root/previs_proj/movie_usd/hidden_time/hidden_time.usda`
      * Scene: `.../scene_1/scene_1.usda`
      * Shot: `.../scene_1/shot_1/shot_1.usda`
      * Object: `.../scene_1/objects/object_1.usda`
3.  생성된 GLB/geometry USD/텍스처는 `scene_n/objects/assets/{object_name}/`에서 직접 확인할 수 있습니다.

