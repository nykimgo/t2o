# 객체 생성 파이프라인 성능 실험 런북 — A(프리뷰 렌더 게이팅) / C(decimation_target 하향)

> 대상: `/home/sr/previs_proj` (4090×4 서버), `object_generate.sh` 3단계 파이프라인.
> 이 문서는 **2026-08-05 실측 베이스라인**과, 병목 A·C 를 검증하는 실험 절차만 담는다.
> 실행 환경·함정 일반론은 `../../RUN_GUIDE.md`, `docs/RUN_GUIDE.md` 참조.

---

## 0. 베이스라인 (2026-08-05 실측)

`T2I_GPU=1 ./object_generate.sh tmp/htest/hidden_time.usda --no-filter` (객체 3개, 시드 랜덤)

**총 779.0s (13분)**

> ⚠️ **이 표는 실험 A 적용 전(2026-08-05) 값이다.** 당시 기본 formats 가 `glb mp4 jpg` 였으므로
> `render 122.5s` 가 포함돼 있다. A 적용 후 기본 실행(`formats=glb`)에서는 **이 122.5s 가 사라진다**
> → 같은 조건이면 **약 656s** 가 새 기준선이다(미실측 추정, 렌더분만 차감).
> 이후 회차와 비교할 때 formats 를 맞추거나 `render_time` 을 제외하고 볼 것.

| 단계 | 시간 | 비중 |
|---|---:|---:|
| ① USD 파싱 | 0.9s | 0.1% |
| ② 추론 | 760.0s | 97.6% |
| ③ GLB→USD 병합 | 18.0s | 2.3% |

②의 내부 분해:

| 구간 | 시간 | 비중(전체) |
|---|---:|---:|
| **save (`to_glb`)** | **345.7s** | **44%** |
| T2I (FLUX) | 124.3s | 16% |
| **render (프리뷰 영상)** | **122.5s** | **16%** |
| I2O (TRELLIS.2) | 81.3s | 10% |
| TRELLIS.2 모델 로드 | 78.6s | 10% (1회) |

객체별 (시드 랜덤이라 편차가 큼):

| 객체 | 시드 | I2O | render | **save** | 합계 |
|---|---:|---:|---:|---:|---:|
| object_1 seagull | 224530 | 22.9s | 40.0s | 24.0s | 129.2s |
| object_2 smartphone | 853458 | 23.7s | 40.9s | 49.4s | 158.4s |
| object_3 car | 407317 | 34.7s | 41.6s | **272.2s** | 386.2s |

산출물은 셋 다 **~95만 면 / GLB 36MB / geometry.usda 92MB** 로 수렴한다
(`decimation_target=1,000,000` 에 걸림).

---

## 1. 공통 준비

### 1-1. 원본 보호 — 반드시 사본에서 돌린다

⚠️ `movie_usd/hidden_time` 의 `object_N.usda` · `geometry.usda` 는 **원 서버 완성본(정답 레퍼런스)** 이다.
파이프라인 ③단계는 원본 USD 에 geometry 참조를 **주입(덮어쓰기)** 하므로 원본으로 돌리면 레퍼런스가 날아간다.

```bash
cd /home/sr/previs_proj
rm -rf tmp/htest
rsync -a --exclude 'objects/assets/' movie_usd/hidden_time/ tmp/htest/   # ~2초, 1.2G
```

`--exclude objects/assets/` 는 이전 산출물을 빼서 깨끗한 상태에서 재생성시키기 위함이다.
**매 실험 회차마다** 아래로 초기화한다(누적되면 파일명에 `_1` 접미사가 붙고 ③단계가 엉킨다):

```bash
rm -rf /home/sr/previs_proj/tmp/htest/scene_1/objects/assets
```

### 1-2. env

```bash
source /home/sr/miniconda3/etc/profile.d/conda.sh
conda activate trellis2      # 런처는 activate 를 스스로 하지 않는다
```

### 1-3. ★ 시드 고정 — 이 실험의 전제

기본값이 `seed: random` 이라 **같은 객체도 회차마다 save 가 24s↔272s 로 튄다.**
시드를 고정하지 않으면 A·C 의 효과가 편차에 완전히 묻힌다.

```bash
# 런처 '--' 뒤 인자는 2단계(json_parse_and_inference.py)로 그대로 전달된다
T2I_GPU=1 ./object_generate.sh tmp/htest/hidden_time.usda --no-filter -- --seed 20260805
```

`--seed` 는 **모든 객체에 동일 시드**를 적용한다(객체별 시드가 아님).
→ 베이스라인 표(§0)의 숫자는 랜덤 시드라 **직접 비교 대상이 아니다.**
**반드시 같은 고정 시드로 before 를 다시 측정**한 뒤 after 와 비교할 것.

### 1-4. 타임스탬프 로거

```bash
mkdir -p /home/sr/previs_proj/tmp/perf && cat > /home/sr/previs_proj/tmp/perf/stamp.py <<'EOF'
#!/usr/bin/env python3
"""stdin 각 줄에 [절대시각 / 시작후경과 / 직전줄과의간격] 을 붙여 출력."""
import sys, time
t0 = time.time(); prev = t0
for line in sys.stdin:
    now = time.time()
    sys.stdout.write(f"[{time.strftime('%H:%M:%S', time.localtime(now))} "
                     f"+{now-t0:8.2f}s d={now-prev:6.2f}s] {line.rstrip()}\n")
    sys.stdout.flush(); prev = now
EOF
```

실행 템플릿 (`RUNTAG` 만 바꿔 쓴다):

```bash
cd /home/sr/previs_proj
RUNTAG=C_250k
rm -rf tmp/htest/scene_1/objects/assets
T2I_GPU=1 stdbuf -oL -eL ./object_generate.sh tmp/htest/hidden_time.usda --no-filter \
  -- --seed 20260805 2>&1 \
  | stdbuf -oL python3 -u tmp/perf/stamp.py > tmp/perf/$RUNTAG.log 2>&1
```

### 1-5. 숫자 읽는 법

**구간별 시간**은 run 디렉터리의 `results.csv` 에 객체별로 다 들어 있다
(`t2i_time, i2o_time, render_time, save_time, total_time`).

```bash
cat > /home/sr/previs_proj/tmp/perf/report.py <<'EOF'
"""사용법: python report.py <run_dir> [<sandbox_assets_dir>]"""
import csv, glob, os, sys
run = sys.argv[1]
assets = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/sr/previs_proj/tmp/htest/scene_1/objects/assets"
tot = {k: 0.0 for k in ("t2i_time", "i2o_time", "render_time", "save_time", "total_time")}
print(f"{'object':<12}{'seed':>9}{'i2o':>8}{'render':>9}{'save':>9}{'total':>9}")
for r in csv.DictReader(open(os.path.join(run, "results.csv"))):
    for k in tot: tot[k] += float(r[k])
    print(f"{r['object_name']:<12}{r['seed']:>9}{float(r['i2o_time']):>8.1f}"
          f"{float(r['render_time']):>9.1f}{float(r['save_time']):>9.1f}"
          f"{float(r['total_time']):>9.1f}")
print(f"{'ALL':<12}{'':>9}{tot['i2o_time']:>8.1f}{tot['render_time']:>9.1f}"
      f"{tot['save_time']:>9.1f}{tot['total_time']:>9.1f}   (T2I {tot['t2i_time']:.1f})")
try:
    import trimesh
    print("\n--- 산출물 ---")
    for f in sorted(glob.glob(os.path.join(assets, "*/*.glb"))):
        m = trimesh.load(f, force="mesh", process=False)
        usd = glob.glob(os.path.join(os.path.dirname(f), "*.geometry.usda"))
        print(f"{os.path.basename(f):<30} faces={len(m.faces):>8}  "
              f"glb={os.path.getsize(f)/1e6:>6.1f}MB  "
              f"usd={(os.path.getsize(usd[0])/1e6 if usd else 0):>6.1f}MB")
except ImportError:
    pass
EOF

# 사용
python /home/sr/previs_proj/tmp/perf/report.py \
  "$(ls -d /home/sr/previs_proj/t2o_pipeline/t2o_results/TRELLIS.2-4B/*/run_* | tail -1)"
```

**단계 경계(①/②/③)와 모델 로드 시간**은 로그 타임스탬프에서 읽는다:

```bash
grep -E "1/3|2/3|3/3|Loading TRELLIS|전체 파이프라인 완료" tmp/perf/$RUNTAG.log | grep -v Rendering
```

### 1-6. 실험 순서 규칙

- ~~**A 와 C 를 동시에 바꾸지 말 것.**~~ → **A 는 2026-08-06 적용 완료.** 이제 C 만 남았다.
- **C 의 품질 평가에는 프리뷰가 필요하므로 YAML `output.formats` 에 `mp4`/`jpg` 를 명시한다.**
  A 적용으로 기본값이 `glb` 가 됐고 렌더가 formats 로 게이팅되므로, 안 적으면 프리뷰가 안 나온다(§3-2).
- 회차마다 디스크가 ~400MB 쌓인다. `t2o_results/` 와 샌드박스 assets 를 주기적으로 정리.

---

## 2. 실험 A — 프리뷰 렌더를 `formats` 로 게이팅  ✅ **적용 완료 (2026-08-06)**

> **측정 없이 채택했다.** 원인·효과가 코드로 자명하고(요청하지 않은 프리뷰를 렌더),
> GLB 산출물 경로와 무관해 리스크가 없다고 판단. before/after 실측은 생략했다.
> 아래 §2-1~2-4 는 판단 근거와 패치 내용의 기록으로 남긴다.
>
> **함께 바뀐 것 — 기본 formats 가 `glb` 로 변경됐다:**
> - `object_generate.sh:129` — `${TRELLIS_FORMATS:-glb mp4 jpg}` → `${TRELLIS_FORMATS:-glb}`
> - `json_parse_and_inference.py:249` — argparse 기본값 `['glb','ply','mp4','jpg']` → `['glb']`
>   (`ply` 는 v2 에 gaussian 이 없어 무의미하므로 함께 제거)
>
> **기본 실행에서 없어지는 것**: 턴테이블 `*_pbr.mp4` 1개 + 썸네일 `*_00Ns.jpg` 3장.
> **그대로 남는 것**: `*.glb`, `*.glb.meta.json`, **`*_ref.png`(T2I 참조 이미지 — formats 와
> 무관하게 항상 저장. 3D 가 이상할 때 이미지 문제인지 리프트 문제인지 가르는 디버깅 1순위)**,
> `results.csv`, `generation.json`, `run_manifest.json`, geometry.usda + 텍스처 + USD 주입.
>
> 프리뷰가 필요하면 `TRELLIS_FORMATS="glb mp4 jpg"` 로 예전과 동일하게 나온다.

### 2-1. 현상 (실증 완료)

`trellis2_inference_core.py:252` 의 렌더 게이트가 `formats` 를 보지 않고 `envmap` 만 본다:

```python
video = None
if self.envmap is not None:              # ← formats 와 무관
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=self.envmap))
```

`TRELLIS_FORMATS="glb"` 로 재실행해 확인한 결과 — **render 41.4s 를 그대로 지불**했고
(`glb mp4 jpg` 실행의 41.7s 와 동일), `:304` 의 `or video is not None` 때문에
**JPG 썸네일까지 그대로 생성**됐다. 실제로 빠진 건 mp4 쓰기(~2s)뿐이다.

렌더는 120프레임 프리뷰 영상(~3.86 it/s ≈ 31s) + PBR 후처리로 객체당 ~41s 고정.

### 2-2. 패치 (2곳)

**(1) `previz_pipeline/trellis2_inference_core.py:250-258`**

```python
        # --- Stage C: PBR preview render (best-effort) ---
        render_start = time.time()
        video = None
-        if self.envmap is not None:
+        # 프리뷰(mp4/jpg)를 실제로 요청했을 때만 렌더한다. 이 게이트가 없으면
+        # --formats glb 로도 120프레임 렌더(객체당 ~41s)를 그대로 지불한다.
+        want_preview = ("mp4" in formats) or ("jpg" in formats)
+        if want_preview and self.envmap is not None:
             try:
                 video = render_utils.make_pbr_vis_frames(
                     render_utils.render_video(mesh, envmap=self.envmap))
             except Exception as e:
                 logging.warning(f"⚠️ PBR render failed (skipping): {e}")
         render_time = time.time() - render_start
```

**(2) `previz_pipeline/trellis2_inference_core.py:304`** — jpg 가 formats 를 무시하는 문제

```python
-        if ("jpg" in formats or video is not None) and video is not None:
+        if "jpg" in formats and video is not None:
```

### 2-3. 적용 후 확인 (측정 대신 수행한 것)

정적 검사만 했다. 실행 측정은 생략.

```bash
python -m py_compile previz_pipeline/trellis2_inference_core.py \
                     previz_pipeline/json_parse_and_inference.py   # OK
bash -n /home/sr/previs_proj/object_generate.sh                     # OK
python previz_pipeline/json_parse_and_inference.py --help | grep -A2 formats   # 기본값 glb 확인
```

### 2-4. 나중에 검증한다면 (선택)

다음 전체 실행 때 아래만 확인하면 충분하다. 별도 회차를 잡을 필요는 없다.

| 항목 | 기대 |
|---|---|
| `results.csv` 의 `render_time` | **≈ 0s** (기존 객체당 ~41s) |
| **GLB 면수 / 파일 크기** | 시드가 같다면 패치 전과 **완전 동일** |
| ③단계 결과 | `✅ 성공: N개`, `❌ 오류: 0개` |
| `previews/` 하위 | `*_ref.png` 만 있고 mp4·jpg 없음(=의도) |

산출물이 달라지면 렌더 외 경로를 건드린 것이다 → 롤백(§5).

**프리뷰 회귀 확인**(프리뷰를 다시 쓸 일이 생기면):
`TRELLIS_FORMATS="glb mp4 jpg"` 로 1회 돌려 mp4·jpg 가 예전처럼 생성되는지 본다.

---

## 3. 실험 C — `decimation_target` 하향

### 3-1. 현상

`trellis2_inference_core.py:276` 의 기본값이 **1,000,000 면**인데,
`json_parse_and_inference.py:119` 는 `postprocessing` 에 `texture_size` 만 넣고
`decimation_target` 은 넘기지 않는다 → 항상 기본값 1M 이 적용된다.

실측: 세 객체 모두 목표치에 붙어서 나온다.

| 객체 | 면수 | GLB | geometry.usda | save |
|---|---:|---:|---:|---:|
| seagull | 926,908 | 35.9MB | 93.8MB | 24.0s |
| smartphone | 938,850 | 35.6MB | 93.3MB | 49.4s |
| car | 971,047 | 36.1MB | 92.0MB | 272.2s |

**스마트폰조차 93.9만 면**이다. 프리비즈 용도에 과하며, save 시간·파일 크기·③단계
변환 시간이 모두 여기에 비례한다.

### 3-2. ★ 코드 수정 불필요 — YAML config 로 주입한다

`json_parse_and_inference.py:297` 의 `--config` 경로는 YAML 을 `yaml.safe_load` 로
그대로 읽어 `config` 로 쓰고, 코어는 `postprocessing_config.get("decimation_target", …)`
로 소비한다. 따라서 YAML 만으로 스윕이 가능하다.

⚠️ **`--config` 를 주면 `--seed` · `--texture_size` · `--formats` 인자가 무시된다**
(해당 대입이 `else` 분기 안에 있음). YAML 에 **전부 명시**해야 한다.

```bash
mkdir -p /home/sr/previs_proj/tmp/perf/cfg
cat > /home/sr/previs_proj/tmp/perf/cfg/dec_250k.yaml <<'EOF'
# TRELLIS.2(v2) 용. sampler 파라미터는 넣지 않는다 — 모델 기본값이 E2E 검증된 유일 구성.
generation:
  seed: 20260805
output:
  formats: [glb, mp4, jpg]      # ★ C 의 품질 평가에 프리뷰가 필요하므로 명시적으로 켠다
postprocessing:
  texture_size: 1024            # 기본과 동일하게 유지(교란 방지)
  decimation_target: 250000     # ← 이 값만 회차마다 바꾼다
EOF
```

> ⚠️ **실험 A 적용(2026-08-06) 이후 달라진 점 — `formats` 를 반드시 명시할 것.**
> 이제 기본값이 `glb` 이고, **프리뷰는 `formats` 에 `mp4`/`jpg` 가 있을 때만 렌더된다.**
> C 는 면수별 결과를 **눈으로 비교**하는 실험이므로 위처럼 켜 두어야 한다.
> 빠뜨리면 `previews/` 에 `*_ref.png`(T2I 참조 이미지)만 남아 품질 판단을 할 수 없다.
>
> 시간 해석 시 주의: 프리뷰를 켜면 회차마다 **객체당 ~41s 의 `render_time` 이 다시 붙는다.**
> 이는 `decimation_target` 과 무관한 고정비다. **`save_time` 은 렌더와 분리 계측되므로
> C 의 효과는 `save_time` 으로 보면 되고**, `total_time` 끼리 비교할 때만 렌더분을 감안하면 된다.

### 3-2-1. 파일럿 실측 (2026-08-05, 이 경로가 실제로 동작함을 확인)

`decimation_target: 100000`, `formats: [glb]`, seagull 1개로 검증:

| 항목 | 기본(1M) | 파일럿(100k) | 변화 |
|---|---:|---:|---:|
| 실제 면수 | 926,908 | **98,745** | 목표치에 정확히 붙음 |
| GLB | 35.9MB | **6.5MB** | 5.5× ↓ |
| geometry.usda | 93.8MB | **10.4MB** | 9× ↓ |
| save | 24.0s | **10.5s** | 2.3× ↓ |
| ③단계 | 6.5s | **1.25s** | 5× ↓ |

③단계까지 `성공 1개 / 오류 0개` 로 완주했다. 로그에 `📄 Loaded YAML config` 가 찍힌다.
**단, 이 파일럿은 품질을 평가하지 않았다**(`formats: [glb]` 라 프리뷰가 없음).
시간·용량 효과는 확인됐고, 남은 것은 §3-5 의 품질 판단이다.
시드가 달라(20260805 vs 224530) save 24.0s↔10.5s 비교는 참고치이며,
정식 비교는 §3-3 처럼 동일 YAML 경로·동일 시드로 기준점을 다시 잡아야 한다.

실행:

```bash
cd /home/sr/previs_proj
RUNTAG=C_250k; rm -rf tmp/htest/scene_1/objects/assets
T2I_GPU=1 TRELLIS_CONFIG=tmp/perf/cfg/dec_250k.yaml stdbuf -oL -eL \
  ./object_generate.sh tmp/htest/hidden_time.usda --no-filter 2>&1 \
  | stdbuf -oL python3 -u tmp/perf/stamp.py > tmp/perf/$RUNTAG.log 2>&1
```

첫 회차에서 **YAML 이 실제로 먹었는지 반드시 확인**한다(오타 시 조용히 기본값으로 돈다):

```bash
grep -E "Loaded YAML config" tmp/perf/C_250k.log      # 이 줄이 없으면 적용 안 된 것
python tmp/perf/report.py "$(ls -d t2o_pipeline/t2o_results/TRELLIS.2-4B/*/run_* | tail -1)"
# → faces 가 목표치 근방으로 내려왔는지 확인
```

### 3-3. 스윕

`decimation_target` 만 바꿔 5회차. 기준점(1,000,000)도 **동일 YAML 경로로** 돌려야
공정한 비교가 된다(인자 경로 vs config 경로 차이를 제거).

| 회차 | decimation_target |
|---|---:|
| C_1M (기준) | 1,000,000 |
| C_500k | 500,000 |
| C_250k | 250,000 |
| C_100k | 100,000 |
| C_50k | 50,000 |

회차당 대략 8~13분, 전체 1시간 내외.

### 3-4. 측정 항목

| 항목 | 출처 |
|---|---|
| `save_time` (객체별·합계) | `results.csv` |
| 실제 면수 | `report.py` (trimesh) |
| GLB / geometry.usda 크기 | `report.py` |
| ③단계 시간 | 로그의 `3/3` → `전체 파이프라인 완료` 간격 |
| 총 시간 | 로그 마지막 줄 경과 |

**car(object_3)가 최악 케이스**이므로 판단은 car 기준으로 한다.

### 3-5. 품질 평가 (이 실험의 핵심 — 시간보다 이쪽이 결정적)

각 회차의 프리뷰로 비교한다:
`t2o_results/TRELLIS.2-4B/<날짜>/<run>/previews/scene_1/object_*/`

| 체크 | 무엇을 보나 |
|---|---|
| 실루엣 | 갈매기 날개 끝·다리, 자동차 필러/사이드미러 같은 얇은 구조가 뭉개지는가 |
| 표면 | 평면(스마트폰 몸체·차체)에 각진 패싯이 보이는가 |
| 텍스처 | UV 재전개로 텍스처가 흐려지거나 이음새가 생기는가 (`texture_size` 는 고정이므로 면수 영향만 봄) |
| USD 합성 | ③단계 주입 후 `object_N.usda` 를 usdview 로 열어 정상 표시되는지 |

프리뷰 mp4 를 나란히 두고 보면 판단이 빠르다. **"프리비즈에서 이 정도면 충분한가"**
가 기준이며, 이는 시간 수치가 아니라 제품 판단이다.

### 3-6. 결과 기록

| 회차 | target | car 면수 | car save | save 합계 | ③단계 | 총 시간 | GLB/USD 크기 | 품질 판정 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| C_1M | 1,000,000 | | | | | | | 기준 |
| C_500k | 500,000 | | | | | | | |
| C_250k | 250,000 | | | | | | | |
| C_100k | 100,000 | | | | | | | |
| C_50k | 50,000 | | | | | | | |

**결론란**: 채택 값 = ____ / 근거 = ____

### 3-7. 채택 시 배선 (실험이 아니라 반영 단계)

YAML 은 실험용이다. 값이 정해지면 상시 적용되도록 배선한다 — 택 1:

1. `json_parse_and_inference.py` 에 `--decimation_target` 인자 추가
   (`:255` 의 `--texture_size` 를 그대로 흉내내고, `:308` 옆에 대입 한 줄 추가),
   런처에 `TRELLIS_DECIMATION_TARGET` 환경변수 노출.
2. 또는 `trellis2_inference_core.py:276` 의 기본값 자체를 변경.

1번을 권한다 — 객체 종류에 따라 조정 여지를 남기고, 기본값 변경은 다른 호출 경로에도
영향을 주기 때문이다.

---

## 4. A + C 합산 검증

A 채택 + C 채택값으로 1회 완주해 상호작용이 없는지 본다.

C 채택값을 정한 뒤, **프리뷰를 끄고**(`formats: [glb]`) 1회 완주해 실사용 시간을 잡는다.
A 는 이미 적용돼 있으므로 formats 만 `[glb]` 로 두면 렌더가 건너뛰어진다.

```bash
RUNTAG=AC_final; rm -rf tmp/htest/scene_1/objects/assets
T2I_GPU=1 TRELLIS_CONFIG=tmp/perf/cfg/final.yaml stdbuf -oL -eL \
  ./object_generate.sh tmp/htest/hidden_time.usda --no-filter 2>&1 \
  | stdbuf -oL python3 -u tmp/perf/stamp.py > tmp/perf/$RUNTAG.log 2>&1
```

| 항목 | 베이스라인(A 이전) | A+C | 절감 |
|---|---:|---:|---:|
| 총 시간 | 779.0s | | |
| render 합계 | 122.5s | **0s (A 로 확정)** | −122.5s |
| save 합계 | 345.7s | | |
| ③단계 | 18.0s | | |
| GLB 크기(평균) | 35.9MB | | |

③단계가 정상(`성공 3개 / 오류 0개`)이고 `object_N.usda` 합성이 깨지지 않았는지 확인.

---

## 5. 롤백

- A (적용 완료): 되돌리려면 4곳 —
  `trellis2_inference_core.py` 의 렌더 게이트·jpg 조건 2곳,
  기본값 2곳(`object_generate.sh:129`, `json_parse_and_inference.py:249`).
  **부분 롤백도 가능**: 코드는 두고 `TRELLIS_FORMATS="glb mp4 jpg"` 만 주면 예전 동작이 된다.
- C: `TRELLIS_CONFIG` 를 빼면 즉시 기본값(1M)으로 복귀. 코드를 건드리지 않았다면 롤백 불필요.
- 산출물: 실험은 전부 `tmp/htest` 사본에서만 이뤄지므로 `movie_usd/` 원본은 영향 없음.
  (검증: `ls -la movie_usd/hidden_time/scene_1/objects/*.usda` 의 mtime 이 실험 전과 같아야 함)

---

## 6. 손대지 않은 것 (참고)

같은 측정에서 나온 다른 병목이지만 이 문서 범위 밖이다:

- **T2I 124.3s** — FLUX bf16 이 24GB 카드에 안 들어가 `enable_model_cpu_offload()` 로 도는 게 원인.
  transformer 를 유휴 GPU 2장에 샤딩·완전 상주시키면 **장당 42~44s → 3.0~3.6s** (실측).
  가중치·dtype 동일이라 품질 영향 없음. 별도 실험으로 진행할 것.
- **TRELLIS.2 로드 78.6s** — run 당 1회 고정비. 상주 워커로만 회수 가능.
- **I2O 81.3s** — 실제 3D 생성. 샘플러 기본값이 E2E 검증된 유일 구성이라 건드리면 품질 리스크.
- ③단계 18.0s 는 이미 최적화된 값이다(GLB→USD 네이티브 변환기의 numpy 벌크 경로 적용 후).
