# A100 실험 인프라 재구성 (경량 번들)

회사 서버라 대용량 전송 불가 → **이 번들엔 코드/텍스트/설정만** 담았고,
무거운 것(env·모델·VLM·ollama)은 전부 **A100서 재구성**한다. 아래 순서대로.

## 번들 내용물
```
RECONSTRUCT_ON_A100.md            ← 이 문서
prompt_metrics/install.sh         ← ❌항목1: VQA 채점 env 재구축 스크립트
prompt_metrics/pip_freeze.reference.txt  ← 4090 라이브 핀 119개 전량(대조용)
fix_configs.sh                    ← ⚠️항목4: config stale 경로 치환
prompt_lab_requirements.txt       ← ❌항목3(.venv) 재생성용 (CV 의존만, 초경량)
vqa_scores/phase2_vqa_inanimate.json     ← ⚠️항목5: 무생물 ablation 원점수(재생성 안 하려면 이거)
vqa_scores/phase2_vqa_creatures.json     ← ⚠️항목5: 생명체 ablation 원점수
```

---

## 1) ❌ prompt_metrics env (VQAScore 채점 — objective A/B/C 채점 필수)
크기 6.4G라 복사 불가 → 재구축(arch 의존 없음, A100 sm_80 그대로 동작):
```bash
bash prompt_metrics/install.sh          # conda env 'prompt_metrics' 생성
```
- VLM `clip-flant5-xl`(~4G)·`ImageReward.pt`(~2G)는 **첫 채점 실행 시 자동 다운로드** → 전송 불필요.
- 핀 대조가 필요하면 `pip_freeze.reference.txt`(4090 실측 전량)와 비교.

## 2) ❌ ollama + gpt-oss:20b (B/auto 모드 필수)
13G snap store라 복사 번거로움 → **A100서 재취득**:
```bash
# ollama 설치 후
ollama pull gpt-oss:20b
```
- reasoning 모델이라 콜당 느림(~99s@4090). prompt_lab 사용 시 `llm.params.max_tokens=4096` 넉넉히.

## 3) ❌ prompt_lab/.venv (복사하면 절대경로 깨짐)
133M venv에 `/home/sr` 경로 박힘 → 복사 금지. 둘 중 택1:
- **(권장·빠름)** trellis2 env 재사용: `export PROMPT_LAB_PYTHON=<trellis2 python 경로>`
- 재생성: `python -m venv prompt_lab/.venv && prompt_lab/.venv/bin/pip install -r prompt_lab_requirements.txt`
  (의존이 PyYAML/numpy/Pillow/pytest 뿐 — CV 탐지기만, 학습모델 없음)

## 4) ⚠️ config stale FLUX 경로
`image_creature_ablation.yaml`·`image_template_ablation.yaml`·`image_seagull_tpose_auto.yaml`
6줄이 dev `/home/sr/previs_proj/...` → A100 경로로 수정 필요:
```bash
bash fix_configs.sh /root/previs_proj    # ← A100 실제 repo 루트로 인자 확인!
```

## 5) ⚠️ runs 원점수 (재분석용 — 선택)
768장 이미지·events는 못 옮김(1.2G). 하지만 **VQA 원점수 JSON은 이 번들에 포함**(각 ~9KB).
무생물/생명체 ablation을 **이미지 재생성 없이 재집계/LOO 재분석**하려면 `vqa_scores/*.json` 사용.
결론만 필요하면 `prompt_lab/docs/EXPERIMENT_LOG.md`로 충분(스킵 가능).
> ⚠️ 이미지 원본이 필요한 재분석은 config대로 **재생성**해야 함(원본 768장은 전송 제외됨).

---

## 완료 확인 (Definition of Done)
- [ ] `conda run -n prompt_metrics python -c "import t2v_metrics"` 통과
- [ ] `ollama list` 에 gpt-oss:20b
- [ ] `PROMPT_LAB_PYTHON` 지정 or `.venv` 재생성
- [ ] `grep -rn /home/sr configs/` 결과 없음(경로 치환 완료)
- [ ] FLUX/TRELLIS.2 가중치 HF 재다운(게이트 승인 `raengs`) — A100_HANDOFF §4

---

## ✅ A100 실측 재구성 결과 (2026-07-28, 완료)

전 항목 재구성 완료. 실행 시 알아둘 실측값·정정:

- **prompt_lab 위치**: 정리하면서 `t2o_pipeline/prompt_lab/` 로 옮겼다(공용 루트 정리). 그래서 `fix_configs.sh`
  는 `t2o_pipeline/` 에서 실행: `cd t2o_pipeline && bash docs/a100_reconstruct/fix_configs.sh /root/previs_proj`.
- **① prompt_metrics env**: 생성 완료. `conda run -n prompt_metrics python -c "import t2v_metrics"` → OK, cuda True.
- **② ollama gpt-oss:20b**: 호스트 `ollama` 컨테이너에 pull 완료(`ollama list` 확인). 실제 응답 확인(hello).
  ⚠️ **엔드포인트는 `http://172.17.0.1:11434`** (도커 브리지 게이트웨이). previs-prep 은 ollama 컨테이너와
  네트워크가 달라 `ollama` 호스트명은 안 됨. prompt_lab config·파이프라인 모두 이 IP 사용.
  - prompt_lab: config `endpoint: http://172.17.0.1:11434/v1/chat/completions` 로 치환함.
  - 파이프라인 `--filter`: `OLLAMA_HOST=http://172.17.0.1:11434` 지정(trellis2 에 `ollama` 파이썬 패키지 설치함).
- **③ .venv**: 재생성 대신 **trellis2 재사용**. `export PROMPT_LAB_PYTHON=/root/miniconda3/envs/trellis2/bin/python`.
  trellis2 에 없던 `pytest` 추가함. `--test` → **184 passed, 1 skipped** 확인.
- **④ config 경로**: `/home/sr` → `/root` 치환 완료(3개 config). stale 없음.
- **⑤ vqa_scores**: `docs/a100_reconstruct/vqa_scores/{inanimate,creatures}.json` 에 보관(재집계/LOO 재분석용).
- **FLUX 54G / TRELLIS.2 16G**: 이미 HF 재다운 완료(hf_models/).

### prompt_lab 실행 예 (실측)
```bash
docker exec -it previs-prep bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate trellis2
export PROMPT_LAB_PYTHON=/root/miniconda3/envs/trellis2/bin/python
export OLLAMA_HOST=http://172.17.0.1:11434
cd /root/previs_proj/t2o_pipeline
./prompt_lab.sh --test            # 184 passed 기대
# ablation/auto 실행 시 VQA 채점은 prompt_metrics env 데몬을 subprocess 로 호출(메인 프로세스, Phase2)
```
