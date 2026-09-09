"""E2E functional test: ERNIE (same-env subprocess) -> TRELLIS.2 -> GLB.
Run in trellis2 with PYTHONPATH=previz_pipeline:trellis2_src.
Use T2I_GPU for a separate T2I card while TRELLIS.2 is resident.
T2I_MODEL_PATH may override the local ERNIE-Image-Turbo directory.
"""
import os
from pathlib import Path
from trellis2_inference_core import Trellis2InferenceCore
from t2i_config import DEFAULT_T2I_MODEL_PATH
from t2i_prompt_builder import build_t2i_prompt

_REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("E2E_OUT", _REPO_ROOT / "t2o_results" / "smoke_out" / "e2e"))
OUT.mkdir(parents=True, exist_ok=True)

core = Trellis2InferenceCore(
    t2i_model_path=os.environ.get(
        "T2I_MODEL_PATH", DEFAULT_T2I_MODEL_PATH),
)
# Minimal state normally set by _process_file_batch:
core.run_id = "e2e_test"
core.output_base = OUT

core.load_pipeline()

# 프로덕션과 동일하게 §9 템플릿을 거친다. 맨 프롬프트를 그대로 넘기면 prompt
# enhancer(기본 ON)가 장면 전체를 지어내 단일 객체 검증이 되지 않는다 —
# 격리 문구(single object/neutral background)는 빌더가 넣는다.
_prompt, _ = build_t2i_prompt(
    object_name="armchair", appearance="worn brown leather",
    base_description="a worn leather armchair", category=None,
    rig_type="static_object")

result = core._generate_single(
    prompt=_prompt,
    predefined_name="e2e_chair",
    config={"seed": 42},
    formats=["glb", "mp4"],
    postprocessing_config={"texture_size": 2048, "simplify_target": 16777216},
    llm_model="test",
    record_context={
        "scene": "e2e", "shot": "shot_unknown", "target_dir_name": "e2e_chair",
        "target_type": "item", "object_name": "e2e_chair", "run_id": "e2e_test",
    },
)

print("=== E2E RESULT ===")
for k in ("success", "generation_time", "render_time", "save_time", "total_time", "save_path"):
    print(f"  {k}: {result.get(k)}")
print("  saved_files:")
for f in result.get("saved_files", []):
    sz = os.path.getsize(f) / 1e6 if os.path.exists(f) else 0
    print(f"    {f}  ({sz:.2f} MB)")
assert result.get("success") and any(f.endswith(".glb") for f in result.get("saved_files", [])), "E2E FAILED"
print("E2E_OK")
