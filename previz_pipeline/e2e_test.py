"""E2E functional test for Trellis2InferenceCore.
Exercises the full glue: prompt -> FLUX (subprocess/t2i env) -> TRELLIS.2 ->
o_voxel to_glb (PNG). Uses FLUX.1-schnell (Apache-2.0), which is what the
shipping configuration must use; the earlier dev-only run was a functional
stand-in from before the schnell gate was accepted.
To isolate a schnell-specific failure, point T2I_MODEL_PATH at FLUX.1-dev —
dev is non-commercial and for diagnosis only.
Run in the trellis2 env with PYTHONPATH=previz_pipeline:trellis2_src.
"""
import os
from pathlib import Path
from trellis2_inference_core import Trellis2InferenceCore

_REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("E2E_OUT", _REPO_ROOT / "t2o_results" / "smoke_out" / "e2e"))
OUT.mkdir(parents=True, exist_ok=True)

core = Trellis2InferenceCore(
    t2i_model_path=os.environ.get(
        "T2I_MODEL_PATH", str(_REPO_ROOT / "hf_models" / "FLUX.1-schnell")),
)
# Minimal state normally set by _process_file_batch:
core.run_id = "e2e_test"
core.output_base = OUT

core.load_pipeline()

result = core._generate_single(
    prompt="a worn leather armchair",
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
