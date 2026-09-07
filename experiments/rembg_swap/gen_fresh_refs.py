"""현행 배송 경로(§9 프롬프트 빌더 + FLUX.1-schnell)로 신선한 레퍼런스 생성.

기존 t2o_results 의 레퍼런스 11장은 프롬프트 하드닝 이전(7~8월 초) 산출물이라
다중 객체 장면이 섞여 있다. rembg 교체 판정은 지금 배송되는 프롬프트가 만드는
이미지에서 해야 하므로, 정확히 그 경로로 다시 뽑는다.

라우팅 분기 커버: bird / static_object ×3 / quadruped.

Run:
    conda activate trellis2
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=previz_pipeline python experiments/rembg_swap/gen_fresh_refs.py
"""
from pathlib import Path

from t2i_prompt_builder import build_t2i_prompt
from text_to_image import TextToImage

OBJECTS = [
    # (name, appearance, base_description, rig_type)
    ("seagull", "white and grey feathers", "a seagull", "bird"),
    ("smartphone", "black slate with metal frame", "a modern smartphone", "static_object"),
    ("car", "silver compact SUV", "a passenger car", "static_object"),
    ("armchair", "worn brown leather", "a worn leather armchair", "static_object"),
    ("dog", "golden retriever coat", "a large dog", "quadruped"),
]

out_dir = Path(__file__).resolve().parent / "fresh_refs"
out_dir.mkdir(parents=True, exist_ok=True)

t2i = TextToImage()
t2i.load()
for name, appearance, desc, rig in OBJECTS:
    prompt, body_plan = build_t2i_prompt(
        object_name=name, appearance=appearance, base_description=desc,
        category=None, rig_type=rig)
    for seed in (42, 777):
        dest = out_dir / f"{name}_{seed}.png"
        if dest.exists():
            continue
        t2i.generate(prompt, seed=seed, out_path=str(dest))
        print(f"[gen] {dest.name}  (rig={rig}, body_plan={body_plan})")
print("done")
