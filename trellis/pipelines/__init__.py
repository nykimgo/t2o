from . import samplers
from .trellis_image_to_3d import TrellisImageTo3DPipeline
from .trellis_text_to_3d import TrellisTextTo3DPipeline


def _get_hf_models_dir():
    """Get the Hugging Face models directory in the project root."""
    import os
    # Find project root by locating trellis/ directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # current_dir is trellis/pipelines/, go up to trellis/, then to project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    models_dir = os.path.join(project_root, "hf_models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        models_dir = _get_hf_models_dir()
        config_file = hf_hub_download(path, "pipeline.json", cache_dir=models_dir)

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
