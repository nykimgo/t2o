import importlib

__attributes = {
    'SparseStructureEncoder': 'sparse_structure_vae',
    'SparseStructureDecoder': 'sparse_structure_vae',
    
    'SparseStructureFlowModel': 'sparse_structure_flow',
    
    'SLatEncoder': 'structured_latent_vae',
    'SLatGaussianDecoder': 'structured_latent_vae',
    'SLatRadianceFieldDecoder': 'structured_latent_vae',
    'SLatMeshDecoder': 'structured_latent_vae',
    'ElasticSLatEncoder': 'structured_latent_vae',
    'ElasticSLatGaussianDecoder': 'structured_latent_vae',
    'ElasticSLatRadianceFieldDecoder': 'structured_latent_vae',
    'ElasticSLatMeshDecoder': 'structured_latent_vae',
    
    'SLatFlowModel': 'structured_latent_flow',
    'ElasticSLatFlowModel': 'structured_latent_flow',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


def _get_hf_models_dir():
    """Get the Hugging Face models directory in the project root."""
    import os
    # Find project root by locating trellis/ directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # current_dir is trellis/models/, go up to trellis/, then to project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    models_dir = os.path.join(project_root, "hf_models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def from_pretrained(path: str, **kwargs):
    """
    Load a model from a pretrained checkpoint.

    Args:
        path: The path to the checkpoint. Can be either local path or a Hugging Face model name.
              NOTE: config file and model file should take the name f'{path}.json' and f'{path}.safetensors' respectively.
        **kwargs: Additional arguments for the model constructor.
    """
    import os
    import json
    from safetensors.torch import load_file
    is_local = os.path.exists(f"{path}.json") and os.path.exists(f"{path}.safetensors")

    if is_local:
        config_file = f"{path}.json"
        model_file = f"{path}.safetensors"
    else:
        from huggingface_hub import hf_hub_download
        import logging
        models_dir = _get_hf_models_dir()
        path_parts = path.split('/')
        repo_id = f'{path_parts[0]}/{path_parts[1]}'
        model_name = '/'.join(path_parts[2:])
        
        # 디버깅 로그
        logging.info(f"📥 Loading model: {path}")
        
        if model_name:
            # 먼저 로컬 캐시에서만 찾기 시도 (네트워크 접근 없음)
            try:
                config_file = hf_hub_download(repo_id, f"{model_name}.json", cache_dir=models_dir, local_files_only=True)
                model_file = hf_hub_download(repo_id, f"{model_name}.safetensors", cache_dir=models_dir, local_files_only=True)
            except (FileNotFoundError, OSError):
                # 로컬에 없으면 네트워크에서 다운로드
                logging.info(f"   📡 로컬 캐시에 없어서 네트워크에서 다운로드합니다")
                config_file = hf_hub_download(repo_id, f"{model_name}.json", cache_dir=models_dir, force_download=False)
                model_file = hf_hub_download(repo_id, f"{model_name}.safetensors", cache_dir=models_dir, force_download=False)
        else:
            # If no model_name, assume it's just repo_id/model_name format
            try:
                config_file = hf_hub_download(repo_id, "model.json", cache_dir=models_dir, local_files_only=True)
                model_file = hf_hub_download(repo_id, "model.safetensors", cache_dir=models_dir, local_files_only=True)
            except (FileNotFoundError, OSError):
                # 로컬에 없으면 네트워크에서 다운로드
                logging.info(f"   📡 로컬 캐시에 없어서 네트워크에서 다운로드합니다")
                config_file = hf_hub_download(repo_id, "model.json", cache_dir=models_dir, force_download=False)
                model_file = hf_hub_download(repo_id, "model.safetensors", cache_dir=models_dir, force_download=False)


    with open(config_file, 'r') as f:
        config = json.load(f)
    model = __getattr__(config['name'])(**config['args'], **kwargs)
    model.load_state_dict(load_file(model_file))

    return model


# For Pylance
if __name__ == '__main__':
    from .sparse_structure_vae import (
        SparseStructureEncoder, 
        SparseStructureDecoder,
    )
    
    from .sparse_structure_flow import SparseStructureFlowModel
    
    from .structured_latent_vae import (
        SLatEncoder,
        SLatGaussianDecoder,
        SLatRadianceFieldDecoder,
        SLatMeshDecoder,
        ElasticSLatEncoder,
        ElasticSLatGaussianDecoder,
        ElasticSLatRadianceFieldDecoder,
        ElasticSLatMeshDecoder,
    )
    
    from .structured_latent_flow import (
        SLatFlowModel,
        ElasticSLatFlowModel,
    )
