from typing import *
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
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

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            import logging
            models_dir = Pipeline._get_hf_models_dir()
            
            # 디버깅 로그
            import logging
            logging.info(f"📥 Downloading pipeline: {path}")
            logging.info(f"   cache_dir: {models_dir}")
            logging.info(f"   cache_dir exists: {os.path.exists(models_dir)}")
            
            # 먼저 로컬 캐시에서만 찾기 시도 (네트워크 접근 없음)
            try:
                config_file = hf_hub_download(path, "pipeline.json", cache_dir=models_dir, local_files_only=True)
            except (FileNotFoundError, OSError):
                # 로컬에 없으면 네트워크에서 다운로드
                logging.info(f"   📡 로컬 캐시에 없어서 네트워크에서 다운로드합니다")
                config_file = hf_hub_download(path, "pipeline.json", cache_dir=models_dir)
            
            # 실제 다운로드된 경로 로깅
            logging.info(f"   ✅ pipeline.json downloaded to: {config_file}")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            # ../TRELLIS-image-large 같은 상대 경로를 미리 절대 경로로 변환
            if str(v).startswith("../"):
                # ../TRELLIS-image-large -> JeffreyXiang/TRELLIS-image-large
                model_name = str(v)[3:]  # "../" 제거
                if "TRELLIS-image-large" in model_name:
                    model_path = f"JeffreyXiang/{model_name}"
                else:
                    # 다른 ../ 경로는 path의 조직명 사용
                    org_name = path.split('/')[0] if '/' in path else "microsoft"
                    model_path = f"{org_name}/{model_name}"
            else:
                # 상대 경로가 아니면 path와 결합
                model_path = f"{path}/{v}"

            _models[k] = models.from_pretrained(model_path)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
