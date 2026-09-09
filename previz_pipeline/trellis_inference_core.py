"""Record-batch 3D generation core — 경로/이름/매니페스트/CSV 공통 헬퍼.

원래 TRELLIS v1(text-to-3D)의 실행 코어였으나, v1 백엔드는 제거되었다.
지금은 Trellis2InferenceCore(trellis2_inference_core.py)가 상속해
배치 순회·출력 구조·기록(generation.json/CSV/GLB meta)을 재사용하는 베이스다.
load_pipeline()/_generate_single() 은 서브클래스가 구현한다.
"""
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from bilingual import pick_lang


class TrellisInferenceCore:
    """TRELLIS model-based 3D generation core functionality for record-based processing"""
    
    def __init__(self, model_path: str = "microsoft/TRELLIS.2-4B",
                 base_output_dir: str = os.environ.get(
                     "TRELLIS_BASE_OUTPUT",
                     str(Path(__file__).resolve().parents[1] / "t2o_results"))):
        """
        Args:
            model_path: TRELLIS model path (local path or HuggingFace model name)
            base_output_dir: Base directory for output files
        """
        self.model_path = model_path
        self.base_output_dir = Path(base_output_dir)
        self.pipeline = None
        self.results_data: List[Dict] = []
        self.object_name_counter: Dict[str, int] = {}
        self.run_id: Optional[str] = None
        
        # 모델명 추출 (경로에서 마지막 부분)
        self.model_name = self._extract_model_name(model_path)
        
        # 현재 날짜
        self.current_date = datetime.now().strftime('%Y%m%d')
        
        # 출력 디렉토리 구조
        self.output_base = self.base_output_dir / "output" / self.model_name / self.current_date
        
        # 로깅 설정
        self._setup_logging()
    
    def _extract_model_name(self, model_path: str) -> str:
        """모델 경로에서 모델명 추출"""
        if '/' in model_path:
            # HuggingFace 형태 (microsoft/TRELLIS-text-xlarge) 또는 경로
            model_name = model_path.split('/')[-1]
        else:
            model_name = model_path
        
        # microsoft/ 접두사 제거
        if model_name.startswith('microsoft-'):
            model_name = model_name[10:]
        
        return model_name
    
    def _setup_logging(self):
        """로깅 설정"""
        # logs 폴더 생성
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        log_file_path = logs_dir / 'trellis_inference.log'
        
        # basicConfig가 이미 호출되었을 수 있으므로, FileHandler만 추가
        root_logger = logging.getLogger()
        
        # 이미 trellis_inference.log FileHandler가 있는지 확인
        log_file_exists = any(
            isinstance(h, logging.FileHandler) and str(log_file_path) in h.baseFilename
            for h in root_logger.handlers
        )
        
        if not log_file_exists:
            # FileHandler 추가
            file_handler = logging.FileHandler(str(log_file_path))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(file_handler)
        
        # basicConfig가 아직 호출되지 않았다면 호출
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(str(log_file_path))
                ]
            )
    
    def load_yaml_config(self, yaml_filename: str) -> Optional[Dict]:
        """Load YAML configuration file"""
        try:
            import yaml
        except ImportError:
            logging.error("❌ PyYAML not installed. Install with: pip install PyYAML")
            return None
            
        try:
            yaml_path = Path(yaml_filename)
            
            if not yaml_path.exists():
                logging.error(f"❌ YAML file not found: {yaml_filename}")
                return None
            
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logging.info(f"📄 Loaded YAML config: {yaml_filename}")
            return config
            
        except Exception as e:
            logging.error(f"❌ Failed to load YAML config {yaml_filename}: {e}")
            return None
    
    def load_pipeline(self) -> None:
        """서브클래스(Trellis2InferenceCore)가 구현한다."""
        raise NotImplementedError("load_pipeline() 은 백엔드 서브클래스가 구현한다")

    def generate_unique_name(self, base_prompt: str) -> str:
        """Generate unique object name based on prompt"""
        print(f'base_prompt: {base_prompt}')
        words = base_prompt.lower().split()
        clean_words = [word.strip('.,!?;:"()[]{}') for word in words if word.strip('.,!?;:"()[]{}')]
        english_words = [word for word in clean_words if word.isascii() and word.isalpha()]
        
        if not english_words:
            base_name = "object"
        else:
            base_name = "_".join(english_words[:3])
        
        if base_name in self.object_name_counter:
            self.object_name_counter[base_name] += 1
            return f"{base_name}_{self.object_name_counter[base_name]:02d}"
        else:
            self.object_name_counter[base_name] = 1
            return f"{base_name}_01"
    
    def _sanitize_path_segment(self, name: Optional[str], fallback: str) -> str:
        """Sanitize strings for directory/file names."""
        if not name:
            name = fallback
        safe = ''.join(ch if (ch.isalnum() or ch in ('-', '_')) else '_' for ch in name)
        safe = safe.strip('_-')
        return safe or fallback

    def _is_placeholder_shot(self, shot_name: str) -> bool:
        """shot 정보가 없는 scene canonical 항목인지 판별합니다."""
        return shot_name in ('shot_unknown', 'unknown', '')

    def _resolve_output_base(self, output_dir: str) -> Path:
        """출력 베이스 경로를 결정합니다. run 디렉토리면 그대로 사용합니다."""
        if output_dir == "./outputs":
            return self.base_output_dir / "output" / self.model_name / self.current_date
        path = Path(output_dir)
        if path.name.startswith("run_") or (path / "run_manifest.json").exists():
            return path
        return path / self.model_name / self.current_date

    def _preview_output_path(self, scene_name: str, shot_name: str, target_dir_name: str) -> Path:
        """미리보기(ply/mp4/jpg) 저장 디렉토리를 구성합니다."""
        base = self.output_base / "previews"
        if self._is_placeholder_shot(shot_name):
            return base / scene_name / target_dir_name
        return base / scene_name / shot_name / target_dir_name

    def _write_generation_json(
        self,
        preview_dir: Path,
        *,
        prompt: str,
        context: Dict,
        seed: int,
        glb_path: Optional[str],
        preview_files: List[str],
    ) -> None:
        """객체별 생성 provenance를 preview 폴더에 저장합니다."""
        generation = {
            "run_id": self.run_id,
            "object_path": context.get("object_path") or context.get("file_identifier"),
            "object_name": context.get("object_name"),
            "prompt_used": prompt,
            "t2i_prompt": context.get("t2i_prompt"),
            "description_en": context.get("description_en"),
            "seed": seed,
            "glb_path": glb_path,
            "preview_files": preview_files,
            "timestamp": datetime.now().isoformat(),
        }
        out_path = preview_dir / "generation.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(generation, f, ensure_ascii=False, indent=2)

    def _write_glb_meta(self, glb_path: Path, meta: Dict) -> None:
        """GLB 옆에 .meta.json sidecar를 기록합니다."""
        meta_path = glb_path.with_suffix(glb_path.suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _build_file_prefix(self, scene_name: str, shot_name: str, label_name: str, seed: int) -> str:
        """출력 파일명 접두사를 구성합니다. shot 미지정 시 shot 세그먼트를 생략합니다."""
        if self._is_placeholder_shot(shot_name):
            return f"{scene_name}_{label_name}_{seed}"
        return f"{scene_name}_{shot_name}_{label_name}_{seed}"

    def _resolve_asset_label_name(self, item: Dict, fallback: str = 'item') -> str:
        """출력 파일명에 사용할 라벨(name_en 우선)을 결정합니다."""
        for key in ('name_en', 'category', 'target_type', 'object_name'):
            value = item.get(key)
            if value and str(value).strip():
                return self._sanitize_path_segment(str(value).strip(), fallback)
        name = item.get('name')
        if name:
            picked = pick_lang(name, 'en')
            if picked.strip():
                return self._sanitize_path_segment(picked.strip(), fallback)
        return fallback

    def process_batch_from_records(self, records: List[Dict], config: Dict, output_dir: str = "./outputs") -> None:
        """
        Process prompts from in-memory records (e.g., JSON-derived data).

        Args:
            records: List of dicts containing at least a 'prompt'. Optional keys:
                - object_name
                - seed
                - llm_model
                - file_identifier
                - scene
                - shot
                - target_name
                - target_type
                - category
                - name_en
            config: Generation configuration dict (same structure as YAML config).
            output_dir: Target directory for outputs in this run.
        """
        if not records:
            logging.warning("❌ No records provided for processing")
            return

        # file_identifier는 내부에서 사용하지 않으므로 안전하게 채워줌
        normalized_records = []
        for record in records:
            if 'prompt' not in record or not str(record['prompt']).strip():
                logging.warning("⚠️ Skipping record without a prompt")
                continue
            normalized_records.append({
                'prompt': record.get('prompt'),
                'object_name': record.get('object_name'),
                'seed': record.get('seed', config.get('generation', {}).get('seed', "random")),
                'llm_model': record.get('llm_model'),
                'file_identifier': record.get('file_identifier'),
                'scene': record.get('scene'),
                'shot': record.get('shot'),
                'target_name': record.get('target_name') or record.get('object_name'),
                'target_type': record.get('target_type'),
                'category': record.get('category'),
                'name_en': record.get('name_en'),
                'usd_file_path': record.get('usd_file_path'),
                'run_id': record.get('run_id'),
                'object_path': record.get('object_path') or record.get('file_identifier'),
                't2i_prompt': record.get('t2i_prompt'),
                'description_en': record.get('description_en'),
            })

        if not normalized_records:
            logging.warning("❌ No valid records remaining after normalization")
            return

        self._process_file_batch(normalized_records, config, output_dir)

    def _process_file_batch(self, file_data: List[Dict], config: Dict, output_dir: str) -> None:
        """Process a batch of file data with individual settings"""
        self.output_base = self._resolve_output_base(output_dir)
        if file_data and file_data[0].get('run_id'):
            self.run_id = file_data[0]['run_id']
        elif not self.run_id:
            self.run_id = os.environ.get('RUN_ID')
        
        # 출력 디렉토리 생성
        self.output_base.mkdir(parents=True, exist_ok=True)

        generation_config = config.get('generation', {})
        output_config = config.get('output', {})
        postprocessing_config = config.get('postprocessing', {})
        
        formats = output_config.get('formats', ['glb'])
        
        for i, item in enumerate(file_data, 1):
            prompt = item['prompt']
            object_name = item['object_name']
            predefined_name = object_name

            seed = item['seed']
            llm_model = item.get('llm_model')
            scene_name = self._sanitize_path_segment(item.get('scene'), 'scene_unknown')
            shot_name = self._sanitize_path_segment(item.get('shot'), 'shot_unknown')
            target_dir_name = self._sanitize_path_segment(item.get('target_name') or predefined_name, 'item_unknown')
            target_type = item.get('target_type') or 'object'
            
            logging.info(f"\n🎯 [{i}/{len(file_data)}] Processing: '{prompt}'")
            if predefined_name:
                logging.info(f"📋 Object name: {predefined_name}")
            if llm_model:
                logging.info(f"🤖 LLM model: {llm_model}")
            logging.info(f"🎲 Seed: {seed}")
            
            try:
                # 개별 생성 설정
                individual_config = generation_config.copy()
                individual_config['seed'] = seed
                
                result = self._generate_single(
                    prompt=prompt,
                    predefined_name=predefined_name,
                    config=individual_config,
                    formats=formats,
                    postprocessing_config=postprocessing_config,
                    llm_model=llm_model,
                    record_context={
                        'scene': scene_name,
                        'shot': shot_name,
                        'target_dir_name': target_dir_name,
                        'target_type': target_type,
                        'name_en': item.get('name_en'),
                        'category': item.get('category'),
                        'object_name': object_name,
                        'usd_file_path': item.get('usd_file_path'),
                        'object_path': item.get('object_path') or item.get('file_identifier'),
                        'file_identifier': item.get('file_identifier'),
                        't2i_prompt': item.get('t2i_prompt'),
                        'description_en': item.get('description_en'),
                        'run_id': item.get('run_id') or self.run_id,
                    }
                )
                
                self.results_data.append(result)
                logging.info(f"✅ Completed: {result['object_name']}")
                
            except Exception as e:
                error_result = {
                    'prompt': prompt,
                    'object_name': predefined_name or 'error',
                    'seed': seed,
                    'model_name': self.model_name,
                    'llm_model': llm_model,
                    'generation_time': 0.0,
                    'total_time': 0.0,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.results_data.append(error_result)
                logging.error(f"❌ Failed: {prompt} - {e}")
        
        # Save results to CSV
        self._save_results_to_csv()

        # 단계별 소요 시간 요약 (T2I / I2O 는 trellis2 백엔드에서만 분리 계측됨)
        _ok = [r for r in self.results_data if r.get('success')]
        if _ok:
            _sum = lambda k: sum(float(r.get(k) or 0.0) for r in _ok)
            logging.info("⏱️  ===== 2단계(생성) 소요 시간 =====")
            if any('t2i_time' in r for r in _ok):
                logging.info(f"⏱️    T2I 합계: {_sum('t2i_time'):.1f}s")
                logging.info(f"⏱️    I2O 합계: {_sum('i2o_time'):.1f}s")
            else:
                logging.info(f"⏱️    생성(T2I+I2O) 합계: {_sum('generation_time'):.1f}s")
            logging.info(f"⏱️    프리뷰 렌더 합계: {_sum('render_time'):.1f}s")
            logging.info(f"⏱️    저장(GLB/mp4/jpg) 합계: {_sum('save_time'):.1f}s")
            logging.info(f"⏱️    객체 {len(_ok)}개 합계: {_sum('total_time'):.1f}s "
                         f"(객체당 평균 {_sum('total_time') / len(_ok):.1f}s)")

    def _get_unique_filename(self, directory: Path, base_filename: str) -> str:
        """
        중복되는 파일명이 있으면 파일명 뒤에 001, 002, ... 형태로 숫자를 붙여서 유니크한 파일명 반환
        
        Args:
            directory: 파일이 저장될 디렉토리
            base_filename: 기본 파일명 (확장자 포함)
        
        Returns:
            유니크한 파일명 (확장자 포함)
        """
        filepath = directory / base_filename
        
        if not filepath.exists():
            return base_filename
        
        # 파일명과 확장자 분리
        name_part = base_filename.rsplit('.', 1)[0]  # 확장자 제거
        ext_part = '.' + base_filename.rsplit('.', 1)[1] if '.' in base_filename else ''  # 확장자
        
        # 중복되는 경우 숫자 접미사 추가
        counter = 1
        while True:
            new_filename = f"{name_part}_{counter:03d}{ext_part}"
            new_filepath = directory / new_filename
            
            if not new_filepath.exists():
                return new_filename
            
            counter += 1
            
            # 안전장치: 999개를 넘어가면 중단
            if counter > 999:
                # 타임스탬프를 추가하여 강제로 유니크하게 만듦
                timestamp = datetime.now().strftime('%H%M%S')
                new_filename = f"{name_part}_{timestamp}{ext_part}"
                return new_filename

    def _generate_single(self, prompt: str, predefined_name: Optional[str], config: Dict, formats: List[str], postprocessing_config: Dict, llm_model: Optional[str] = None, record_context: Optional[Dict[str, str]] = None) -> Dict:
        """서브클래스(Trellis2InferenceCore)가 구현한다."""
        raise NotImplementedError("_generate_single() 은 백엔드 서브클래스가 구현한다")

    def _save_results_to_csv(self) -> None:
        """Save results to CSV file with specified naming format"""
        if not self.results_data:
            return
        
        try:
            import pandas as pd
        except ImportError:
            logging.warning("⚠️ pandas not installed, skipping CSV save")
            return
        
        # run 디렉토리면 results.csv, 아니면 기존 타임스탬프 파일명
        if self.run_id or (self.output_base / "run_manifest.json").exists():
            csv_path = self.output_base / "results.csv"
        else:
            current_time = datetime.now().strftime('%H%M%S')
            csv_filename = f"results_{self.model_name}_{self.current_date}_{current_time}.csv"
            csv_path = self.output_base / csv_filename
        
        results_df = pd.DataFrame(self.results_data)
        results_df.to_csv(csv_path, index=False)
        
        logging.info(f"📊 Results saved to: {csv_path}")
        
        # Print summary
        successful = sum(1 for r in self.results_data if r.get('success', False))
        total = len(self.results_data)
        avg_time = sum(r.get('generation_time', 0) for r in self.results_data if r.get('success', False)) / max(successful, 1)
        
        logging.info(f"✅ Summary: {successful}/{total} successful, avg time: {avg_time:.1f}s")
        logging.info(f"📁 All files saved in: {self.output_base}")

