import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

# Repo-relative defaults so the same checkout runs on any server.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HF_MODELS = _REPO_ROOT / 'hf_models'
_DEFAULT_BASE_OUTPUT = os.environ.get(
    'TRELLIS_BASE_OUTPUT', str(_REPO_ROOT / 't2o_results'))

from bilingual import pick_lang
from trellis_inference_core import TrellisInferenceCore
try:
    # v2 image-to-3D backend (TRELLIS.2 + FLUX). Import-guarded so the v1 path
    # still works in envs where trellis2/o_voxel aren't installed.
    from trellis2_inference_core import Trellis2InferenceCore
except Exception:
    Trellis2InferenceCore = None


def _canonical_object_path(object_path: str) -> str:
    parts = [p for p in object_path.replace('\\', '/').split('/') if p]
    filtered = [p for p in parts if not (p.lower().startswith('shot_'))]
    return '/'.join(filtered) if filtered else object_path


def _log_skipped_items(stage: str, skipped: Dict[str, List[str]]) -> None:
    total = sum(len(paths) for paths in skipped.values())
    if not total:
        return
    labels = {
        "shot_override_meta_only": "shot override (scene canonical과 중복, geometry 상속)",
        "no_prompt": "TRELLIS 프롬프트 없음",
        "target_filter": "타겟 필터 불일치",
    }
    logging.info("ℹ️ %s 건너뜀: %d개", stage, total)
    for reason, paths in skipped.items():
        logging.info("  - %s: %d개", labels.get(reason, reason), len(paths))
        if reason == "shot_override_meta_only":
            continue
        for path in paths:
            logging.info("      · %s", path)


def _sanitize_name(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    # USD path가 들어오는 경우 마지막 컴포넌트만 사용
    name = value.split('/')[-1]
    if not name:
        name = value
    # 공백 및 특수문자 정리
    cleaned = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in name)
    cleaned = cleaned.strip('_-')
    return cleaned or None


def _extract_scene_shot(identifier: Optional[str]) -> Tuple[str, str]:
    if not identifier:
        return "scene_unknown", "shot_unknown"
    parts = [p for p in identifier.replace('\\', '/').split('/') if p]
    scene = next((p for p in parts if p.lower().startswith('scene_')), None)
    shot = next((p for p in parts if p.lower().startswith('shot_')), None)
    return scene or "scene_unknown", shot or "shot_unknown"


def _derive_scene_canonical_usd_path(item: Dict[str, Any], usd_root: Optional[Path]) -> Optional[str]:
    """JSON에 usd_file_path가 없을 때 object_path와 USD 루트로 scene canonical 경로를 유도합니다."""
    existing = item.get('usd_file_path')
    if existing:
        return existing
    if usd_root is None:
        return None

    object_name = item.get('object_name') or item.get('actor_name')
    object_path = item.get('object_path') or item.get('actor_path', '')
    if not object_name or not object_path:
        return None

    parts = [p for p in object_path.replace('\\', '/').split('/') if p]
    scene = next((p for p in parts if p.lower().startswith('scene_')), None)
    if not scene:
        return None

    candidate = usd_root / scene / "objects" / f"{object_name}.usda"
    return str(candidate.resolve())


def _build_default_config(seed: Any, formats: List[str],
                          backend: str = 'trellis2') -> Dict[str, Any]:
    """백엔드별 기본 생성 설정.

    v1 과 v2 는 샘플러 파라미터의 이름과 개수가 다르다. v1 설정을 v2 로 그대로
    넘기면 `SparseStructureFlowModel.forward() got an unexpected keyword
    argument 'cfg_strength'` 로 죽는다.

      v1: slat_sampler_params            / cfg_strength
      v2: shape_slat_sampler_params
          + tex_slat_sampler_params      / guidance_strength

    v2 는 샘플러 파라미터를 비워 모델 기본값을 쓴다 — E2E 로 검증된 유일한 구성이
    그것이다(previz_pipeline/e2e_test.py). v1 의 cfg_strength=7.5 를 v2 의
    guidance_strength(기본 3.0)로 옮겨 적을 근거가 없어 옮기지 않았다.
    튜닝이 필요하면 v2 키 이름으로 명시할 것.
    """
    generation: Dict[str, Any] = {'seed': seed}
    if backend != 'trellis2':
        generation['sparse_structure_sampler_params'] = {
            'steps': 12, 'cfg_strength': 7.5,
        }
        generation['slat_sampler_params'] = {
            'steps': 12, 'cfg_strength': 7.5,
        }

    postprocessing: Dict[str, Any] = {'texture_size': 1024}
    if backend != 'trellis2':
        # v1 은 비율(0.95), v2 는 목표 face 수(simplify_target)라 의미가 다르다.
        postprocessing['simplify'] = 0.95

    return {
        'generation': generation,
        'output': {'formats': formats},
        'postprocessing': postprocessing,
    }


def load_augmented_records(
    json_path: Path,
    prefer_t2i_prompt: bool,
    allow_original_fallback: bool,
    target_filter: Optional[str],
    max_items: Optional[int],
    seed_value: Any,
    use_json_seed: bool,
    llm_label: Optional[str],
    usd_root: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with json_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    if isinstance(payload, dict) and 'results' in payload:
        entries = payload['results']
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError("JSON 구조를 파악할 수 없습니다. 'results' 리스트가 필요합니다.")

    records: List[Dict[str, Any]] = []
    skipped: Dict[str, List[str]] = defaultdict(list)
    for idx, item in enumerate(entries):
        item_path = item.get('object_path') or item.get('actor_path') or f"item_{idx}"

        if target_filter and item.get('target') != target_filter:
            skipped["target_filter"].append(item_path)
            continue

        # Scene canonical만 TRELLIS 생성 대상 (스펙 2.1, #3)
        if item.get('object_scope') == 'shot' or item.get('_pipeline') == 'meta_only':
            skipped["shot_override_meta_only"].append(item_path)
            continue

        # 프롬프트 우선순위: t2i_prompt -> description_en -> description(en)
        prompt_text: Optional[str] = None
        if prefer_t2i_prompt:
            prompt_text = item.get('t2i_prompt')
        if not prompt_text:
            prompt_text = item.get('description_en')
        if not prompt_text and allow_original_fallback:
            prompt_text = pick_lang(item.get('description'), 'en')
        if not prompt_text:
            logging.warning("⚠️ 프롬프트가 없어 건너뜁니다: %s", item_path)
            skipped["no_prompt"].append(item_path)
            continue

        name_candidates = [
            item.get('object_name'),
            item.get('actor_name'),
            _sanitize_name(item.get('object_path')),
            _sanitize_name(item.get('actor_path'))
        ]
        object_name = next((n for n in name_candidates if n), f"usd_item_{idx:03d}")

        record_seed = item.get('seed') if use_json_seed and item.get('seed') is not None else seed_value
        label = llm_label or item.get('llm_model') or "usd_aug"
        file_identifier = item.get('object_path') or item.get('actor_path')
        scene, shot = _extract_scene_shot(file_identifier)
        target_type = item.get('target') or ('object' if item.get('object_path') else 'actor')
        category = item.get('category') or target_type
        name_en = (
            item.get('name_en')
            or pick_lang(item.get('name'), 'en')
            or ''
        )
        target_name = object_name
        usd_file_path = _derive_scene_canonical_usd_path(item, usd_root)

        records.append({
            'prompt': prompt_text,
            'object_name': object_name,
            'seed': record_seed,
            'llm_model': label,
            'file_identifier': file_identifier,
            'object_path': file_identifier,
            'scene': scene,
            'shot': shot,
            'target_name': target_name,
            'target_type': target_type,
            'category': category,
            'name_en': name_en,
            't2i_prompt': item.get('t2i_prompt'),
            'description_en': item.get('description_en'),
            'run_id': run_id,
            'usd_file_path': usd_file_path,
            'usd_relative_path': item.get('usd_relative_path')
        })

        if max_items and len(records) >= max_items:
            break

    _log_skipped_items("Stage 2 (TRELLIS)", skipped)

    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="USD 증강 JSON을 TRELLIS 3D 생성 파이프라인에 투입하는 도구"
    )
    parser.add_argument('--json', required=True, help='usd_parse_and_augment.py에서 생성된 JSON 경로')
    parser.add_argument('--model_path', default='microsoft/TRELLIS-text-xlarge', help='TRELLIS 모델 경로 혹은 HF 모델명')
    parser.add_argument('--config', help='YAML 설정 경로 (미지정 시 기본 설정 사용)')
    parser.add_argument('--output', default='./outputs', help='이번 실행 출력 디렉토리')
    parser.add_argument('--run_dir', help='run 출력 디렉토리 (지정 시 output_base로 직접 사용)')
    parser.add_argument('--run_id', help='run ID (generation.json / GLB meta에 기록)')
    parser.add_argument('--base_output', default=_DEFAULT_BASE_OUTPUT, help='TrellisInferenceCore 기본 출력 베이스 경로')
    parser.add_argument('--usd_root', help='USD 프로젝트 루트 (usd_file_path 미지정 시 scene canonical 경로 유도용)')
    parser.add_argument('--target', choices=['object', 'actor', 'all'], default='object', help='JSON에서 추출할 타겟 유형 (기본: object)')
    parser.add_argument('--max_items', type=int, help='처리할 최대 항목 수')
    parser.add_argument('--prefer_original', action='store_true', help='t2i_prompt보다 원본 prompt를 우선 사용')
    parser.add_argument('--allow_original_fallback', action='store_true', help='t2i_prompt가 없을 때 원본 prompt 사용 허용')
    parser.add_argument('--seed', default='random', help='기본 시드값 (random 또는 정수)')
    parser.add_argument('--seed_from_json', action='store_true', help='JSON 내 seed가 있으면 사용')
    parser.add_argument('--llm_label', default='usd_aug', help='출력 구조에 표시할 LLM 라벨')
    parser.add_argument('--formats', nargs='+', default=['glb', 'ply', 'mp4', 'jpg'], help='저장할 출력 포맷')
    parser.add_argument('--simplify', type=float, default=0.95, help='GLB 단순화 비율 (v1 전용)')
    parser.add_argument('--texture_size', type=int, default=1024, help='텍스처 해상도')
    parser.add_argument('--backend', choices=['trellis', 'trellis2'], default='trellis2',
                        help='3D 생성 백엔드 (trellis=v1 text-to-3D, trellis2=v2 image-to-3D + FLUX)')
    parser.add_argument('--t2i_model_path', default=str(_HF_MODELS / 'FLUX.1-schnell'),
                        help='trellis2 백엔드의 Text→Image(FLUX) 모델 경로')
    parser.add_argument('--pipeline_type', default=None, help='TRELLIS.2 pipeline_type (기본: 모델 default)')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    json_path = Path(args.json).expanduser().resolve()
    if not json_path.exists():
        logging.error("❌ JSON 파일을 찾을 수 없습니다: %s", json_path)
        return 1

    target_filter = None if args.target == 'all' else args.target
    usd_root = Path(args.usd_root).expanduser().resolve() if args.usd_root else None

    if args.backend == 'trellis2':
        if Trellis2InferenceCore is None:
            logging.error("❌ Trellis2InferenceCore를 불러올 수 없습니다 (trellis2 env에서 실행하세요).")
            return 1
        model_path = args.model_path
        if model_path == 'microsoft/TRELLIS-text-xlarge':  # v1 기본값 → v2 로컬 모델
            model_path = str(_HF_MODELS / 'TRELLIS.2-4B')
        manager = Trellis2InferenceCore(
            model_path=model_path,
            base_output_dir=args.base_output,
            t2i_model_path=args.t2i_model_path,
            pipeline_type=args.pipeline_type,
        )
    else:
        manager = TrellisInferenceCore(model_path=args.model_path, base_output_dir=args.base_output)
    if args.run_id:
        manager.run_id = args.run_id

    output_dir = args.run_dir or args.output

    if args.config:
        config = manager.load_yaml_config(args.config)
        if not config:
            logging.error("❌ YAML 설정을 불러오지 못했습니다.")
            return 1
    else:
        config = _build_default_config(args.seed, args.formats, args.backend)
        if args.backend != 'trellis2':
            # v2 는 이 키를 읽지 않는다(simplify_target 을 쓴다). 넣어두면 적용되는
            # 것처럼 보여 오해를 부른다.
            config['postprocessing']['simplify'] = args.simplify
        config['postprocessing']['texture_size'] = args.texture_size

    try:
        records = load_augmented_records(
            json_path=json_path,
            prefer_t2i_prompt=not args.prefer_original,
            allow_original_fallback=True if args.allow_original_fallback or args.prefer_original else False,
            target_filter=target_filter,
            max_items=args.max_items,
            seed_value=args.seed,
            use_json_seed=args.seed_from_json,
            llm_label=args.llm_label,
            usd_root=usd_root,
            run_id=args.run_id,
        )
    except Exception as exc:
        logging.error("❌ JSON 로딩 실패: %s", exc)
        return 1

    if not records:
        logging.error("❌ 처리할 레코드가 없습니다. 필터 조건을 확인하세요.")
        return 1

    logging.info("📊 총 %d개 레코드를 TRELLIS 파이프라인에 전달합니다.", len(records))

    try:
        manager.load_pipeline()
    except Exception as exc:
        logging.error("❌ TRELLIS 파이프라인 로딩 실패: %s", exc)
        return 1

    try:
        manager.process_batch_from_records(records, config, output_dir)
    except Exception as exc:
        logging.error("❌ 배치 처리 중 오류: %s", exc)
        return 1

    logging.info("🎉 모든 처리가 완료되었습니다.")
    logging.info("📁 결과 경로: %s", manager.output_base)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

