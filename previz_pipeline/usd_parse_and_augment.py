"""
USD 파일 파싱 및 프롬프트 증강 통합 스크립트

USD 파일에서 필요한 정보를 추출하고, 중간 저장 없이 바로 LLM으로 증강한 후 최종 JSON을 저장합니다.
"""

import json
import os
import re
import sys
import time
from typing import List, Optional, Dict, Any, Tuple

from bilingual import is_blank, pick_lang

# USD 파서 import
try:
    from usd_parser import parse_usd_file, ENABLE_ACTOR_PARSING
    USD_PARSER_AVAILABLE = True
except ImportError:
    ENABLE_ACTOR_PARSING = False
    USD_PARSER_AVAILABLE = False
    print("⚠️ usd_parser 모듈을 찾을 수 없습니다.")

# Ollama import
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

if not OLLAMA_AVAILABLE and not REQUESTS_AVAILABLE:
    print("⚠️ Ollama를 사용하려면 'ollama' 또는 'requests' 패키지가 필요합니다.")


def load_system_prompt(prompt_file: str = "/path/to/system_prompt") -> str:
    """
    시스템 프롬프트 파일을 로드합니다.
    
    Args:
        prompt_file: 시스템 프롬프트 파일 경로
        
    Returns:
        시스템 프롬프트 문자열
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, prompt_file)
    
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def _determine_response_key(obj: Dict[str, str]) -> Optional[str]:
    """
    LLM 응답 매핑에 사용할 고유 키를 결정합니다.
    """
    target = obj.get("target", "")
    candidate_fields = []
    
    if target == "object":
        candidate_fields = ["object_path", "object_name", "object_id"]
    elif ENABLE_ACTOR_PARSING and target == "actor":
        candidate_fields = ["actor_path", "actor_name", "actor_id"]
    else:
        candidate_fields = [
            "object_path",
            "actor_path",
            "object_name",
            "actor_name",
            "object_id",
            "actor_id",
        ]
    
    for field in candidate_fields:
        value = obj.get(field)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    
    return None


def _build_path_to_obj_map(objects: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    """
    객체들을 빠르게 참조하기 위한 매핑을 생성합니다.
    """
    mapping = {}
    for obj in objects:
        key = _determine_response_key(obj)
        if key and key not in mapping:
            mapping[key] = obj
    return mapping


def _canonical_object_path(object_path: str) -> str:
    """shot override 경로를 scene canonical 경로로 변환합니다."""
    parts = [p for p in object_path.replace("\\", "/").split("/") if p]
    filtered = [p for p in parts if not re.match(r"^shot_\w+$", p)]
    return "/".join(filtered) if filtered else object_path


def _log_skip_summary(stage: str, skipped: Dict[str, List[str]]) -> None:
    """건너뛴 항목을 사유별로 출력합니다."""
    total = sum(len(paths) for paths in skipped.values())
    if not total:
        return
    print(f"[INFO] {stage} 건너뜀: {total}개")
    reason_labels = {
        "shot_override_meta_only": "shot override (scene canonical과 중복, geometry 상속)",
        "no_prompt": "TRELLIS 프롬프트 없음",
        "target_filter": "타겟 필터 불일치",
        "no_path": "object_path/actor_path 없음",
        "no_object_name": "object_name 추출 실패",
        "no_usd": "원본 USD 파일 없음",
        "no_scene_canonical": "shot 경로이나 scene canonical USD 없음",
        "no_glb_dir": "GLB 디렉토리 없음",
        "no_glb": "GLB 파일 없음 (3D 생성 실패 또는 미실행)",
    }
    for reason, paths in skipped.items():
        label = reason_labels.get(reason, reason)
        print(f"  - {label}: {len(paths)}개")
        for path in paths:
            print(f"      · {path}")


def _copy_bilingual_fields(source: Dict[str, Any], target: Dict[str, Any], field_name: str) -> None:
    """nested dict와 _ko/_en 평탄 필드를 target에 복사합니다."""
    if field_name in source:
        target[field_name] = source[field_name]
    for suffix in ("_ko", "_en"):
        flat_key = f"{field_name}{suffix}"
        if flat_key in source:
            target[flat_key] = source[flat_key]


def _has_description(obj: Dict[str, Any]) -> bool:
    return (
        not is_blank(obj.get("description"))
        or bool(str(obj.get("description_en", "")).strip())
        or bool(str(obj.get("description_ko", "")).strip())
    )


def _get_korean_description(obj: Dict[str, Any]) -> str:
    return str(obj.get("description_ko") or pick_lang(obj.get("description"), "ko")).strip()


def _get_english_description(obj: Dict[str, Any]) -> str:
    return str(obj.get("description_en") or pick_lang(obj.get("description"), "en")).strip()


def _get_korean_name(obj: Dict[str, Any]) -> str:
    return str(obj.get("name_ko") or pick_lang(obj.get("name"), "ko")).strip()


def _get_english_name(obj: Dict[str, str]) -> str:
    """필터링/TRELLIS 입력용 영어 name (번역본 우선, 없으면 USD en)."""
    translated = str(obj.get("translated_name", "")).strip()
    if translated:
        return translated
    return str(obj.get("name_en") or pick_lang(obj.get("name"), "en")).strip()


def _parse_translation_entry(entry: Any) -> Tuple[str, str]:
    """
    1단계 번역 LLM 응답 항목에서 translated_description, translated_name을 추출합니다.
    """
    if isinstance(entry, dict):
        return (
            str(entry.get("translated_description", "") or "").strip(),
            str(entry.get("translated_name", "") or "").strip(),
        )
    if isinstance(entry, str):
        return entry.strip(), ""
    return "", ""


def _convert_llm_result_keys(
    result: Dict[str, Any],
    path_to_obj: Dict[str, Dict[str, str]],
    stage_label: str
) -> Dict[str, Any]:
    """
    LLM 응답 키를 실제 경로 키로 변환합니다.
    """
    converted_result = {}
    unmapped_keys = []
    
    for key, value in result.items():
        if key in path_to_obj:
            converted_result[key] = value
            continue
        
        found = False
        for path_key, obj in path_to_obj.items():
            if (
                key == obj.get("description_ko", "") or
                key == _get_korean_description(obj) or
                key == obj.get("description", "") or
                key == obj.get("object_id", "") or
                key == obj.get("actor_id", "") or
                key == obj.get("object_name", "") or
                key == obj.get("actor_name", "") or
                key == obj.get("name", "") or
                key == str(obj.get("object_id", "")) or
                key == str(obj.get("actor_id", ""))
            ):
                converted_result[path_key] = value
                found = True
                print(f"[DEBUG][{stage_label}] 키 매핑: '{key}' -> '{path_key}'")
                break
        
        if not found:
            unmapped_keys.append(key)
            print(f"⚠️[{stage_label}] 키 '{key}'를 매핑할 수 없습니다.")
    
    if unmapped_keys:
        print(f"[WARNING][{stage_label}] 매핑되지 않은 키 {len(unmapped_keys)}개: {unmapped_keys[:3]}...")
    
    return converted_result if converted_result else result


def _unload_ollama_model(model_name: str, base_url: Optional[str]) -> None:
    """
    Ollama 모델을 GPU 메모리에서 언로드합니다.
    
    Args:
        model_name: 언로드할 모델 이름
        base_url: Ollama 서버 URL
    """
    api_url = f"{base_url or 'http://localhost:11434'}/api/generate"
    
    if OLLAMA_AVAILABLE:
        try:
            client = ollama.Client(host=base_url) if base_url else ollama.Client()
            client.generate(
                model=model_name,
                prompt='',
                options={'keep_alive': 0}
            )
            print(f"[INFO] 모델 언로드 완료: {model_name}")
        except Exception as e:
            print(f"[WARNING] 모델 언로드 실패 ({model_name}): {e}")
    elif REQUESTS_AVAILABLE:
        import requests
        try:
            requests.post(
                api_url,
                json={
                    'model': model_name,
                    'prompt': '',
                    'keep_alive': 0
                },
                timeout=10
            )
            print(f"[INFO] 모델 언로드 완료: {model_name}")
        except requests.exceptions.RequestException as e:
            print(f"[WARNING] 모델 언로드 실패 ({model_name}): {e}")
    else:
        print(f"[WARNING] Ollama 클라이언트를 사용할 수 없어 모델 언로드를 건너뜁니다: {model_name}")


def _generate_with_ollama(
    prompt: str,
    model_name: str,
    base_url: Optional[str],
    num_predict: int,
    operation_label: str
) -> str:
    """
    Ollama 또는 requests를 사용해 공통 방식으로 LLM을 호출합니다.
    """
    if OLLAMA_AVAILABLE:
        try:
            client = ollama.Client(host=base_url) if base_url else ollama.Client()
            start_time = time.time()
            response = client.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'temperature': 0.7,
                    'num_predict': num_predict,
                }
            )
            elapsed_time = time.time() - start_time
            print(f"[시간 측정] {operation_label}: {elapsed_time:.2f}초")
            return response['response']
        except Exception as e:
            raise RuntimeError(f"{operation_label} Ollama API 호출 실패: {e}")
    
    if REQUESTS_AVAILABLE:
        import requests
        url = f"{base_url or 'http://localhost:11434'}/api/generate"
        try:
            start_time = time.time()
            response = requests.post(
                url,
                json={
                    'model': model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.7,
                        'num_predict': num_predict,
                    }
                },
                timeout=300
            )
            response.raise_for_status()
            elapsed_time = time.time() - start_time
            print(f"[시간 측정] {operation_label}: {elapsed_time:.2f}초")
            return response.json()['response']
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"{operation_label} Ollama HTTP API 호출 실패: {e}")
    
    raise ImportError(
        "Ollama를 사용하려면 'ollama' 패키지 또는 'requests' 패키지가 필요합니다.\n"
        "설치: pip install ollama 또는 pip install requests"
    )


def _parse_json_response(response_text: str) -> Dict[str, str]:
    """
    LLM 응답에서 JSON 객체를 추출합니다.
    
    Args:
        response_text: LLM이 반환한 텍스트
        
    Returns:
        {"object_path": "augmented", ...} 형식의 딕셔너리
    """
    response_text = response_text.strip()
    
    # 마크다운 코드 블록 제거
    if '```' in response_text:
        json_match = re.search(r'```(?:json)?\s*(\{.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1).strip()
        else:
            if response_text.startswith('```'):
                response_text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.MULTILINE)
                response_text = re.sub(r'\s*```\s*$', '', response_text, flags=re.MULTILINE)
    
    # 직접 JSON 객체인 경우
    if response_text.startswith('{'):
        # 중괄호 매칭으로 완전한 JSON 추출
        brace_count = 0
        json_end = -1
        for i, char in enumerate(response_text):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end > 0:
            try:
                json_str = response_text[:json_end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # 중괄호 매칭 실패 시 전체 시도
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
    
    # JSON 객체 찾기 (중괄호 매칭 사용)
    brace_start = response_text.find('{')
    if brace_start >= 0:
        brace_count = 0
        json_end = -1
        for i in range(brace_start, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end > brace_start:
            try:
                json_str = response_text[brace_start:json_end]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    # 마지막 시도: 정규식으로 찾기
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not parse JSON response: {response_text[:500]}...")


def translate_descriptions_with_ollama(
    parsed_objects: List[Dict[str, str]],
    model_name: str = "gemma3:4b",
    base_url: Optional[str] = None,
    prompt_file: str = "step1_translate_prompt.txt"
) -> Dict[str, Any]:
    """
    한글 description과 name을 영어로 번역합니다.
    """
    if not parsed_objects:
        return {}
    
    objects_with_description = [
        obj for obj in parsed_objects
        if _get_korean_description(obj)
    ]
    
    if not objects_with_description:
        print("[WARNING] 번역할 description이 없습니다.")
        return {}
    
    system_prompt = load_system_prompt(prompt_file)
    objects_for_llm = []
    for obj in objects_with_description:
        llm_obj = {
            "target": obj.get("target", ""),
            "category": obj.get("category", ""),
            "description": _get_korean_description(obj),
            "object_path": obj.get("object_path", ""),
            "actor_path": obj.get("actor_path", "")
        }
        name = _get_korean_name(obj)
        if name:
            llm_obj["name"] = name
        objects_for_llm.append(llm_obj)
    
    input_json = json.dumps(objects_for_llm, ensure_ascii=False, indent=2)
    translation_prompt = f"""{system_prompt}

## Input JSON:
{input_json}

## Instructions:
- Translate each "description" into natural English. Preserve every detail from the original.
- If "name" is present and non-empty, translate it into a concise English noun (e.g., "핸드폰" -> "smartphone", "차량" -> "car", "갈매기" -> "seagull").
- Use either "object_path" or "actor_path" (whichever is non-empty) as the key in your output JSON.
- Return ONLY a JSON object where keys are these paths and values are objects:
  {{"translated_description": "<English description>", "translated_name": "<English noun>"}}
- If "name" is empty or absent in the input item, set "translated_name" to "".
"""
    
    response_text = _generate_with_ollama(
        prompt=translation_prompt,
        model_name=model_name,
        base_url=base_url,
        num_predict=max(len(objects_for_llm) * 200, 1000),
        operation_label="1단계 번역"
    )
    
    print(f'[INFO][번역] RESPONSE TEXT: {response_text}\n\n')
    
    try:
        result = _parse_json_response(response_text)
        print(f"[DEBUG][번역] JSON 파싱 성공: {len(result)}개 항목")
    except Exception as e:
        print(f"[ERROR][번역] JSON 파싱 실패: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    path_to_obj = _build_path_to_obj_map(objects_with_description)
    converted_result = _convert_llm_result_keys(result, path_to_obj, stage_label="번역")
    print(f"[DEBUG][번역] 변환 완료: {len(converted_result)}개")
    
    return converted_result


def augment_batch_with_ollama(
    parsed_objects: List[Dict[str, str]],
    model_name: str = "gemma3:4b",
    system_prompt_file: str = "step2_filter_prompt.txt",
    base_url: Optional[str] = None,
    retry_count: int = 0
) -> Dict[str, str]:
    """
    번역된 description을 기반으로 프롬프트를 필터링합니다.
    """
    if not parsed_objects:
        return {}
    
    objects_with_description = [
        obj for obj in parsed_objects
        if str(obj.get("translated_description", "")).strip()
    ]
    
    if not objects_with_description:
        print("[WARNING] 필터링할 description이 없습니다.")
        return {}
    
    system_prompt = load_system_prompt(system_prompt_file)
    print(f"[INFO] 필터링 대상 객체 수: {len(objects_with_description)}")
    
    objects_for_llm = []
    for obj in objects_with_description:
        llm_obj = {
            "target": obj.get("target", ""),
            "category": obj.get("category", ""),
            "translated_description": obj.get("translated_description", ""),
            "object_path": obj.get("object_path", ""),
            "actor_path": obj.get("actor_path", "")
        }
        english_name = _get_english_name(obj)
        if english_name and obj.get("target") == "object":
            llm_obj["name"] = english_name
        objects_for_llm.append(llm_obj)
    
    input_json = json.dumps(objects_for_llm, ensure_ascii=False, indent=2)
    
    batch_prompt = f"""{system_prompt}

## Input Data:
{input_json}
"""
    
    response_text = _generate_with_ollama(
        prompt=batch_prompt,
        model_name=model_name,
        base_url=base_url,
        num_predict=max(len(objects_with_description) * 300, 1000),
        operation_label="2단계 필터링"
    )
    
    print(f'[INFO][필터링] RESPONSE TEXT: {response_text}\n\n')
    
    try:
        result = _parse_json_response(response_text)
        print(f"[DEBUG][필터링] JSON 파싱 성공: {len(result)}개 항목")
        print(f"[DEBUG][필터링] LLM 응답 키들: {list(result.keys())[:5]}...")
    except Exception as e:
        print(f"[ERROR][필터링] JSON 파싱 실패: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    path_to_obj = _build_path_to_obj_map(objects_with_description)
    print(f"[DEBUG][필터링] 기대하는 키들: {list(path_to_obj.keys())[:5]}...")
    
    result = _convert_llm_result_keys(result, path_to_obj, stage_label="필터링")
    print(f"[DEBUG][필터링] 변환 완료: 최종 결과 {len(result)}개 (입력: {len(objects_with_description)}개)")
    
    missing_paths = [path for path in path_to_obj.keys() if path not in result]
    if missing_paths and len(missing_paths) < len(objects_with_description) and retry_count < 1:
        print(f"[INFO][필터링] 누락된 항목 {len(missing_paths)}개 발견. 재시도 중... (재시도 횟수: {retry_count + 1})")
        missing_objects = [path_to_obj[path] for path in missing_paths]
        retry_result = augment_batch_with_ollama(
            missing_objects,
            model_name=model_name,
            system_prompt_file=system_prompt_file,
            base_url=base_url,
            retry_count=retry_count + 1
        )
        result.update(retry_result)
        print(f"[INFO][필터링] 재시도 완료: 추가로 {len(retry_result)}개 항목 획득")
    elif missing_paths:
        print(f"[WARNING][필터링] 누락된 항목 {len(missing_paths)}개가 여전히 있습니다: {missing_paths[:3]}...")
    
    return result


def _attach_usd_path_fields(source: Dict[str, str], target: Dict[str, Any]) -> None:
    """usd_file_path/usd_relative_path를 복사하고 assets_dir를 유도합니다."""
    usd_file_path = source.get("usd_file_path", "")
    if usd_file_path:
        target["usd_file_path"] = usd_file_path
        object_name = source.get("object_name") or source.get("actor_name", "")
        if object_name and source.get("target") == "object":
            target["assets_dir"] = os.path.normpath(
                os.path.join(os.path.dirname(usd_file_path), "assets", object_name)
            )
    usd_relative_path = source.get("usd_relative_path", "")
    if usd_relative_path:
        target["usd_relative_path"] = usd_relative_path


def extract_essential_info(parsed_objects: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    USD 파싱 결과에서 필수 정보만 추출합니다.
    
    Args:
        parsed_objects: USD 파싱 결과 리스트
        
    Returns:
        필수 정보만 포함된 리스트 (prim 정보, category, prompt)
    """
    essential_list = []
    
    for obj in parsed_objects:
        target = obj.get("target", "")
        # [OBJECT-ONLY] actor 항목 스킵
        if not ENABLE_ACTOR_PARSING:
            if target == "actor" or ("actor_name" in obj and "object_name" not in obj):
                continue
        essential = {}
        
        if target == "object":
            essential["target"] = "object"
            essential["object_path"] = obj.get("object_path", "")
            essential["object_name"] = obj.get("object_name", "")
            essential["object_id"] = obj.get("object_id", "")
            essential["object_scope"] = obj.get("object_scope", "scene")
            essential["category"] = obj.get("category", "")
            _copy_bilingual_fields(obj, essential, "name")
            _copy_bilingual_fields(obj, essential, "description")
        elif ENABLE_ACTOR_PARSING and target == "actor":
            # [OBJECT-ONLY DISABLED] actor의 경우
            essential["target"] = "actor"
            essential["actor_path"] = obj.get("actor_path", "")
            essential["actor_name"] = obj.get("actor_name", "")
            essential["actor_id"] = obj.get("actor_id", "")
            essential["category"] = ""  # actor는 category가 없을 수 있음
            # LLM이 기대하는 형식을 위해 description도 추가
            description = obj.get("description", "")
            essential["description"] = description
            essential["prompt"] = description  # 최종 JSON용
        else:
            # target이 없는 경우 둘 다 시도
            if "object_path" in obj or "object_name" in obj:
                essential["target"] = "object"
                essential["object_path"] = obj.get("object_path", "")
                essential["object_name"] = obj.get("object_name", "")
                essential["object_id"] = obj.get("object_id", "")
                essential["category"] = obj.get("category", "")
                _copy_bilingual_fields(obj, essential, "name")
                _copy_bilingual_fields(obj, essential, "description")
                essential["prompt"] = _get_english_description(obj)
            elif ENABLE_ACTOR_PARSING and ("actor_path" in obj or "actor_name" in obj):
                # [OBJECT-ONLY DISABLED] actor fallback
                essential["target"] = "actor"
                essential["actor_path"] = obj.get("actor_path", "")
                essential["actor_name"] = obj.get("actor_name", "")
                essential["actor_id"] = obj.get("actor_id", "")
                essential["category"] = ""
                # LLM이 기대하는 형식을 위해 description도 추가
                description = obj.get("description", "")
                essential["description"] = description
                essential["prompt"] = description  # 최종 JSON용
            else:
                # 알 수 없는 경우 건너뛰기
                continue

        _attach_usd_path_fields(obj, essential)

        if _has_description(essential):
            essential_list.append(essential)
    
    return essential_list


def parse_and_augment(
    usd_file_path: str,
    output_json_path: str,
    model_name: str = "gemma3:4b",
    filter_model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    parse_type: str = "object",
    enable_translate: bool = False,
    enable_filter: bool = False,
    translation_prompt_file: str = "step1_translate_prompt.txt",
    filter_prompt_file: str = "step2_filter_prompt.txt"
) -> None:
    """
    USD 파일을 파싱하고, 바로 LLM으로 증강한 후 최종 JSON을 저장합니다.
    
    Args:
        usd_file_path: USD 파일 경로
        output_json_path: 출력 JSON 파일 경로
        model_name: 1단계 번역용 Ollama 모델 이름 (기본값: "gemma3:4b")
        filter_model_name: 2단계 필터링용 Ollama 모델 이름 (기본값: None, model_name 사용)
        base_url: Ollama 서버 URL (기본값: None, localhost:11434 사용)
        parse_type: 파싱할 타입 ("object", "actor", "both")
        enable_translate: 1단계 번역 활성화 여부 (기본값: False)
        enable_filter: 2단계 필터링 활성화 여부 (기본값: False)
    """
    if not USD_PARSER_AVAILABLE:
        raise ImportError("USD 파서를 사용할 수 없습니다. usd_parser 모듈을 확인하세요.")
    
    # [OBJECT-ONLY] actor 파싱 비활성화 시 object만 처리
    if not ENABLE_ACTOR_PARSING and parse_type != "object":
        print(f"[INFO] object-only scope: parse_type '{parse_type}' → 'object'로 고정")
        parse_type = "object"
    
    print(f"[INFO] USD 파일 파싱 시작: {usd_file_path}")
    print(f"[INFO] 파싱 타입: {parse_type}")
    
    # 파일 존재 여부 사전 확인
    if not os.path.exists(usd_file_path):
        print(f"❌ USD 파일이 존재하지 않습니다: {usd_file_path}")
        print(f"   현재 작업 디렉토리: {os.getcwd()}")
        raise FileNotFoundError(f"USD 파일을 찾을 수 없습니다: {usd_file_path}")
    
    # 1. USD 파일 파싱
    parsed_objects = parse_usd_file(usd_file_path, parse_type=parse_type)
    print(f"[INFO] 추출된 항목 수: {len(parsed_objects)}")
    
    # 타입별 개수 확인
    objects = [r for r in parsed_objects if "object_name" in r]
    actors = [r for r in parsed_objects if "actor_name" in r] if ENABLE_ACTOR_PARSING else []
    print(f"  - 객체(object): {len(objects)}개")
    if ENABLE_ACTOR_PARSING:
        print(f"  - 액터(actor): {len(actors)}개")
    
    if len(parsed_objects) == 0:
        print(f"⚠️ USD 파일에서 추출된 항목이 없습니다.")
        print(f"   파일 경로: {usd_file_path}")
        print(f"   파싱 타입: {parse_type}")
        print(f"   파일 크기: {os.path.getsize(usd_file_path)} bytes")
        print(f"   파일 읽기 가능: {os.access(usd_file_path, os.R_OK)}")
        print(f"   💡 파일이 올바른 USD 형식인지 확인하거나, 파일 내부의 참조(reference) 경로를 확인하세요.")
        return
    
    # 2-0. Scene canonical만 downstream(LLM/TRELLIS/merge) 대상으로 분리 (스펙 2.1, #3)
    #      Shot override(object_scope == "shot")는 생성/주입 대상이 아니며 메타 참고용으로만 보존합니다.
    scene_objects = [o for o in parsed_objects if o.get("object_scope", "scene") != "shot"]
    shot_objects = [o for o in parsed_objects if o.get("object_scope") == "shot"]
    if shot_objects:
        print(f"[INFO] scene canonical: {len(scene_objects)}개 / shot override(meta-only): {len(shot_objects)}개")

    # 2. 필수 정보만 추출 (중간 JSON 저장 없음) — scene canonical만 LLM 입력으로 사용
    essential_objects = extract_essential_info(scene_objects)
    print(f"[INFO] 필수 정보 추출 완료: {len(essential_objects)}개")
    
    if not essential_objects:
        print("[WARNING] 증강할 항목이 없습니다.")
        print(f"   파싱된 항목은 {len(parsed_objects)}개이지만, 필수 정보(description 등)가 없습니다.")
        return
    
    # 3. 1단계 번역 (옵션)
    if enable_translate:
        translation_model = model_name
        print(f"[INFO] 1단계 번역 시작 (모델: {translation_model})...")
        translation_result = translate_descriptions_with_ollama(
            essential_objects,
            model_name=translation_model,
            base_url=base_url,
            prompt_file=translation_prompt_file
        )
        print(f"[INFO] 1단계 번역 완료: {len(translation_result)}개")
        
        missing_translation = 0
        for obj in essential_objects:
            key = _determine_response_key(obj)
            entry = translation_result.get(key, "") if key else ""
            translated_text, translated_name = _parse_translation_entry(entry)
            if translated_text:
                obj["translated_description"] = translated_text
            else:
                obj["translated_description"] = ""
                missing_translation += 1
            obj["translated_name"] = translated_name
        
        if missing_translation:
            print(f"[WARNING] 번역되지 않은 항목: {missing_translation}개")
    else:
        print(f"[INFO] 1단계 번역 비활성화됨 — USD en 필드를 translated_description으로 사용")
        translation_model = None
        for obj in essential_objects:
            obj["translated_description"] = _get_english_description(obj)
            obj["translated_name"] = _get_english_name(obj)
    
    # 1단계 모델 언로드 여부 결정 (2단계 모델과 다르면 언로드)
    filter_model = filter_model_name if filter_model_name is not None else model_name
    models_are_different = enable_translate and enable_filter and translation_model != filter_model
    
    if models_are_different:
        print(f"[INFO] 1단계와 2단계 모델이 다르므로 1단계 모델 언로드 중...")
        _unload_ollama_model(translation_model, base_url)
    
    # 4. LLM 필터링 (옵션)
    if enable_filter:
        print(f"[INFO] 2단계 필터링 시작 (모델: {filter_model})...")
        augmented_dict = augment_batch_with_ollama(
            essential_objects,
            model_name=filter_model,
            system_prompt_file=filter_prompt_file,
            base_url=base_url
        )
        print(f"[INFO] LLM 증강 완료: {len(augmented_dict)}개")
        
        # 2단계 모델 언로드
        print(f"[INFO] 2단계 모델 언로드 중...")
        _unload_ollama_model(filter_model, base_url)
    else:
        print(f"[INFO] 2단계 필터링 비활성화됨")
        augmented_dict = {}
        
        # 필터링이 비활성화되고 번역이 활성화된 경우 1단계 모델 언로드
        if enable_translate:
            print(f"[INFO] 1단계 모델 언로드 중...")
            _unload_ollama_model(translation_model, base_url)
    
    # 5. 최종 JSON 생성 (필수 정보 + aug_prompt)
    final_results = []
    aug_prompt_count = 0
    translated_count = 0
    description_only_count = 0
    
    for essential in essential_objects:
        target = essential.get("target", "")
        
        # aug_prompt 가져오기 (여러 키 시도)
        aug_prompt = ""
        if target == "object":
            path_key = essential.get("object_path", "")
            # 여러 키로 시도
            aug_prompt = (
                augmented_dict.get(path_key, "") or
                augmented_dict.get(essential.get("object_name", ""), "") or
                augmented_dict.get(essential.get("object_id", ""), "") or
                augmented_dict.get(str(essential.get("object_id", "")), "")
            )
        elif ENABLE_ACTOR_PARSING and target == "actor":
            path_key = essential.get("actor_path", "")
            aug_prompt = (
                augmented_dict.get(path_key, "") or
                augmented_dict.get(essential.get("actor_name", ""), "") or
                augmented_dict.get(essential.get("actor_id", ""), "") or
                augmented_dict.get(str(essential.get("actor_id", "")), "")
            )
        else:
            path_key = essential.get("object_path") or essential.get("actor_path", "")
            aug_prompt = augmented_dict.get(path_key, "")
        
        # 프롬프트 우선순위: aug_prompt -> translated_description -> description
        # aug_prompt가 비어있으면 translated_description 사용
        if not aug_prompt:
            aug_prompt = essential.get("translated_description", "")
            if aug_prompt:
                translated_count += 1
            else:
                description_only_count += 1
        else:
            aug_prompt_count += 1
        
        # 최종 결과 구성
        final_item = {
            "target": target,
        }
        
        if target == "object":
            final_item["object_scope"] = essential.get("object_scope", "scene")
            final_item["object_path"] = essential.get("object_path", "")
            final_item["object_name"] = essential.get("object_name", "")
            final_item["object_id"] = essential.get("object_id", "")
            _copy_bilingual_fields(essential, final_item, "name")
            if essential.get("translated_name"):
                final_item["translated_name"] = essential.get("translated_name", "")
        elif ENABLE_ACTOR_PARSING and target == "actor":
            final_item["actor_path"] = essential.get("actor_path", "")
            final_item["actor_name"] = essential.get("actor_name", "")
            final_item["actor_id"] = essential.get("actor_id", "")
        
        # USD 경로 저장 (scene canonical downstream: TRELLIS/merge)
        if essential.get("usd_file_path"):
            final_item["usd_file_path"] = essential.get("usd_file_path", "")
        if essential.get("usd_relative_path"):
            final_item["usd_relative_path"] = essential.get("usd_relative_path", "")
        if essential.get("assets_dir"):
            final_item["assets_dir"] = essential.get("assets_dir", "")

        final_item["category"] = essential.get("category", "")
        _copy_bilingual_fields(essential, final_item, "description")
        final_item["translated_description"] = essential.get("translated_description", "")
        final_item["aug_prompt"] = aug_prompt
        
        final_results.append(final_item)
    
    # 5-1. Shot override는 메타 참고용으로만 JSON에 보존 (생성/주입 제외, 스펙 6절)
    #      object_scope == "shot" + _pipeline == "meta_only" 마킹으로 downstream에서 필터됩니다.
    for obj in shot_objects:
        meta_item = {
            "target": "object",
            "object_scope": "shot",
            "object_path": obj.get("object_path", ""),
            "object_name": obj.get("object_name", ""),
            "object_id": obj.get("object_id", ""),
            "_pipeline": "meta_only",
            "_skip_reason": "shot_override_meta_only",
            "_skip_detail": "scene canonical과 중복; TRELLIS/merge 제외 (geometry는 scene reference로 상속)",
        }
        canonical_path = _canonical_object_path(obj.get("object_path", ""))
        if canonical_path:
            meta_item["canonical_object_path"] = canonical_path
        if obj.get("name"):
            meta_item["name"] = obj.get("name", "")
        for field in ("appearance", "location", "action"):
            _copy_bilingual_fields(obj, meta_item, field)
        if obj.get("usd_file_path"):
            meta_item["usd_file_path"] = obj.get("usd_file_path", "")
        final_results.append(meta_item)
    
    # 통계 정보 계산
    augmented_success_count = aug_prompt_count
    
    # 타입별 통계
    parsed_objects_count = len(objects)
    parsed_actors_count = len(actors)
    
    # description이 있는 항목 수 (LLM에 보낸 항목)
    with_description_count = len(essential_objects)
    
    # 통계 정보 구성
    statistics = {
        "parsed_total": len(parsed_objects),
        "parsed_objects": parsed_objects_count,
        "parsed_actors": parsed_actors_count,
        "scene_canonical_count": len(scene_objects),
        "shot_meta_only_count": len(shot_objects),
        "with_description": with_description_count,
        "aug_prompt_count": aug_prompt_count,
        "translated_count": translated_count,
        "description_only_count": description_only_count,
        "skipped_shot_override_paths": [
            o.get("object_path", "") for o in shot_objects if o.get("object_path")
        ],
    }
    
    # 5. 최종 JSON 저장 (통계 정보 포함)
    output_data = {
        "statistics": statistics,
        "results": final_results
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 최종 JSON 저장 완료: {output_json_path}")
    print(f"[INFO] 통계:")
    print(f"  - USD 파싱: 총 {len(parsed_objects)}개 (object: {parsed_objects_count}개, actor: {parsed_actors_count}개)")
    print(f"  - description 있음: {with_description_count}개")
    print(f"  - aug_prompt 사용: {aug_prompt_count}개")
    print(f"  - translated_description 사용: {translated_count}개")
    print(f"  - description만 사용: {description_only_count}개")
    print(f"  - shot override(meta-only, TRELLIS/merge 제외): {len(shot_objects)}개")
    print(f"[INFO] 총 {len(final_results)}개 항목 저장됨")


# 사용 예시
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="USD 파일 파싱 및 프롬프트 증강 통합 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # USD 파일 파싱 및 증강 (object와 actor 모두)
  python usd_parse_and_augment.py scene.usd --output results.json
  
  # object만 파싱 및 증강
  python usd_parse_and_augment.py scene.usd --output objects.json --type object
  
  # actor만 파싱 및 증강
  python usd_parse_and_augment.py scene.usd --output actors.json --type actor
  
  # 모델 지정
  python usd_parse_and_augment.py scene.usd --output results.json --model gemma3:4b
        """
    )
    parser.add_argument(
        "usd_file",
        help="USD 파일 경로"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="출력 JSON 파일 경로"
    )
    parser.add_argument(
        "-t", "--type",
        type=str,
        choices=["object", "actor", "both"],
        default="object",
        help="파싱할 타입: object (객체만, 기본값), actor/both (ENABLE_ACTOR_PARSING=True 필요)"
    )
    parser.add_argument(
        "-m", "--model",
        default="gemma3:4b",
        help="1단계 번역용 Ollama 모델 이름 (기본값: gemma3:4b)"
    )
    parser.add_argument(
        "--filter-model",
        default=None,
        help="2단계 필터링용 Ollama 모델 이름 (기본값: --model과 동일)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Ollama 서버 URL (기본값: http://localhost:11434)"
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="1단계 번역 활성화 (한국어 → 영어 번역)"
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="2단계 필터링 활성화 (프롬프트 증강)"
    )
    
    args = parser.parse_args()
    
    try:
        parse_and_augment(
            args.usd_file,
            args.output,
            model_name=args.model,
            filter_model_name=args.filter_model,
            base_url=args.base_url,
            parse_type=args.type,
            enable_translate=args.translate,
            enable_filter=args.filter
        )
        print("\n✅ 완료!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

