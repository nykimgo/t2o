"""이중 언어({ko,en}) 자연어 필드 — t2o_pipeline 로컬 유틸."""
from __future__ import annotations

from typing import Any, Dict

_LANGS = ("ko", "en")


def _is_blank_str(s: Any) -> bool:
    return not isinstance(s, str) or not s.strip() or s.strip().upper() == "NA"


def is_blank(value: Any) -> bool:
    """단일/이중 공통 빈값 판정."""
    if value is None:
        return True
    if isinstance(value, str):
        return _is_blank_str(value)
    if isinstance(value, dict):
        if "ko" in value or "en" in value:
            return all(_is_blank_str(value.get(lang, "")) for lang in _LANGS)
        return len(value) == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    return False


def pick_lang(value: Any, lang: str = "ko") -> str:
    """{ko,en} | str | None 에서 단일 언어 문자열 추출."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        primary = value.get(lang, "")
        if isinstance(primary, str) and primary.strip():
            return primary
        for alt_lang in _LANGS:
            alt = value.get(alt_lang, "")
            if isinstance(alt, str) and alt.strip():
                return alt
        return primary if isinstance(primary, str) else ""
    return str(value)


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def coerce_bilingual(value: Any) -> Dict[str, str]:
    """임의 입력을 {ko,en} 으로 정규화."""
    if isinstance(value, dict) and ("ko" in value or "en" in value):
        ko = _as_str(value.get("ko", ""))
        en = _as_str(value.get("en", ""))
        if not en.strip():
            en = ko
        if not ko.strip():
            ko = en
        return {"ko": ko, "en": en}
    if isinstance(value, str):
        return {"ko": value, "en": value}
    return {"ko": "", "en": ""}


def flatten_bilingual_fields(
    target: Dict[str, Any],
    field_name: str,
    bilingual: Dict[str, str],
) -> None:
    """dict 필드와 _ko/_en 평탄 필드를 target에 기록."""
    if is_blank(bilingual):
        return
    target[field_name] = bilingual
    target[f"{field_name}_ko"] = bilingual.get("ko", "")
    target[f"{field_name}_en"] = bilingual.get("en", "")
