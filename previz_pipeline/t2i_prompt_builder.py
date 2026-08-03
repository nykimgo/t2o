"""Confirmed T2I system-prompt builder (§9, prompt_lab EXPERIMENT_LOG.md, 2026-07 확정).

Two templates, routed by category (생명체는 내부에서 형태 3분기):

  ① 무생물 (inanimate) — ablation+LOO 검증:
     "{base_description}, {appearance}, {object}, single centered object,
      neutral background, full object visible in frame, unoccluded"

  ② 생명체 (creature) — body-plan 분류 후 고정 스캐폴딩 verbatim 삽입 (필드는 object만):
     "{object}, <biped|quadruped|bird 스캐폴딩>"

Body-plan 분류는 **LLM-free**다. Tripo/UniRig 의 rig-type taxonomy(biped/quadruped/bird/…)를
참고하되, 저쪽은 3D 메시 지오메트리로 판정한다(post-mesh). 우리는 메시가 없는 text→image 단계라
그 방식을 그대로 쓸 수 없어서, USD `category` + 객체명(en) 키워드 dict 로 결정론적 분류한다.
open-vocab 동물의 롱테일은 dict 가 못 잡을 수 있어 안전 기본값(quadruped)으로 떨어지며 경고 로그를 남긴다.
정밀도가 필요하면 `classify_body_plan` 에 LLM(ollama) 훅을 끼울 수 있다 (§9: "LLM은 클래스 라벨만").

자세 문구는 §9 고정값 verbatim — 재현성/ablation 최적화를 위해 **변조 금지**.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

# --- §9 고정 문구 (verbatim, do not reword) ---------------------------------
ISOLATION_INANIMATE = (
    "single centered object, neutral background, "
    "full object visible in frame, unoccluded"
)

SCAFFOLDING = {
    "biped": (
        "in a symmetric A-pose, arms angled slightly down and away from the body, "
        "legs straight and slightly apart, front view, neutral background, "
        "single object, full body"
    ),
    "quadruped": (
        "standing naturally on all four legs, all four legs clearly separated and "
        "extended, not tucked, side view, neutral background, single object, "
        "full body in frame"
    ),
    "bird": (
        "with wings fully spread symmetrically, standing, front view, "
        "neutral background, single object, full body in frame"
    ),
    # insect(6족)는 §9에 없던 신규 클래스 — 업스트림 rig_type=insect 대응(잠정값, O/X 검증 전).
    "insect": (
        "with all six legs clearly separated and extended, standing, side view, "
        "neutral background, single object, full body in frame"
    ),
}

# 업스트림 권위 필드(rig_type)의 정식 값. static_object 는 무생물(생명체 아님).
VALID_RIG_TYPES = {"biped", "quadruped", "bird", "insect", "static_object"}


def normalize_rig_type(value) -> str:
    """rig_type 값을 방어적으로 정규화. 'static object'/'Static-Object' 등 → 'static_object'.
    미지값/빈값은 '' 반환(→ 키워드 dict 폴백)."""
    if not value:
        return ""
    v = "_".join(str(value).strip().lower().replace("-", "_").replace(" ", "_").split("_"))
    if v in {"static", "staticobject", "inanimate", "non_creature", "object", "prop"}:
        v = "static_object"
    return v if v in VALID_RIG_TYPES else ""

# --- 생명체 판정 / body-plan 분류용 키워드 (모두 소문자 substring 매칭) --------
_CREATURE_CATEGORIES = {
    "animal", "animals", "creature", "creatures", "bird", "birds", "insect",
    "insects", "fish", "reptile", "amphibian", "mammal", "person", "people",
    "human", "humanoid", "character",
    "동물", "새", "곤충", "물고기", "파충류", "포유류", "사람", "인물", "캐릭터", "인간",
}

_BIRD_KEYWORDS = {
    "bird", "seagull", "gull", "eagle", "hawk", "owl", "sparrow", "pigeon",
    "dove", "duck", "goose", "swan", "penguin", "crow", "raven", "parrot",
    "chicken", "hen", "rooster", "flamingo", "pelican", "heron", "crane",
    "ostrich", "peacock", "falcon", "robin", "magpie", "woodpecker",
    "새", "갈매기", "독수리", "매", "부엉이", "올빼미", "참새", "비둘기", "오리",
    "거위", "백조", "펭귄", "까마귀", "앵무새", "닭", "학", "공작",
}

_BIPED_KEYWORDS = {
    "human", "man", "woman", "boy", "girl", "child", "kid", "person", "people",
    "gorilla", "chimpanzee", "chimp", "ape", "monkey", "orangutan", "humanoid",
    "robot", "android", "kangaroo",
    "사람", "인간", "남자", "여자", "소년", "소녀", "아이", "고릴라", "원숭이", "로봇", "캥거루",
}

_QUADRUPED_KEYWORDS = {
    "dog", "puppy", "cat", "kitten", "horse", "pony", "cow", "bull", "ox",
    "pig", "sheep", "lamb", "goat", "lion", "tiger", "bear", "wolf", "fox",
    "deer", "elephant", "rhino", "rhinoceros", "hippo", "hippopotamus", "zebra",
    "giraffe", "leopard", "cheetah", "panther", "rabbit", "hare", "mouse",
    "rat", "hamster", "camel", "donkey", "buffalo", "bison", "boar", "panda",
    "koala", "raccoon", "squirrel", "hedgehog", "goat",
    "개", "강아지", "고양이", "말", "소", "돼지", "양", "염소", "사자", "호랑이",
    "곰", "늑대", "여우", "사슴", "코끼리", "코뿔소", "얼룩말", "기린", "토끼", "판다",
}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _any_kw(text: str, keywords: set) -> bool:
    return any(kw in text for kw in keywords)


def is_creature(object_name: Optional[str], category: Optional[str],
                target: Optional[str] = "object") -> bool:
    """생명체 여부. actor(사람)는 항상 생명체(biped)."""
    if _norm(target) == "actor":
        return True
    cat = _norm(category)
    if cat in _CREATURE_CATEGORIES:
        return True
    name = _norm(object_name)
    return _any_kw(name, _BIRD_KEYWORDS | _BIPED_KEYWORDS | _QUADRUPED_KEYWORDS)


def classify_body_plan(object_name: Optional[str], category: Optional[str],
                       target: Optional[str] = "object") -> str:
    """생명체를 biped|quadruped|bird 중 하나로 결정론적 분류.

    우선순위: 객체명 키워드(bird>biped>quadruped) → category 힌트 → 안전 기본값(quadruped).
    (LLM 훅 자리: ollama 등으로 라벨만 받아오려면 여기서 분기.)
    """
    if _norm(target) == "actor":
        return "biped"
    name = _norm(object_name)
    if _any_kw(name, _BIRD_KEYWORDS):
        return "bird"
    if _any_kw(name, _BIPED_KEYWORDS):
        return "biped"
    if _any_kw(name, _QUADRUPED_KEYWORDS):
        return "quadruped"

    cat = _norm(category)
    if cat in {"bird", "birds", "새"}:
        return "bird"
    if cat in {"person", "people", "human", "humanoid", "character",
               "사람", "인물", "캐릭터", "인간"}:
        return "biped"
    if cat in {"animal", "animals", "mammal", "creature", "creatures",
               "동물", "포유류"}:
        return "quadruped"

    logging.warning(
        "[t2i_prompt] body-plan 미분류(키워드/카테고리 매칭 실패) → quadruped 기본값 사용: "
        "name=%r category=%r", object_name, category)
    return "quadruped"


def _inanimate_prompt(obj: str, appearance: Optional[str],
                      base_description: Optional[str]) -> str:
    # 무생물: base_description, appearance, object + isolation (빈 값/중복 제거)
    parts = []
    for p in (base_description, appearance, obj):
        p = (p or "").strip()
        if p and p not in parts:
            parts.append(p)
    parts.append(ISOLATION_INANIMATE)
    return ", ".join(parts)


def _creature_prompt(obj: str, scaffolding: str,
                     base_description: Optional[str]) -> str:
    """생명체: {object}, {스캐폴딩}, {base_description}.

    base_description 은 정체성/외형(예: 알비노 갈매기, 매 형상의 주인공 갈매기)을 담으므로
    생명체에도 반드시 포함한다. 스캐폴딩이 이미 'neutral background, single object, full body'
    (격리 역할)를 포함하므로 별도 격리문구는 붙이지 않는다(중복 방지).
    주의: filter OFF 면 base_description 에 동작 표현(예: flying)이 남아 강제 자세와 충돌할 수 있다
    (filter ON 이면 caption 정제로 동작이 제거됨). — prompt_lab 실험 B/C 로 최종 정련 예정.
    """
    lead = (obj or "").strip() or (base_description or "").strip() or "creature"
    parts = [lead, scaffolding]
    bd = (base_description or "").strip()
    if bd and bd not in parts:
        parts.append(bd)
    return ", ".join(p for p in parts if p)


def build_t2i_prompt(object_name: Optional[str], appearance: Optional[str],
                     base_description: Optional[str], category: Optional[str],
                     target: Optional[str] = "object",
                     rig_type: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """§9 확정 템플릿으로 최종 T2I 프롬프트 조립.

    rig_type(업스트림 권위 필드; biped|quadruped|bird|insect|static_object)이 주어지면
    그 값을 신뢰해 라우팅한다. 없거나 미지값이면 category+객체명 키워드 dict 로 폴백한다.

    Returns (prompt, body_plan). body_plan 은 생명체일 때만 값(스캐폴딩 클래스), 무생물이면 None.
    """
    obj = (object_name or "").strip()

    # 1) 업스트림 rig_type 우선 (권위)
    rt = normalize_rig_type(rig_type)
    if rt == "static_object":
        return _inanimate_prompt(obj, appearance, base_description), None
    if rt in SCAFFOLDING:  # biped|quadruped|bird|insect
        return _creature_prompt(obj, SCAFFOLDING[rt], base_description), rt

    # 2) 폴백: 결정론적 키워드 dict
    if is_creature(object_name, category, target):
        plan = classify_body_plan(object_name, category, target)
        return _creature_prompt(obj, SCAFFOLDING[plan], base_description), plan

    return _inanimate_prompt(obj, appearance, base_description), None
