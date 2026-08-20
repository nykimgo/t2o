"""AutoRigging 스캐너 — 영화/씬 usda 에서 리깅 대상(생명체)을 수집해 매니페스트를 만든다.

DESIGN.md §3 명세:
- 캐릭터: <영화>/characters/<char>.usda — T2Character 산출 = 전부 인물 → 무조건 수집 (biped)
- 오브젝트: <영화>/scene_N/objects/object_M.usda — customData rig_type ∉ {static_object, 빈값}
  일 때만 수집. 미지값·누락 → 스킵 + 경고 (조용히 폴백하지 않는다).
- GLB 부재(빈 래퍼) → 스킵 + 사유 기록.

산출: autorig_manifest.json — 이후 모든 단계(리깅·모션·USD 조립)가 이걸 입력으로 받는다.

pxr(usd-core) 필요. usd_parser(previz_pipeline)의 rig_type 정규화 사상을 따르되,
팀 확정 7클래스 enum 을 정본으로 하고 구 enum(bird/insect)은 매핑 후 경고를 남긴다.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pxr import Usd

# 팀 확정 rig_type enum (DESIGN §3-1) + static_object
RIG_TYPES = {"biped", "quadruped", "hexapod", "octopod",
             "avian", "serpentine", "aquatic"}
# 구 스키마 값 매핑 (usd_parser 구 enum: bird/insect) — 매핑 시 경고 기록
LEGACY_MAP = {"bird": "avian", "insect": "hexapod"}


def _normalize_rig_type(value) -> tuple[str, str]:
    """(정규화값, 경고) 반환. 알 수 없는 값은 ('', 경고)."""
    if not value:
        return "", ""
    v = "_".join(str(value).strip().lower().replace("-", "_").replace(" ", "_").split("_"))
    if v in {"static", "staticobject", "inanimate", "prop", "object"}:
        v = "static_object"
    if v == "static_object" or v in RIG_TYPES:
        return v, ""
    if v in LEGACY_MAP:
        return LEGACY_MAP[v], f"구 enum '{v}' → '{LEGACY_MAP[v]}' 매핑"
    return "", f"미지 rig_type '{value}'"


def _read_root_custom_data(usda_path: Path) -> dict:
    stage = Usd.Stage.Open(str(usda_path))
    prim = stage.GetDefaultPrim()
    if not prim:
        prims = [p for p in stage.GetPseudoRoot().GetChildren()]
        prim = prims[0] if prims else None
    return dict(prim.GetCustomData() or {}) if prim else {}


def _find_glb(assets_dir: Path, name: str) -> Path | None:
    """assets/<name>/ 에서 GLB 탐색 — <name>.glb 우선, 없으면 최신 *.glb."""
    if not assets_dir.is_dir():
        return None
    exact = assets_dir / f"{name}.glb"
    if exact.exists():
        return exact
    cands = sorted(assets_dir.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _entry(kind: str, name: str, usda: Path, assets_dir: Path, rig_type: str,
           warnings: list[str]) -> dict:
    glb = _find_glb(assets_dir, name)
    tex = assets_dir / "bin" / "texture.jpg"
    geo = assets_dir / f"{name}.geometry.usda"
    e = {
        "kind": kind, "name": name, "rig_type": rig_type,
        "usda": str(usda), "assets_dir": str(assets_dir),
        "glb": str(glb) if glb else None,
        "texture": str(tex) if tex.exists() else None,
        "geometry_usda": str(geo) if geo.exists() else None,
        "skip": None, "warnings": warnings,
    }
    if glb is None:
        e["skip"] = "glb 부재 (빈 래퍼 또는 3D 미생성)"
    return e


def scan(usd_file: str | Path) -> dict:
    """영화 또는 씬 usda → 매니페스트 dict.

    입력이 <movie>/<movie>.usda 면 영화 전체(캐릭터 + 모든 씬 오브젝트),
    <movie>/scene_N/scene_N.usda 면 그 씬 오브젝트만 (+ 영화 캐릭터는 항상 포함 —
    캐릭터는 영화 단위 소유라 씬 실행에서도 리깅 대상이다).
    """
    usd_file = Path(usd_file).resolve()
    if not usd_file.exists():
        raise FileNotFoundError(usd_file)

    scene_only = None
    movie_root = usd_file.parent
    if re.match(r"^scene_\w+$", movie_root.name):
        scene_only = movie_root.name
        movie_root = movie_root.parent
    movie = movie_root.name

    targets: list[dict] = []

    # 1) 캐릭터 — 무조건 수집 (rig_type=biped 취급)
    chars_dir = movie_root / "characters"
    if chars_dir.is_dir():
        for cu in sorted(chars_dir.glob("*.usda")):
            if ".rt_" in cu.name or ".rig." in cu.name:
                continue  # 본 모듈 산출물(래퍼) 재수집 방지
            name = cu.stem
            targets.append(_entry("character", name, cu,
                                  chars_dir / "assets" / name, "biped", []))

    # 2) 오브젝트 — rig_type 판정
    scene_dirs = ([movie_root / scene_only] if scene_only
                  else sorted(d for d in movie_root.glob("scene_*") if d.is_dir()))
    for sd in scene_dirs:
        obj_dir = sd / "objects"
        if not obj_dir.is_dir():
            continue
        for ou in sorted(obj_dir.glob("*.usda")):
            if ".rt_" in ou.name or ".rig." in ou.name:
                continue
            name = ou.stem
            cd = _read_root_custom_data(ou)
            raw = cd.get("rig_type", "")
            rig_type, warn = _normalize_rig_type(raw)
            warnings = [w for w in [warn] if w]
            if rig_type == "static_object":
                continue                       # 명시적 정적 오브젝트 — 조용히 제외
            if not rig_type:
                # 미지값·누락 → 스킵 + 경고 (조용한 폴백 금지)
                e = _entry("object", name, ou, obj_dir / "assets" / name, "", warnings)
                e["skip"] = (f"rig_type 미지값 '{raw}'" if raw else "rig_type 누락")
                targets.append(e)
                continue
            targets.append(_entry("object", name, ou,
                                  obj_dir / "assets" / name, rig_type, warnings))

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "input_usd": str(usd_file),
        "movie_root": str(movie_root),
        "movie": movie,
        "scene_only": scene_only,
        "targets": targets,
        "n_collect": sum(1 for t in targets if not t["skip"]),
        "n_skip": sum(1 for t in targets if t["skip"]),
    }


def write_manifest(manifest: dict, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AutoRigging 대상 스캔")
    ap.add_argument("usd_file")
    ap.add_argument("--out", default="autorig_manifest.json")
    a = ap.parse_args()
    m = scan(a.usd_file)
    write_manifest(m, a.out)
    print(f"[scan] 수집 {m['n_collect']} / 스킵 {m['n_skip']} -> {a.out}")
    for t in m["targets"]:
        mark = "SKIP" if t["skip"] else "OK  "
        extra = f" ({t['skip']})" if t["skip"] else ""
        warn = f"  ⚠ {'; '.join(t['warnings'])}" if t["warnings"] else ""
        print(f"  [{mark}] {t['kind']:9s} {t['name']:20s} rig_type={t['rig_type'] or '-'}{extra}{warn}")
