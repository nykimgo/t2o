"""실행 리포트 + 원본 불변 검증 (DESIGN §7-3, §8 T3).

실행 전 movie 트리 전 파일 해시 스냅샷 → 실행 후 동일 경로 재해시 비교.
본 모듈은 새 파일 추가만 허용하므로, 스냅샷에 있던 파일의 해시가 하나라도
변하면 FAIL (원본 훼손).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(movie_root: str | Path) -> dict:
    """movie 트리의 모든 기존 파일 → {상대경로: sha256}."""
    movie_root = Path(movie_root)
    out = {}
    for p in sorted(movie_root.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(movie_root))] = _sha256(p)
    return out


def verify_unchanged(movie_root: str | Path, snap: dict) -> dict:
    """스냅샷 대비 변경/삭제 파일 목록. 비어 있으면 원본 불변 PASS."""
    movie_root = Path(movie_root)
    changed, missing = [], []
    for rel, digest in snap.items():
        p = movie_root / rel
        if not p.exists():
            missing.append(rel)
        elif _sha256(p) != digest:
            changed.append(rel)
    added = [str(p.relative_to(movie_root)) for p in sorted(movie_root.rglob("*"))
             if p.is_file() and str(p.relative_to(movie_root)) not in snap]
    return {"pass": not changed and not missing,
            "changed": changed, "missing": missing, "added": added}


def write_report(run_dir: str | Path, manifest: dict, results: list[dict],
                 originals: dict | None) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") == "skip"]
    failed = [r for r in results if r.get("status") == "fail"]

    report = {
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "input_usd": manifest["input_usd"],
        "movie": manifest["movie"],
        "summary": {"ok": len(ok), "skip": len(skipped), "fail": len(failed)},
        "originals_unchanged": originals,
        "results": results,
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2))

    lines = [f"# AutoRigging 리포트 — {manifest['movie']}", "",
             f"- 입력: `{manifest['input_usd']}`",
             f"- 완료: {report['finished_at']}",
             f"- 성공 {len(ok)} · 스킵 {len(skipped)} · 실패 {len(failed)}"]
    if originals is not None:
        mark = "PASS" if originals["pass"] else "**FAIL**"
        lines.append(f"- 원본 불변 검증: {mark} "
                     f"(변경 {len(originals['changed'])} · 삭제 {len(originals['missing'])} "
                     f"· 추가 {len(originals['added'])})")
        for rel in originals["changed"]:
            lines.append(f"  - 변경됨(위반): `{rel}`")
    lines.append("")
    lines.append("| 대상 | rig_type | 상태 | 검증 | 비고 |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        v = r.get("validation") or {}
        vs = ""
        if v:
            vs = (f"anim {v.get('anim_samples', 0)}샘플 / "
                  f"tex {'OK' if v.get('texture_ok') else 'MISS'} / "
                  f"bbox {v.get('bbox_ratio', '-')}x "
                  f"→ {'PASS' if v.get('pass') else 'FAIL'}")
        note = r.get("reason") or "; ".join(v.get("notes", [])) or ""
        lines.append(f"| {r['name']} | {r.get('rig_type', '-')} | {r['status']} | {vs} | {note} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    return run_dir / "report.json"
