"""AutoRigging 오케스트레이터 — 스캔 → 리깅(배치 subprocess) → 기본모션 → USD 조립 → 리포트.

실행 (unirig env, LD_LIBRARY_PATH 제거 필수 — 런처 autorig_generate.sh 가 처리):
  python autorigging/autorig.py <movie.usda|scene.usda> [--dry-run] [--only 이름]
                                [--no-motion] [--run-dir DIR] [--seed 12345]

단계 설계 (DESIGN §4):
- 리깅은 rig_service.py 를 --daemon subprocess 로 1회 띄워 전 에셋을 배치 처리하고
  stdin 닫힘과 함께 종료시킨다 (프로세스 수명 = VRAM/RAM unload 보장, 상주 금지).
- 모션+USD 조립은 본 프로세스(bpy+pxr 공존)에서 에셋별 순차 수행.
- 실패한 에셋은 스킵하고 전체는 계속 (리포트에 사유 기록).
- 멱등성: rigs FBX·rt_default 래퍼가 이미 있으면 skip (exists).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import scan as SCAN            # noqa: E402
import report as REPORT        # noqa: E402

MARK = "##RIG## "


# ---------------------------------------------------------------- 리깅 (배치 subprocess)
def rig_batch(jobs: list[dict], seed: int, log) -> dict[str, dict]:
    """rig_service 데몬을 subprocess 로 띄워 전 에셋 배치 리깅. {name: 결과} 반환.

    jobs: [{name, glb, out_dir}] — out_dir = assets/<이름>/rigs/
    """
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.pop("LD_LIBRARY_PATH", None)       # usd/lib 가 bpy MaterialX·libcuda 를 가림 (실측)

    proc = subprocess.Popen(
        [sys.executable, "-u", str(HERE / "rig_service.py"), "--daemon", "--seed", str(seed)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env, cwd=str(HERE))

    results: dict[str, dict] = {}

    def read_mark_line():
        for line in proc.stdout:
            log(line.rstrip())
            if line.startswith(MARK):
                return json.loads(line[len(MARK):])
        raise RuntimeError("rig_service 가 응답 없이 종료됨")

    try:
        ready = read_mark_line()           # {"ready": true, "load_sec": ...}
        log(f"[rig] 모델 로드 완료 ({ready.get('load_sec')}s) — {len(jobs)}건 배치 시작")
        for j in jobs:
            proc.stdin.write(json.dumps(
                {"input": j["glb"], "output_dir": j["out_dir"], "skin": True}) + "\n")
            proc.stdin.flush()
            results[j["name"]] = read_mark_line()
    finally:
        try:
            proc.stdin.close()             # stdin 닫힘 → 데몬 종료 = unload
        except Exception:
            pass
        proc.wait(timeout=120)
    return results


# ---------------------------------------------------------------- 메인
def main():
    ap = argparse.ArgumentParser(description="AutoRigging 파이프라인")
    ap.add_argument("usd_file")
    ap.add_argument("--dry-run", action="store_true", help="스캔 결과만 출력")
    ap.add_argument("--only", default="", help="이름 부분일치 필터")
    ap.add_argument("--no-motion", action="store_true", help="리깅까지만")
    ap.add_argument("--run-dir", default="", help="리포트/매니페스트 출력 디렉터리")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    t0 = time.time()
    run_dir = Path(a.run_dir) if a.run_dir else Path.cwd() / "autorig_out" / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "autorig.log"
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    # 1) 스캔
    manifest = SCAN.scan(a.usd_file)
    SCAN.write_manifest(manifest, run_dir / "autorig_manifest.json")
    log(f"[scan] 수집 {manifest['n_collect']} / 스킵 {manifest['n_skip']} "
        f"({manifest['movie']}) -> {run_dir / 'autorig_manifest.json'}")
    for t in manifest["targets"]:
        mark = "SKIP" if t["skip"] else "OK  "
        extra = f" ({t['skip']})" if t["skip"] else ""
        warn = f"  ⚠ {'; '.join(t['warnings'])}" if t["warnings"] else ""
        log(f"  [{mark}] {t['kind']:9s} {t['name']:20s} rig_type={t['rig_type'] or '-'}{extra}{warn}")
    if a.dry_run:
        log("[dry-run] 종료")
        return 0

    targets = [t for t in manifest["targets"] if not t["skip"]]
    if a.only:
        targets = [t for t in targets if a.only in t["name"]]
        log(f"[filter] --only '{a.only}' → {len(targets)}건")

    results: list[dict] = []
    for t in manifest["targets"]:
        if t["skip"]:
            results.append({"name": t["name"], "rig_type": t["rig_type"],
                            "status": "skip", "reason": t["skip"]})

    # 2) 원본 스냅샷 (불변 검증용 — §7-3)
    log("[hash] 원본 스냅샷 생성 중...")
    snap = REPORT.snapshot(manifest["movie_root"])
    log(f"[hash] {len(snap)}개 파일 스냅샷 완료")

    # 3) 리깅 — 이미 rigs 가 있으면 skip (멱등성 T4)
    rig_jobs, rig_res = [], {}
    for t in targets:
        rigs_dir = Path(t["assets_dir"]) / "rigs"
        rigged = rigs_dir / f"{t['name']}_rigged.fbx"
        if rigged.exists():
            log(f"[rig] skip (exists): {rigged.name}")
            rig_res[t["name"]] = {"rigged_fbx": str(rigged),
                                  "skeleton_fbx": str(rigs_dir / f"{t['name']}_skeleton.fbx"),
                                  "error": None, "cached": True}
        else:
            rig_jobs.append({"name": t["name"], "glb": t["glb"], "out_dir": str(rigs_dir)})
    if rig_jobs:
        log(f"[rig] {len(rig_jobs)}건 배치 리깅 (모델 로드 ~10s + 에셋당 ~10-22s)")
        rig_res.update(rig_batch(rig_jobs, a.seed, log))

    # 4) 모션 + USD 조립 (bpy — 리깅 subprocess 종료 후 임포트)
    if not a.no_motion:
        import default_motion as DM        # noqa: PLC0415 — bpy 는 리깅 후 로드
        import usd_assemble as UA          # noqa: PLC0415

        for t in targets:
            name = t["name"]
            r = rig_res.get(name, {})
            if r.get("error") or not r.get("rigged_fbx"):
                results.append({"name": name, "rig_type": t["rig_type"],
                                "status": "fail", "reason": f"리깅 실패: {r.get('error')}"})
                continue
            wrapper = Path(t["usda"]).parent / f"{name}.{UA.MOTION_SUFFIX}.usda"
            if wrapper.exists():
                log(f"[motion] skip (exists): {wrapper.name}")
                results.append({"name": name, "rig_type": t["rig_type"],
                                "status": "skip", "reason": "rt_default 래퍼 존재 (멱등)"})
                continue
            try:
                log(f"[motion] {name} ({t['rig_type']}) 기본모션 생성...")
                motion = DM.build(r["rigged_fbx"], t["rig_type"])
                out = UA.assemble(t, motion, Path(r["rigged_fbx"]))
                v = out["validation"]
                log(f"[usd] {name}: anim {v['anim_samples']}샘플 | "
                    f"tex {'OK' if v['texture_ok'] else 'MISS'} | "
                    f"bbox {v['bbox_ratio']}x | {'PASS' if v['pass'] else 'FAIL'}")
                results.append({"name": name, "rig_type": t["rig_type"],
                                "status": "ok" if v["pass"] else "fail",
                                "reason": None if v["pass"] else "검증 FAIL",
                                "rigs": {k: r.get(k) for k in ("skeleton_fbx", "rigged_fbx")},
                                "layer": out["layer"], "wrapper": out["wrapper"],
                                "motion": out["manifest"]["motion"],
                                "motion_grade": out["manifest"]["motion_grade"],
                                "validation": v})
            except Exception as e:  # 한 에셋 실패가 전체를 죽이면 안 됨 (T7)
                log(f"[fail] {name}: {e!r}")
                results.append({"name": name, "rig_type": t["rig_type"],
                                "status": "fail", "reason": repr(e)[:300]})
    else:
        for t in targets:
            r = rig_res.get(t["name"], {})
            status = "fail" if r.get("error") else "ok"
            results.append({"name": t["name"], "rig_type": t["rig_type"],
                            "status": status, "reason": r.get("error"),
                            "rigs": {k: r.get(k) for k in ("skeleton_fbx", "rigged_fbx")}})

    # 5) 원본 불변 검증 + 리포트
    log("[hash] 원본 불변 검증 중...")
    originals = REPORT.verify_unchanged(manifest["movie_root"], snap)
    if originals["pass"]:
        log(f"[hash] PASS — 기존 파일 {len(snap)}개 불변, 추가 {len(originals['added'])}개")
    else:
        log(f"[hash] ❌ FAIL — 변경 {originals['changed']} 삭제 {originals['missing']}")
    rp = REPORT.write_report(run_dir, manifest, results, originals)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    log(f"[done] 성공 {n_ok} / 실패 {n_fail} / 스킵 "
        f"{sum(1 for r in results if r['status'] == 'skip')} "
        f"— {time.time() - t0:.0f}s, 리포트: {rp}")
    log_f.close()
    # 에셋 단위 실패는 전체를 죽이지 않지만(T7), 원본 훼손은 하드 실패다.
    return 1 if not originals["pass"] else 0


if __name__ == "__main__":
    sys.exit(main())
