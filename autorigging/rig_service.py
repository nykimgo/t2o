"""UniRig 상주 리깅 서비스 — 개발자 에셋 확인용 (모델 1회 로드 + extract 1회).

기존 실험 래퍼(EXP1_rigging/s3_rig.py)는 메시마다 셸 스크립트를 새 프로세스로 띄워
① 파이썬/torch/bpy 임포트+체크포인트 로드(~25s)를 스켈레톤·스킨 각각 반복하고
② extract(데시메이션 50k·복셀화)도 두 단계가 각각 다시 돌렸다(skin 쪽 force_override).
이 서비스는 UniRig 코드를 수정하지 않고 run.py 의 조립 로직만 인프로세스로 재구성해
스켈레톤·스킨 모델을 상주시키고 extract 를 에셋당 1회만 돈다.

실행 (unirig env 필수, GPU 1장):
  # 배치: 여러 에셋을 한 번의 로드로 처리
  CUDA_VISIBLE_DEVICES=0 /home/sr/miniconda3/envs/unirig/bin/python rig_service.py \
      asset1.glb asset2.glb --out-dir ./rigs

  # 데몬: stdin 으로 JSON 라인을 받아 즉시 리깅 (개발 툴 연동용)
  CUDA_VISIBLE_DEVICES=0 /home/sr/miniconda3/envs/unirig/bin/python rig_service.py --daemon
  → stdin:  {"input": "/abs/path/mesh.glb", "output_dir": "/abs/out", "skin": true}
  → stdout: ##RIG## {"ready": true}                      (기동 완료 신호)
            ##RIG## {"input": ..., "skeleton_fbx": ..., "rigged_fbx": ...,
                     "extract_sec": ..., "skeleton_sec": ..., "skin_sec": ..., "error": null}

주의:
- UniRig 저장소 위치는 env UNIRIG_DIR (기본: EXP1_rigging/UniRig). 상대 config 경로 때문에
  프로세스 cwd 를 그쪽으로 옮기므로 입출력 경로는 절대경로로 넘겨라 (CLI 가 절대화해 준다).
- 시드 기본 12345 = s3_rig.py 와 동일 (동등성 비교 가능).
- 로컬 패치 전제: UniRig run.py 의 safe_globals(Box), ar.py 의 user_mode
  predict_skeleton.npz 저장 (MANIFEST §4). 스킨은 그 npz 를 소비한다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 기본값: 이관 패키지의 UniRig 저장소 (로컬 패치 2건 적용본). env UNIRIG_DIR 로 오버라이드.
_DEFAULT_UNIRIG = Path(__file__).resolve().parent.parent / "autorigging_transfer" / "unirig"
UNIRIG_DIR = Path(os.environ.get("UNIRIG_DIR", str(_DEFAULT_UNIRIG)))
MARK = "##RIG## "

# UniRig 는 'configs/...' 상대경로와 'src' 패키지 임포트를 전제한다.
os.chdir(UNIRIG_DIR)
sys.path.insert(0, str(UNIRIG_DIR))

import torch  # noqa: E402
import yaml  # noqa: E402
from box import Box  # noqa: E402

try:  # torch>=2.6 weights_only 차단 해제 (run.py 로컬 패치와 동일)
    torch.serialization.add_safe_globals([Box])
except AttributeError:
    pass

import lightning as L  # noqa: E402

from src.data.datapath import Datapath  # noqa: E402
from src.data.dataset import UniRigDatasetModule, DatasetConfig  # noqa: E402
from src.data.extract import get_files, extract_builtin  # noqa: E402
from src.data.transform import TransformConfig  # noqa: E402
from src.inference.download import download  # noqa: E402
from src.model.parse import get_model  # noqa: E402
from src.system.parse import get_system, get_writer  # noqa: E402
from src.tokenizer.parse import get_tokenizer  # noqa: E402
from src.tokenizer.spec import TokenizerConfig  # noqa: E402

SKELETON_TASK = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
SKIN_TASK = "configs/task/quick_inference_unirig_skin.yaml"
FACES_TARGET = 50000              # extract.sh 기본값과 동일 (조건 간 동일 적용)
REQUIRE_SUFFIX = ["obj", "fbx", "FBX", "dae", "glb", "gltf", "vrm"]


def _load_yaml(path: str) -> Box:
    return Box(yaml.safe_load(open(path, "r")))


class _Stage:
    """task yaml 하나(스켈레톤 또는 스킨)의 상주 상태: 모델+시스템(가중치 로드 완료)."""

    def __init__(self, task_path: str):
        task = _load_yaml(task_path)
        assert task.mode == "predict", task_path
        self.task = task
        self.data_name = task.components.get("data_name", "raw_data.npz")

        data_config = _load_yaml(os.path.join("configs/data", task.components.data + ".yaml"))
        transform_config = _load_yaml(
            os.path.join("configs/transform", task.components.transform + ".yaml"))
        self.predict_dataset_config = DatasetConfig.parse(
            config=data_config.predict_dataset_config).split_by_cls()
        self.predict_transform_config = TransformConfig.parse(
            config=transform_config.predict_transform_config)

        tokenizer_config = task.components.get("tokenizer", None)
        self.tokenizer_config = None
        tokenizer = None
        if tokenizer_config is not None:
            self.tokenizer_config = TokenizerConfig.parse(config=_load_yaml(
                os.path.join("configs/tokenizer", task.components.tokenizer + ".yaml")))
            tokenizer = get_tokenizer(config=self.tokenizer_config)

        model_config = _load_yaml(os.path.join("configs/model", task.components.model + ".yaml"))
        self.model = get_model(tokenizer=tokenizer, **model_config)

        system_config = _load_yaml(os.path.join("configs/system", task.components.system + ".yaml"))
        self.system = get_system(**system_config, model=self.model,
                                 optimizer_config=None, loss_config=None,
                                 scheduler_config=None, steps_per_epoch=1)

        # ★ 상주의 핵심: 체크포인트를 지금 1회만 로드. 이후 predict 는 ckpt_path=None.
        ckpt_path = download(task.resume_from_checkpoint)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.system.load_state_dict(ckpt["state_dict"], strict=True)
        self.system.eval()
        del ckpt

        self.trainer_config = dict(task.get("trainer", {}))
        self.writer_config = dict(task.get("writer", {}))

    def predict(self, npz_files: list, npz_dir: str, output_fbx: str,
                data_name: str | None = None):
        """npz(extract 산출) → FBX. Trainer/writer 는 호출당 새로 만들지만 모델은 재사용."""
        writer_config = dict(self.writer_config)
        writer_config["npz_dir"] = npz_dir
        writer_config["output_dir"] = None
        writer_config["output_name"] = output_fbx
        writer_config["user_mode"] = True
        writer = get_writer(**writer_config,
                            order_config=self.predict_transform_config.order_config)

        data = UniRigDatasetModule(
            process_fn=self.model._process_fn,
            predict_dataset_config=self.predict_dataset_config,
            predict_transform_config=self.predict_transform_config,
            tokenizer_config=self.tokenizer_config,
            debug=False,
            data_name=data_name or self.data_name,
            datapath=Datapath(files=npz_files, cls=None),
            cls=None,
        )
        trainer = L.Trainer(callbacks=[writer], logger=False,
                            enable_progress_bar=False, enable_model_summary=False,
                            **self.trainer_config)
        trainer.predict(self.system, datamodule=data, ckpt_path=None,
                        return_predictions=False)


class UniRigService:
    """스켈레톤+스킨 상주. rig() 호출당: extract 1회 → 스켈레톤 → (선택) 스킨."""

    def __init__(self):
        t0 = time.time()
        self.skeleton = _Stage(SKELETON_TASK)
        self.skin = _Stage(SKIN_TASK)
        self.load_sec = time.time() - t0

    def rig(self, input_path: str, output_dir: str, skin: bool = True,
            seed: int = 12345) -> dict:
        input_path = str(Path(input_path).resolve())
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem
        npz_root = str(out / "npz")
        skeleton_fbx = str(out / f"{stem}_skeleton.fbx")
        rigged_fbx = str(out / f"{stem}_rigged.fbx")
        r = {"input": input_path, "skeleton_fbx": None, "rigged_fbx": None,
             "extract_sec": None, "skeleton_sec": None, "skin_sec": None, "error": None}
        try:
            L.seed_everything(seed, workers=True)

            t0 = time.time()                                   # ① extract — 에셋당 1회
            files = get_files(data_name="raw_data.npz", inputs=input_path,
                              input_dataset_dir=None, output_dataset_dir=npz_root,
                              require_suffix=REQUIRE_SUFFIX,
                              force_override=True, warning=False)
            extract_builtin(output_folder=npz_root, target_count=FACES_TARGET,
                            num_runs=1, id=0,
                            time=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
                            files=files)
            npz_files = [f[1] for f in files]
            r["extract_sec"] = round(time.time() - t0, 1)

            t1 = time.time()                                   # ② 스켈레톤 (+predict_skeleton.npz)
            self.skeleton.predict(npz_files, npz_root, skeleton_fbx)
            if not Path(skeleton_fbx).exists():
                raise RuntimeError("skeleton FBX 미생성")
            r["skeleton_fbx"] = skeleton_fbx
            r["skeleton_sec"] = round(time.time() - t1, 1)

            if skin:                                           # ③ 스킨 (extract 재실행 없음)
                t2 = time.time()
                self.skin.predict(npz_files, npz_root, rigged_fbx,
                                  data_name="predict_skeleton.npz")
                if not Path(rigged_fbx).exists():
                    raise RuntimeError("rigged FBX 미생성")
                r["rigged_fbx"] = rigged_fbx
                r["skin_sec"] = round(time.time() - t2, 1)
        except Exception as e:  # 한 에셋 실패가 서비스를 죽이면 안 됨
            r["error"] = repr(e)[:300]
        finally:
            torch.cuda.empty_cache()
        return r


def main():
    ap = argparse.ArgumentParser(description="UniRig 상주 리깅 서비스")
    ap.add_argument("inputs", nargs="*", help="입력 메시 (glb/obj/fbx/...)")
    ap.add_argument("--out-dir", default="./rigs")
    ap.add_argument("--no-skin", action="store_true", help="스켈레톤만")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--daemon", action="store_true",
                    help="stdin JSON 라인 모드 (개발 툴 연동)")
    args = ap.parse_args()
    # cwd 가 UNIRIG_DIR 로 바뀌기 전 기준의 상대경로 입력을 위해 원 cwd 로 절대화
    orig_cwd = os.environ.get("PWD", os.getcwd())

    def absolutize(p):
        return p if os.path.isabs(p) else os.path.join(orig_cwd, p)

    svc = UniRigService()
    print(f"[rig_service] 모델 로드 {svc.load_sec:.1f}s (스켈레톤+스킨, 이후 상주)",
          file=sys.stderr, flush=True)

    if args.daemon:
        print(MARK + json.dumps({"ready": True, "load_sec": round(svc.load_sec, 1)}),
              flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                res = svc.rig(absolutize(req["input"]),
                              absolutize(req.get("output_dir", "./rigs")),
                              skin=req.get("skin", True),
                              seed=req.get("seed", args.seed))
            except Exception as e:
                res = {"error": repr(e)[:300]}
            print(MARK + json.dumps(res, ensure_ascii=False), flush=True)
        return

    if not args.inputs:
        ap.error("입력 메시를 주거나 --daemon 을 켜라")
    total0 = time.time()
    for p in args.inputs:
        res = svc.rig(absolutize(p), absolutize(args.out_dir),
                      skin=not args.no_skin, seed=args.seed)
        print(json.dumps(res, ensure_ascii=False), flush=True)
    n = len(args.inputs)
    print(f"[rig_service] {n}건 완료, 로드 {svc.load_sec:.1f}s + "
          f"처리 {time.time()-total0:.1f}s ({(time.time()-total0)/n:.1f}s/에셋)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
