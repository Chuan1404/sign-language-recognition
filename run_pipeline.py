"""Run pose training -> frame extraction -> RGB training -> multimodal fine-tuning."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
STAGES = ("pose", "extract", "rgb", "multimodal")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", help="Checkpoints, RGB cache and logs for this run")
    p.add_argument("--data_path", default=str(PROJECT / "datasets/processed/wlasl_features_v2"))
    p.add_argument("--label_path", default=str(PROJECT / "datasets/annotations/WLASL100"))
    p.add_argument("--video_dir", default=str(PROJECT / "datasets/raw/WLASL/videos"))
    p.add_argument("--val_mode", default="test")
    p.add_argument("--pose_epochs", type=int, default=100)
    p.add_argument("--rgb_epochs", type=int, default=100)
    p.add_argument("--multimodal_epochs", type=int, default=30)
    p.add_argument("--pose_batch_size", type=int, default=8)
    p.add_argument("--rgb_batch_size", type=int, default=4)
    p.add_argument("--multimodal_batch_size", type=int, default=4)
    p.add_argument("--pose_lr", type=float, default=1e-4)
    p.add_argument("--rgb_lr", type=float, default=1e-4)
    p.add_argument("--multimodal_lr", type=float, default=1e-5)
    p.add_argument("--multimodal_pose_lr", type=float, default=1e-6)
    p.add_argument("--pose_weight", type=float, default=0.5)
    p.add_argument("--finetune_pose", action="store_true")
    p.add_argument("--fixed_k", type=int, default=10, help="Number of important anchors before gap filling; 0 uses score threshold")
    p.add_argument("--max_frames", type=int, default=100)
    p.add_argument("--max_frame_gap", type=int, default=3)
    p.add_argument("--image_size", type=int, default=224, help="RGB cache resolution; training dataset resizes to 112")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_at", choices=STAGES, default="pose", help="Run this stage and all subsequent stages, using earlier artifacts in run_dir")
    p.add_argument("--dry_run", action="store_true", help="Print commands without training or writing files")
    args = p.parse_args(argv)
    if args.start_at != "pose" and not args.run_dir:
        p.error("--start_at requires an existing --run_dir")
    positive = (args.pose_epochs, args.rgb_epochs, args.multimodal_epochs,
                args.pose_batch_size, args.rgb_batch_size, args.multimodal_batch_size,
                args.max_frames, args.image_size, args.pose_lr, args.rgb_lr,
                args.multimodal_lr, args.multimodal_pose_lr)
    if any(value <= 0 for value in positive):
        p.error("Epochs, batch sizes, frame budget, image size and learning rates must be positive")
    if args.fixed_k < 0 or args.fixed_k > args.max_frames or args.max_frame_gap < 0:
        p.error("Require 0 <= fixed_k <= max_frames and max_frame_gap >= 0")
    if not 0 < args.pose_weight < 1:
        p.error("Require 0 < pose_weight < 1")
    if args.run_dir is None:
        args.run_dir = str(PROJECT / "outputs/pipeline" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    for key in ("run_dir", "data_path", "label_path", "video_dir"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    return args


def build_plan(args):
    run_dir = Path(args.run_dir)
    pose = run_dir / "pose_best.pt"
    rgb = run_dir / "rgb_best.pt"
    multimodal = run_dir / "multimodal_best.pt"
    cache = run_dir / "selected_rgb"
    shared = ["--data_path", args.data_path, "--label_path", args.label_path]
    validation = ["--val_mode", args.val_mode]
    def command(script, options):
        return [sys.executable, "-u", str(PROJECT / script), *map(str, options)]
    splits = list(dict.fromkeys(["train", args.val_mode, "test"]))
    plan = [
        ("pose", command("train_selector.py", shared + validation + [
            "--output", pose, "--epochs", args.pose_epochs,
            "--batch_size", args.pose_batch_size, "--lr", args.pose_lr, "--seed", args.seed]), pose),
        ("extract", command("src/extract_features/extract_important_frame.py", shared + [
            "--checkpoint", pose, "--video_dir", args.video_dir, "--out_dir", cache,
            "--batch_size", args.pose_batch_size, "--fixed_k", args.fixed_k,
            "--max_frames", args.max_frames, "--max_frame_gap", args.max_frame_gap,
            "--image_size", args.image_size, "--crop", "hands", "--seed", args.seed,
            "--splits", *splits]), cache),
        ("rgb", command("main_train.py", shared + validation + [
            "--rgb_dir", cache, "--output", rgb, "--epochs", args.rgb_epochs,
            "--batch_size", args.rgb_batch_size, "--lr", args.rgb_lr]), rgb),
        ("multimodal", command("main_multimodal.py", shared + validation + [
            "--rgb_dir", cache, "--pose_checkpoint", pose, "--rgb_checkpoint", rgb,
            "--output", multimodal, "--epochs", args.multimodal_epochs,
            "--batch_size", args.multimodal_batch_size, "--lr", args.multimodal_lr,
            "--pose_lr", args.multimodal_pose_lr, "--pose_weight", args.pose_weight,
            "--seed", args.seed] + (["--finetune_pose"] if args.finetune_pose else [])), multimodal),
    ]
    return plan


def require_artifact(stage, path):
    if stage == "extract":
        if not path.is_dir() or not any(path.glob("*/rgb_frames.npy")):
            raise RuntimeError(f"No RGB cache produced: {path}")
    elif not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing checkpoint after {stage}: {path}")


def run_stage(stage, command, run_dir, env):
    print(f"\nRunning {stage}: {subprocess.list2cmdline(command)}", flush=True)
    with open(run_dir / f"{stage}.log", "w", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
    if code != 0:
        raise RuntimeError(f"Stage {stage} failed (exit {code}); see {run_dir / (stage + '.log')}")


def main(args):
    plan = build_plan(args)
    first = STAGES.index(args.start_at)
    print(f"Run directory: {args.run_dir}")
    if args.dry_run:
        for stage, command, _ in plan[first:]:
            print(f"[{stage}] {subprocess.list2cmdline(command)}")
        return
    for _, command, _ in plan:
        if not Path(command[2]).is_file():
            raise FileNotFoundError(command[2])
    for path in (args.data_path, args.label_path, args.video_dir):
        if not Path(path).is_dir():
            raise FileNotFoundError(path)
    for stage, _, artifact in plan[:first]:
        require_artifact(stage, artifact)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "pipeline_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join([str(PROJECT), str(PROJECT / "src"), env.get("PYTHONPATH", "")])
    for stage, command, artifact in plan[first:]:
        run_stage(stage, command, run_dir, env)
        require_artifact(stage, artifact)
    print(f"\nPipeline complete. Final checkpoint: {plan[-1][2]}")


if __name__ == "__main__":
    try:
        main(parse_args())
    except (RuntimeError, FileNotFoundError) as error:
        print(f"Pipeline stopped: {error}", file=sys.stderr)
        sys.exit(1)
