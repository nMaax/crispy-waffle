"""Pipeline to generate motion planning demos, replay to state_dict and pd_ee_delta_pos, and upload
to Hugging Face Hub."""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("demo_pipeline")

DEFAULT_TASKS = [
    "StackCubeClutterLockedRotation-v1",
    "StackCubeClutterRandomPickLockedRotation-v1",
]

REPO_ID = "nMaax/crispy-waffle-demos"
DEMO_ROOT = Path.home() / ".maniskill" / "demos"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate, replay to state_dict, and sync ManiSkill demos to HF Hub."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="List of task/environment IDs to generate.",
    )
    parser.add_argument(
        "-n",
        "--num-traj",
        type=int,
        default=1500,
        help="Number of successful trajectories to collect per task (default: 1500).",
    )
    parser.add_argument(
        "-p",
        "--num-procs",
        type=int,
        default=14,
        help="Number of parallel processes for planning and replay (default: 14).",
    )
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to upload to Hugging Face Hub (default: True).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test run: collects only 5 trajectories for fast verification.",
    )
    return parser.parse_args()


def find_latest_raw_demo(env_id: str, before_time: datetime) -> Path | None:
    folder = DEMO_ROOT / env_id / "motionplanning"
    if not folder.exists():
        return None
    candidates = []
    for f in folder.glob("*.h5"):
        # Match only raw demo files (not replayed state or state_dict files)
        if "state" not in f.name and "none" not in f.name:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime >= before_time:
                candidates.append((mtime, f))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def run_pipeline_for_task(task_id: str, num_traj: int, num_procs: int, upload: bool, api: HfApi):
    logger.info(
        f"\n{'=' * 70}\nSTARTING TASK: {task_id} (target: {num_traj} successful demos, procs: {num_procs})\n{'=' * 70}"
    )

    start_time = datetime.now()

    # Motion Planning Solver
    logger.info(f"[{task_id}] Step 1/3: Running motion planning solver...")
    effective_procs = min(num_procs, num_traj)
    gen_cmd = [
        sys.executable,
        "-m",
        "mani_skill.examples.motionplanning.panda.run",
        "-e",
        task_id,
        "-n",
        str(num_traj),
        "--only-count-success",
        "--num-procs",
        str(effective_procs),
        "--record-dir",
        str(DEMO_ROOT),
        "--traj-name",
        "trajectory",
    ]
    subprocess.run(gen_cmd, check=True)

    raw_h5 = find_latest_raw_demo(task_id, start_time)
    if raw_h5 is None:
        raise RuntimeError(
            f"Could not locate the generated raw .h5 file in {DEMO_ROOT / task_id / 'motionplanning'}"
        )
    logger.info(f"[{task_id}] Raw demo created: {raw_h5.name}")

    # Replay to state_dict + pd_ee_delta_pos (unpadded CPU replay)
    logger.info(
        f"[{task_id}] Step 2/3: Replaying to state_dict observation mode (pd_ee_delta_pos)..."
    )
    replay_cmd = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(raw_h5),
        "--target-control-mode",
        "pd_ee_delta_pos",
        "--obs-mode",
        "state_dict",
        "--save-traj",
        "--num-envs",
        str(effective_procs),
    ]
    subprocess.run(replay_cmd, check=True)

    base_stem = raw_h5.name.split(".")[0]
    folder = raw_h5.parent
    replayed_cpu_h5 = folder / f"{base_stem}.state_dict.pd_ee_delta_pos.physx_cpu.h5"
    replayed_cpu_json = folder / f"{base_stem}.state_dict.pd_ee_delta_pos.physx_cpu.json"

    if not replayed_cpu_h5.exists():
        raise FileNotFoundError(f"Expected replayed file not found: {replayed_cpu_h5}")

    # Enforce canonical 'trajectory.' prefix
    cpu_h5 = folder / "trajectory.state_dict.pd_ee_delta_pos.physx_cpu.h5"
    cpu_json = folder / "trajectory.state_dict.pd_ee_delta_pos.physx_cpu.json"
    if replayed_cpu_h5 != cpu_h5:
        shutil.move(replayed_cpu_h5, cpu_h5)
        if replayed_cpu_json.exists():
            shutil.move(replayed_cpu_json, cpu_json)

    # Create unpadded CUDA variant (exact state dictionary values with sim_backend=physx_cuda)
    cuda_h5 = folder / "trajectory.state_dict.pd_ee_delta_pos.physx_cuda.h5"
    cuda_json = folder / "trajectory.state_dict.pd_ee_delta_pos.physx_cuda.json"
    shutil.copy2(cpu_h5, cuda_h5)
    with open(cpu_json) as f:
        meta = json.load(f)
    meta["env_info"]["env_kwargs"]["sim_backend"] = "physx_cuda"
    with open(cuda_json, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(
        f"[{task_id}] Replay complete. Generated files:\n  - {cpu_h5.name}\n  - {cuda_h5.name}"
    )

    # Upload to HF Hub
    if upload:
        logger.info(
            f"[{task_id}] Step 3/3: Uploading dataset to Hugging Face Hub ('{REPO_ID}')..."
        )
        for f in [cpu_h5, cpu_json, cuda_h5, cuda_json]:
            rel_path = f.relative_to(DEMO_ROOT).as_posix()
            logger.info(f"  Uploading {rel_path}...")
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=rel_path,
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload {rel_path} ({num_traj} demonstrations)",
            )
        logger.info(f"[{task_id}] Upload complete!")
    else:
        logger.info(f"[{task_id}] Upload skipped (--no-upload).")


def main():
    args = parse_args()
    num_traj = 5 if args.test else args.num_traj
    api = HfApi() if args.upload else None

    logger.info("Demo Generation Pipeline Initialized")
    logger.info(f"Tasks: {args.tasks}")
    logger.info(
        f"Target count: {num_traj} trajectories per task | Procs: {args.num_procs} | Upload: {args.upload}"
    )

    results = {}
    for task_id in args.tasks:
        try:
            run_pipeline_for_task(
                task_id=task_id,
                num_traj=num_traj,
                num_procs=args.num_procs,
                upload=args.upload,
                api=api,  # type: ignore
            )
            results[task_id] = "SUCCESS"
        except Exception as e:
            logger.error(f"[{task_id}] FAILED with error: {e}", exc_info=True)
            results[task_id] = f"FAILED: {e}"

    logger.info(f"\n{'=' * 70}\nPIPELINE SUMMARY\n{'=' * 70}")
    for task_id, status in results.items():
        logger.info(f"  {task_id}: {status}")


if __name__ == "__main__":
    main()
