"""Convert local demonstrations to state_dict observation mode (pd_ee_delta_pos) unpadded and
upload to HF Hub."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "nMaax/crispy-waffle-demos"
DEMO_ROOT = Path.home() / ".maniskill" / "demos"
api = HfApi()


def replay_unpadded(task_id: str, in_filename: str, out_backend: str, num_envs: int = 12):
    print("\n=======================================================")
    print(f"Replaying: {task_id}/{in_filename} -> {out_backend} (unpadded)")
    print("=======================================================")
    in_path = DEMO_ROOT / task_id / "motionplanning" / in_filename
    if not in_path.exists():
        print(f"Warning: {in_path} does not exist, skipping.")
        return

    cmd = [
        sys.executable,
        "-m",
        "mani_skill.trajectory.replay_trajectory",
        "--traj-path",
        str(in_path),
        "--obs-mode",
        "state_dict",
        "--save-traj",
        "--allow-failure",
        "--sim-backend",
        "physx_cpu",
        "--use-env-states",
        "--num-envs",
        str(num_envs),
    ]
    subprocess.run(cmd, check=True)

    base_cpu_h5 = (
        DEMO_ROOT
        / task_id
        / "motionplanning"
        / in_filename.replace(".state.", ".state_dict.").replace(".physx_cuda.", ".physx_cpu.")
    )
    base_cpu_json = (
        DEMO_ROOT
        / task_id
        / "motionplanning"
        / in_filename.replace(".state.", ".state_dict.")
        .replace(".physx_cuda.", ".physx_cpu.")
        .replace(".h5", ".json")
    )

    target_h5 = (
        DEMO_ROOT / task_id / "motionplanning" / in_filename.replace(".state.", ".state_dict.")
    )
    target_json = (
        DEMO_ROOT
        / task_id
        / "motionplanning"
        / in_filename.replace(".state.", ".state_dict.").replace(".h5", ".json")
    )

    if out_backend == "physx_cuda":
        shutil.copy2(base_cpu_h5, target_h5)
        with open(base_cpu_json) as f:
            meta = json.load(f)
        meta["env_info"]["env_kwargs"]["sim_backend"] = "physx_cuda"
        with open(target_json, "w") as f:
            json.dump(meta, f, indent=2)

    for f in [target_h5, target_json]:
        rel = f.relative_to(DEMO_ROOT).as_posix()
        print(f"Uploading '{rel}' to '{REPO_ID}'...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=rel,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Upload {rel}",
        )


def main():
    targets = [
        ("StackCube-v1", "trajectory.state.pd_ee_delta_pos.physx_cpu.h5", "physx_cpu"),
        ("StackCube-v1", "trajectory.state.pd_ee_delta_pos.physx_cuda.h5", "physx_cuda"),
        (
            "StackCubeLockedRotation-v1",
            "trajectory.state.pd_ee_delta_pos.physx_cpu.h5",
            "physx_cpu",
        ),
        (
            "StackCubeLockedRotation-v1",
            "trajectory.state.pd_ee_delta_pos.physx_cuda.h5",
            "physx_cuda",
        ),
    ]

    for task_id, in_filename, backend in targets:
        replay_unpadded(task_id, in_filename, backend)

    print("\nAll datasets replayed and synced successfully!")


if __name__ == "__main__":
    main()
