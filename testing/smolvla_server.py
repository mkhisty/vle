"""Serve the harness's SmolVLA LIBERO policy from the shared workspace venv."""

import argparse
import logging
from pathlib import Path

from vla_eval.model_servers.lerobot import LeRobotModelServer
from vla_eval.model_servers.serve import serve
from vla_eval.specs import STATE_EEF_POS_AA_GRIP, DimSpec


class SmolVLALiberoServer(LeRobotModelServer):
    def get_observation_spec(self) -> dict[str, DimSpec]:
        spec = super().get_observation_spec()
        spec["state"] = STATE_EEF_POS_AA_GRIP
        return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="lerobot/smolvla_libero")
    parser.add_argument("--revision", default="a24f1f38")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    logging.basicConfig(level=logging.INFO)
    from huggingface_hub import snapshot_download

    checkpoint = args.checkpoint
    if not Path(checkpoint).is_dir():
        checkpoint = snapshot_download(repo_id=checkpoint, revision=args.revision)
    server = SmolVLALiberoServer(
        policy_type="smolvla",
        checkpoint=checkpoint,
        image_keys={
            "agentview": "observation.images.camera1",
            "wrist": "observation.images.camera2",
        },
        state_key="observation.state",
        device=args.device,
        chunk_size=args.chunk_size,
        quat_no_antipodal=True,
        max_rows=1,
    )
    serve(server, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
