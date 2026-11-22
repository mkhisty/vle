"""Generate a saved instruction set for the first LIBERO-Goal task."""

import argparse
import copy
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--null", action="store_true", help="Repeat the input without importing ERT"
    )
    parser.add_argument(
        "--instruction", default="open the middle drawer of the cabinet"
    )
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(os.environ["ERT_OUTPUT_DIR"])
    )
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")

    if args.null:
        instructions = [args.instruction] * args.count
    else:
        from generate_instructions import EmbodiedRedTeamModel

        ert = EmbodiedRedTeamModel(embedding_model=None, num_rejection_samples=1)
        instructions = ert(args.instruction, num_instructions=args.count)

    if len(instructions) != args.count or not all(
        isinstance(text, str) for text in instructions
    ):
        raise ValueError(
            "Generator must return the requested number of instruction strings"
        )
    import yaml
    from vla_eval.cli.config_loader import load_config

    config = load_config(
        str(
            Path(os.environ["VLA_EVAL_DIR"])
            / "configs/benchmarks/libero/goal_instruction_pilot.yaml"
        )
    )
    template = config["benchmarks"][0]
    config["benchmarks"] = []
    for index, instruction in enumerate(instructions):
        benchmark = copy.deepcopy(template)
        benchmark["subname"] = f"instruction_{index:03d}"
        benchmark["params"]["instruction_override"] = instruction
        config["benchmarks"].append(benchmark)
    config["output_dir"] = str(args.output_dir.resolve() / "evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "instructions.json").write_text(
        json.dumps(instructions, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "eval.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(f"Saved {len(instructions)} instructions and eval.yaml in {args.output_dir}")


if __name__ == "__main__":
    main()
