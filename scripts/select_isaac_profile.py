"""Select the native profile for a new declared single-block reference."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--campaign-cases", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--policy-source-run")
    parser.add_argument("--policy-socket")
    args, _ = parser.parse_known_args()
    selected = "isaacsim_physx"
    if args.campaign_cases and args.case and not (args.policy_source_run or args.policy_socket):
        try:
            declaration = json.loads(args.campaign_cases.read_text())
            matches = [
                entry["row"] for entry in declaration["cases"] if entry["row"]["case"] == args.case
            ]
        except (OSError, ValueError, KeyError, TypeError):
            # The normal task parser will report malformed inputs.
            matches = []
        if len(matches) == 1 and matches[0]["category"] == "single_block":
            selected = "newton_steel"
    print(selected)


if __name__ == "__main__":
    main()
