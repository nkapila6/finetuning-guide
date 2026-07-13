import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ftguide - hands-on finetuning guide for small models"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Run training (placeholder - use the notebook)")
    sub.add_parser("curate", help="Run data curation (placeholder - use the notebook)")
    sub.add_parser("info", help="Show package info")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "info":
        print("ftguide v0.1.0 - hands-on finetuning guide for small models")
        print(
            "Use the Jupyter notebook (notebooks/finetuning_guide.ipynb) for the full walkthrough."
        )
        return 0

    print(f"Command '{args.command}' is a placeholder.")
    print(
        "Use the Jupyter notebook (notebooks/finetuning_guide.ipynb) for the full walkthrough."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
