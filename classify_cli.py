import sys
from pathlib import Path

from model import ResumeSeniorityClassifier


def main() -> int:
    """
    Main classification function.

    Returns:
        Exit code (0 - success, 1 - error)
    """
    if len(sys.argv) != 3:
        print(
            f"Usage: python {sys.argv[0]} path/to/x_data.npy path/to/y_data.npy",
            file=sys.stderr,
        )
        return 1

    x_path = Path(sys.argv[1])
    y_path = Path(sys.argv[2])

    if not x_path.exists():
        print(f"Error: File not found: {x_path}", file=sys.stderr)
        return 1

    if not y_path.exists():
        print(f"Error: File not found: {y_path}", file=sys.stderr)
        return 1

    if x_path.suffix.lower() != ".npy":
        print(f"Error: Expected .npy file, got: {x_path.suffix}", file=sys.stderr)
        return 1

    if y_path.suffix.lower() != ".npy":
        print(f"Error: Expected .npy file, got: {y_path.suffix}", file=sys.stderr)
        return 1

    try:
        classifier = ResumeSeniorityClassifier(str(x_path), str(y_path))
        classifier.run()
        return 0
    except Exception as e:
        print(f"Error during classification: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
