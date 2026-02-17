import sys
from pathlib import Path

from pipeline import ResumeDatasetPipeline


def main() -> int:
    """
    Main pipeline function.

    Returns:
        Exit code (0 - success, 1 - error)
    """
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} path/to/hh.csv", file=sys.stderr)
        return 1

    csv_path = Path(sys.argv[1])

    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        return 1

    if csv_path.suffix.lower() != ".csv":
        print(f"Error: Expected .csv file, got: {csv_path.suffix}", file=sys.stderr)
        return 1

    try:
        pipeline = ResumeDatasetPipeline(str(csv_path))
        pipeline.run()
        return 0
    except Exception as e:
        print(f"Error during pipeline execution: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
