"""Run the full Iris analysis pipeline from the command line."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iris_analysis.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Iris analysis pipeline.")
    parser.add_argument("--data", default="data/iris.csv", type=Path)
    parser.add_argument("--figures", default="outputs/figures", type=Path)
    parser.add_argument("--report", default="reports/paper.md", type=Path)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    args = parser.parse_args()

    result = run_pipeline(
        data_path=args.data,
        figure_dir=args.figures,
        report_path=args.report,
        random_state=args.random_state,
        test_size=args.test_size,
    )
    evaluation = result["evaluation"]
    print(f"Dataset saved to: {result['data_path']}")
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Macro F1: {evaluation['f1_macro']:.4f}")
    print(f"Report: {result['report_path']}")
    for name, path in result["figures"].items():
        print(f"Figure [{name}]: {path}")


if __name__ == "__main__":
    main()
