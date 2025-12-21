#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path


def run_step(script_path: Path) -> int:
    cmd = [sys.executable, str(script_path)]
    print("\n" + "=" * 70)
    print(f"▶ Running: {script_path.as_posix()}")
    print("=" * 70)
    proc = subprocess.run(cmd, cwd=str(project_root()), text=True)
    return proc.returncode


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def must_exist(path_str: str) -> bool:
    p = project_root() / path_str
    return p.exists()


def main():
    root = project_root()

    scripts_dir = root / "scripts"
    baseline_script = scripts_dir / "run_baseline.py"
    enhanced_script = scripts_dir / "run_enhanced.py"
    report_script = scripts_dir / "generate_report.py"

    for s in [baseline_script, enhanced_script, report_script]:
        if not s.exists():
            print(f"❌ Missing script: {s.as_posix()}")
            return

    (root / "evaluation" / "results").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)

    rc = run_step(baseline_script)
    if rc != 0:
        print("\n❌ Baseline step failed.")
        return

    if not must_exist("evaluation/results/baseline.json"):
        print("\n❌ baseline.json was not created. Expected: evaluation/results/baseline.json")
        return
    print("\n✅ Found: evaluation/results/baseline.json")

    rc = run_step(enhanced_script)
    if rc != 0:
        print("\n❌ Enhanced step failed.")
        return

    if not must_exist("evaluation/results/enhanced_simple.json"):
        print("\n❌ enhanced_simple.json was not created. Expected: evaluation/results/enhanced_simple.json")
        return
    print("\n✅ Found: evaluation/results/enhanced_simple.json")

    rc = run_step(report_script)
    if rc != 0:
        print("\n❌ Report generation step failed.")
        return

    if not must_exist("docs/enhancement_report.md"):
        print("\n❌ enhancement_report.md was not created. Expected: docs/enhancement_report.md")
        return
    print("\n✅ Found: docs/enhancement_report.md")

    # Final summary
    print("\n" + "=" * 70)
    print("✅ FULL EVALUATION PIPELINE COMPLETE")
    print("=" * 70)
    print("Artifacts:")
    print("  - evaluation/results/baseline.json")
    print("  - evaluation/results/enhanced_simple.json")
    print("  - docs/enhancement_report.md")
    print("\nRun from project root:")
    print("  python scripts/run_full_evaluation.py")


if __name__ == "__main__":
    main()
