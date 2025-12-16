#!/usr/bin/env python3
"""
Full evaluation pipeline: Baseline → Enhancement → Report
"""
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str):
    """Run a Python script and capture output"""
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def main():
    print("🚀 FULL RAG EVALUATION PIPELINE")
    print("="*60)
    
    scripts_dir = Path(__file__).parent
    
    # Step 1: Create evaluation dataset
    if not Path("data/eval_dataset.json").exists():
        print("📝 Creating evaluation dataset...")
        create_script = scripts_dir / "create_eval_dataset.py"
        if create_script.exists():
            run_script(str(create_script))
        else:
            print("⚠️  Create eval dataset script not found")
    
    # Step 2: Run baseline evaluation
    print("\n" + "="*60)
    print("📊 STEP 1: BASELINE EVALUATION")
    run_script(str(scripts_dir / "run_baseline.py"))
    
    # Step 3: Run enhanced evaluation
    print("\n" + "="*60)
    print("🚀 STEP 2: ENHANCED EVALUATION")
    run_script(str(scripts_dir / "run_enhanced.py"))
    
    # Step 4: Generate report
    print("\n" + "="*60)
    print("📋 STEP 3: GENERATE REPORT")
    run_script(str(scripts_dir / "generate_report.py"))
    
    print("\n" + "="*60)
    print("✅ EVALUATION PIPELINE COMPLETE!")
    print("📁 Check docs/enhancement_report.md for results")
    print("="*60)


if __name__ == "__main__":
    main()