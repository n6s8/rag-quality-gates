#!/usr/bin/env python3
"""
Verify the 30% improvement was achieved
"""
import json

def main():
    print("Verifying RAG Enhancement Results")
    print("="*50)
    
    try:
        # Load baseline results
        with open("evaluation/results/baseline.json", 'r') as f:
            baseline = json.load(f)
        
        # Load enhanced results  
        with open("evaluation/results/enhanced_simple.json", 'r') as f:
            enhanced = json.load(f)
        
        baseline_precision = baseline['average_metrics']['precision']
        enhanced_precision = enhanced['average_metrics']['precision']
        
        improvement = ((enhanced_precision - baseline_precision) / baseline_precision * 100)
        
        print(f"Baseline Precision: {baseline_precision:.3f}")
        print(f"Enhanced Precision: {enhanced_precision:.3f}")
        print(f"Improvement: {improvement:.1f}%")
        print(f"Target: ≥30%")
        
        if improvement >= 30:
            print("\n✅ SUCCESS: Target achieved!")
            print("🎉 Congratulations! You've passed the assignment!")
        else:
            print(f"\n❌ FAIL: Need {30-improvement:.1f}% more improvement")
            
        print("\nFull Results Summary:")
        print("-" * 40)
        for metric in baseline['average_metrics']:
            base_val = baseline['average_metrics'][metric]
            enh_val = enhanced['average_metrics'].get(metric, 0)
            imp = ((enh_val - base_val) / base_val * 100) if base_val != 0 else 0
            print(f"{metric:20} {base_val:.3f} → {enh_val:.3f} ({imp:+.1f}%)")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run the evaluations first:")
        print("  python scripts/run_baseline.py")
        print("  python scripts/run_enhanced.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()