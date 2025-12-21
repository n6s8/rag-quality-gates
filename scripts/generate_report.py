import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


CORE_METRICS = [
    ("precision", "Retrieval Precision"),
    ("recall", "Retrieval Recall"),
    ("relevance", "Answer Relevance"),
    ("hallucination", "Hallucination Score"),
    ("response_time", "Response Time (s)"),
]

ANALYSIS_METRICS = [
    ("interpretation_score", "Interpretation Score"),
    ("historical_context_score", "Historical Context Score"),
    ("explanation_depth", "Explanation Depth"),
    ("thematic_analysis", "Thematic Analysis"),
    ("interpretation_quality", "Interpretation Quality"),
]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_avg_metrics(obj: Dict[str, Any]) -> Dict[str, float]:
    avg = obj.get("average_metrics") or {}
    out: Dict[str, float] = {}
    for k, v in avg.items():
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def _get_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = obj.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _pct_change(baseline: Optional[float], enhanced: Optional[float]) -> Optional[float]:
    if baseline is None or enhanced is None:
        return None
    if baseline == 0:
        return None
    return (enhanced - baseline) / baseline * 100.0


def _fmt_float(x: Optional[float], digits: int = 3) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def _fmt_pct(x: Optional[float], digits: int = 1, signed: bool = True) -> str:
    if x is None:
        return "N/A"
    sign = "+" if (signed and x >= 0) else ""
    return f"{sign}{x:.{digits}f}%"


def _md_table(rows):
    header = "| Metric | Baseline | Enhanced | Improvement |\n|--------|----------|----------|-------------|\n"
    body = ""
    for r in rows:
        body += f"| {r['label']} | {r['baseline']} | {r['enhanced']} | {r['improvement']} |\n"
    return header + body


def build_report(
    baseline_path: Path,
    enhanced_path: Path,
    output_path: Path,
    target_metric_key: str = "precision",
    threshold_pct: float = 30.0,
) -> str:
    baseline_obj = _load_json(baseline_path)
    enhanced_obj = _load_json(enhanced_path)

    baseline_avg = _get_avg_metrics(baseline_obj)
    enhanced_avg = _get_avg_metrics(enhanced_obj)

    baseline_meta = _get_meta(baseline_obj)
    enhanced_meta = _get_meta(enhanced_obj)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    baseline_n = baseline_obj.get("test_queries") or baseline_obj.get("metadata", {}).get("test_queries") or "N/A"
    enhanced_n = enhanced_obj.get("test_queries") or enhanced_obj.get("metadata", {}).get("test_queries") or "N/A"

    enhanced_top_k = enhanced_meta.get("top_k", "N/A")
    baseline_top_k = baseline_meta.get("top_k", "N/A")

    target_baseline = baseline_avg.get(target_metric_key)
    target_enhanced = enhanced_avg.get(target_metric_key)
    target_impr = _pct_change(target_baseline, target_enhanced)

    status = "SUCCESS" if (target_impr is not None and target_impr >= threshold_pct) else "FAIL"

    core_rows = []
    for key, label in CORE_METRICS:
        b = baseline_avg.get(key)
        e = enhanced_avg.get(key)
        impr = _pct_change(b, e)
        core_rows.append(
            {
                "label": label,
                "baseline": _fmt_float(b),
                "enhanced": _fmt_float(e),
                "improvement": _fmt_pct(impr),
            }
        )

    analysis_rows = []
    for key, label in ANALYSIS_METRICS:
        if key in baseline_avg or key in enhanced_avg:
            b = baseline_avg.get(key)
            e = enhanced_avg.get(key)
            impr = _pct_change(b, e)
            analysis_rows.append(
                {
                    "label": label,
                    "baseline": _fmt_float(b),
                    "enhanced": _fmt_float(e),
                    "improvement": _fmt_pct(impr),
                }
            )

    target_label = dict(CORE_METRICS).get(target_metric_key, target_metric_key)

    report = []
    report.append("# RAG Enhancement Evaluation Report\n")
    report.append(f"**Generated on:** {now}\n")

    report.append("## Executive Summary\n")
    report.append(
        "This report compares the **baseline** RAG system against an **enhanced** version.\n\n"
        "The evaluation is fully automated: the pipeline generates JSON artifacts (baseline/enhanced), "
        "and this script builds the Markdown report from those artifacts.\n"
    )

    report.append("## Key Findings\n")
    report.append(f"- Primary target metric: **{target_label}**\n")
    report.append(f"- Improvement: **{_fmt_pct(target_impr)}** (requirement: >= **{threshold_pct:.1f}%**)\n")
    report.append(f"- Status: **{status}**\n")
    report.append("- Additional quality metrics (interpretation/context) included when available.\n")

    report.append("\n## Test Setup\n")
    report.append(f"- Baseline results file: `{baseline_path.as_posix()}`\n")
    report.append(f"- Enhanced results file: `{enhanced_path.as_posix()}`\n")
    report.append(f"- Number of test queries (baseline): {baseline_n}\n")
    report.append(f"- Number of test queries (enhanced): {enhanced_n}\n")
    report.append(f"- top_k (baseline): {baseline_top_k}\n")
    report.append(f"- top_k (enhanced): {enhanced_top_k}\n")

    report.append("\n## Metric Comparison (Core)\n\n")
    report.append(_md_table(core_rows))

    report.append("\n## Target Metric Analysis\n\n")
    report.append(f"### {target_label} Improvement: {_fmt_pct(target_impr)}\n\n")
    report.append(f"**Requirement:** >= {_fmt_pct(threshold_pct, signed=False)} improvement for minimal pass score\n\n")
    report.append(f"**Status:** **{status}**\n\n")
    report.append("**Values:**\n")
    report.append(f"- Baseline {target_label}: {_fmt_float(target_baseline)}\n")
    report.append(f"- Enhanced {target_label}: {_fmt_float(target_enhanced)}\n")
    if target_baseline is not None and target_enhanced is not None:
        report.append(f"- Absolute difference: {_fmt_float(target_enhanced - target_baseline)}\n")

    if analysis_rows:
        report.append("\n## Metric Comparison (Interpretation and Context Quality)\n\n")
        report.append(_md_table(analysis_rows))

    report.append("\n## Enhancement Summary (What Changed)\n\n")
    report.append(
        "Describe your enhancement here (manually) in terms of:\n"
        "- what was changed in retrieval (e.g., intent-aware retrieval, hybrid retrieval, re-ranking)\n"
        "- how prompts/context formatting changed\n"
        "- why this should improve the chosen target metric\n"
    )

    report.append("\n## Notes on Validity and Limitations\n\n")
    report.append(
        "- Small evaluation sets can introduce variance; interpret improvements with this in mind.\n"
        "- Some analysis-quality metrics can be heuristic; they are useful for regression testing but do not fully replace human judgment.\n"
        "- Track trade-offs: improvements in precision can sometimes affect recall, relevance, or hallucination score.\n"
    )

    report.append("\n## Files Produced by the Evaluation Pipeline\n\n")
    report.append(f"- Baseline: `{baseline_path.as_posix()}`\n")
    report.append(f"- Enhanced: `{enhanced_path.as_posix()}`\n")
    report.append(f"- Report: `{output_path.as_posix()}`\n")

    report.append("\n## Next Steps\n\n")
    report.append(
        "1. Expand evaluation coverage (more queries, especially for interpretation/historical context).\n"
        "2. Consider adding a second-stage re-ranker or hybrid retrieval (vector + metadata) for harder queries.\n"
        "3. Improve hallucination guardrails (force answers to stay inside retrieved context, add refusal policy when context is missing).\n"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(report), encoding="utf-8")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="evaluation/results/baseline.json")
    parser.add_argument("--enhanced", default="evaluation/results/enhanced_simple.json")
    parser.add_argument("--out", default="docs/enhancement_report.md")
    parser.add_argument("--target-metric", default="precision")
    parser.add_argument("--threshold", type=float, default=30.0)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    enhanced_path = Path(args.enhanced)
    output_path = Path(args.out)

    report_path = build_report(
        baseline_path=baseline_path,
        enhanced_path=enhanced_path,
        output_path=output_path,
        target_metric_key=args.target_metric,
        threshold_pct=args.threshold,
    )

    print(f"Report generated: {report_path}")

    baseline_obj = _load_json(baseline_path)
    enhanced_obj = _load_json(enhanced_path)

    baseline_avg = _get_avg_metrics(baseline_obj)
    enhanced_avg = _get_avg_metrics(enhanced_obj)

    b = baseline_avg.get(args.target_metric)
    e = enhanced_avg.get(args.target_metric)
    impr = _pct_change(b, e)

    status = (
        f"SUCCESS (>= {args.threshold:.1f}%)"
        if (impr is not None and impr >= args.threshold)
        else f"FAIL (< {args.threshold:.1f}%)"
    )

    print("\nREPORT SUMMARY:")
    print(f"   {args.target_metric} Improvement: {_fmt_pct(impr)}")
    print(f"   Status: {status}")


if __name__ == "__main__":
    main()
