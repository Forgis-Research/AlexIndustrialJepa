#!/usr/bin/env python3
"""
validate_figure.py — Post-compile quality checker for TikZ figures.

Checks:
  1. PDF page dimensions (within NeurIPS column bounds)
  2. Companion .tex file: font sizes, color compliance, positioning
  3. Companion .log file: overfull/underfull boxes, warnings
  4. Overlap risk heuristic (node placement analysis)

Usage:
    python validate_figure.py <figure.pdf>
    python validate_figure.py <figure.pdf> --json    # machine-readable output
    python validate_figure.py <figure.pdf> --strict   # exit 1 on any warning

Requires: PyPDF2 (optional, for PDF dimension check)
          Falls back to pdfinfo CLI if PyPDF2 unavailable.
"""

import sys
import re
import json
import subprocess
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field, asdict


@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "WARN", "FAIL"
    message: str
    details: list = field(default_factory=list)


def check_pdf_dimensions(pdf_path: Path) -> CheckResult:
    """Check PDF is within NeurIPS column bounds."""
    width_pts, height_pts = None, None

    # Try PyPDF2 first
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        page = reader.pages[0]
        box = page.mediabox
        width_pts = float(box.width)
        height_pts = float(box.height)
    except (ImportError, Exception):
        pass

    # Fallback to pdfinfo CLI
    if width_pts is None:
        try:
            result = subprocess.run(
                ["pdfinfo", str(pdf_path)],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if "Page size" in line:
                    nums = re.findall(r"[\d.]+", line)
                    if len(nums) >= 2:
                        width_pts = float(nums[0])
                        height_pts = float(nums[1])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if width_pts is None:
        return CheckResult(
            "PDF Dimensions", "WARN",
            "Could not read PDF dimensions (install PyPDF2 or poppler-utils)"
        )

    width_in = width_pts / 72.0
    height_in = height_pts / 72.0

    details = [f"{width_in:.2f}\" x {height_in:.2f}\" ({width_pts:.0f} x {height_pts:.0f} pts)"]

    # NeurIPS full column = 5.5in, page height ~9in usable
    if width_in > 7.0:
        return CheckResult("PDF Dimensions", "FAIL",
                           f"Too wide: {width_in:.2f}\" (max ~5.5\" for column)", details)
    if height_in > 10.0:
        return CheckResult("PDF Dimensions", "FAIL",
                           f"Too tall: {height_in:.2f}\" (will dominate the page)", details)
    if width_in > 5.7:
        return CheckResult("PDF Dimensions", "WARN",
                           f"Width {width_in:.2f}\" exceeds NeurIPS column (5.5\")", details)

    return CheckResult("PDF Dimensions", "PASS",
                       f"{width_in:.2f}\" x {height_in:.2f}\"", details)


def check_font_sizes(tex_path: Path) -> CheckResult:
    """Count distinct font size commands — max 2 allowed (+ math scripts)."""
    if not tex_path.exists():
        return CheckResult("Font Sizes", "WARN", f"No .tex file at {tex_path}")

    content = tex_path.read_text(encoding="utf-8", errors="replace")

    size_cmds = re.findall(
        r"\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)\b",
        content
    )
    counts = Counter(size_cmds)
    distinct = len(counts)

    details = [f"{cmd}: {n}x" for cmd, n in counts.most_common()]

    if distinct > 3:
        return CheckResult("Font Sizes", "FAIL",
                           f"{distinct} distinct sizes (max 2 + math scripts)", details)
    if distinct > 2:
        return CheckResult("Font Sizes", "WARN",
                           f"{distinct} sizes — verify 3rd is math-only", details)
    return CheckResult("Font Sizes", "PASS", f"{distinct} distinct sizes", details)


def check_color_compliance(tex_path: Path) -> CheckResult:
    """All colors must come from \\definecolor, no inline hex."""
    if not tex_path.exists():
        return CheckResult("Color Compliance", "WARN", "No .tex file")

    content = tex_path.read_text(encoding="utf-8", errors="replace")

    # Find inline color specs that aren't in \definecolor lines
    lines = content.splitlines()
    violations = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("\\definecolor"):
            continue
        # Check for raw hex in color= or HTML{} contexts
        if re.search(r"#[A-Fa-f0-9]{6}", stripped):
            violations.append(f"  Line {i}: {stripped[:80]}")

    if violations:
        return CheckResult("Color Compliance", "FAIL",
                           f"{len(violations)} inline color literal(s)", violations[:5])
    return CheckResult("Color Compliance", "PASS", "All colors via \\definecolor")


def check_positioning(tex_path: Path) -> CheckResult:
    """Audit absolute vs relative positioning."""
    if not tex_path.exists():
        return CheckResult("Positioning", "WARN", "No .tex file")

    content = tex_path.read_text(encoding="utf-8", errors="replace")

    # Absolute: at (number, number) — not calc-derived
    abs_positions = re.findall(r"at\s*\(\s*-?[\d.]+\s*,\s*-?[\d.]+\s*\)", content)
    calc_positions = re.findall(r"at\s*\(\$", content)
    rel_positions = re.findall(r"(above|below|left|right)\s*=\s*[\d.]+", content)

    risky = len(abs_positions) - len(calc_positions)
    total_rel = len(rel_positions) + len(calc_positions)

    details = [
        f"Absolute (raw): {len(abs_positions)}",
        f"Calc-derived: {len(calc_positions)}",
        f"Relative (positioning lib): {len(rel_positions)}",
    ]

    if risky > 3:
        return CheckResult("Positioning", "WARN",
                           f"{risky} absolute placements — overlap risk", details)
    return CheckResult("Positioning", "PASS",
                       f"{total_rel} relative, {risky} absolute", details)


def check_margins(tex_path: Path) -> CheckResult:
    """Verify standalone border >= 8pt."""
    if not tex_path.exists():
        return CheckResult("Margins", "WARN", "No .tex file")

    content = tex_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"border\s*=\s*(\d+)\s*pt", content)

    if not match:
        return CheckResult("Margins", "WARN", "No explicit border= found in \\documentclass")

    border = int(match.group(1))
    if border < 8:
        return CheckResult("Margins", "FAIL",
                           f"Border {border}pt < 8pt minimum",
                           [f"Change to border={max(8, border)}pt"])
    return CheckResult("Margins", "PASS", f"Border: {border}pt")


def check_log_warnings(log_path: Path) -> CheckResult:
    """Parse .log for overfull/underfull box warnings."""
    if not log_path.exists():
        return CheckResult("LaTeX Warnings", "WARN", f"No .log file at {log_path}")

    content = log_path.read_text(encoding="utf-8", errors="replace")

    overfull = re.findall(r"(Overfull \\[hv]box.*)", content)
    underfull = re.findall(r"(Underfull \\[hv]box.*)", content)
    other_warns = re.findall(r"(LaTeX Warning:.*)", content)

    total = len(overfull) + len(underfull)
    details = []
    if overfull:
        details.append(f"Overfull: {len(overfull)}")
        details.extend(f"  {w[:80]}" for w in overfull[:3])
    if underfull:
        details.append(f"Underfull: {len(underfull)}")
        details.extend(f"  {w[:80]}" for w in underfull[:3])
    if other_warns:
        details.append(f"Other warnings: {len(other_warns)}")

    if total > 0:
        return CheckResult("LaTeX Warnings", "WARN",
                           f"{total} box warning(s)", details)
    return CheckResult("LaTeX Warnings", "PASS", "Clean compile — no box warnings")


def check_overlap_heuristic(tex_path: Path) -> CheckResult:
    """Heuristic: detect nodes that may overlap based on spacing values."""
    if not tex_path.exists():
        return CheckResult("Overlap Heuristic", "WARN", "No .tex file")

    content = tex_path.read_text(encoding="utf-8", errors="replace")

    # Find spacing values in positioning: right=Xcm, below=Xcm, etc.
    spacings = re.findall(
        r"(above|below|left|right)\s*=\s*([\d.]+)\s*(cm|pt|mm|em)",
        content
    )

    tight_gaps = []
    for direction, value, unit in spacings:
        val = float(value)
        # Convert to approximate pt for comparison
        if unit == "cm":
            val_pt = val * 28.35
        elif unit == "mm":
            val_pt = val * 2.835
        elif unit == "em":
            val_pt = val * 10  # rough
        else:
            val_pt = val

        if val_pt < 6:  # less than 6pt gap
            tight_gaps.append(f"{direction}={value}{unit} (~{val_pt:.1f}pt)")

    # Check inner sep values
    inner_seps = re.findall(r"inner\s+sep\s*=\s*([\d.]+)\s*(pt|cm|mm)?", content)
    tight_padding = []
    for value, unit in inner_seps:
        val = float(value)
        if unit == "cm":
            val_pt = val * 28.35
        elif unit == "mm":
            val_pt = val * 2.835
        else:
            val_pt = val
        if val_pt < 3:
            tight_padding.append(f"inner sep={value}{unit or 'pt'} (~{val_pt:.1f}pt)")

    issues = tight_gaps + tight_padding
    if issues:
        return CheckResult("Overlap Heuristic", "WARN",
                           f"{len(issues)} tight spacing(s) detected",
                           issues[:10])
    return CheckResult("Overlap Heuristic", "PASS", "All spacings above minimum thresholds")


def check_accessibility(tex_path: Path) -> CheckResult:
    """Check that color is not the sole distinguishing cue."""
    if not tex_path.exists():
        return CheckResult("Accessibility", "WARN", "No .tex file")

    content = tex_path.read_text(encoding="utf-8", errors="replace")

    # Count distinct fill colors used
    fills = re.findall(r"fill\s*=\s*(\w+)", content)
    draw_colors = re.findall(r"draw\s*=\s*(\w+)", content)
    unique_fills = set(fills) - {"white", "none"}
    unique_draws = set(draw_colors)

    # Check for text labels on colored elements (good sign)
    has_labels = bool(re.findall(r"\\textbf|\\text\w*\{", content))
    has_line_styles = bool(re.findall(r"dashed|dotted|densely|loosely", content))

    details = [
        f"Distinct fills: {len(unique_fills)}",
        f"Has text labels: {has_labels}",
        f"Has line style variation: {has_line_styles}",
    ]

    if len(unique_fills) > 3 and not has_labels:
        return CheckResult("Accessibility", "WARN",
                           "Many fill colors but no text labels — add redundant cues",
                           details)
    return CheckResult("Accessibility", "PASS", "Color + labels/styles for redundancy", details)


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_figure.py <figure.pdf> [--json] [--strict]")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    output_json = "--json" in sys.argv
    strict = "--strict" in sys.argv

    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        sys.exit(1)

    tex_path = pdf_path.with_suffix(".tex")
    log_path = pdf_path.with_suffix(".log")

    checks = [
        check_pdf_dimensions(pdf_path),
        check_margins(tex_path),
        check_font_sizes(tex_path),
        check_color_compliance(tex_path),
        check_positioning(tex_path),
        check_overlap_heuristic(tex_path),
        check_accessibility(tex_path),
        check_log_warnings(log_path),
    ]

    # Force UTF-8 output on Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    if output_json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        fails = 0
        warns = 0
        for c in checks:
            if c.status == "PASS":
                icon = "[PASS]"
            elif c.status == "WARN":
                icon = "[WARN]"
                warns += 1
            else:
                icon = "[FAIL]"
                fails += 1

            print(f"  {icon} {c.name}: {c.message}")
            for d in c.details:
                print(f"      {d}")

        print()
        total_issues = fails + warns
        if total_issues == 0:
            print("\033[32mALL CHECKS PASSED\033[0m")
        else:
            print(f"\033[33m{fails} fail(s), {warns} warning(s)\033[0m")

        if strict and total_issues > 0:
            sys.exit(1)
        elif fails > 0:
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
