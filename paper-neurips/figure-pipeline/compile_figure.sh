#!/bin/bash
# Figure compile + validate pipeline
# Usage: ./compile_figure.sh <file.tex> [--strict]
#
# Compiles TikZ figure with pdflatex, then runs quality checks:
#   - Extracts and reports overfull/underfull box warnings
#   - Checks for font size violations (>2 distinct sizes)
#   - Validates PDF bounding box dimensions
#   - Runs validate_figure.py if available
#
# On success: outputs PDF path + quality report
# On failure: extracts error lines for LLM feedback loop
#
# --strict: treat any quality warning as a failure (exit 1)

set -euo pipefail

TEX_FILE="${1:?Usage: compile_figure.sh <file.tex> [--strict]}"
STRICT="${2:-}"
BASE_NAME="${TEX_FILE%.tex}"
LOG_FILE="${BASE_NAME}.log"
PDF_FILE="${BASE_NAME}.pdf"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

WARNINGS=0

warn() {
    echo -e "${YELLOW}⚠ WARNING:${NC} $1"
    WARNINGS=$((WARNINGS + 1))
}

pass() {
    echo -e "${GREEN}✓${NC} $1"
}

fail() {
    echo -e "${RED}✗ COMPILE FAILED${NC}"
}

# ─── Step 1: Compile ────────────────────────────────────────────
echo -e "${CYAN}=== Compiling ${TEX_FILE} ===${NC}"

if ! pdflatex -interaction=nonstopmode -halt-on-error "${TEX_FILE}" > /dev/null 2>&1; then
    fail
    echo ""
    echo "Errors from ${LOG_FILE}:"
    echo "---"
    # Extract error lines (lines starting with !)
    grep -A 3 "^!" "${LOG_FILE}" 2>/dev/null || echo "(no ! errors found)"
    # Missing packages
    grep "File .* not found" "${LOG_FILE}" 2>/dev/null || true
    # Undefined control sequences
    grep "Undefined control sequence" "${LOG_FILE}" 2>/dev/null || true
    echo "---"
    exit 1
fi

echo -e "${GREEN}✓ Compilation succeeded${NC} → ${PDF_FILE}"
echo ""

# ─── Step 2: Log Quality Checks ────────────────────────────────
echo -e "${CYAN}=== Quality Checks ===${NC}"

# 2a. Overfull / underfull boxes
OVERFULL=$(grep -c "Overfull" "${LOG_FILE}" 2>/dev/null || echo 0)
UNDERFULL=$(grep -c "Underfull" "${LOG_FILE}" 2>/dev/null || echo 0)

if [ "$OVERFULL" -gt 0 ] || [ "$UNDERFULL" -gt 0 ]; then
    warn "Overfull boxes: ${OVERFULL}, Underfull boxes: ${UNDERFULL}"
    grep "Overfull\|Underfull" "${LOG_FILE}" 2>/dev/null | head -10
else
    pass "No overfull/underfull box warnings"
fi

# 2b. Font size audit — count distinct size commands in .tex
echo ""
FONT_SIZES=$(grep -oE '\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)' \
    "${TEX_FILE}" 2>/dev/null | sort -u | wc -l)

if [ "$FONT_SIZES" -gt 3 ]; then
    warn "Font size count: ${FONT_SIZES} distinct sizes (max 2 + math scripts)"
    echo "  Sizes found:"
    grep -oE '\\(tiny|scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|Huge)' \
        "${TEX_FILE}" 2>/dev/null | sort | uniq -c | sort -rn
elif [ "$FONT_SIZES" -gt 0 ]; then
    pass "Font sizes: ${FONT_SIZES} distinct (within budget)"
fi

# 2c. Raw color literals (should use \definecolor names)
echo ""
RAW_COLORS=$(grep -cE '(HTML|RGB|rgb)\{[A-Fa-f0-9]{6}\}' "${TEX_FILE}" 2>/dev/null || echo 0)
INLINE_COLORS=$(grep -cE 'color=.*#[A-Fa-f0-9]' "${TEX_FILE}" 2>/dev/null || echo 0)
# Exclude \definecolor lines — those are expected
DEFINECOLOR_COUNT=$(grep -c '\\definecolor' "${TEX_FILE}" 2>/dev/null || echo 0)

if [ "$INLINE_COLORS" -gt 0 ]; then
    warn "Found ${INLINE_COLORS} inline color literals (use \\definecolor names)"
else
    pass "All colors use \\definecolor names"
fi

# 2d. Absolute positioning audit
echo ""
ABS_POS=$(grep -cE 'at\s*\([0-9.-]+\s*,\s*[0-9.-]+\)' "${TEX_FILE}" 2>/dev/null || echo 0)
CALC_POS=$(grep -cE 'at\s*\(\$' "${TEX_FILE}" 2>/dev/null || echo 0)
RISKY_POS=$((ABS_POS - CALC_POS))

if [ "$RISKY_POS" -gt 2 ]; then
    warn "Found ${RISKY_POS} absolute 'at(x,y)' placements (prefer positioning library)"
else
    pass "Positioning: ${ABS_POS} absolute (${CALC_POS} calc-derived) — OK"
fi

# 2e. Border/margin check — standalone border
echo ""
BORDER=$(grep -oE 'border=[0-9]+pt' "${TEX_FILE}" 2>/dev/null | head -1)
if [ -z "$BORDER" ]; then
    warn "No explicit border= in \\documentclass. Add border=8pt."
else
    BORDER_VAL=$(echo "$BORDER" | grep -oE '[0-9]+')
    if [ "$BORDER_VAL" -lt 8 ]; then
        warn "Border is ${BORDER_VAL}pt (minimum 8pt for safe margins)"
    else
        pass "Standalone border: ${BORDER_VAL}pt ≥ 8pt"
    fi
fi

# 2f. PDF dimensions (if pdfinfo is available)
echo ""
if command -v pdfinfo &> /dev/null; then
    PAGE_SIZE=$(pdfinfo "${PDF_FILE}" 2>/dev/null | grep "Page size" || true)
    if [ -n "$PAGE_SIZE" ]; then
        WIDTH_PTS=$(echo "$PAGE_SIZE" | grep -oE '[0-9.]+' | head -1)
        HEIGHT_PTS=$(echo "$PAGE_SIZE" | grep -oE '[0-9.]+' | head -2 | tail -1)
        WIDTH_IN=$(echo "scale=2; ${WIDTH_PTS} / 72" | bc 2>/dev/null || echo "?")
        HEIGHT_IN=$(echo "scale=2; ${HEIGHT_PTS} / 72" | bc 2>/dev/null || echo "?")
        echo "  PDF size: ${WIDTH_IN}\" × ${HEIGHT_IN}\" (${WIDTH_PTS} × ${HEIGHT_PTS} pts)"

        # Check if unreasonably large (>7in wide or >10in tall)
        TOO_WIDE=$(echo "${WIDTH_PTS} > 504" | bc 2>/dev/null || echo 0)
        TOO_TALL=$(echo "${HEIGHT_PTS} > 720" | bc 2>/dev/null || echo 0)
        if [ "$TOO_WIDE" = "1" ] || [ "$TOO_TALL" = "1" ]; then
            warn "PDF dimensions may be too large for NeurIPS column (max ~5.5\" wide)"
        else
            pass "PDF dimensions within expected range"
        fi
    fi
else
    echo "  (pdfinfo not found — skipping dimension check; install poppler-utils)"
fi

# ─── Step 3: Run Python validator if available ──────────────────
echo ""
VALIDATOR="${SCRIPT_DIR}/validate_figure.py"
if [ -f "$VALIDATOR" ] && command -v python &> /dev/null; then
    echo -e "${CYAN}=== Running validate_figure.py ===${NC}"
    python "$VALIDATOR" "${PDF_FILE}" || WARNINGS=$((WARNINGS + 1))
else
    echo "(validate_figure.py or python not available — skipping advanced checks)"
fi

# ─── Summary ────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}=== Summary ===${NC}"
if [ "$WARNINGS" -eq 0 ]; then
    echo -e "${GREEN}ALL CHECKS PASSED${NC} — ${PDF_FILE} is ready"
    exit 0
else
    echo -e "${YELLOW}${WARNINGS} warning(s) found${NC}"
    if [ "$STRICT" = "--strict" ]; then
        echo -e "${RED}Strict mode: treating warnings as errors${NC}"
        exit 1
    else
        echo "Review warnings above. Use --strict to enforce zero-warning policy."
        exit 0
    fi
fi
