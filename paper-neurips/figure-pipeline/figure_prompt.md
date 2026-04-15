# Figure Creation Agent — Design Bible

> Every figure in this paper must meet the standards below.
> The agent MUST run the **Self-Check Protocol** (§10) after every figure
> and report each check as PASS / FAIL before delivering the PDF.

---

## §1  Scope & Workflow

```
color_schema.json   ← canonical palette (never invent ad-hoc colors)
figure_prompt.md    ← this file (design rules + self-check)
compile_figure.sh   ← compile .tex → .pdf + validation
validate_figure.py  ← automated post-compile quality checks
```

**Lifecycle of a figure:**
1. Draft `.tex` using rules below + `color_schema.json` colors.
2. Compile: `bash compile_figure.sh <file>.tex`
3. Validate: `python validate_figure.py <file>.pdf`
4. Run §10 self-check (agent reads own output & verifies).
5. Copy final PDF to `../figures/`.

---

## §2  LaTeX Boilerplate (mandatory)

```latex
\documentclass[border=8pt]{standalone}          % 8pt = safe margin
\usepackage{tikz}
\usepackage{amsmath,amssymb}
\usetikzlibrary{
  arrows.meta, positioning, shapes.geometric,
  fit, calc, decorations.pathreplacing, backgrounds
}
```

- **No other packages.** Must compile with `pdflatex` on stock MiKTeX / TeX Live.
- Target widths: **5.5 in** (NeurIPS full column) or **2.7 in** (half column).
- All colors MUST be `\definecolor` from `color_schema.json` hex values.
  Never use raw RGB/HTML literals inline.

---

## §3  Typography

| Rule | Spec |
|------|------|
| **Max font sizes** | Exactly **2**: one body size + one small annotation size. A 3rd size is allowed ONLY for math sub/superscripts. |
| Body text | `\sffamily\small` (≈ 9pt in standalone) |
| Annotations | `\sffamily\scriptsize` (≈ 7pt) or `\sffamily\footnotesize` (≈ 8pt) |
| Minimum | Nothing below `\scriptsize`. If it can't fit at `\scriptsize`, move it to the caption. |
| Font family | Sans-serif only (`\sffamily`). Matches NeurIPS body. |
| Math | `$...$` inline. Macros must match paper.tex exactly (`\mathbf{h}`, `\mathrm{RUL}`, etc.). |

**Hard rule:** Count every `\font...` and `\...size` command in your `.tex`.
If more than 2 distinct sizes (+ math scripts), simplify before compiling.

---

## §4  Visual Hierarchy

- **One focal point per figure.** The most important element should be largest,
  most saturated, or most central.
- Importance tiers:
  1. **Primary** — saturated fill from palette, full border. (Encoders, predictor, loss.)
  2. **Secondary** — light fill (25% opacity), thin border. (Representation vectors.)
  3. **Tertiary** — gray/dashed, minimal. (Annotations, downstream notes.)
- Max **5–6 distinct visual elements** at primary/secondary level.
  If you need more, use subfigures `(a)`, `(b)`.

---

## §5  Layout, Spacing & Anti-Overlap Rules

> **The #1 quality killer in LLM-generated figures is element overlap.**
> These rules exist to prevent it categorically.

### 5.1  Mandatory Gaps

| Between | Minimum gap |
|---------|-------------|
| Adjacent boxes | **8pt** (`right=0.8cm` or equivalent) |
| Box border → internal text | **4pt** padding (`inner sep=4pt`) |
| Label → nearest box edge | **3pt** (`above=0.1cm`, `right=0.08cm`, etc.) |
| Parallel arrows | **6pt** vertical separation |
| Arrow label → arrow line | **2pt** |

### 5.2  Positioning Strategy

- **Always use `positioning` library** (`right=Xcm of node`).
  Never place nodes with absolute `at (x,y)` unless computing coordinates
  from other nodes via `calc`.
- **Never eyeball coordinates.** If two elements look close, increase the gap.
  Generous whitespace > cramped precision.
- For multi-row layouts: define a grid with explicit vertical spacing
  (`below=1.0cm`), then fill columns.

### 5.3  Overlap Prevention Checklist (run mentally before compile)

For every node N:
1. Is N placed with `positioning` relative to another node? ✓
2. Does N have explicit `minimum height` and `minimum width`? ✓
3. Is the gap to every neighbor ≥ the minimum from §5.1? ✓
4. If N has a label, is the label placed with `above=`, `below=`, `right=`,
   `left=` with explicit offset? ✓
5. Do any arrows cross over N? If yes, reroute or add a bend. ✓

### 5.4  Arrow Routing

- **Straight first** (horizontal or vertical). One 90° bend is acceptable.
- **Never diagonal** unless it's the only option (rare).
- **No crossings.** If two arrows would cross, reroute one with
  `([yshift=Xpt]node.east)` or use a waypoint coordinate.
- Arrow labels: place with `node[midway, above=2pt]` or `node[midway, right=2pt]`.
  Never `node[midway]` alone (risks overlap with the arrow line).

---

## §6  Color

All colors from `color_schema.json`. No exceptions.

| Usage | Color(s) |
|-------|----------|
| Context encoder | `cprimary` border, `bgencoder` fill |
| Target encoder | `caccent` border, `bgtarget` fill |
| Predictor | `csecondary` border, `bgpredictor` fill |
| Loss | `cwarning` border, `bgloss` fill |
| Data arrows | `carrow` (solid) |
| EMA / non-data | `cneutralmid` (dashed) |
| Labels / text | `cneutraldark` |
| Annotations | `cneutralmid` or `cneutrallight` |

### 6.1  Color Rules

- **Max 4–5 hues** per figure. The palette has 4 semantic hues
  (blue, orange, teal, yellow) + neutrals. That's your budget.
- Background fills: **pastel only** (`bg*` colors or `!25` variants).
  Never saturated fills on large areas.
- Saturated color: reserved for **small accents** (arrow tips, tiny labels, border lines).
- **Accessibility:** every color distinction must also be encoded by
  shape, line style, or text label. Never rely on color alone.

---

## §7  Academic Style

- **Flat design only.** No gradients, no shadows, no 3D, no glow.
- Rounded corners: **3–4pt max**, and only on semantic blocks.
  Data/vector nodes can use 2pt. Don't round everything.
- Line weights:
  - Box borders: `0.5–0.8pt`
  - Arrows: `0.7–1.0pt`
  - Annotations/separators: `0.3–0.4pt`
- White/transparent background. No gray canvas.

---

## §8  Information Density

- **Every element must earn its place.** If removing it loses zero information, delete it.
- Prefer **direct labels** over legends. Labels on/near the element > separate legend box.
- Keep text inside boxes **short** (≤ 3 words per line, ≤ 4 lines per box).
  Longer descriptions belong in the caption.
- Math notation must match `paper.tex` exactly — same macros, same symbols.

---

## §9  Common Mistakes — Reject Immediately

| Mistake | Fix |
|---------|-----|
| Text overflows box boundary | Increase `minimum width` / `minimum height`, or shorten text |
| Two labels overlap each other | Increase gap; move one label to opposite side |
| Arrow passes through a node | Reroute with waypoint coordinate |
| More than 2 font sizes | Remove the extra size; use the smaller of the two |
| Saturated fill on a large box | Switch to `bg*` or `!25` variant |
| Diagonal spaghetti arrows | Straighten; use 90° bends |
| Element placed with absolute `at(x,y)` not derived from calc | Use `positioning` relative placement |
| Decorative element (adds no information) | Delete it |
| Box sizes vary wildly for equal-importance items | Normalize `minimum width` / `minimum height` |
| Color is only distinguishing cue (no shape/label backup) | Add redundant encoding |

---

## §10  Self-Check Protocol (MANDATORY)

**After every figure, before delivering, the agent MUST verify each item
and report the result as a structured checklist.**

```
SELF-CHECK REPORT — <figure_name>.tex
──────────────────────────────────────
[ ] 1. FONT COUNT: List every \font/\size command used.
       ≤ 2 distinct sizes (+ math scripts)?           → PASS / FAIL
[ ] 2. OVERLAP SCAN: For every node, confirm gap to nearest
       neighbor ≥ minimum (§5.1). List any pair < threshold.
                                                       → PASS / FAIL
[ ] 3. ARROW CROSSINGS: Count arrow-arrow crossings.
       Zero?                                           → PASS / FAIL
[ ] 4. ARROW-THROUGH-NODE: Does any arrow path pass through
       a node it shouldn't?                            → PASS / FAIL
[ ] 5. TEXT OVERFLOW: Any node where text exceeds the
       minimum width/height?                           → PASS / FAIL
[ ] 6. COLOR COMPLIANCE: All colors from color_schema.json?
       No raw hex/RGB literals?                        → PASS / FAIL
[ ] 7. MAX HUES: ≤ 5 distinct hues used?              → PASS / FAIL
[ ] 8. ACCESSIBILITY: Every color-encoded distinction also
       has shape/line-style/label backup?              → PASS / FAIL
[ ] 9. READABILITY AT 50%: Would every label still be
       legible if the figure were printed at 50%?
       (Proxy: nothing below \scriptsize)              → PASS / FAIL
[ ] 10. GRAYSCALE: Would all elements remain distinguishable
        in grayscale? (Proxy: hue pairs have different
        luminance — check bg fills)                    → PASS / FAIL
[ ] 11. INFORMATION DENSITY: Any element that could be
        removed without losing information?            → PASS / FAIL
[ ] 12. MARGINS: standalone border ≥ 8pt? No content
        within 5pt of the crop edge?                   → PASS / FAIL
[ ] 13. POSITIONING: All nodes use relative positioning
        (positioning library) or calc-derived coords?  → PASS / FAIL
[ ] 14. COMPILE CLEAN: Zero overfull/underfull box warnings
        in the .log file?                              → PASS / FAIL

VERDICT: ALL PASS → deliver.  ANY FAIL → fix and re-check.
```

**If any check fails, the agent must fix the issue and re-run the full
self-check. Do not deliver a figure with any FAIL.**

---

## §11  Figure Specifications for This Paper

### Figure 1: Trajectory JEPA Architecture
- File: `trajectory_jepa_architecture.tex` (exists, update in place)
- Two-branch layout:
  - Top: x_past → Context Encoder → h_past → Predictor → ĥ_fut → L₁ Loss
  - Bottom: x_future → Target Encoder (EMA) → h_fut → L₁ Loss
  - EMA: dashed vertical arrow between encoders
  - Downstream: h_past → linear → RUL(t)
  - Horizon k: diamond input to predictor

### Additional figures
- Specify each new figure here before creation.
- Include: purpose, layout sketch, key elements, data source.

---

## §12  Color Schema Quick Reference

```
PRIMARY (blue)     #2D5F8A / light #5B8DB8 / dark #1A3A5C
SECONDARY (orange) #C25B28 / light #E8925F
ACCENT (teal)      #3A8C6E / light #6BB89A
WARNING (yellow)   #D4A843
NEUTRALS           dark #333333 / mid #888888 / light #CCCCCC
BACKGROUNDS        encoder #E8F0F8 / predictor #FFF3E8 /
                   target #E8F8F0 / loss #FFF8E8 / general #F5F5F5
ARROW              #555555
```
