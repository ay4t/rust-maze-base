# 📘 Rust Maze Base Designer (v0.1)
Design labyrinth-style Rust bases with multi-TC support, dead-end TC placement, and armored metal decoy walls. Outputs a clean 1920×1920 PNG floor plan and a structured JSON.

## ✨ Features
- 2D maze generation on a flexible grid (auto-expands if constraints require more space).
- TC placement strictly at maze dead-ends, with minimum TC-to-TC distance and a safety margin from outer borders.
- Armored metal (blue) walls automatically wrap each TC, plus additional blue walls randomly sprinkled as decoys to mislead raiders.
- Crystal-clear export: PNG floor plan (1920×1920) + JSON metadata (grid, TC positions, entrance, material estimate).

## 🛠️ Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## 🚀 Quick Start
```bash
python rust_maze_designer.py \
  --size 12x12 \
  --tc 3 \
  --complexity complex \
  --material metal \
  --seed 42 \
  --tc-border-margin 3 \
  --outdir .
```

## 🎛️ CLI Options
- `--size WxH`  Grid size (e.g., 12x12). Auto-expands if constraints (dead-ends/distance) require more room.
- `--tc N`      Number of TCs (1–5). Placed at dead-ends only.
- `--complexity` Maze complexity flag (affects generation seed/styling in future versions).
- `--material`  Base wall material label for legend display (stone|metal|armored).
- `--seed`      RNG seed for reproducible output.
- `--tc-border-margin` Minimum distance (in cells) of TC from the outer border. Default: 1.
- `--outdir`    Output directory for PNG and JSON.

## 🗺️ Output
- PNG: `maze_design_YYYYmmddHHMMSS.png` at 1920×1920.
- JSON: `maze_design_YYYYmmddHHMMSS.json` with fields: `grid`, `params`, `tc`, `entrance`, `materials_estimate`, `notes`.

## 🎨 Legend
- Red square: TC (Tool Cupboard) position.
- Orange circle: Entrance.
- Blue line: Armored metal walls (see decoy pattern below).
- Grey line: Regular walls.

## 🧩 TC Placement Rules
- TCs are placed only at dead-ends for maximum raid confusion.
- Enforces minimum TC-to-TC distance (Manhattan).
- Keeps a safe margin from the outer border (`--tc-border-margin`).
- If constraints cannot be satisfied, the grid auto-expands up to a safe cap.

## 🛡️ Armored Metal (Blue) Decoy Pattern
- TC-adjacent walls are automatically drawn in blue to represent armored metal.
- Additional blue walls are randomly sprinkled across the maze as decoys.
- Note: "Armored metal pattern is sprinkled randomly to mislead raiders."

## 📐 Canvas & Layout
- Fixed canvas: 1920×1920.
- Grid is rendered on the left panel with proper margins; legend lives on the right panel—no overlap.
- Font sizes are increased for readability. Uses DejaVuSans if available (falls back gracefully).

## ⚠️ Notes & Roadmap
- Disconnected structures and building privilege simulation are simplified in v0.1 (distance-based heuristics). Future versions will add proper privilege radius and structural stability checks.
- Material and raid cost estimates are currently coarse (wall segment–based). Will be refined.

