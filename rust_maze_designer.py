#!/usr/bin/env python3
"""
Rust Maze Base Designer (v0.1)
- Generator maze grid sederhana + penempatan multi TC dengan constraint jarak minimal 7 foundations (grid squares)
- Output: PNG (floor plan sederhana) + JSON koordinat elemen

Catatan:
- Versi awal: fokus pada kerangka kerja yang bisa langsung jalan.
- Validasi "disconnected structure" Rust masih disederhanakan (menggunakan jarak >= 7 grid sebagai pendekatan awal). Iterasi berikutnya bisa menambah pemisahan struktur aktual.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Set

from PIL import Image, ImageDraw, ImageFont

# Optional: jika NetworkX dibutuhkan untuk validasi ke depan
try:
    import networkx as nx  # noqa: F401
except Exception:
    nx = None  # tidak wajib di v0.1

Coord = Tuple[int, int]


@dataclass
class Config:
    width: int = 12
    height: int = 12
    tc_count: int = 3
    complexity: str = "complex"  # simple|medium|complex
    material: str = "metal"  # stone|metal|armored
    seed: int | None = None
    min_tc_distance: int = 7  # minimal jarak manhattan antar TC
    cell_px: int = 40  # default; tidak digunakan saat canvas fixed
    wall_thickness: int = 3
    canvas_px: int = 1920  # ukuran kanvas persegi (HD 1920x1920)
    legend_width: int = 360  # panel legend di sisi kanan
    margin: int = 40  # margin di sekeliling grid
    tc_border_margin: int = 1  # jarak minimal TC dari dinding terluar
    outdir: str = "."


class Maze:
    def __init__(self, w: int, h: int, rng: random.Random):
        self.w = w
        self.h = h
        self.rng = rng
        # Representasi: tiap sel punya dinding di 4 sisi. Kita simpan dinding yang masih ada.
        # dinding vertical: antara (x,y) dan (x+1,y) untuk x in [0..w-2], y in [0..h-1]
        # dinding horizontal: antara (x,y) dan (x,y+1) untuk x in [0..w-1], y in [0..h-2]
        self.v_walls: Set[Tuple[int, int]] = {(x, y) for x in range(w - 1) for y in range(h)}
        self.h_walls: Set[Tuple[int, int]] = {(x, y) for x in range(w) for y in range(h - 1)}

    def carve_recursive_backtracking(self):
        visited = [[False] * self.h for _ in range(self.w)]

        def neighbors(cx: int, cy: int) -> List[Tuple[int, int, str]]:
            dirs = []
            if cx > 0:
                dirs.append((cx - 1, cy, "L"))
            if cx < self.w - 1:
                dirs.append((cx + 1, cy, "R"))
            if cy > 0:
                dirs.append((cx, cy - 1, "U"))
            if cy < self.h - 1:
                dirs.append((cx, cy + 1, "D"))
            self.rng.shuffle(dirs)
            return dirs

        stack = []
        start = (self.rng.randrange(self.w), self.rng.randrange(self.h))
        stack.append(start)
        visited[start[0]][start[1]] = True

        while stack:
            x, y = stack[-1]
            nbrs = [(nx, ny, d) for (nx, ny, d) in neighbors(x, y) if not visited[nx][ny]]
            if not nbrs:
                stack.pop()
                continue
            nx_, ny_, d = nbrs[0]
            # Hancurkan dinding antara (x,y) dan (nx_,ny_)
            if d == "L":
                # antara (nx_,y) dan (x,y) adalah dinding vertical di (nx_, y)
                self.v_walls.discard((nx_, y))
            elif d == "R":
                self.v_walls.discard((x, y))
            elif d == "U":
                self.h_walls.discard((x, ny_))
            elif d == "D":
                self.h_walls.discard((x, y))

            visited[nx_][ny_] = True
            stack.append((nx_, ny_))

    def random_entrance(self) -> Tuple[Coord, Coord]:
        # Buat 1 pintu masuk di tepi: pilih sel tepi dan buka dinding keluar
        edges = []
        for x in range(self.w):
            edges.append((x, 0, "U"))
            edges.append((x, self.h - 1, "D"))
        for y in range(self.h):
            edges.append((0, y, "L"))
            edges.append((self.w - 1, y, "R"))
        x, y, side = self.rng.choice(edges)
        if side == "L" and (x - 1, y) not in self.v_walls and x > 0:
            pass
        # Paksa buka dinding di sisi tepi
        if side == "L" and x > 0:
            self.v_walls.discard((x - 1, y))
        elif side == "R" and x < self.w - 1:
            self.v_walls.discard((x, y))
        elif side == "U" and y > 0:
            self.h_walls.discard((x, y - 1))
        elif side == "D" and y < self.h - 1:
            self.h_walls.discard((x, y))
        return (x, y), {"L": (-1, 0), "R": (1, 0), "U": (0, -1), "D": (0, 1)}[side]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def choose_tc_positions(w: int, h: int, count: int, min_dist: int, rng: random.Random, border_margin: int = 0) -> List[Coord]:
    positions: List[Coord] = []
    attempts = 0
    max_attempts = 5000
    while len(positions) < count and attempts < max_attempts:
        attempts += 1
        cand = (rng.randrange(w), rng.randrange(h))
        if cand in positions:
            continue
        # hindari tepi luar
        if not (border_margin <= cand[0] <= w - 1 - border_margin and border_margin <= cand[1] <= h - 1 - border_margin):
            continue
        if all(manhattan(cand, p) >= min_dist for p in positions):
            positions.append(cand)
    if len(positions) < count:
        # fallback: longgar sedikit (kurangi min_dist) agar selalu dapat
        while len(positions) < count and min_dist > 2:
            min_dist -= 1
            positions = choose_tc_positions(w, h, count, min_dist, rng, border_margin)
            if len(positions) >= count:
                break
    return positions[:count]


def choose_random_cells(w: int, h: int, k: int, rng: random.Random, exclude: Set[Coord] | None = None) -> List[Coord]:
    exclude = exclude or set()
    all_cells = [(x, y) for x in range(w) for y in range(h) if (x, y) not in exclude]
    rng.shuffle(all_cells)
    return all_cells[:k]


def count_open_neighbors(maze: Maze, x: int, y: int) -> int:
    open_count = 0
    # Left: open jika tidak ada dinding vertical di (x-1, y)
    if x > 0 and (x - 1, y) not in maze.v_walls:
        open_count += 1
    # Right: open jika tidak ada dinding vertical di (x, y)
    if x < maze.w - 1 and (x, y) not in maze.v_walls:
        open_count += 1
    # Up: open jika tidak ada dinding horizontal di (x, y-1)
    if y > 0 and (x, y - 1) not in maze.h_walls:
        open_count += 1
    # Down: open jika tidak ada dinding horizontal di (x, y)
    if y < maze.h - 1 and (x, y) not in maze.h_walls:
        open_count += 1
    return open_count


def get_dead_end_cells(maze: Maze) -> List[Coord]:
    return [(x, y) for x in range(maze.w) for y in range(maze.h) if count_open_neighbors(maze, x, y) == 1]


def pick_positions_from_candidates(cands: List[Coord], count: int, min_dist: int, rng: random.Random) -> List[Coord]:
    picks: List[Coord] = []
    pool = cands[:]
    rng.shuffle(pool)
    for c in pool:
        if all(manhattan(c, p) >= min_dist for p in picks):
            picks.append(c)
            if len(picks) >= count:
                break
    return picks


def draw_floor_plan(cfg: Config, maze: Maze, entrance_cell: Coord, tcs: List[Coord], out_png: str):
    # Kanvas fixed 1920x1920, grid ditempatkan di kiri dengan margin; legend di panel kanan terpisah
    canvas_size = (cfg.canvas_px, cfg.canvas_px)
    img = Image.new("RGB", canvas_size, (255, 255, 255))
    dr = ImageDraw.Draw(img)

    # Hitung ukuran grid agar muat di area (canvas - legend - margin)
    grid_max_w = cfg.canvas_px - cfg.legend_width - 2 * cfg.margin
    grid_max_h = cfg.canvas_px - 2 * cfg.margin
    cell_px = max(10, min(grid_max_w // cfg.width, grid_max_h // cfg.height))

    grid_wpx = cfg.width * cell_px
    grid_hpx = cfg.height * cell_px
    grid_x0 = cfg.margin
    grid_y0 = cfg.margin
    grid_x1 = grid_x0 + grid_wpx
    grid_y1 = grid_y0 + grid_hpx

    # Frame grid
    dr.rectangle([(grid_x0, grid_y0), (grid_x1, grid_y1)], outline=(0, 0, 0), width=cfg.wall_thickness)

    # Tentukan dinding yang akan berwarna biru (armored):
    # 1) Dinding yang membungkus TC
    # 2) Beberapa dinding acak sebagai decoy
    col_armored = (65, 105, 225)  # biru
    blue_v: Set[Tuple[int, int]] = set()
    blue_h: Set[Tuple[int, int]] = set()

    for (txc, tyc) in tcs:
        # kiri
        if txc > 0 and (txc - 1, tyc) in maze.v_walls:
            blue_v.add((txc - 1, tyc))
        # kanan
        if txc < cfg.width - 1 and (txc, tyc) in maze.v_walls:
            blue_v.add((txc, tyc))
        # atas
        if tyc > 0 and (txc, tyc - 1) in maze.h_walls:
            blue_h.add((txc, tyc - 1))
        # bawah
        if tyc < cfg.height - 1 and (txc, tyc) in maze.h_walls:
            blue_h.add((txc, tyc))

    # Decoy: pilih sebagian dinding sisa secara acak
    rng_local = random.Random(cfg.seed)
    cand_v = list(maze.v_walls - blue_v)
    cand_h = list(maze.h_walls - blue_h)
    total_walls = len(cand_v) + len(cand_h)
    decoy_total = min(max(4, cfg.tc_count * 6), max(0, total_walls // 20))  # proporsional + batas bawah kecil
    kv = min(len(cand_v), decoy_total // 2)
    kh = min(len(cand_h), decoy_total - kv)
    if kv > 0:
        blue_v.update(rng_local.sample(cand_v, kv))
    if kh > 0:
        blue_h.update(rng_local.sample(cand_h, kh))

    # Walls dengan offset grid; gunakan warna biru untuk armored
    # Vertical walls
    for (x, y) in maze.v_walls:
        x0 = grid_x0 + (x + 1) * cell_px
        y0 = grid_y0 + y * cell_px
        x1 = x0
        y1 = grid_y0 + (y + 1) * cell_px
        color = col_armored if (x, y) in blue_v else (120, 120, 120)
        dr.line([(x0, y0), (x1, y1)], fill=color, width=cfg.wall_thickness)
    # Horizontal walls
    for (x, y) in maze.h_walls:
        x0 = grid_x0 + x * cell_px
        y0 = grid_y0 + (y + 1) * cell_px
        x1 = grid_x0 + (x + 1) * cell_px
        y1 = y0
        color = col_armored if (x, y) in blue_h else (120, 120, 120)
        dr.line([(x0, y0), (x1, y1)], fill=color, width=cfg.wall_thickness)

    def cell_rect(c: Coord, pad: int = 8) -> Tuple[int, int, int, int]:
        cx, cy = c
        x0 = grid_x0 + cx * cell_px + pad
        y0 = grid_y0 + cy * cell_px + pad
        x1 = grid_x0 + (cx + 1) * cell_px - pad
        y1 = grid_y0 + (cy + 1) * cell_px - pad
        return (x0, y0, x1, y1)

    # Legend colors
    col_tc = (220, 20, 60)       # red
    col_door = (65, 105, 225)    # blue
    col_storage = (60, 179, 113) # green
    col_entrance = (255, 165, 0) # orange

    # Entrance marker
    dr.ellipse(cell_rect(entrance_cell, pad=14), outline=col_entrance, width=cfg.wall_thickness)

    # TCs
    for t in tcs:
        dr.rectangle(cell_rect(t, pad=10), fill=None, outline=col_tc, width=cfg.wall_thickness)

    # Legend panel di kanan
    panel_x0 = grid_x1 + cfg.margin // 2
    panel_x1 = cfg.canvas_px - cfg.margin
    panel_y0 = cfg.margin
    panel_y1 = cfg.canvas_px - cfg.margin
    dr.rectangle([(panel_x0, panel_y0), (panel_x1, panel_y1)], outline=(0, 0, 0), width=1)

    # Font untuk legend (besar). Coba TrueType DejaVuSans, fallback ke default.
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_item = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font_title = ImageFont.load_default()
        font_item = ImageFont.load_default()

    # Isi legend
    tx = panel_x0 + 20
    ty = panel_y0 + 20
    dr.text((tx, ty), "Legenda:", fill=(0, 0, 0), font=font_title)
    ty += 40
    # TC marker
    dr.line([(tx, ty), (tx + 40, ty)], fill=col_tc, width=cfg.wall_thickness)
    dr.text((tx + 50, ty - 14), "TC", fill=(0, 0, 0), font=font_item)
    ty += 34
    # Entrance marker (circle)
    r = 10
    dr.ellipse([(tx, ty - r), (tx + 2 * r, ty + r)], outline=col_entrance, width=cfg.wall_thickness)
    dr.text((tx + 50, ty - 14), "Entrance", fill=(0, 0, 0), font=font_item)
    ty += 34
    # Armored metal marker (blue)
    dr.line([(tx, ty), (tx + 40, ty)], fill=col_armored, width=cfg.wall_thickness)
    dr.text((tx + 50, ty - 14), "Armored metal (blue)", fill=(0, 0, 0), font=font_item)
    ty += 34
    # Note in English
    note = "Note: Armored metal pattern is sprinkled randomly to mislead raiders."
    dr.text((tx, ty), note, fill=(0, 0, 0), font=font_item)

    img.save(out_png)


def estimate_material(cfg: Config, maze: Maze) -> Dict[str, int]:
    # Estimasi sangat sederhana berbasis jumlah segment dinding tersisa
    wall_segments = len(maze.v_walls) + len(maze.h_walls)
    base = {
        "stone": 50,
        "metal": 80,
        "armored": 120,
    }.get(cfg.material, 50)
    total = wall_segments * base
    return {
        "wall_segments": wall_segments,
        "material_unit": cfg.material,
        "estimated_cost": total,
    }


def parse_size(s: str) -> Tuple[int, int]:
    try:
        parts = s.lower().split("x")
        if len(parts) != 2:
            raise ValueError
        w = int(parts[0])
        h = int(parts[1])
        if not (2 <= w <= 20 and 2 <= h <= 20):
            raise ValueError
        return w, h
    except Exception:
        raise argparse.ArgumentTypeError("Format --size harus seperti 12x12, batas 2..20")


def build(cfg: Config):
    rng = random.Random(cfg.seed)

    # Upaya penempatan TC, jika gagal karena ruang sempit/dead-end kurang -> lebarkan grid otomatis
    max_cap = 30  # batas aman pelebaran
    w, h = cfg.width, cfg.height
    tc_positions: List[Coord] = []
    while True:
        # Bangun maze untuk ukuran saat ini dan cari dead-end
        maze = Maze(w, h, rng)
        maze.carve_recursive_backtracking()
        entrance_cell, _ = maze.random_entrance()

        # Kandidat dead-end dengan menjaga margin border
        dead_ends = get_dead_end_cells(maze)
        dead_ends = [c for c in dead_ends if (cfg.tc_border_margin <= c[0] <= w - 1 - cfg.tc_border_margin and cfg.tc_border_margin <= c[1] <= h - 1 - cfg.tc_border_margin)]

        # Pilih TC dari dead-end candidates dengan jarak minimal
        tc_positions = pick_positions_from_candidates(dead_ends, cfg.tc_count, cfg.min_tc_distance, rng)
        if len(tc_positions) >= cfg.tc_count:
            break
        if w >= max_cap or h >= max_cap:
            # jika tetap gagal pada batas, gunakan yang ada (fallback)
            break
        # perbesar grid dan coba lagi
        w += 1
        h += 1

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    png_name = f"maze_design_{ts}.png"
    json_name = f"maze_design_{ts}.json"
    out_png = os.path.join(cfg.outdir, png_name)
    out_json = os.path.join(cfg.outdir, json_name)

    # Render hanya grid + TC + entrance
    # Update config sementara untuk lebar/tinggi akhir saat menggambar dan menulis metadata
    cfg_final = Config(**{**asdict(cfg), "width": w, "height": h})
    draw_floor_plan(cfg_final, maze, entrance_cell, tc_positions, out_png)

    data = {
        "grid": {"width": w, "height": h},
        "params": asdict(cfg_final),
        "tc": [{"x": x, "y": y} for (x, y) in tc_positions],
        "entrance": {"x": entrance_cell[0], "y": entrance_cell[1]},
        "materials_estimate": estimate_material(cfg_final, maze),
        "notes": [
            "Doors/Storage dihapus sesuai permintaan.",
            "TC ditempatkan di ujung jalan buntu (dead-end).",
            "TC dihindarkan dari border dengan margin.",
            "Grid otomatis melebar jika perlu ruang/dead-end cukup untuk overlap TC.",
        ],
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return out_png, out_json


def main():
    p = argparse.ArgumentParser(description="Rust Maze Base Designer")
    p.add_argument("--size", type=parse_size, default=(12, 12), help="Ukuran grid, contoh: 12x12 (min 2x2, max 20x20)")
    p.add_argument("--tc", type=int, default=3, help="Jumlah TC (1-5)")
    p.add_argument("--complexity", type=str, default="complex", choices=["simple", "medium", "complex"], help="Tingkat kompleksitas maze (sementara dekoratif)")
    p.add_argument("--material", type=str, default="metal", choices=["stone", "metal", "armored"], help="Tipe material dinding")
    p.add_argument("--seed", type=int, default=None, help="Seed RNG untuk reproduksibilitas")
    p.add_argument("--outdir", type=str, default=".", help="Direktori output file")
    p.add_argument("--tc-border-margin", type=int, default=1, help="Jarak minimal TC dari dinding terluar (grid)")

    args = p.parse_args()

    width, height = args.size
    tc = max(1, min(5, int(args.tc)))

    cfg = Config(
        width=width,
        height=height,
        tc_count=tc,
        complexity=args.complexity,
        material=args.material,
        seed=args.seed,
        outdir=args.outdir,
        tc_border_margin=max(0, int(args.tc_border_margin)),
    )

    os.makedirs(cfg.outdir, exist_ok=True)
    png, js = build(cfg)
    print(f"OK: generated {png} and {js}")


if __name__ == "__main__":
    main()
