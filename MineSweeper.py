#!/usr/bin/env python3  # (Shebang ignored on Windows—run via `python minesweeper_solver.py …`)
"""Minesweeper Solver – debug‑friendly version (2025‑06‑30)

Goal of this revision
---------------------
* **Robust rozlišení zakrytá × odhalená „0“** – bez falešných nul na neotevřených
  dlaždicích.
* Odstraněny zbytky duplicitního kódu a dokončen `main()`.
* Funkce `classify()` je teď přehledná, všechny průběžné proměnné mají lokální
  jména, žádné nedeklarované `edge_cnt`.

> Výpočet pravděpodobností min je stále placeholder – zaměřujeme se na spolehlivou
> vizuální klasifikaci políček. Jakmile rozeznávání stabilně funguje, můžeme
> přidat skutečný solver.
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError

UNKNOWN, FLAGGED = -2, -1  # interní kódy
Coord = Tuple[int, int]

@dataclass
class SolverResult:
    probs: np.ndarray
    best:  Coord

# ─────────────────────────── OCR setup ────────────────────────────────────────

def setup_tesseract(cmd: str | None) -> bool:
    if cmd:
        exe = Path(cmd)
        if not exe.exists():
            sys.exit(f"[ERR] tesseract.exe not found at {cmd}")
        pytesseract.pytesseract.tesseract_cmd = str(exe)
    try:
        pytesseract.get_tesseract_version()
        return True
    except TesseractNotFoundError:
        print('[WARN] Tesseract not found – OCR disabled')
        return False

# ─────────────────── template fallback (0–8) ─────────────────────────────────-
TPL_DIR = Path(__file__).with_suffix('').parent / 'templates'
DIGIT_TPL: dict[int, np.ndarray] = {}
if TPL_DIR.exists():
    for n in range(9):
        g = cv2.imread(str(TPL_DIR / f"{n}.png"), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            DIGIT_TPL[n] = g
else:
    print('[INFO] templates/ folder not found – OCR fallback disabled')


def color_digit(bgr: np.ndarray) -> int | None:
    """Return Minesweeper digit (1‑8) by dominant HSV colour.
    Thresholds refined so that:
        * bright red   (H∈[0,6]∪[174,180], S>120, V>130)      → 3
        * dark red/brown (H∈[0,15], S>100, V 60‑130)           → 5
        * bright blue  (H 100‑135, S>120, V>120)               → 1
        * navy blue    (H 100‑135, S>120, V  60‑120)           → 4
        * green       (H 45‑85,  S>120, V>100)                → 2
        * teal/cyan   (H 85‑100, S>120, V>100)                → 6
        * black       (S<50, V<60)                            → 7
        * grey        (S<50, 80<V<160)                        → 8
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (s > 50) & (v > 60)
    if np.count_nonzero(mask) < 5:
        return None
    hvals = h[mask]
    smean = int(np.mean(s[mask]))
    vmean = int(np.mean(v[mask]))
    hist = np.histogram(hvals, bins=180, range=(0,180))[0]
    peak = hist.argmax()

    # Unsaturated cases first (7,8)
    if smean < 50:
        if vmean < 60:
            return 7
        if 80 < vmean < 160:
            return 8
        return None

    # Red / Brown / 3 or 5
    if peak < 15 or peak > 170:
        return 3 if vmean > 130 else 5

    # Green 2
    if 45 < peak < 85:
        return 2

    # Cyan 6
    if 85 < peak < 100:
        return 6

    # Blue family 1 / 4 (bright vs dark)
    if 100 < peak < 135:
        return 1 if vmean > 120 else 4

    return None  # málo barevných bodů → 7 nebo 8 nebo neznámé

def detect_digit(gray: np.ndarray, bgr: np.ndarray, ocr_ok: bool) -> int | None:
    """Return digit 0‑8 or None.
    Priority: colour‑based → template → Tesseract."""
    cd = color_digit(bgr)
    if cd is not None:
        return cd

    if DIGIT_TPL:
        best, score = None, 1.0
        for d, tmpl in DIGIT_TPL.items():
            m = cv2.matchTemplate(gray, tmpl, cv2.TM_SQDIFF_NORMED).min()
            if m < score:
                best, score = d, m
        if best is not None and score < 0.25:
            return best

    if ocr_ok:
        txt = pytesseract.image_to_string(
            gray,
            config='--psm 10 --oem 3 -c tessedit_char_whitelist=012345678'
        ).strip()
        if txt.isdigit():
            v = int(txt)
            if 0 <= v <= 8:
                return v
    return None

# ─────────────────────── cell classification ─────────────────────────────────

def is_flag(bgr: np.ndarray) -> bool:
    """Heuristic flag detector.
    • klasická vlajka zaujímá podstatně větší červenou plochu než číslice „3“.
    • navíc obsahuje tmavou (černou) tyčku‑podstavec.
    Vrací True, pokud počet sytě červených *a* tmavých pixelů
    překročí relativní práh podle velikosti dlaždice.
    """
    h, w = bgr.shape[:2]
    pix = h * w

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0,70,70), (12,255,255)) | cv2.inRange(hsv, (170,70,70), (180,255,255))
    red_cnt = int(red.sum() / 255)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    black_cnt = int(np.count_nonzero(gray < 40))

    dark_thresh = max(int(pix * 0.04), 15)
    return red_cnt > pix * 0.20 and black_cnt > dark_thresh


def classify(bgr: np.ndarray, ocr_ok: bool) -> int:
    """Classify a single Minesweeper tile."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 1️⃣ Digit via colour / template / OCR
    digit = detect_digit(gray, bgr, ocr_ok)
    if digit is not None:
        return digit

    # 2️⃣ Flag (after digit to avoid 3→flag omyl)
    if is_flag(bgr):
        return FLAGGED

    # 3️⃣ Blank revealed 0
    edges = cv2.Canny(gray, 40, 120)
    if gray.mean() > 185 and edges.sum() / 255 < 0.08 * (gray.shape[0]*gray.shape[1]):
        return 0

    # 4️⃣ Covered
    tl, tr, bl, br = gray[1, 1], gray[1, -2], gray[-2, 1], gray[-2, -2]
    if (max(tl, tr, bl, br) - min(tl, tr, bl, br)) > 40:
        return UNKNOWN

    return UNKNOWN

# ───────────────────────── grid helpers ───────────────────────────────────────

def split(img: np.ndarray, rows: int, cols: int):
    h, w = img.shape[:2]
    ch, cw = h // rows, w // cols
    return [[img[r*ch:(r+1)*ch, c*cw:(c+1)*cw] for c in range(cols)] for r in range(rows)]


def build_board(cells, rows, cols, ocr_ok):
    board = np.empty((rows, cols), int)
    for r in range(rows):
        for c in range(cols):
            board[r, c] = classify(cells[r][c], ocr_ok)
    return board

# ────────────────────────── dummy solver ─────────────────────────────────────-

def solve(board: np.ndarray) -> SolverResult:
    probs = np.full(board.shape, np.nan)
    mask = board == UNKNOWN
    probs[mask] = 0.5  # uniform placeholder
    best = tuple(np.argwhere(mask)[0]) if mask.any() else (-1, -1)
    return SolverResult(probs, best)

# ───────────────────────── debug vizualizace ─────────────────────────────────

def save_debug(img: np.ndarray, board: np.ndarray):
    rows, cols = board.shape
    h, w = img.shape[:2]
    ch, cw = h // rows, w // cols
    vis = img.copy()
    for r in range(rows):
        for c in range(cols):
            cv2.rectangle(vis, (c*cw, r*ch), (c*cw+cw, r*ch+ch), (0,255,0), 1)
            val = board[r, c]
            col = (0,0,255) if val == FLAGGED else (0,0,0)
            cv2.putText(vis, str(val), (c*cw+2, r*ch+ch-4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    cv2.imwrite('debug_grid.png', vis)

# ───────────────────────── overlay pravděpodobností ──────────────────────────

def draw_overlay(img: np.ndarray, probs: np.ndarray, best: Coord):
    rows, cols = probs.shape
    h, w = img.shape[:2]
    ch, cw = h // rows, w // cols
    vis = img.copy()
    for r in range(rows):
        for c in range(cols):
            if math.isnan(probs[r, c]):
                continue
            cv2.putText(vis, f"{probs[r, c]:.2f}", (c*cw+2, r*ch+ch-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1, cv2.LINE_AA)
    if best != (-1, -1):
        br, bc = best
        cv2.rectangle(vis, (bc*cw, br*ch), (bc*cw+cw, br*ch+ch), (255,0,0), 2)
    return vis

# ───────────────────────── CLI & entrypoint ─────────────────────────────────--

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument('image')
    p.add_argument('--rows', type=int, required=True)
    p.add_argument('--cols', type=int, required=True)
    p.add_argument('--show', action='store_true')
    p.add_argument('--tesscmd')
    p.add_argument('--out', default='overlay.png')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_cli()
    img = cv2.imread(args.image)
    if img is None:
        sys.exit('Cannot read image')

    ocr = setup_tesseract(args.tesscmd)
    board = build_board(split(img, args.rows, args.cols), args.rows, args.cols, ocr)
    save_debug(img, board)

    digit_cnt = int(((board >= 0) & (board <= 8)).sum())
    print(f'[INFO] recognised digits: {digit_cnt}')

    res = solve(board)
    overlay = draw_overlay(img, res.probs, res.best)
    cv2.imwrite(args.out, overlay)
    if args.show:
        cv2.imshow('overlay', overlay);
        cv2.waitKey(0)
        cv2.destroyAllWindows()
