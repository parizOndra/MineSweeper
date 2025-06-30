# Minesweeper Solver

This repository contains a proof‑of‑concept Python script for analysing a Minesweeper screenshot. It attempts to recognise the digits / flags and (in later versions) should compute recommended moves.

## Files

- `MineSweeper.py` – main script performing the image processing.
- `Board.jpg` – example screenshot of a Minesweeper board.
- `debug_grid.png` – output with recognised values for each tile.
- `overlay.png` – output with overlaid probabilities (currently only placeholders).

## Requirements

- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/) (`cv2`)
- [NumPy](https://numpy.org/)
- [pytesseract](https://pypi.org/project/pytesseract/) and the `tesseract` executable for OCR (optional).

Install dependencies for example with:

```bash
pip install opencv-python numpy pytesseract
```

You must also have the `tesseract` binary in your PATH or specify it with `--tesscmd`.

## Usage

```bash
python MineSweeper.py <image> --rows <rows> --cols <cols> [--show] [--tesscmd <path>] [--out <file>]
```

- `image` – path to the screenshot of the board.
- `--rows` and `--cols` – dimensions of the grid.
- `--show` – display the resulting overlay in a window.
- `--tesscmd` – optional path to `tesseract.exe` (useful on Windows).
- `--out` – filename for the overlay image (default `overlay.png`).

Example (with the included board image):

```bash
python MineSweeper.py Board.jpg --rows 16 --cols 30 --show
```

The script will save `debug_grid.png` with recognised numbers and `overlay.png` with a probability overlay. The solver performs a few basic deterministic deductions, marking tiles that must contain a mine with probability **1.0** and safe tiles with **0.0**. Remaining unknowns keep a default 0.5 probability. Revealed tiles are shown with a fixed 0.0 probability.

## Notes

The script uses a simple heuristic to detect flags, looking for a vertical black mast topped with a red pennant. This avoids confusing flags with the red digit "3". The code currently contains a few placeholder pieces and may require adjustments for your environment. For example, a folder `templates/` with digit images can be used as a fallback when OCR is not available.
