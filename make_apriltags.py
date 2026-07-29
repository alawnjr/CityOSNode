#!/usr/bin/env python3
"""Generate printable AprilTag PDFs at an exact physical size.

    python3 make_apriltags.py 3 4 --size-mm 235 --paper a3 -o tags.pdf

One tag per page, laid out so the printed black square measures EXACTLY
--size-mm across. That number is what goes in SMARTROOM_TAG_SIZE_MM, and
getting it wrong scales every PnP translation by the same factor — a tag
printed at half its declared size puts the camera at half its true range.

Two things this is careful about:

SIZE CONVENTION. "Tag size" is ambiguous in the wild (black square? including
the white quiet zone? edge to centre?). This sidesteps the question by
generating with cv2.aruco, the same library realsense_extrinsics.py detects
with: cv2.aruco.detectMarkers returns the OUTER CORNERS OF THE BLACK BORDER,
and generateImageMarker(borderBits=1) produces an image whose extent is exactly
that border. So --size-mm is the full width of the black square, and generator
and detector agree by construction rather than by assumption.

PRINT SCALING. The page carries a 100 mm reference bar and tick marks aligned
to the tag's edges. Print at 100% ("Actual size", NOT "Fit to page"), then
measure the bar. If it is not 100 mm the printer scaled the page and the tag is
the wrong size — rescale or reprint rather than trusting it.

Sizing guidance (see CLAUDE.md): pose error scales as 1/tag_pixels, and
tag_px = f * size / range. The D455 sets the requirement, since its 90 deg FOV
gives it half the D435's focal length: it needs ~310 mm for 100 px at 2 m where
the D435 manages on ~145 mm.

A quiet zone eats into the sheet, so the largest black square each size holds is
smaller than the paper suggests — letter 173, A4 168, tabloid 224, A3 238,
A2 336, A1 475 mm (print --help to recompute). Measured against the tags
currently in the room, A3 at 235 mm puts the worst case (the D455's view of the
floor tag) at ~67 px, just over the 60 px gate; A2 at 336 mm takes it to ~95 px.
Floor tags want the larger end — a grazing view foreshortens them ~2x.

Requires opencv (opencv-python-headless, already in requirements.txt). The PDF
itself is written with the stdlib, so there is no reportlab dependency.
"""

import argparse
import math
import sys
import zlib
from pathlib import Path

MM = 72.0 / 25.4          # PostScript points per millimetre
# (width, height) in mm, portrait
REF_BAR_MM = 100.0       # printed scale-check bar
PAPER = {"a4": (210.0, 297.0), "a3": (297.0, 420.0), "a2": (420.0, 594.0),
         "a1": (594.0, 841.0), "letter": (215.9, 279.4), "tabloid": (279.4, 431.8)}


def tag_bitmap(tag_id, module_px=24):
    """(8-bit greyscale bitmap, module count) for one 36h11 tag.

    The bitmap is an exact integer multiple of the module count so every module
    is a whole number of pixels — with /Interpolate off, that prints crisp."""
    try:
        import cv2
        import numpy as np  # noqa: F401 - cv2 needs it present
    except ImportError as exc:
        sys.exit(f"opencv is required to generate the tag bitmaps: {exc}\n"
                 f"  pip install opencv-python-headless")
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    # 36h11 carries a 6x6 payload; borderBits=1 adds the one-module black border
    # that detectMarkers keys on, giving 8 modules across.
    modules = int(getattr(d, "markerSize", 6)) + 2
    try:
        img = cv2.aruco.generateImageMarker(d, int(tag_id), modules * module_px, borderBits=1)
    except cv2.error as exc:
        sys.exit(f"tag id {tag_id} is not in the 36h11 family (it holds 587 ids): {exc}")
    return img, modules


class Pdf:
    """Minimal PDF writer — just enough for placed images and text."""

    def __init__(self):
        self.objs = [None]        # object numbers are 1-based
        self.root = None

    def add(self, body):
        self.objs.append(body if isinstance(body, bytes) else body.encode("latin-1"))
        return len(self.objs) - 1

    def reserve(self):
        """Claim an object number now, fill it in later with set().

        Pages must be referenced by every page's /Parent but can only be written
        once its /Kids are known."""
        self.objs.append(None)
        return len(self.objs) - 1

    def set(self, num, body):
        self.objs[num] = body if isinstance(body, bytes) else body.encode("latin-1")
        return num

    def stream(self, dict_entries, data):
        head = f"<< {dict_entries} /Length {len(data)} >>\nstream\n".encode("latin-1")
        return self.add(head + data + b"\nendstream")

    def render(self):
        buf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = {}
        for num in range(1, len(self.objs)):
            offsets[num] = len(buf)
            buf += f"{num} 0 obj\n".encode("latin-1") + self.objs[num] + b"\nendobj\n"
        xref = len(buf)
        count = len(self.objs)
        buf += f"xref\n0 {count}\n".encode("latin-1")
        buf += b"0000000000 65535 f \n"
        for num in range(1, count):
            buf += f"{offsets[num]:010d} 00000 n \n".encode("latin-1")
        buf += (f"trailer\n<< /Size {count} /Root {self.root} 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n").encode("latin-1")
        return bytes(buf)


def esc(text):
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def page_content(size_mm, page_mm, modules, label_lines, quiet_modules=1,
                 ref_mm=REF_BAR_MM):
    """Content stream: the tag, edge ticks, a print-scale reference bar, labels."""
    pw, ph = page_mm
    quiet_mm = size_mm / modules * quiet_modules
    x = (pw - size_mm) / 2.0                  # centred horizontally
    # sit the tag high on the page, leaving room for the ruler and labels below
    y = ph - quiet_mm - size_mm - 6.0
    ops = []
    # the tag itself
    ops.append(f"q {size_mm * MM:.4f} 0 0 {size_mm * MM:.4f} {x * MM:.4f} {y * MM:.4f} cm /Im1 Do Q")
    # Tick marks aligned to the tag's edges, placed OUTSIDE the quiet zone so
    # they cannot be mistaken for tag structure. Measure between a pair to check
    # the printed size. The side pair needs a margin the sheet may not have — a
    # 235mm tag on A3 leaves 31mm, less than the quiet zone plus tick length —
    # so draw it only when it fits rather than off the edge of the page.
    tick, gap = 8.0, quiet_mm + 3.0
    ops.append("0.5 w 0 G")
    for ex in (x, x + size_mm):                       # marks the WIDTH
        ops.append(f"{ex * MM:.4f} {(y - gap) * MM:.4f} m {ex * MM:.4f} {(y - gap - tick) * MM:.4f} l S")
    side_ticks = x - gap - tick >= 2.0
    if side_ticks:                                    # marks the HEIGHT
        for ey in (y, y + size_mm):
            ops.append(f"{(x - gap) * MM:.4f} {ey * MM:.4f} m "
                       f"{(x - gap - tick) * MM:.4f} {ey * MM:.4f} l S")
    # print-scale reference bar with end caps
    by = y - gap - tick - 12.0
    bx = (pw - ref_mm) / 2.0
    ops.append(f"1 w {bx * MM:.4f} {by * MM:.4f} m {(bx + ref_mm) * MM:.4f} {by * MM:.4f} l S")
    for e in (bx, bx + ref_mm):
        ops.append(f"{e * MM:.4f} {(by - 2.5) * MM:.4f} m {e * MM:.4f} {(by + 2.5) * MM:.4f} l S")
    # labels
    label_lines = list(label_lines) + [
        ("Ticks mark the black square's four edges." if side_ticks else
         "Ticks below mark the black square's left and right edges (it is square, "
         "so that is its height too)."),
        "Keep the white margin around the tag clean - the detector needs it.",
    ]
    ty = by - 12.0
    for i, line in enumerate(label_lines):
        size = 11 if i == 0 else 8
        ops.append(f"BT /F1 {size} Tf {12 * MM:.4f} {(ty - i * 5.0) * MM:.4f} Td ({esc(line)}) Tj ET")
    return "\n".join(ops).encode("latin-1")


FURNITURE_MM = 60.0      # ticks + reference bar + label block, below the tag
TILE_LABEL_MM = 16.0     # strip under a tile for its cut label and scale bar
TILE_BAR_MM = 50.0       # per-tile scale check (a 100mm bar need not fit a tile)


def tile_grid(size_mm, paper, quiet_modules=1, modules=8, margin_mm=10.0):
    """How to split one tag across sheets: (ncols, nrows, tile_w, tile_h, canvas).

    The tiled canvas is the tag PLUS its quiet zone, so no cut line ever falls on
    the black border — a cut that strays a millimetre into the border changes
    where detectMarkers thinks the corner is, and the pose follows. Tiles are
    equal-sized rather than filling each sheet and leaving a runt at the end:
    equal tiles mean every join is the same, which is far easier to assemble
    accurately.

    `margin_mm` is the printer's non-printable border. Most inkjets and lasers
    cannot go closer than ~5mm to the paper edge; 10mm is a safe default."""
    pw, ph = PAPER[paper]
    canvas = size_mm + 2.0 * size_mm / modules * quiet_modules
    usable_w = pw - 2.0 * margin_mm
    usable_h = ph - 2.0 * margin_mm - TILE_LABEL_MM
    if usable_w <= 0 or usable_h <= 0:
        sys.exit(f"margin {margin_mm} mm leaves no printable area on {paper}")
    ncols = math.ceil(canvas / usable_w - 1e-9)
    nrows = math.ceil(canvas / usable_h - 1e-9)
    return ncols, nrows, canvas / ncols, canvas / nrows, canvas


def tile_content(size_mm, page_mm, modules, tag_id, col, row, grid, quiet_modules=1,
                 margin_mm=10.0):
    """One tile of a tiled tag: the clipped slice, cut lines, and assembly label."""
    ncols, nrows, tw, th, canvas = grid
    pw, ph = page_mm
    quiet_mm = size_mm / modules * quiet_modules
    # content window: centred across, pushed to the top so the label strip fits
    cx0 = (pw - tw) / 2.0
    cy0 = ph - margin_mm - th
    # canvas origin in sheet coordinates; row 0 is the TOP row as printed
    sx = cx0 - col * tw
    sy = cy0 - (nrows - 1 - row) * th
    ops = ["q",
           f"{cx0 * MM:.4f} {cy0 * MM:.4f} {tw * MM:.4f} {th * MM:.4f} re W n",
           f"{size_mm * MM:.4f} 0 0 {size_mm * MM:.4f} "
           f"{(sx + quiet_mm) * MM:.4f} {(sy + quiet_mm) * MM:.4f} cm /Im1 Do",
           "Q"]
    # CROP MARKS, never a cut line across the tile. A stroked rectangle on the
    # tile boundary is centred on the path, so half of it lands on the artwork —
    # stitched back together that reads as grey lines through the tag and the
    # detector rejects it outright (measured: reassembly with them found no tag
    # at all). So mark the corners from OUTSIDE and cut along a straightedge laid
    # between opposing marks; no ink ever touches the tag.
    ops.append("0.25 w 0 G")
    gap, mark = 1.0, 5.0
    for ex in (cx0, cx0 + tw):
        for ey in (cy0, cy0 + th):
            dx = -1.0 if ex == cx0 else 1.0
            dy = -1.0 if ey == cy0 else 1.0
            ops.append(f"{(ex + dx * gap) * MM:.4f} {ey * MM:.4f} m "
                       f"{(ex + dx * (gap + mark)) * MM:.4f} {ey * MM:.4f} l S")
            ops.append(f"{ex * MM:.4f} {(ey + dy * gap) * MM:.4f} m "
                       f"{ex * MM:.4f} {(ey + dy * (gap + mark)) * MM:.4f} l S")
    # scale bar + label in the strip below
    by = cy0 - 7.0
    bx = cx0
    ops.append(f"1 w {bx * MM:.4f} {by * MM:.4f} m {(bx + TILE_BAR_MM) * MM:.4f} {by * MM:.4f} l S")
    for e in (bx, bx + TILE_BAR_MM):
        ops.append(f"{e * MM:.4f} {(by - 2.0) * MM:.4f} m {e * MM:.4f} {(by + 2.0) * MM:.4f} l S")
    text = (f"tag {tag_id}  |  row {row + 1} of {nrows}, col {col + 1} of {ncols}  |  "
            f"black square {size_mm:g} mm  |  bar = {TILE_BAR_MM:g} mm, PRINT AT 100%  |  "
            f"cut between the corner marks, butt edges, tape behind")
    ops.append(f"BT /F1 7 Tf {bx * MM:.4f} {(by - 9.0) * MM:.4f} Td ({esc(text)}) Tj ET")
    return "\n".join(ops).encode("latin-1")


def max_size_mm(paper, modules=8, quiet_modules=1):
    """Largest black square that fits `paper` with its quiet zone and labels."""
    pw, ph = PAPER[paper]
    q = 1.0 + 2.0 * quiet_modules / modules          # tag+quiet as a multiple of size
    return min(pw / q, (ph - FURNITURE_MM) / q)


def build(tag_ids, size_mm, paper, module_px, out_path, quiet_modules=1,
          tile=False, margin_mm=10.0):
    pw, ph = PAPER[paper]
    pdf = Pdf()
    font = pdf.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    pages_ref = pdf.reserve()          # written last, but every page's /Parent needs it now
    page_objs = []
    grid = None
    for tag_id in tag_ids:
        img, modules = tag_bitmap(tag_id, module_px)
        h, w = img.shape[:2]
        limit = max_size_mm(paper, modules, quiet_modules)
        if tile:
            grid = tile_grid(size_mm, paper, quiet_modules, modules, margin_mm)
        elif size_mm > limit:
            fits = [p for p in PAPER if max_size_mm(p, modules, quiet_modules) >= size_mm]
            hint = (f"{min(fits, key=lambda p: PAPER[p][0] * PAPER[p][1])} or larger"
                    if fits else "a plotter / print shop")
            sys.exit(
                f"a {size_mm:g} mm tag needs {size_mm * (1 + 2 * quiet_modules / modules):.0f} mm "
                f"of paper once its {size_mm / modules * quiet_modules:.0f} mm white quiet zone "
                f"is included, plus {FURNITURE_MM:.0f} mm for the scale bar and labels.\n"
                f"  {paper} tops out at {limit:.0f} mm -- use --size-mm {limit:.0f} for this "
                f"sheet, --paper {hint}, or --tile to split it across several sheets.")
        image = pdf.stream(
            f"/Type /XObject /Subtype /Image /Width {w} /Height {h} "
            f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Interpolate false "
            f"/Filter /FlateDecode",
            zlib.compress(img.tobytes(), 9))
        labels = [
            f"AprilTag 36h11  id {tag_id}",
            f"black square = {size_mm:g} mm across  ->  SMARTROOM_TAG_SIZE_MM={size_mm:g}",
            "PRINT AT 100% / ACTUAL SIZE - do NOT use Fit to Page.",
            f"Then measure the bar above: it must be exactly {REF_BAR_MM:g} mm. If it is not,",
            "the printer rescaled the page and this tag is the wrong size.",
        ]
        def page(content_obj):
            return pdf.add(
                f"<< /Type /Page /Parent {pages_ref} 0 R "
                f"/MediaBox [0 0 {pw * MM:.4f} {ph * MM:.4f}] "
                f"/Resources << /XObject << /Im1 {image} 0 R >> /Font << /F1 {font} 0 R >> >> "
                f"/Contents {content_obj} 0 R >>")

        if tile:
            ncols, nrows = grid[0], grid[1]
            for row in range(nrows):          # top row first, reading order
                for col in range(ncols):
                    page_objs.append(page(pdf.stream("", tile_content(
                        size_mm, (pw, ph), modules, tag_id, col, row, grid,
                        quiet_modules, margin_mm))))
        else:
            page_objs.append(page(pdf.stream("", page_content(
                size_mm, (pw, ph), modules, labels, quiet_modules))))
    kids = " ".join(f"{n} 0 R" for n in page_objs)
    pdf.set(pages_ref, f"<< /Type /Pages /Count {len(page_objs)} /Kids [{kids}] >>")
    pdf.root = pdf.add(f"<< /Type /Catalog /Pages {pages_ref} 0 R >>")
    if any(o is None for o in pdf.objs[1:]):
        raise AssertionError("an object number was reserved but never filled in")
    Path(out_path).write_bytes(pdf.render())
    return pw, ph, len(page_objs)


def main():
    ap = argparse.ArgumentParser(
        description="Generate printable 36h11 AprilTag PDFs at an exact physical size.")
    ap.add_argument("ids", nargs="+", type=int, help="tag ids to generate, one per page")
    ap.add_argument("--size-mm", type=float, default=235.0,
                    help="width of the BLACK SQUARE in mm (default 235, the most that "
                         "fits A3 with its quiet zone)")
    ap.add_argument("--paper", default="a3", choices=sorted(PAPER),
                    help="sheet size (default a3)")
    ap.add_argument("--module-px", type=int, default=24,
                    help="pixels per tag module; higher = larger file, no sharper (default 24)")
    ap.add_argument("--tile", action="store_true",
                    help="split each tag across several sheets (for tags larger than the "
                         "paper); cut between the corner marks and butt the pieces together")
    ap.add_argument("--margin-mm", type=float, default=10.0,
                    help="printer's non-printable border, --tile only (default 10)")
    ap.add_argument("--quiet-modules", type=int, default=1,
                    help="modules of white margin around the tag (AprilTag needs 1; default 1)")
    ap.add_argument("-o", "--out", default="apriltags.pdf", help="output PDF path")
    args = ap.parse_args()

    pw, ph, n = build(args.ids, args.size_mm, args.paper, args.module_px, args.out,
                      args.quiet_modules, args.tile, args.margin_mm)
    print(f"{args.out}: {n} page(s) of {pw:.0f}x{ph:.0f} mm, "
          f"tag black square {args.size_mm:g} mm", file=sys.stderr)
    if args.tile:
        nc, nr, tw, th, canvas = tile_grid(args.size_mm, args.paper, args.quiet_modules,
                                           8, args.margin_mm)
        print(f"tiled {nc}x{nr} per tag ({nc * nr} sheets each): {canvas:.0f} mm of "
              f"artwork in {tw:.0f}x{th:.0f} mm pieces", file=sys.stderr)
    print(f"ids: {', '.join(str(i) for i in args.ids)}", file=sys.stderr)


if __name__ == "__main__":
    main()
