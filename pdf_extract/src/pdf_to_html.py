#!/usr/bin/env python3
"""Extract vector-PDF geometry, text, and images into a visual cell HTML file.

The extractor is deterministic and offline.  It reads embedded PDF text and
axis-aligned vector paths; it does not perform OCR or call an LLM.  Detected
table cells keep their page coordinates, while text outside cells is retained
in an invisible page-level fallback layer for debugging.
"""
from __future__ import annotations

import html
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Iterable

import pdfplumber


DEFAULT_MIN_LINE_RATIO = 0.005
DEFAULT_MIN_RECT_AREA_RATIO = 0.00005
DEFAULT_AXIS_TOLERANCE = 1.25
DEFAULT_MERGE_TOLERANCE = 2.5


def _snap(value: float, step: float = 0.25) -> float:
    return round(float(value) / step) * step


def _covers(a0: float, a1: float, b0: float, b1: float, tolerance: float) -> bool:
    return a0 <= b0 + tolerance and a1 >= b1 - tolerance


def _segment(x0: float, y0: float, x1: float, y1: float, axis_tolerance: float):
    dx, dy = float(x1) - float(x0), float(y1) - float(y0)
    if abs(dy) <= axis_tolerance and abs(dx) >= abs(dy):
        start, end = sorted((float(x0), float(x1)))
        y = (float(y0) + float(y1)) / 2.0
        return "h", start, y, end, y
    if abs(dx) <= axis_tolerance and abs(dy) > abs(dx):
        start, end = sorted((float(y0), float(y1)))
        x = (float(x0) + float(x1)) / 2.0
        return "v", x, start, x, end
    return None


def _merge_segments(segments: Iterable[tuple], horizontal: bool, tolerance: float):
    coordinates = []
    for segment in segments:
        if horizontal:
            _, x0, y, x1, _ = segment
            coordinates.append((y, x0, x1))
        else:
            _, x, y0, _, y1 = segment
            coordinates.append((x, y0, y1))
    coordinates.sort(key=lambda item: (item[0], item[1], item[2]))

    bands = []
    for coordinate, start, end in coordinates:
        if bands and abs(bands[-1][0] - coordinate) <= tolerance:
            bands[-1][1].append((start, end))
        else:
            bands.append((coordinate, [(start, end)]))
    merged = []
    for coordinate, intervals in bands:
        runs = []
        # Sort along the line AFTER grouping near-equal x/y coordinates.
        # Otherwise a later parallel edge may start above an earlier segment
        # and incorrectly bridge a genuine gap (creating phantom cell walls).
        for start, end in sorted(intervals):
            if runs and start <= runs[-1][1] + tolerance:
                runs[-1] = (runs[-1][0], max(runs[-1][1], end))
            else:
                runs.append((start, end))
        merged.extend((coordinate, start, end) for start, end in runs)
    if horizontal:
        return [(start, coordinate, end, coordinate)
                for coordinate, start, end in merged if end > start]
    return [(coordinate, start, coordinate, end)
            for coordinate, start, end in merged if end > start]


def _curve_segments(page, axis_tolerance: float):
    for curve in page.curves:
        points = list(curve.get("pts") or [])
        if len(points) < 2:
            continue
        pairs = list(zip(points, points[1:]))
        if curve.get("close", False) or curve.get("closed", False):
            pairs.append((points[-1], points[0]))
        for first, second in pairs:
            if len(first) < 2 or len(second) < 2:
                continue
            segment = _segment(
                first[0], first[1], second[0], second[1], axis_tolerance
            )
            if segment:
                yield segment


def page_segments(page, min_line_ratio: float, axis_tolerance: float,
                  merge_tolerance: float):
    """Collect usable horizontal/vertical edges from lines, rects and curves."""
    minimum_length = max(page.width, page.height) * max(0.0, min_line_ratio)
    horizontal, vertical = [], []

    def add(x0, y0, x1, y1) -> None:
        segment = _segment(x0, y0, x1, y1, axis_tolerance)
        if not segment:
            return
        length = (segment[3] - segment[1] if segment[0] == "h"
                  else segment[4] - segment[2])
        if length >= minimum_length:
            (horizontal if segment[0] == "h" else vertical).append(segment)

    for line in page.lines:
        add(
            line.get("x0", 0), line.get("top", line.get("y0", 0)),
            line.get("x1", 0), line.get("bottom", line.get("y1", 0)),
        )
    for rectangle in page.rects:
        x0, x1 = float(rectangle["x0"]), float(rectangle["x1"])
        top, bottom = float(rectangle["top"]), float(rectangle["bottom"])
        add(x0, top, x1, top)
        add(x0, bottom, x1, bottom)
        add(x0, top, x0, bottom)
        add(x1, top, x1, bottom)
    for segment in _curve_segments(page, axis_tolerance):
        length = (segment[3] - segment[1] if segment[0] == "h"
                  else segment[4] - segment[2])
        if length >= minimum_length:
            (horizontal if segment[0] == "h" else vertical).append(segment)

    return (
        _merge_segments(horizontal, True, merge_tolerance),
        _merge_segments(vertical, False, merge_tolerance),
    )


def extract_words(page) -> list[dict]:
    words = []
    for word in page.extract_words(
        keep_blank_chars=True, use_text_flow=False, x_tolerance=1, y_tolerance=3
    ):
        text = (word.get("text") or "").strip()
        if not text:
            continue
        x0, x1 = float(word["x0"]), float(word["x1"])
        y0, y1 = float(word["top"]), float(word["bottom"])
        words.append({
            "text": text,
            "x0": x0,
            "x1": x1,
            "y0": y0,
            "y1": y1,
            "cx": (x0 + x1) / 2,
            "cy": (y0 + y1) / 2,
            "height": max(1.0, y1 - y0),
        })
    return words


def _intersection_map(horizontal, vertical, tolerance: float):
    intersections = {}
    for horizontal_edge in horizontal:
        hx0, hy, hx1, _ = horizontal_edge
        for vertical_edge in vertical:
            vx, vy0, _, vy1 = vertical_edge
            if (vy0 - tolerance <= hy <= vy1 + tolerance
                    and hx0 - tolerance <= vx <= hx1 + tolerance):
                key = _snap(vx), _snap(hy)
                item = intersections.setdefault(key, {"h": [], "v": []})
                item["h"].append(horizontal_edge)
                item["v"].append(vertical_edge)
    return intersections


def _connects(first, second, intersections, horizontal: bool,
              tolerance: float) -> bool:
    if first not in intersections or second not in intersections:
        return False
    if horizontal:
        start, end = sorted((first[0], second[0]))
        return any(
            _covers(edge[0], edge[2], start, end, tolerance)
            for edge in intersections[first]["h"]
        )
    start, end = sorted((first[1], second[1]))
    return any(
        _covers(edge[1], edge[3], start, end, tolerance)
        for edge in intersections[first]["v"]
    )


def find_smallest_rectangles(horizontal, vertical, tolerance: float):
    """Find leaf rectangles by walking intersections instead of H²×V² search."""
    intersections = _intersection_map(horizontal, vertical, tolerance)
    rows, columns = defaultdict(list), defaultdict(list)
    for x, y in intersections:
        rows[y].append(x)
        columns[x].append(y)
    for values in rows.values():
        values.sort()
    for values in columns.values():
        values.sort()

    result = set()
    for x, y in sorted(intersections, key=lambda point: (point[1], point[0])):
        right = [rx for rx in rows[y] if rx > x + tolerance]
        below = [by for by in columns[x] if by > y + tolerance]
        best = None
        for bottom_y in below:
            bottom_left = _snap(x), _snap(bottom_y)
            if not _connects((x, y), bottom_left, intersections, False, tolerance):
                continue
            for right_x in right:
                top_right = _snap(right_x), _snap(y)
                bottom_right = _snap(right_x), _snap(bottom_y)
                if (bottom_right not in intersections
                        or not _connects((x, y), top_right, intersections, True, tolerance)
                        or not _connects(top_right, bottom_right, intersections, False, tolerance)
                        or not _connects(bottom_left, bottom_right, intersections, True, tolerance)):
                    continue
                area = (right_x - x) * (bottom_y - y)
                if best is None or area < best[0]:
                    best = area, (x, y, right_x, bottom_y)
        if best:
            result.add(tuple(round(value, 3) for value in best[1]))
    return sorted(result, key=lambda rectangle: (rectangle[1], rectangle[0]))


def _group_lines(words, tolerance: float = 3.0):
    rows = []
    for word in sorted(words, key=lambda item: (item["cy"], item["x0"])):
        row = next(
            (item for item in rows if abs(item[0] - word["cy"])
             <= max(tolerance, word["height"] * 0.6)),
            None,
        )
        if row is None:
            row = [word["cy"], []]
            rows.append(row)
        row[1].append(word)
    return [
        sorted(row[1], key=lambda item: item["x0"])
        for row in sorted(rows, key=lambda item: item[0])
    ]


def _contains(outer, inner, tolerance: float = 2.5) -> bool:
    outer_area = (outer[2] - outer[0]) * (outer[3] - outer[1])
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] >= inner[2] - tolerance
        and outer[3] >= inner[3] - tolerance
        and outer_area > inner_area + tolerance
    )


def _cell_records(rectangles, words, page_width: float, page_height: float,
                  min_area_ratio: float, tolerance: float):
    """Assign each word to its smallest real cell; retain unassigned page text."""
    page_area = page_width * page_height
    rectangles = [
        rectangle for rectangle in rectangles
        if ((rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])
            >= page_area * min_area_ratio)
    ]
    rectangles.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    owners = defaultdict(list)
    assigned = set()
    for word in words:
        for index, (x0, y0, x1, y1) in enumerate(rectangles):
            if (x0 - tolerance <= word["cx"] <= x1 + tolerance
                    and y0 - tolerance <= word["cy"] <= y1 + tolerance):
                owners[index].append(word)
                assigned.add(id(word))
                break

    records = [
        {
            "x0": rectangles[index][0],
            "y0": rectangles[index][1],
            "x1": rectangles[index][2],
            "y1": rectangles[index][3],
            "words": cell_words,
        }
        # Empty cells are structural evidence: blank sequence columns on a
        # continuation page, empty measurements, and inner-table containers.
        # Dropping them turns aligned rows into seemingly incompatible grids.
        for index, cell_words in sorted(
            ((index, owners[index]) for index in range(len(rectangles))), key=lambda item: (
                rectangles[item[0]][1], rectangles[item[0]][0]
            )
        )
    ]
    real_cells = list(records)
    for cell in real_cells:
        outer = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
        cell["is_container"] = any(
            other is not cell and _contains(
                outer,
                (other["x0"], other["y0"], other["x1"], other["y1"]),
                tolerance,
            )
            for other in real_cells
        )

    outside = [word for word in words if id(word) not in assigned]
    if outside:
        records.append({
            "x0": 0.0,
            "y0": 0.0,
            "x1": page_width,
            "y1": page_height,
            "words": outside,
            "is_page_text": True,
        })
    return records


def _plain_text(cell) -> str:
    lines = []
    for row in _group_lines(cell["words"]):
        output, previous = [], cell["x0"]
        for word in row:
            gap = max(0.0, word["x0"] - previous)
            spaces = int(round(gap / max(word["height"] * 0.55, 1.0)))
            if output and spaces == 0:
                spaces = 1
            output.append(" " * min(32, max(0, spaces)) + word["text"])
            previous = word["x1"]
        lines.append("".join(output).strip())
    return "\n".join(lines)


def _cell_html(cell, scale: float) -> str:
    x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
    spans = []
    for row in _group_lines(cell["words"]):
        for word in row:
            size = max(6.0, min(24.0, word["height"] * scale * 0.9))
            style = (
                f"left:{(word['x0'] - x0) * scale:.1f}px;"
                f"top:{(word['y0'] - y0) * scale:.1f}px;"
                f"font-size:{size:.1f}px;"
                f"line-height:{max(size, word['height'] * scale):.1f}px"
            )
            spans.append(
                f'<span class="word" style="{style}">{html.escape(word["text"])}</span>'
            )
    style = (
        f"left:{x0 * scale:.1f}px;top:{y0 * scale:.1f}px;"
        f"width:{(x1 - x0) * scale:.1f}px;height:{(y1 - y0) * scale:.1f}px"
    )
    classes = ["cell"]
    if cell.get("is_page_text"):
        classes.append("page-text")
    if cell.get("is_container"):
        classes.append("container")
    background = cell.get('background', '')
    if background:
        style += f';background:{background}'
    attrs = (f' data-background="{background}"'
             f' data-background-coverage="{cell.get("background_coverage", 0)}"')
    return f'<div class="{" ".join(classes)}"{attrs} style="{style}">{"".join(spans)}</div>'


def _parse_pages(specification: str | None, page_count: int) -> set[int]:
    if not specification:
        return set(range(1, page_count + 1))
    selected = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            selected.update(range(max(1, int(start)), min(page_count, int(end)) + 1))
        else:
            selected.add(int(token))
    return selected


def _image_record(page_number: int, source: str, bbox, scale: float) -> dict:
    x0, y0, x1, y1 = map(float, bbox)
    return {
        "page": page_number,
        "src": source,
        "bbox": [x0, y0, x1, y1],
        "left": x0 * scale,
        "top": y0 * scale,
        "width": (x1 - x0) * scale,
        "height": (y1 - y0) * scale,
    }


def _save_images_with_pymupdf(pdf: Path, image_dir: Path,
                              scale: float) -> list[dict] | None:
    try:
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
    except ImportError:
        return None

    result = []
    document = pymupdf.open(str(pdf))
    try:
        for page_number, page in enumerate(document, 1):
            image_index = 0
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 1 or not block.get("image"):
                    continue
                image_index += 1
                extension = str(block.get("ext") or "png").lower()
                if extension not in {
                    "png", "jpg", "jpeg", "jpx", "webp", "tiff", "bmp"
                }:
                    extension = "png"
                target = image_dir / (
                    f"page-{page_number:04d}-image-{image_index:03d}.{extension}"
                )
                target.write_bytes(block["image"])
                result.append(_image_record(
                    page_number, f"images/{target.name}", block["bbox"], scale
                ))
    finally:
        document.close()
    return result


def _save_images_with_pdfplumber(pdf: Path, image_dir: Path,
                                 scale: float) -> list[dict]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Image export requires PyMuPDF, or pdfplumber plus Pillow"
        ) from error

    result = []
    with pdfplumber.open(str(pdf)) as document:
        for page_number, page in enumerate(document.pages, 1):
            for image_index, image in enumerate(page.images or [], 1):
                stream = image.get("stream")
                if stream is None:
                    continue
                try:
                    decoded = Image.open(BytesIO(stream.get_data()))
                    target = image_dir / (
                        f"page-{page_number:04d}-image-{image_index:03d}.png"
                    )
                    decoded.save(target, format="PNG")
                except Exception:
                    # Some PDF image streams are not browser-decodable images.
                    continue
                bbox = [
                    image.get("x0", 0), image.get("top", 0),
                    image.get("x1", 0), image.get("bottom", 0),
                ]
                result.append(_image_record(
                    page_number, f"images/{target.name}", bbox, scale
                ))
    return result


def _save_images(pdf: Path, output: Path, scale: float) -> list[dict]:
    image_dir = output.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    result = _save_images_with_pymupdf(pdf, image_dir, scale)
    return result if result is not None else _save_images_with_pdfplumber(
        pdf, image_dir, scale
    )


def _inject_images(html_text: str, images: list[dict]) -> str:
    for image in images:
        page_start = html_text.find(
            f'<section class="page" data-page="{image["page"]}"'
        )
        if page_start < 0:
            continue
        marker = '<div class="canvas">'
        canvas = html_text.find(marker, page_start)
        if canvas < 0:
            continue
        insert_at = canvas + len(marker)
        tag = (
            f'<img class="pdf-image" data-image="{html.escape(image["src"])}" '
            f'src="{html.escape(image["src"])}" '
            f'style="position:absolute;left:{image["left"]:.1f}px;'
            f'top:{image["top"]:.1f}px;width:{image["width"]:.1f}px;'
            f'height:{image["height"]:.1f}px;z-index:2" alt="PDF image">\n'
        )
        html_text = html_text[:insert_at] + tag + html_text[insert_at:]
    return html_text.replace(
        "</style>",
        ".pdf-image{object-fit:contain;pointer-events:none}\n</style>",
        1,
    )


def extract(
    pdf: Path,
    output: Path,
    *,
    pages: str | None = None,
    min_line_ratio: float = DEFAULT_MIN_LINE_RATIO,
    min_rect_area_ratio: float = DEFAULT_MIN_RECT_AREA_RATIO,
    axis_tol: float = DEFAULT_AXIS_TOLERANCE,
    merge_tol: float = DEFAULT_MERGE_TOLERANCE,
    scale: float = 1.35,
    white_tolerance: float = .02,
) -> None:
    """Write geometry-preserving cell HTML and decoded images for ``pdf``."""
    sections = []
    with pdfplumber.open(str(pdf)) as document:
        selected = _parse_pages(pages, len(document.pages))
        for page_number, page in enumerate(document.pages, 1):
            if page_number not in selected:
                continue
            words = extract_words(page)
            horizontal, vertical = page_segments(
                page, min_line_ratio, axis_tol, merge_tol
            )
            corners = _intersection_map(horizontal, vertical, merge_tol)
            rectangles = find_smallest_rectangles(
                horizontal, vertical, merge_tol
            )
            cells = _cell_records(
                rectangles,
                words,
                page.width,
                page.height,
                min_rect_area_ratio,
                merge_tol,
            )
            from background import annotate_cells
            annotate_cells(cells, page, white_tolerance=white_tolerance)
            bits = [
                f'<section class="page" data-page="{page_number}" id="page-{page_number}" '
                f'style="width:{page.width * scale:.1f}px;'
                f'height:{page.height * scale:.1f}px">',
                f'<div class="page-label">第 {page_number} 页 · '
                f'edges={len(horizontal) + len(vertical)} · '
                f'corners={len(corners)} · cells={len(cells)}</div>',
                '<div class="canvas">',
            ]
            for x0, y0, x1, _ in horizontal:
                bits.append(
                    f'<i class="rule h" style="left:{x0 * scale:.1f}px;'
                    f'top:{y0 * scale:.1f}px;width:{(x1 - x0) * scale:.1f}px"></i>'
                )
            for x0, y0, _, y1 in vertical:
                bits.append(
                    f'<i class="rule v" style="left:{x0 * scale:.1f}px;'
                    f'top:{y0 * scale:.1f}px;height:{(y1 - y0) * scale:.1f}px"></i>'
                )
            plain = []
            for index, cell in enumerate(cells, 1):
                bits.append(_cell_html(cell, scale))
                plain.append(f"[{index}] {_plain_text(cell)}")
            bits.extend([
                "</div></section>",
                '<details class="debug"><summary>按闭合矩形提取的文本顺序（含页面兜底框）</summary>',
                "<pre>" + html.escape("\n".join(plain)) + "</pre></details>",
            ])
            sections.append("\n".join(bits))

    css = """<meta charset="utf-8"><title>矢量 PDF 表格结构提取</title><style>
*{box-sizing:border-box}body{margin:0;background:#d8d8d8;font-family:Arial,'Microsoft YaHei',sans-serif;color:#222}
.page{position:relative;margin:24px auto;background:#fff;box-shadow:0 2px 8px #777;page-break-after:always;overflow:hidden}
.page-label{position:absolute;left:5px;top:3px;padding:1px 4px;background:rgba(255,255,255,.82);font-size:11px;color:#555;z-index:8}
.canvas{position:absolute;inset:0}.rule{position:absolute;background:#777;display:block;z-index:1}.rule.h{height:1px}.rule.v{width:1px}
.cell{position:absolute;border:1px solid rgba(30,110,210,.72);background:rgba(210,230,255,.16);z-index:3;overflow:hidden}
.cell.page-text,.cell.container{border:0;background:transparent;outline:0;pointer-events:none}.cell.page-text{z-index:2}
.word{position:absolute;white-space:pre;z-index:4;background:rgba(255,255,220,.72)}
.debug{width:min(1100px,96vw);margin:-12px auto 24px;background:#fff;padding:6px 10px;font-size:12px}
pre{white-space:pre-wrap;background:#f5f5f5;padding:8px;max-height:320px;overflow:auto}
@media print{body{background:#fff}.page{margin:0;box-shadow:none}.debug{display:none}}
</style>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        '<!doctype html><html><head>' + css + '</head><body>'
        + "\n".join(sections) + '</body></html>',
        encoding="utf-8",
    )
    images = _save_images(pdf, output, scale)
    output.write_text(
        _inject_images(output.read_text(encoding="utf-8"), images),
        encoding="utf-8",
    )
