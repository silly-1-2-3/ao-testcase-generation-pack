#!/usr/bin/env python3
"""Command-line orchestration for PDF -> cell HTML -> structured outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pdf_to_html
import structured


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="input vector PDF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--headers", default=structured.DEFAULT_HEADERS,
        help="comma-separated columns; use | between alternative header rows",
    )
    parser.add_argument("--tolerance", type=float, default=structured.DEFAULT_TOLERANCE)
    parser.add_argument("--pages", help="optional page list/range, e.g. 1,3-5")
    parser.add_argument("--scale", type=float, default=1.35)
    parser.add_argument("--min-line-ratio", type=float,
                        default=pdf_to_html.DEFAULT_MIN_LINE_RATIO)
    parser.add_argument("--min-rect-area-ratio", type=float,
                        default=pdf_to_html.DEFAULT_MIN_RECT_AREA_RATIO)
    parser.add_argument("--axis-tolerance", type=float,
                        default=pdf_to_html.DEFAULT_AXIS_TOLERANCE)
    parser.add_argument("--merge-tolerance", type=float,
                        default=pdf_to_html.DEFAULT_MERGE_TOLERANCE)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_html = args.output_dir / "document.cells.html"
    output_html = args.output_dir / "document.structured.html"
    full_json = args.output_dir / "document.structured.long.json"
    compact_json = args.output_dir / "document.structured.short.json"

    pdf_to_html.extract(
        args.pdf, cell_html, pages=args.pages,
        min_line_ratio=args.min_line_ratio,
        min_rect_area_ratio=args.min_rect_area_ratio,
        axis_tol=args.axis_tolerance,
        merge_tol=args.merge_tolerance,
        scale=args.scale,
    )
    payload = structured.extract(cell_html, args.headers, args.tolerance)
    structured.write_outputs(payload, output_html, full_json, compact_json)

    print(f"[ok] tables={len(payload['tables'])} images={len(payload['images'])}")
    print(f"[ok] html: {output_html}")
    print(f"[ok] long json: {full_json}")
    print(f"[ok] short json: {compact_json}")
    print(f"[debug] cell html: {cell_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
