#!/usr/bin/env python3
"""Command-line orchestration for PDF -> cell HTML -> structured outputs."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pdf_to_html
import structured
import review
import templates
import cards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, nargs='?', help="input vector PDF")
    parser.add_argument('--from-cells', type=Path, help='reuse intermediate HTML in the same output directory')
    parser.add_argument('--overrides', type=Path, help='document-bound direction overrides from review UI/model')
    parser.add_argument('--white-tolerance', type=float, default=.02,
                        help='RGB distance from white (0..1); default .02 retains light gray')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--headers", default=None,
        help="comma-separated columns; use | between alternative header rows",
    )
    parser.add_argument('--file-template','-ft',help='file type matched against template regexes')
    parser.add_argument('--templates',type=Path,default=templates.DEFAULT_FILE,help='UTF-8 regex ::=>:: headers text file')
    parser.add_argument('--card-headers',default='工种,序号,工序内容|序号,项目内容',help='header groups whose rows become generation cards')
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
    parser = build_parser()
    args = parser.parse_args()
    if args.headers is None:
        args.headers = templates.resolve(args.file_template,args.templates) if args.file_template else structured.DEFAULT_HEADERS
    if bool(args.pdf) == bool(args.from_cells):
        parser.error('provide either PDF or --from-cells, not both')
    if not 0 <= args.white_tolerance <= 1:
        parser.error('--white-tolerance must be in [0,1]')
    if args.from_cells and args.from_cells.resolve() != (args.output_dir/'document.cells.html').resolve():
        parser.error('--from-cells must be document.cells.html in --output-dir, to preserve image links')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cell_html = args.output_dir / "document.cells.html"
    output_html = args.output_dir / "document.structured.html"
    full_json = args.output_dir / "document.structured.long.json"
    compact_json = args.output_dir / "document.structured.short.json"

    if args.pdf:
        source = args.output_dir/'source.pdf'
        if args.pdf.resolve()!=source.resolve(): shutil.copyfile(args.pdf,source)
        pdf_to_html.extract(
            args.pdf, cell_html, pages=args.pages,
            min_line_ratio=args.min_line_ratio,
            min_rect_area_ratio=args.min_rect_area_ratio,
            axis_tol=args.axis_tolerance,
            merge_tol=args.merge_tolerance,
            scale=args.scale, white_tolerance=args.white_tolerance,
        )
    decisions = review.load_overrides(args.overrides, review.identity(cell_html, args.headers, args.tolerance))
    tables = []
    payload = structured.extract(cell_html, args.headers, args.tolerance, overrides=decisions, review=tables)
    structured.write_outputs(payload, output_html, full_json, compact_json)
    review.write_review(cell_html, args.headers, args.tolerance, tables, decisions)
    compact=structured.compact_payload(payload)
    card_data=cards.make_cards(compact,'local',args.pdf.name if args.pdf else 'document',args.card_headers)
    (args.output_dir/'document.cards.json').write_text(json.dumps(card_data,ensure_ascii=False,indent=2),encoding='utf8')

    print(f"[ok] tables={len(payload['tables'])} images={len(payload['images'])}")
    print(f"[ok] html: {output_html}")
    print(f"[ok] long json: {full_json}")
    print(f"[ok] short json: {compact_json}")
    print(f"[debug] cell html: {cell_html}")
    print(f"[review] {args.output_dir / 'document.review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
