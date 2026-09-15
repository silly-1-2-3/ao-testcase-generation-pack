#!/usr/bin/env python3
"""Turn geometry-preserving cell HTML into nested AO tables.

This module contains the current table semantics in one place: exact header
selection, nested-grid discovery, cross-page sequence inheritance, image/text
reading order, human-readable HTML rendering, and compact model JSON output.
"""
from __future__ import annotations

import copy
import html
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


DEFAULT_TOLERANCE = 2.5
DEFAULT_HEADERS = (
    "工种,序号,工序内容|序号,工序内容|序号,检查操作程序,响应及显示"
)


def _norm(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").replace("\u3000", ""))
    return text.strip(" :：;；/\\")


def _style(style: str) -> dict[str, float | str]:
    result: dict[str, float | str] = {}
    for key, value in re.findall(r"([\w-]+)\s*:\s*([^;]+)", style or ""):
        value = value.strip()
        match = re.match(r"[-+]?\d+(?:\.\d+)?", value)
        result[key] = float(match.group(0)) if match else value
    return result


def _bbox_from_style(style: str) -> list[float] | None:
    values = _style(style)
    keys = "left", "top", "width", "height"
    if not all(isinstance(values.get(key), (int, float)) for key in keys):
        return None
    left, top = float(values["left"]), float(values["top"])
    return [
        left,
        top,
        left + float(values["width"]),
        top + float(values["height"]),
    ]


def _area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _width(bbox: list[float]) -> float:
    return max(1.0, bbox[2] - bbox[0])


def _contains(outer: list[float], inner: list[float], tolerance: float) -> bool:
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] >= inner[2] - tolerance
        and outer[3] >= inner[3] - tolerance
        and _area(outer) > _area(inner) + tolerance
    )


def _overlap_x(first: list[float], second: list[float]) -> float:
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0]))


@dataclass
class ImageRef:
    page: int
    src: str
    bbox: list[float] | None = None
    width: float | None = None
    height: float | None = None
    alt: str = "PDF image"


@dataclass
class Cell:
    page: int
    index: int
    bbox: list[float]
    text: str = ""
    images: list[ImageRef] = field(default_factory=list)


@dataclass
class Table:
    page: int
    header_cells: list[Cell]
    rows: list[list[Cell]]
    bbox: list[float]
    nested: list["Table"] = field(default_factory=list)
    parent_cell: Cell | None = None
    title: str = ""

    @property
    def headers(self) -> list[str]:
        return [cell.text.strip() for cell in self.header_cells]


class CellHtmlParser(HTMLParser):
    """Read real cells and embedded-image positions from intermediate HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.pages: dict[int, list[Cell]] = {}
        self.images: list[ImageRef] = []
        self.page = 0
        self.cell: Cell | None = None
        self.cell_depth = 0
        self.cell_parts: list[tuple[float, float, str]] = []
        self.span_depth = 0
        self.span_style: dict[str, float | str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {key: value or "" for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        if tag == "section" and "page" in classes:
            self.page = int(attributes.get("data-page") or len(self.pages) + 1)
            self.pages.setdefault(self.page, [])
            return
        if self.page <= 0:
            return
        if tag == "div" and "cell" in classes and "page-text" not in classes:
            bbox = _bbox_from_style(attributes.get("style", ""))
            if bbox:
                self.cell = Cell(self.page, len(self.pages[self.page]), bbox)
                self.cell_depth = 1
                self.cell_parts = []
            return
        if self.cell is not None:
            self.cell_depth += 1
            if tag == "span" and "word" in classes:
                self.span_depth = self.cell_depth
                self.span_style = _style(attributes.get("style", ""))
        if tag in {"img", "image"}:
            self.images.append(ImageRef(
                self.page,
                attributes.get("src", ""),
                _bbox_from_style(attributes.get("style", "")),
                alt=attributes.get("alt") or "PDF image",
            ))

    def handle_endtag(self, tag: str) -> None:
        if self.cell is None:
            return
        if tag == "span":
            self.span_depth = 0
            self.span_style = {}
        if tag == "div":
            self.cell_depth -= 1
            if self.cell_depth == 0:
                self.cell_parts.sort(key=lambda item: (item[0], item[1]))
                self.cell.text = "\n".join(
                    text.strip() for _, _, text in self.cell_parts if text.strip()
                )
                self.pages[self.page].append(self.cell)
                self.cell = None
                self.cell_parts = []
        elif self.cell_depth:
            self.cell_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.cell is None or not data.strip():
            return
        top = float(self.span_style.get("top", 0.0)) if self.span_depth else 0.0
        left = float(self.span_style.get("left", 0.0)) if self.span_depth else 0.0
        self.cell_parts.append((top, left, data))


def load_cell_html(path: Path) -> tuple[dict[int, list[Cell]], list[ImageRef]]:
    parser = CellHtmlParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.pages, parser.images


def _cluster_rows(cells: list[Cell], tolerance: float) -> list[list[Cell]]:
    groups: list[list[Cell]] = []
    for cell in sorted(cells, key=lambda item: (item.bbox[1], item.bbox[0])):
        row = next(
            (group for group in groups
             if abs(group[0].bbox[1] - cell.bbox[1]) <= tolerance),
            None,
        )
        if row is None:
            groups.append([cell])
        else:
            row.append(cell)
    for row in groups:
        row.sort(key=lambda item: item.bbox[0])
    return sorted(groups, key=lambda group: min(item.bbox[1] for item in group))


def parse_header_specs(value: str) -> list[list[str]]:
    if not value or value.strip().lower() in {"*", "all", "通配", "全部"}:
        return [["*"]]
    specifications = []
    for alternative in value.split("|"):
        labels = [
            item for item in re.split(r"[,，、]+", alternative) if _norm(item)
        ]
        if len(labels) == 1 and re.search(r"\s+", alternative.strip()):
            labels = [
                item for item in re.split(r"\s+", alternative.strip())
                if _norm(item)
            ]
        specifications.append([_norm(item) for item in labels])
    return specifications or [["*"]]


def _matches_header(values: list[str], specifications: list[list[str]]) -> bool:
    actual = [value for value in (_norm(item) for item in values) if value]
    for wanted in specifications:
        if wanted == ["*"]:
            return True
        if len(actual) < len(wanted):
            continue
        for start in range(len(actual) - len(wanted) + 1):
            if actual[start:start + len(wanted)] == wanted:
                return True
    return False


def _header_candidates(cells: list[Cell], specifications: list[list[str]],
                       tolerance: float) -> list[list[Cell]]:
    result = []
    for row in _cluster_rows(cells, tolerance):
        nonempty = [cell for cell in row if _norm(cell.text)]
        if _matches_header([cell.text for cell in nonempty], specifications):
            result.append(nonempty)
    return result


def _candidate_table(header: list[Cell], all_cells: list[Cell],
                     tolerance: float) -> Table:
    x0 = min(cell.bbox[0] for cell in header)
    x1 = max(cell.bbox[2] for cell in header)
    header_bottom = max(cell.bbox[3] for cell in header)
    eligible = [
        cell for cell in all_cells
        if cell not in header
        and cell.bbox[1] >= header_bottom - tolerance
        and _overlap_x(cell.bbox, [x0, 0, x1, cell.bbox[3]]) > tolerance
    ]
    selected_rows = []
    frontier = header_bottom
    for row in _cluster_rows(eligible, tolerance):
        top = min(cell.bbox[1] for cell in row)
        if selected_rows and top > frontier + tolerance * 2:
            break
        usable = [
            cell for cell in row
            if _overlap_x(cell.bbox, [x0, 0, x1, cell.bbox[3]]) > tolerance
        ]
        if usable:
            selected_rows.append(usable)
            frontier = max(frontier, max(cell.bbox[3] for cell in usable))
    bottom = max(
        [cell.bbox[3] for cell in header]
        + [cell.bbox[3] for row in selected_rows for cell in row]
    )
    return Table(
        header[0].page,
        header,
        selected_rows,
        [x0, min(cell.bbox[1] for cell in header), x1, bottom],
    )


def _step_like(text: str) -> bool:
    return bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", text or ""))


def _automatic_grids(cells: list[Cell], tolerance: float) -> list[Table]:
    rows = _cluster_rows(cells, tolerance)
    candidates = []
    for index, row in enumerate(rows):
        header = [cell for cell in row if _norm(cell.text)]
        if len(header) < 2 or _step_like(header[0].text):
            continue
        x0 = min(cell.bbox[0] for cell in header)
        x1 = max(cell.bbox[2] for cell in header)
        data_rows = []
        frontier = max(cell.bbox[3] for cell in header)
        for next_row in rows[index + 1:]:
            top = min(cell.bbox[1] for cell in next_row)
            if data_rows and top > frontier + tolerance * 2:
                break
            usable = [
                cell for cell in next_row
                if _overlap_x(cell.bbox, [x0, 0, x1, cell.bbox[3]]) > tolerance
            ]
            if usable:
                data_rows.append(usable)
                frontier = max(frontier, max(cell.bbox[3] for cell in usable))
        if not data_rows:
            continue
        bbox = [x0, min(cell.bbox[1] for cell in header), x1, frontier]
        if any(_area(cell.bbox) > _area(bbox) * 0.72 for cell in header):
            continue
        title = ""
        if (index and len(rows[index - 1]) == 1
                and abs(rows[index - 1][0].bbox[3] - header[0].bbox[1])
                <= tolerance * 3):
            title = rows[index - 1][0].text.strip()
        candidates.append(Table(header[0].page, header, data_rows, bbox, title=title))

    unique = []
    for table in sorted(candidates, key=lambda item: (item.bbox[1], -_area(item.bbox))):
        if any(
            _contains(other.bbox, table.bbox, tolerance)
            and abs(other.bbox[1] - table.bbox[1]) <= tolerance
            for other in unique
        ):
            continue
        unique.append(table)
    return unique


def _deduplicate(tables: list[Table], tolerance: float) -> list[Table]:
    result = []
    for table in sorted(tables, key=lambda item: (item.bbox[1], -_area(item.bbox))):
        if any(
            all(abs(a - b) <= tolerance for a, b in zip(existing.bbox, table.bbox))
            for existing in result
        ):
            continue
        result.append(table)
    return result


def _attach_nested(tables: list[Table], cells: list[Cell], tolerance: float) -> None:
    for child in sorted(tables, key=lambda item: _area(item.bbox)):
        parents = [
            parent for parent in tables
            if parent is not child and _contains(parent.bbox, child.bbox, tolerance)
        ]
        if not parents:
            continue
        parent = min(parents, key=lambda item: _area(item.bbox))
        owners = [cell for cell in cells if _contains(cell.bbox, child.bbox, tolerance)]
        if owners:
            child.parent_cell = min(owners, key=lambda cell: _area(cell.bbox))
        else:
            # A continued outer table can contain only an inner grid on the
            # new page, without a large physical parent cell around it.  The
            # last header cell is only a parent marker here; the cross-page
            # merge later selects the real destination column by x overlap.
            child.parent_cell = parent.header_cells[-1]
        if child not in parent.nested:
            parent.nested.append(child)


def _remove_child_rows(table: Table, candidates: list[Table], tolerance: float) -> None:
    children = [
        child for child in candidates
        if child is not table and _contains(table.bbox, child.bbox, tolerance)
    ]
    child_cells = {
        id(cell)
        for child in children
        for row in [child.header_cells] + child.rows
        for cell in row
    }
    table.rows = [
        [cell for cell in row if id(cell) not in child_cells]
        for row in table.rows
    ]
    table.rows = [row for row in table.rows if row]


def _assign_images(cells: list[Cell], images: list[ImageRef],
                   tolerance: float) -> None:
    for image in images:
        if not image.bbox:
            continue
        center_x = (image.bbox[0] + image.bbox[2]) / 2
        center_y = (image.bbox[1] + image.bbox[3]) / 2
        owners = [
            cell for cell in cells
            if cell.bbox[0] - tolerance <= center_x <= cell.bbox[2] + tolerance
            and cell.bbox[1] - tolerance <= center_y <= cell.bbox[3] + tolerance
        ]
        if owners:
            min(owners, key=lambda cell: _area(cell.bbox)).images.append(image)


def _inherit_sequence_columns(pages: list[list[Table]]) -> None:
    state: dict[tuple[str, int], str] = {}
    for tables in pages:
        for table in tables:
            key = "|".join(_norm(cell.text) for cell in table.header_cells)
            sequence_columns = [
                index for index, cell in enumerate(table.header_cells)
                if _norm(cell.text) in {"序号", "编号", "工步"}
            ]
            for row in table.rows:
                for column in sequence_columns:
                    if column >= len(row):
                        continue
                    cell = row[column]
                    if cell.text.strip():
                        state[(key, column)] = cell.text.strip()
                    elif (key, column) in state:
                        cell.text = state[(key, column)]


def _column_for_cell(table: Table, cell: Cell) -> int:
    """Map a cell to the column bands defined by the repeated header row."""
    overlaps = [_overlap_x(cell.bbox, header.bbox) for header in table.header_cells]
    return max(range(len(overlaps)), key=lambda index: overlaps[index]) if overlaps else 0


def _sequence_columns(table: Table) -> list[int]:
    return [
        index for index, cell in enumerate(table.header_cells)
        if _norm(cell.text) in {"序号", "编号", "工步"}
    ]


def _content_column(table: Table) -> int:
    for index, cell in reversed(list(enumerate(table.header_cells))):
        if _norm(cell.text) in {"工序内容", "项目内容", "内容"}:
            return index
    return max(0, len(table.header_cells) - 1)


def _has_sequence_value(table: Table) -> bool:
    sequence = set(_sequence_columns(table))
    return any(
        cell.text.strip() and _column_for_cell(table, cell) in sequence
        for row in table.rows for cell in row
    )


def _merge_cell_text(target: Cell, continuation: Cell) -> None:
    text = continuation.text.strip()
    if text:
        target.text = (target.text.rstrip() + "\n" + text).strip()
    continuations = getattr(target, "continuations", [])
    continuations.append(continuation)
    setattr(target, "continuations", continuations)


def _merge_cross_page_tables(roots: list[Table]) -> list[Table]:
    """Merge repeated-header pages whose sequence column is intentionally blank.

    Header cell x ranges act as persistent column bands.  A same-header table
    on the immediately following page is a continuation only when it has no
    value in a sequence column.  Text cells are appended to the matching last
    row columns, and inner tables are tagged for the previous content column.
    """
    merged: list[Table] = []
    latest: dict[tuple[str, ...], tuple[Table, int]] = {}
    for table in sorted(roots, key=lambda item: (item.page, item.bbox[1], item.bbox[0])):
        signature = tuple(_norm(cell.text) for cell in table.header_cells)
        previous_state = latest.get(signature)
        can_continue = (
            bool(_sequence_columns(table))
            and not _has_sequence_value(table)
            and previous_state is not None
            and table.page == previous_state[1] + 1
        )
        if not can_continue:
            merged.append(table)
            latest[signature] = (table, table.page)
            continue

        previous = previous_state[0]
        target_row = previous.rows[-1] if previous.rows else []
        for row in table.rows:
            for source_cell in row:
                column = _column_for_cell(table, source_cell)
                targets = [
                    cell for cell in target_row
                    if _column_for_cell(previous, cell) == column
                ]
                if targets:
                    _merge_cell_text(targets[-1], source_cell)
                else:
                    target_row.append(source_cell)

        destination_column = _content_column(previous)
        for child in table.nested:
            setattr(child, "cross_page_column", destination_column)
            previous.nested.append(child)
        latest[signature] = (previous, table.page)
    return merged


def _cell_json(table: Table, cell: Cell, row_number: int) -> dict[str, Any]:
    overlaps = [_overlap_x(cell.bbox, header.bbox) for header in table.header_cells]
    column = _column_for_cell(table, cell)
    colspan = max(1, sum(value > 1.0 for value in overlaps[column:]))
    result: dict[str, Any] = {
        "row": row_number,
        "column": column,
        "colspan": colspan,
        "text": cell.text,
        "bbox": [round(value, 2) for value in cell.bbox],
        "_source_page": cell.page,
    }
    if cell.images:
        result["images"] = [
            {
                "page": image.page,
                "src": image.src,
                "bbox": image.bbox,
                "alt": image.alt,
            }
            for image in cell.images
        ]
    continuations = getattr(cell, "continuations", [])
    if continuations:
        result["_continuations"] = [
            {
                "page": item.page,
                "bbox": [round(value, 2) for value in item.bbox],
                "text": item.text,
                "images": [
                    {
                        "page": image.page,
                        "src": image.src,
                        "bbox": image.bbox,
                        "alt": image.alt,
                    }
                    for image in item.images
                ],
            }
            for item in continuations
        ]
    return result


def _table_json(table: Table) -> dict[str, Any]:
    result = {
        "page": table.page,
        "bbox": [round(value, 2) for value in table.bbox],
        "title": table.title,
        "headers": table.headers,
        "rows": [
            [_cell_json(table, cell, row_number) for cell in row]
            for row_number, row in enumerate(table.rows, 1)
        ],
        "nested_tables": [_table_json(child) for child in table.nested],
    }
    if hasattr(table, "cross_page_column"):
        result["_cross_page_column"] = int(table.cross_page_column)
    return result


def _extract_tables(pages: dict[int, list[Cell]], images: list[ImageRef],
                    header_spec: str, tolerance: float) -> dict[str, Any]:
    specifications = parse_header_specs(header_spec)
    wildcard = specifications == [["*"]]
    page_tables = []
    for page_number in sorted(pages):
        cells = pages[page_number]
        _assign_images(
            cells,
            [image for image in images if image.page == page_number],
            tolerance,
        )
        explicit = [
            _candidate_table(header, cells, tolerance)
            for header in _header_candidates(cells, specifications, tolerance)
        ]
        automatic = _automatic_grids(cells, tolerance)
        if not wildcard:
            automatic = [
                candidate for candidate in automatic
                if any(_contains(parent.bbox, candidate.bbox, tolerance)
                       for parent in explicit)
            ]
            automatic = [
                candidate for candidate in automatic
                if not any(
                    parent.bbox[1] < candidate.bbox[1]
                    and min(_width(parent.bbox), _width(candidate.bbox))
                    / max(_width(parent.bbox), _width(candidate.bbox)) >= 0.90
                    for parent in explicit
                )
            ]
            automatic = [
                candidate for candidate in automatic
                if not any(
                    other is not candidate
                    and other.bbox[1] < candidate.bbox[1]
                    <= other.bbox[3] + tolerance
                    and min(_width(other.bbox), _width(candidate.bbox))
                    / max(_width(other.bbox), _width(candidate.bbox)) >= 0.90
                    for other in automatic
                )
            ]
        candidates = _deduplicate(explicit + automatic, tolerance)
        _attach_nested(candidates, cells, tolerance)
        for table in candidates:
            _remove_child_rows(table, candidates, tolerance)
        page_tables.append(candidates)

    _inherit_sequence_columns(page_tables)
    roots = [
        table for tables in page_tables for table in tables
        if table.parent_cell is None
    ]
    roots = _merge_cross_page_tables(roots)
    return {
        "format": "ao-structured-v2",
        "selection": {
            "headers": header_spec,
            "matched_tables": len(roots),
            "geometry_only": True,
        },
        "images": [
            {
                "page": image.page,
                "src": image.src,
                "bbox": image.bbox,
                "width": image.width,
                "height": image.height,
                "alt": image.alt,
            }
            for image in images
        ],
        "tables": [_table_json(table) for table in roots],
    }


def _postprocess_nested_columns(table: dict, tolerance: float) -> None:
    titles = {
        child.get("title", "") for child in table.get("nested_tables", [])
        if child.get("title")
    }
    if titles:
        table["rows"] = [
            row for row in table.get("rows", [])
            if not (len(row) == 1 and row[0].get("text", "").strip() in titles)
        ]
    for child in table.get("nested_tables", []):
        if child.get("rows"):
            first_row = child["rows"][0]
            first_y0 = min(
                (cell.get("bbox", [0, 0, 0, 0])[1] for cell in first_row),
                default=child["bbox"][1],
            )
            first_y1 = max(
                (cell.get("bbox", [0, 0, 0, 0])[3] for cell in first_row),
                default=child["bbox"][1],
            )
            first_height = max(1.0, first_y1 - first_y0)
            child_bbox = child.get("bbox", [0, 0, 0, 0])
            moved = None
            for row in table.get("rows", []):
                for cell in list(row):
                    bbox = cell.get("bbox", [0, 0, 0, 0])
                    height = bbox[3] - bbox[1]
                    if (
                        bbox[2] <= child_bbox[0] + tolerance
                        and bbox[1] >= first_y0 - tolerance
                        and bbox[3] <= first_y1 + tolerance
                        and height <= first_height * 3
                        and cell.get("text", "").strip()
                        and len(cell.get("text", "")) < 40
                    ):
                        moved = cell
                        row.remove(cell)
                        break
                if moved:
                    break
            if moved:
                child["headers"] = [""] + list(child.get("headers", []))
                for row in child["rows"]:
                    for cell in row:
                        cell["column"] = int(cell.get("column", 0)) + 1
                    row.insert(0, {
                        "row": row[0].get("row", 1),
                        "column": 0,
                        "colspan": 1,
                        "text": moved["text"],
                        "bbox": moved["bbox"],
                    })
        _postprocess_nested_columns(child, tolerance)


def _move_nested_tables_into_cells(table: dict, tolerance: float) -> None:
    table["rows"] = [row for row in table.get("rows", []) if row]
    remaining = []
    for child in list(table.get("nested_tables", [])):
        forced_column = child.pop("_cross_page_column", None)
        if forced_column is not None:
            owners = [
                cell for row in table.get("rows", []) for cell in row
                if int(cell.get("column", -1)) == int(forced_column)
            ]
        else:
            child_bbox = child.get("bbox", [0, 0, 0, 0])
            owners = [
                cell for row in table.get("rows", []) for cell in row
                if _contains(cell.get("bbox", [0, 0, 0, 0]), child_bbox, tolerance)
            ]
        if owners:
            owner = (owners[-1] if forced_column is not None
                     else min(owners, key=lambda cell: _area(cell["bbox"])))
            owner.setdefault("nested_tables", []).append(child)
        else:
            remaining.append(child)
    table["nested_tables"] = remaining
    for row in table.get("rows", []):
        for cell in row:
            for child in cell.get("nested_tables", []):
                _move_nested_tables_into_cells(child, tolerance)
    for child in remaining:
        _move_nested_tables_into_cells(child, tolerance)


def _prune_empty(table: dict) -> bool:
    table["nested_tables"] = [
        child for child in table.get("nested_tables", []) if _prune_empty(child)
    ]
    meaningful_row = False
    for row in table.get("rows", []):
        for cell in row:
            cell["nested_tables"] = [
                child for child in cell.get("nested_tables", []) if _prune_empty(child)
            ]
            if (
                str(cell.get("text", "")).strip()
                or cell.get("images")
                or cell.get("nested_tables")
            ):
                meaningful_row = True
    return meaningful_row or bool(table.get("nested_tables"))


@dataclass
class Fragment:
    top: float
    left: float
    text: str
    line_height: float


@dataclass
class CellLayout:
    page: int
    bbox: list[float]
    fragments: list[Fragment] = field(default_factory=list)


class CellLayoutParser(HTMLParser):
    """Read absolute cell boxes and relative word positions from cell HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.page = 0
        self.cells: list[CellLayout] = []
        self.cell: CellLayout | None = None
        self.cell_depth = 0
        self.span_depth = 0
        self.span_style: dict[str, float | str] = {}
        self.span_text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attributes = {key: value or "" for key, value in attrs}
        classes = set(attributes.get("class", "").split())
        if tag == "section" and "page" in classes:
            self.page = int(attributes.get("data-page") or self.page + 1)
            return
        if self.page <= 0:
            return
        if tag == "div" and "cell" in classes and "page-text" not in classes:
            bbox = _bbox_from_style(attributes.get("style", ""))
            if bbox:
                self.cell = CellLayout(self.page, bbox)
                self.cell_depth = 1
            return
        if self.cell is None:
            return
        self.cell_depth += 1
        if tag == "span" and "word" in classes:
            self.span_depth = self.cell_depth
            self.span_style = _style(attributes.get("style", ""))
            self.span_text = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None and self.span_depth and data:
            self.span_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.cell is None:
            return
        if tag == "span" and self.span_depth:
            text = "".join(self.span_text).strip()
            if text:
                self.cell.fragments.append(Fragment(
                    self.cell.bbox[1] + float(self.span_style.get("top", 0.0)),
                    self.cell.bbox[0] + float(self.span_style.get("left", 0.0)),
                    text,
                    float(self.span_style.get("line-height", 12.0)),
                ))
            self.span_depth = 0
            self.span_style = {}
            self.span_text = []
        self.cell_depth -= 1
        if tag == "div" and self.cell_depth == 0:
            self.cells.append(self.cell)
            self.cell = None


def _same_bbox(first, second, tolerance: float = 1.0) -> bool:
    return len(first) == len(second) and all(
        abs(float(a) - float(b)) <= tolerance for a, b in zip(first, second)
    )


def _join_words(words: list[Fragment]) -> str:
    result = ""
    for word in sorted(words, key=lambda item: item.left):
        if (result and result[-1].isascii() and result[-1].isalnum()
                and word.text[0].isascii() and word.text[0].isalnum()):
            result += " "
        result += word.text
    return result


def _text_lines(fragments: list[Fragment]) -> list[dict]:
    lines: list[list[Fragment]] = []
    for fragment in sorted(fragments, key=lambda item: (item.top, item.left)):
        if not lines:
            lines.append([fragment])
            continue
        reference = sum(item.top for item in lines[-1]) / len(lines[-1])
        tolerance = max(2.5, min(6.0, fragment.line_height * 0.35))
        if abs(fragment.top - reference) <= tolerance:
            lines[-1].append(fragment)
        else:
            lines.append([fragment])
    result = []
    for line in lines:
        text = _join_words(line)
        if text:
            result.append({
                "type": "text",
                "top": min(item.top for item in line),
                "left": min(item.left for item in line),
                "text": text,
            })
    return result


def _ordered_content(layout: CellLayout | None, cell: dict) -> list[dict]:
    events = _text_lines(layout.fragments) if layout else []
    if not events and str(cell.get("text", "")).strip():
        events.append({
            "type": "text",
            "top": float(cell["bbox"][1]),
            "left": float(cell["bbox"][0]),
            "text": cell["text"],
        })
    for image in cell.get("images", []):
        bbox = image.get("bbox") or cell.get("bbox", [0, 0, 0, 0])
        events.append({
            "type": "image",
            "top": float(bbox[1]),
            "left": float(bbox[0]),
            "src": image.get("src", ""),
            "page": image.get("page"),
            "bbox": bbox,
            "alt": image.get("alt", "PDF image"),
        })
    events.sort(key=lambda item: (item["top"], item["left"], item["type"] != "text"))

    content = []
    for event in events:
        if event["type"] == "text" and content and content[-1]["type"] == "text":
            content[-1]["text"] += "\n" + event["text"]
        else:
            content.append({
                key: value for key, value in event.items()
                if key not in {"top", "left"}
            })
    return content


def _enrich_reading_order(payload: dict, cell_html: Path,
                          tolerance: float = 1.0) -> None:
    parser = CellLayoutParser()
    parser.feed(cell_html.read_text(encoding="utf-8"))

    def enrich_table(table: dict) -> None:
        page = int(table.get("page", 0))
        for row in table.get("rows", []):
            for cell in row:
                source_page = int(cell.pop("_source_page", page))
                layout = next(
                    (item for item in parser.cells
                     if item.page == source_page
                     and _same_bbox(item.bbox, cell.get("bbox", []), tolerance)),
                    None,
                )
                cell["content"] = _ordered_content(layout, cell)
                for continuation in cell.pop("_continuations", []):
                    continuation_page = int(continuation.get("page", source_page))
                    continuation_layout = next(
                        (item for item in parser.cells
                         if item.page == continuation_page
                         and _same_bbox(
                             item.bbox, continuation.get("bbox", []), tolerance
                         )),
                        None,
                    )
                    cell["content"].extend(_ordered_content(
                        continuation_layout,
                        continuation,
                    ))
                for child in cell.get("nested_tables", []):
                    enrich_table(child)
        for child in table.get("nested_tables", []):
            enrich_table(child)

    for table in payload.get("tables", []):
        enrich_table(table)
    payload["format"] = "ao-structured-v3-reading-order"


def extract(cell_html: Path, headers: str, tolerance: float) -> dict[str, Any]:
    """Return the full nested representation for an intermediate cell HTML."""
    pages, images = load_cell_html(cell_html)
    payload = _extract_tables(pages, images, headers, tolerance)
    for table in payload["tables"]:
        _postprocess_nested_columns(table, tolerance)
        _move_nested_tables_into_cells(table, tolerance)
    payload["tables"] = [table for table in payload["tables"] if _prune_empty(table)]
    payload["selection"]["matched_tables"] = len(payload["tables"])
    _enrich_reading_order(payload, cell_html)
    return payload


def _image_link(image: dict) -> str:
    source = html.escape(str(image.get("src", "")), quote=True)
    return f'<a class="image-link" href="{source}">这里有一个图片</a>'


def _render_cell(cell: dict, level: int) -> str:
    content = cell.get("content")
    parts = []
    if content:
        for item in content:
            if item.get("type") == "image":
                parts.append(f'<div class="cell-image">{_image_link(item)}</div>')
            elif item.get("type") == "text":
                text = html.escape(str(item.get("text", ""))).replace("\n", "<br>")
                parts.append(f'<div class="cell-text">{text}</div>')
    else:
        parts.append(html.escape(str(cell.get("text", ""))).replace("\n", "<br>"))
        parts.extend(
            f'<div class="cell-image">{_image_link(image)}</div>'
            for image in cell.get("images", [])
        )
    parts.extend(_render_table(table, level + 1)
                 for table in cell.get("nested_tables", []))
    return "".join(parts)


def _render_table(table: dict, level: int = 2) -> str:
    heading_level = min(6, level)
    headers = table.get("headers", [])
    title = html.escape(" / ".join(map(str, headers)))
    output = [
        f'<section class="table-block"><h{heading_level}>{title}</h{heading_level}>',
        "<table><thead><tr>",
    ]
    output.extend(f"<th>{html.escape(str(value))}</th>" for value in headers)
    output.append("</tr></thead><tbody>")
    for row in table.get("rows", []):
        output.append("<tr>")
        output.extend(f"<td>{_render_cell(cell, level)}</td>" for cell in row)
        output.append("</tr>")
    output.append("</tbody></table>")
    output.extend(_render_table(child, level + 1)
                  for child in table.get("nested_tables", []))
    output.append("</section>")
    return "".join(output)


def render_html(payload: dict) -> str:
    image_links = "".join(
        f'<li>第 {image.get("page")} 页：{_image_link(image)}</li>'
        for image in payload.get("images", []) if image.get("src")
    )
    body = "".join(_render_table(table) for table in payload.get("tables", []))
    return (
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>AO 结构化文档</title><style>"
        "body{font:15px system-ui,'Microsoft YaHei',sans-serif;max-width:1200px;margin:2em auto;padding:0 1em}"
        ".table-block{margin:1.5em 0;padding:.5em;border-left:3px solid #789}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #888;padding:.4em;vertical-align:top}"
        "table table{margin:.6em 0;background:#f7fbff}.cell-text{white-space:normal}"
        ".cell-image{margin:.55em 0;padding:.35em;background:#fff8dc;border-left:3px solid #e5a000}"
        ".image-link{font-weight:600;color:#075cab}</style></head><body>"
        + (f"<h1>图片资源</h1><ul>{image_links}</ul>" if image_links else "")
        + body
        + "</body></html>"
    )


def _compact_content_item(item: dict) -> dict:
    if item.get("type") == "image":
        return {"type": "image", "src": item.get("src", "")}
    return {"text": item.get("text", "")}


def _compact(value: Any) -> Any:
    if isinstance(value, list):
        return [_compact(item) for item in value]
    if not isinstance(value, dict):
        return value
    is_cell = "content" in value and "row" in value and "column" in value
    result = {}
    for key, child in value.items():
        if key == "bbox":
            continue
        if is_cell and key in {"text", "images"}:
            continue
        if is_cell and key == "content":
            result[key] = [_compact_content_item(item) for item in child]
            continue
        compacted = _compact(child)
        if key == "nested_tables" and compacted == []:
            continue
        result[key] = compacted
    return result


def compact_payload(payload: dict) -> dict:
    """Remove geometry and duplicate cell fields while preserving semantics."""
    return _compact(copy.deepcopy(payload))


def write_outputs(payload: dict, output_html: Path, full_json: Path,
                  compact_json: Path) -> None:
    for path in (output_html, full_json, compact_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(render_html(payload), encoding="utf-8")
    full_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    compact_json.write_text(
        json.dumps(compact_payload(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
