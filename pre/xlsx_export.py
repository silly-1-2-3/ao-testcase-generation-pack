"""Build a styled AO result workbook with Python's standard library only.

The workbook deliberately keeps the final rows, untouched model rows, and
manual device-retrieval edits in separate worksheets so every change remains
auditable.  All text is written as an inline string (never as an Excel
formula), which also prevents formula-injection through model/user text.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Iterable
from xml.sax.saxutils import escape



CANONICAL_COLUMNS = (
    "步骤层级",
    "说明",
    "注意事项",
    "操作内容",
    "操作对象",
    "操作目的",
    "是否同时发送",
    "多判据组合条件",
    "是否使用设备",
    "操作类型",
    "判据类型",
    "判据范围",
    "判据描述",
    "左值",
    "右值",
    "单位",
    "设备类型",
    "设备单元号",
    "设备指令号",
    "设备参数",
    "判据关联标志",
)

COLUMN_WIDTHS = {
    "行号": 8,
    "步骤层级": 12,
    "说明": 32,
    "注意事项": 26,
    "操作内容": 24,
    "操作对象": 22,
    "操作目的": 26,
    "是否同时发送": 13,
    "多判据组合条件": 18,
    "是否使用设备": 13,
    "操作类型": 12,
    "判据类型": 14,
    "判据范围": 15,
    "判据描述": 28,
    "左值": 12,
    "右值": 12,
    "单位": 10,
    "设备类型": 20,
    "设备单元号": 20,
    "设备指令号": 28,
    "设备参数": 24,
    "判据关联标志": 17,
}

AUDIT_COLUMNS = (
    "修改序号",
    "输出行",
    "原设备类型",
    "新设备类型",
    "原设备指令号",
    "新设备指令号",
    "设备单元号",
    "设备参数",
    "候选主键",
    "候选功能说明",
    "BM25排名",
    "BGE排名",
    "BGE分数",
    "BGE间隔",
    "原检索决策",
    "应用时间",
)

AUDIT_WIDTHS = {
    "修改序号": 10,
    "输出行": 10,
    "原设备类型": 20,
    "新设备类型": 20,
    "原设备指令号": 28,
    "新设备指令号": 28,
    "设备单元号": 20,
    "设备参数": 22,
    "候选主键": 18,
    "候选功能说明": 36,
    "BM25排名": 12,
    "BGE排名": 12,
    "BGE分数": 13,
    "BGE间隔": 13,
    "原检索决策": 32,
    "应用时间": 24,
}

_ILLEGAL_XML = re.compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff]"
)


def _xml_text(value: Any) -> str:
    text = _stringify(value)
    text = _ILLEGAL_XML.sub("\ufffd", text)
    # Excel cells have a 32,767-character limit.  Leave a visible marker.
    if len(text) > 32_000:
        text = text[:31_980] + "…（内容已截断）"
    return escape(text, {'"': "&quot;"})


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return str(value)


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _cell_xml(
    row: int,
    column: int,
    value: Any,
    style: int,
) -> str:
    reference = f"{_column_name(column)}{row}"
    if isinstance(value, bool):
        return (
            f'<c r="{reference}" s="{style}" t="b">'
            f"<v>{1 if value else 0}</v></c>"
        )
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    ):
        return f'<c r="{reference}" s="{style}"><v>{value}</v></c>'
    return (
        f'<c r="{reference}" s="{style}" t="inlineStr">'
        f'<is><t xml:space="preserve">{_xml_text(value)}</t></is></c>'
    )


def _row_xml(
    row_number: int,
    values: Iterable[Any],
    styles: Iterable[int] | int,
    *,
    height: float | None = None,
) -> str:
    values_list = list(values)
    if isinstance(styles, int):
        style_list = [styles] * len(values_list)
    else:
        style_list = list(styles)
        if len(style_list) != len(values_list):
            raise ValueError("Cell style count does not match value count")
    attributes = f' r="{row_number}"'
    if height is not None:
        attributes += f' ht="{height}" customHeight="1"'
    cells = "".join(
        _cell_xml(row_number, index, value, style_list[index - 1])
        for index, value in enumerate(values_list, start=1)
    )
    return f"<row{attributes}>{cells}</row>"


def _ordered_columns(
    original_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()
    for row in (*original_rows, *final_rows):
        for key in row:
            key = str(key)
            if key.startswith("_"):
                continue
            if key not in seen:
                seen.add(key)
                discovered.append(key)
    # Always export the complete SFT schema in a stable order.  This keeps
    # workbooks comparable even when a particular AO leaves many fields blank.
    ordered = list(CANONICAL_COLUMNS)
    ordered.extend(column for column in discovered if column not in ordered)
    return ordered


def _columns_xml(
    columns: list[str],
    widths: dict[str, float],
) -> str:
    values = []
    for index, column in enumerate(columns, start=1):
        width = widths.get(column)
        if width is None:
            width = max(12, min(40, len(str(column)) * 2 + 4))
        values.append(
            f'<col min="{index}" max="{index}" width="{width}" '
            'customWidth="1"/>'
        )
    return "<cols>" + "".join(values) + "</cols>"


def _metadata_rows(
    *,
    column_count: int,
    title: str,
    variant_label: str,
    model_name: str,
    exported_at: str,
    prompt_mode: str,
    generation: dict[str, Any],
    modifications_count: int,
    ao_text: str,
    description: str,
) -> tuple[list[str], list[str]]:
    last_column = _column_name(column_count)
    rows = [
        _row_xml(1, [title] + [""] * (column_count - 1), [1] * column_count, height=30),
        _row_xml(2, [""] * column_count, 0, height=8),
    ]

    metadata_1 = [""] * column_count
    metadata_1[0] = "模型版本"
    metadata_1[1] = variant_label
    metadata_1[3] = "模型服务名"
    metadata_1[4] = model_name
    metadata_1[6] = "人工修改"
    metadata_1[7] = modifications_count
    styles_1 = [0] * column_count
    for index in (0, 3, 6):
        styles_1[index] = 2
    for index in (1, 2, 4, 5, *range(7, column_count)):
        styles_1[index] = 3
    rows.append(_row_xml(3, metadata_1, styles_1, height=24))

    metadata_2 = [""] * column_count
    metadata_2[0] = "导出时间"
    metadata_2[1] = exported_at
    metadata_2[3] = "Prompt 模式"
    metadata_2[4] = prompt_mode
    metadata_2[6] = "生成参数"
    metadata_2[7] = generation
    styles_2 = [0] * column_count
    for index in (0, 3, 6):
        styles_2[index] = 2
    for index in (1, 2, 4, 5, *range(7, column_count)):
        styles_2[index] = 3
    rows.append(_row_xml(4, metadata_2, styles_2, height=28))

    ao_values = ["AO 原文", ao_text] + [""] * (column_count - 2)
    ao_styles = [2] + [3] * (column_count - 1)
    rows.append(_row_xml(5, ao_values, ao_styles, height=54))

    description_values = ["工作表说明", description] + [""] * (
        column_count - 2
    )
    description_styles = [2] + [3] * (column_count - 1)
    rows.append(
        _row_xml(
            6,
            description_values,
            description_styles,
            height=34,
        )
    )
    rows.append(_row_xml(7, [""] * column_count, 0, height=8))

    merges = [
        f"A1:{last_column}1",
        "B3:C3",
        "E3:F3",
        f"H3:{last_column}3",
        "B4:C4",
        "E4:F4",
        f"H4:{last_column}4",
        f"B5:{last_column}5",
        f"B6:{last_column}6",
    ]
    return rows, merges


def _result_sheet_xml(
    *,
    title: str,
    ao_text: str,
    description: str,
    rows: list[dict[str, Any]],
    columns: list[str],
    variant_label: str,
    model_name: str,
    exported_at: str,
    prompt_mode: str,
    generation: dict[str, Any],
    modifications_count: int,
    modified_indexes: set[int],
    selected: bool,
) -> str:
    table_columns = ["行号", *columns]
    column_count = len(table_columns)
    metadata, merges = _metadata_rows(
        column_count=column_count,
        title=title,
        variant_label=variant_label,
        model_name=model_name,
        exported_at=exported_at,
        prompt_mode=prompt_mode,
        generation=generation,
        modifications_count=modifications_count,
        ao_text=ao_text,
        description=description,
    )
    header_row = 8
    xml_rows = metadata + [
        _row_xml(
            header_row,
            table_columns,
            4,
            height=32,
        )
    ]
    for index, row in enumerate(rows):
        if index in modified_indexes:
            style = 8
        elif str(row.get("步骤层级", "")).strip() == "执行步骤":
            style = 7
        else:
            style = 5 if index % 2 == 0 else 6
        values = [index + 1, *[row.get(column, "") for column in columns]]
        xml_rows.append(
            _row_xml(
                header_row + index + 1,
                values,
                style,
                height=42,
            )
        )

    last_column = _column_name(column_count)
    last_row = header_row + max(1, len(rows))
    selected_attr = ' tabSelected="1"' if selected else ""
    merge_xml = "".join(f'<mergeCell ref="{value}"/>' for value in merges)
    auto_filter = (
        f'<autoFilter ref="A{header_row}:{last_column}{last_row}"/>'
        if rows
        else ""
    )
    return _xml_document(
        f"""
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:{last_column}{last_row}"/>
  <sheetViews>
    <sheetView workbookViewId="0" showGridLines="0"{selected_attr}>
      <pane ySplit="{header_row}" topLeftCell="A{header_row + 1}"
            activePane="bottomLeft" state="frozen"/>
      <selection pane="bottomLeft" activeCell="A{header_row + 1}"
                 sqref="A{header_row + 1}"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  {_columns_xml(table_columns, COLUMN_WIDTHS)}
  <sheetData>{''.join(xml_rows)}</sheetData>
  {auto_filter}
  <mergeCells count="{len(merges)}">{merge_xml}</mergeCells>
  <pageMargins left="0.25" right="0.25" top="0.5" bottom="0.5"
               header="0.2" footer="0.2"/>
  <pageSetup orientation="landscape" fitToWidth="1" fitToHeight="0"/>
</worksheet>
"""
    )


def _audit_sheet_xml(
    *,
    modifications: list[dict[str, Any]],
    variant_label: str,
    model_name: str,
    exported_at: str,
) -> str:
    columns = list(AUDIT_COLUMNS)
    column_count = len(columns)
    last_column = _column_name(column_count)
    rows = [
        _row_xml(
            1,
            ["设备检索人工修改审计"] + [""] * (column_count - 1),
            1,
            height=30,
        ),
        _row_xml(2, [""] * column_count, 0, height=8),
        _row_xml(
            3,
            [
                "模型版本",
                variant_label,
                "",
                "模型服务名",
                model_name,
                "",
                "导出时间",
                exported_at,
                *([""] * (column_count - 8)),
            ],
            [2, 3, 3, 2, 3, 3, 2, *([3] * (column_count - 7))],
            height=24,
        ),
        _row_xml(
            4,
            ["审计说明", (
                "仅记录用户在网页中明确点击“应用此候选”产生的修改；"
                "模型原始结果保存在“原始模型结果”工作表。"
            )] + [""] * (column_count - 2),
            [2] + [3] * (column_count - 1),
            height=36,
        ),
        _row_xml(5, [""] * column_count, 0, height=8),
        _row_xml(6, columns, 9, height=32),
    ]
    merges = [
        f"A1:{last_column}1",
        "B3:C3",
        "E3:F3",
        f"H3:{last_column}3",
        f"B4:{last_column}4",
    ]

    for audit_index, modification in enumerate(
        sorted(
            modifications,
            key=lambda item: int(item.get("row_index", 0) or 0),
        ),
        start=1,
    ):
        original = modification.get("original")
        original = original if isinstance(original, dict) else {}
        final = modification.get("final")
        final = final if isinstance(final, dict) else {}
        candidate = modification.get("candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        values = [
            audit_index,
            int(modification.get("row_index", 0) or 0) + 1,
            original.get("设备类型", ""),
            final.get("设备类型", ""),
            original.get("设备指令号", ""),
            final.get("设备指令号", ""),
            final.get("设备单元号", original.get("设备单元号", "")),
            final.get("设备参数", original.get("设备参数", "")),
            candidate.get("设备指令主键", ""),
            candidate.get("设备指令功能说明", ""),
            candidate.get("bm25_rank"),
            candidate.get("bge_rank"),
            candidate.get("bge_score"),
            candidate.get("bge_margin_to_second"),
            modification.get("source_decision", ""),
            modification.get("applied_at", ""),
        ]
        rows.append(
            _row_xml(
                6 + audit_index,
                values,
                8 if audit_index % 2 else 6,
                height=40,
            )
        )

    if not modifications:
        rows.append(
            _row_xml(
                7,
                ["本次导出没有人工应用设备检索候选。"]
                + [""] * (column_count - 1),
                10,
                height=28,
            )
        )
        merges.append(f"A7:{last_column}7")

    last_row = 6 + max(1, len(modifications))
    merge_xml = "".join(f'<mergeCell ref="{value}"/>' for value in merges)
    auto_filter = (
        f'<autoFilter ref="A6:{last_column}{last_row}"/>'
        if modifications
        else ""
    )
    return _xml_document(
        f"""
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:{last_column}{last_row}"/>
  <sheetViews>
    <sheetView workbookViewId="0" showGridLines="0">
      <pane ySplit="6" topLeftCell="A7" activePane="bottomLeft"
            state="frozen"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  {_columns_xml(columns, AUDIT_WIDTHS)}
  <sheetData>{''.join(rows)}</sheetData>
  {auto_filter}
  <mergeCells count="{len(merges)}">{merge_xml}</mergeCells>
  <pageMargins left="0.25" right="0.25" top="0.5" bottom="0.5"
               header="0.2" footer="0.2"/>
  <pageSetup orientation="landscape" fitToWidth="1" fitToHeight="0"/>
</worksheet>
"""
    )


def _xml_document(body: str) -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' + body


def _styles_xml() -> str:
    return _xml_document(
        """
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="5">
    <font><sz val="10"/><name val="Aptos"/><family val="2"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="16"/><name val="Aptos Display"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font>
    <font><b/><color rgb="FF1E3A5F"/><sz val="10"/><name val="Aptos"/></font>
    <font><b/><color rgb="FF9A3412"/><sz val="10"/><name val="Aptos"/></font>
  </fonts>
  <fills count="10">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF14213D"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE8F0FA"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF2563EB"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFF7FAFC"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF4E5"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFE8F7ED"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF7E22CE"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFEDD5"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="3">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left/><right/><top/><bottom style="thin"><color rgb="FFDCE4EF"/></bottom><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFB8C5D6"/></left>
      <right style="thin"><color rgb="FFB8C5D6"/></right>
      <top style="thin"><color rgb="FFB8C5D6"/></top>
      <bottom style="thin"><color rgb="FFB8C5D6"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="11">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"><alignment vertical="top"/></xf>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="0" fontId="3" fillId="3" borderId="2" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="2" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="4" borderId="2" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="6" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="7" borderId="1" xfId="0" applyFill="1" applyBorder="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="2" fillId="8" borderId="2" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="4" fillId="9" borderId="2" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
  <dxfs count="0"/>
  <tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>
</styleSheet>
"""
    )


def build_ao_workbook(
    *,
    variant: str,
    model_name: str,
    ao_text: str,
    prompt_mode: str,
    original_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    modifications: list[dict[str, Any]],
    generation: dict[str, Any],
) -> bytes:
    """Return a complete XLSX workbook as bytes."""

    variant_label = (
        "Qwen3.5-9B Base + LoRA"
        if variant == "lora"
        else "Qwen3.5-9B Base"
    )
    exported_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    created_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    columns = _ordered_columns(original_rows, final_rows)
    modified_indexes = {
        int(item.get("row_index", -1))
        for item in modifications
        if isinstance(item, dict)
        and str(item.get("row_index", "")).lstrip("-").isdigit()
    }
    common_description = (
        "绿色行为用户基于设备检索候选主动修改的行；"
        "橙色行为执行步骤。"
    )
    final_sheet = _result_sheet_xml(
        title=f"AO 结构化测试用例 · 最终结果 · {variant_label}",
        ao_text=ao_text,
        description=common_description,
        rows=final_rows,
        columns=columns,
        variant_label=variant_label,
        model_name=model_name,
        exported_at=exported_at,
        prompt_mode=prompt_mode,
        generation=generation,
        modifications_count=len(modifications),
        modified_indexes=modified_indexes,
        selected=True,
    )
    original_sheet = _result_sheet_xml(
        title=f"AO 结构化测试用例 · 原始模型结果 · {variant_label}",
        ao_text=ao_text,
        description=(
            "此工作表保存模型最初生成的结构化结果，"
            "不包含人工设备候选修改。"
        ),
        rows=original_rows,
        columns=columns,
        variant_label=variant_label,
        model_name=model_name,
        exported_at=exported_at,
        prompt_mode=prompt_mode,
        generation=generation,
        modifications_count=0,
        modified_indexes=set(),
        selected=False,
    )
    audit_sheet = _audit_sheet_xml(
        modifications=modifications,
        variant_label=variant_label,
        model_name=model_name,
        exported_at=exported_at,
    )

    content_types = _xml_document(
        """
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""
    )
    root_relationships = _xml_document(
        """
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""
    )
    workbook = _xml_document(
        """
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView xWindow="120" yWindow="120" windowWidth="24000" windowHeight="15000"/></bookViews>
  <sheets>
    <sheet name="最终结果" sheetId="1" r:id="rId1"/>
    <sheet name="原始模型结果" sheetId="2" r:id="rId2"/>
    <sheet name="修改审计" sheetId="3" r:id="rId3"/>
  </sheets>
  <calcPr calcId="191029" fullCalcOnLoad="1"/>
</workbook>
"""
    )
    workbook_relationships = _xml_document(
        """
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>
"""
    )
    core_properties = _xml_document(
        f"""
<cp:coreProperties
  xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
  xmlns:dc="http://purl.org/dc/elements/1.1/"
  xmlns:dcterms="http://purl.org/dc/terms/"
  xmlns:dcmitype="http://purl.org/dc/dcmitype/"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{_xml_text(f"AO 测试用例 · {variant_label}")}</dc:title>
  <dc:creator>AO Test Case R&amp;D Platform</dc:creator>
  <cp:lastModifiedBy>AO Test Case R&amp;D Platform</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created_at}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created_at}</dcterms:modified>
</cp:coreProperties>
"""
    )
    app_properties = _xml_document(
        """
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>AO Test Case R&amp;D Platform</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>3</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="3" baseType="lpstr">
      <vt:lpstr>最终结果</vt:lpstr>
      <vt:lpstr>原始模型结果</vt:lpstr>
      <vt:lpstr>修改审计</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>1.0</AppVersion>
</Properties>
"""
    )

    output = BytesIO()
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        parts = {
            "[Content_Types].xml": content_types,
            "_rels/.rels": root_relationships,
            "docProps/core.xml": core_properties,
            "docProps/app.xml": app_properties,
            "xl/workbook.xml": workbook,
            "xl/_rels/workbook.xml.rels": workbook_relationships,
            "xl/styles.xml": _styles_xml(),
            "xl/worksheets/sheet1.xml": final_sheet,
            "xl/worksheets/sheet2.xml": original_sheet,
            "xl/worksheets/sheet3.xml": audit_sheet,
        }
        for path, content in parts.items():
            archive.writestr(path, content.encode("utf-8"))
    return output.getvalue()
