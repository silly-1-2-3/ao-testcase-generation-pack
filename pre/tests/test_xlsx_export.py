from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO

from pre.xlsx_export import build_ao_workbook


class XlsxExportTests(unittest.TestCase):
    def build(self, modifications=True) -> bytes:
        original = [
            {
                "步骤层级": "步骤",
                "说明": "测量直流电压",
                "操作内容": "测量",
            },
            {
                "步骤层级": "执行步骤",
                "设备类型": "测试表",
                "设备单元号": "测试点A",
                "设备指令号": "hallucinated_voltage",
                "设备参数": '["28V"]',
            },
        ]
        final = [dict(row) for row in original]
        audits = []
        if modifications:
            final[1]["设备类型"] = "数字万用表"
            final[1]["设备指令号"] = "measure_dc_voltage"
            audits.append(
                {
                    "row_index": 1,
                    "source_decision": "review_unknown_identifier",
                    "original": original[1],
                    "final": final[1],
                    "candidate": {
                        "设备指令主键": "CMD-001",
                        "设备指令号": "measure_dc_voltage",
                        "设备类型": "数字万用表",
                        "设备指令功能说明": "测量指定测试点的直流电压",
                        "bm25_rank": 1,
                        "bge_rank": 1,
                        "bge_score": 0.72,
                        "bge_margin_to_second": 0.21,
                    },
                    "applied_at": "2026-07-30T17:30:00+08:00",
                }
            )
        return build_ao_workbook(
            variant="lora",
            model_name="qwen35-lora",
            ao_text="这是一条用于验证 Excel 导出的完整 AO 指令文本。",
            prompt_mode="compressed",
            original_rows=original,
            final_rows=final,
            modifications=audits,
            generation={"temperature": 0, "top_p": 1},
        )

    def test_package_is_complete_and_all_xml_is_well_formed(self) -> None:
        data = self.build()
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = set(archive.namelist())
            required = {
                "[Content_Types].xml",
                "_rels/.rels",
                "xl/workbook.xml",
                "xl/_rels/workbook.xml.rels",
                "xl/styles.xml",
                "xl/worksheets/sheet1.xml",
                "xl/worksheets/sheet2.xml",
                "xl/worksheets/sheet3.xml",
            }
            self.assertTrue(required <= names)
            for name in names:
                if name.endswith(".xml") or name.endswith(".rels"):
                    ET.fromstring(archive.read(name))

    def test_final_original_and_audit_are_kept_separately(self) -> None:
        data = self.build()
        with zipfile.ZipFile(BytesIO(data)) as archive:
            workbook = archive.read("xl/workbook.xml").decode("utf-8")
            final = archive.read("xl/worksheets/sheet1.xml").decode("utf-8")
            original = archive.read("xl/worksheets/sheet2.xml").decode("utf-8")
            audit = archive.read("xl/worksheets/sheet3.xml").decode("utf-8")

        for sheet_name in ("最终结果", "原始模型结果", "修改审计"):
            self.assertIn(sheet_name, workbook)
        self.assertIn("measure_dc_voltage", final)
        self.assertIn("hallucinated_voltage", original)
        self.assertIn("hallucinated_voltage", audit)
        self.assertIn("measure_dc_voltage", audit)
        self.assertNotIn("<f>", final)

    def test_unmodified_export_has_visible_empty_audit_notice(self) -> None:
        data = self.build(modifications=False)
        with zipfile.ZipFile(BytesIO(data)) as archive:
            audit = archive.read("xl/worksheets/sheet3.xml").decode("utf-8")
        self.assertIn("本次导出没有人工应用设备检索候选", audit)


if __name__ == "__main__":
    unittest.main()
