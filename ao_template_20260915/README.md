# AO 矢量范本、变量模板与五组模拟数据

直接打开 `index.html` 查看所有文件入口。原有 PDF 和 `ao-testcase-generation` 仓库均未覆盖。

## 文件

新增 `cover_template/cover.pdf` 与 `cover.short.json`，是一页独立的封面变量模板，可放在工序模板之前。保留编号、版次、预估天数、机种代号、密级、有效性、名称/说明，以及换版说明、图样/技术文件/设备表、工作说明和划改记录。原版“机种代号”是同一个字段的两行标签，不额外发明独立“代号”字段。

封面 JSON 用 `fields` 存放元信息，`tables` 沿用工序简版 JSON 的表格/单元格结构，`change_record` 存放划改字段；变量全部与 PDF 一一核对。它是人工定义的模板字段映射，不声称现有工序表头提取器已经能自动识别这些无框的封面字段。`document.cells.html` 是实际 PDF 回读版式；封面的 `document.structured.html` 是字段映射预览。重新制作封面运行 `python scripts/ao_template/build_cover.py`，再运行 `make_index.py` 更新总览。原工序模板和五组示例不变。

| 目录 / 文件 | 内容 | 页数 | ao.jsonl 来源 ID |
| --- | --- | ---: | --- |
| `reference_refined.pdf` | 对原扫描 PDF 重新逐页观察后的版式修订；保留第 8 页图片演示 | 8 | 原 AO 范本 |
| `template/template.pdf` + `template.short.json` | 去除示例值的变量模板 | 3 | 不使用业务样例值 |
| `01_flight_control/sample.pdf` + `sample.short.json` | 后缘襟翼卸载测试；含字母子项记录表 | 6 | 2424 |
| `02_autoflight/sample.pdf` + `sample.short.json` | 维护监控、马赫配平；含多层操作列表 | 6 | 1151、1161 |
| `03_communication/sample.pdf` + `sample.short.json` | ACARS、乘务员控制面板 | 6 | 1371、1423 |
| `04_airdata_compass/sample.pdf` + `sample.short.json` | 大气数据、罗盘校准；含加压约束表 | 7 | 4153、4176 |
| `05_pneumatic_hydraulic/sample.pdf` + `sample.short.json` | 气源渗漏、PTU 测试；含长表跨页 | 7 | 4492、3196 |

每个目录还包含：

- `document.cells.html`：PDF 回读的几何版式中间结果。
- `document.structured.html`：回读后的结构化结果，便于看表格包含关系。
- `document.structured.long.json`：含 bbox 的回读结果。
- `document.structured.short.json`：去除 bbox 的回读结果。
- `author.json`：排版程序使用的内容树，**不是** `.short.json`；保留文字、表格和图片的排列顺序。
- `sources.json`：选中的完整原始条目，仅五组样例有此文件。
- `images/`：实际提取的图片；模板含一张图片，五组业务样例没有额外伪造的业务图片。

五组 `sample.short.json` 均由实际 PDF 回读产生，不是直接拿排版数据冒充提取结果。模板的 `template.short.json` 在回读结果基础上，只把图片 `src` 改成 `$image_path$`；其他回读文件保留真实相对路径，可直接查看图片。

## 这次修改了什么

参考版正文改为仿宋、标题用宋体，依据原扫描版恢复正文大小、左上对齐、段落空行和分级编号。第四页重排 40.1 / 40.2 / 40.3；第五页按实际列边界排记录表；保留第六页原本较小的表内字号。原版扫描造成的倾斜、污点、裁切不刻意复制。本版仍是人工视觉重建，不声称字符级完全一致。

模板只保留结构不同的示范，不重复原版 30/40 等仅值不同的外层表：

1. 正文、图片前文、图片、图片后文，以及设备表。
2. 外层工序内容 → 子步骤表 → 某个子步骤内的记录表。
3. 图样、技术文件、辅助件等不同字段的附表。为保持模板简洁，这些附表集中放在同一工序内容中，不照搬原范本封面与末页。

## 变量如何对应

`template/variables.json` 列出了全部变量。`$work_type$` 对应“工种”单元格，`$index$` 对应“序号”单元格；`$project_name$` / `$project_code$` / `$raw_content$` 位于“工序内容”。后缀 `2`、`3` 区分不同示范行，不能全部绑定成同一个值。

`$sub_id$`、`$sub_content$` 是子步骤；`$item$`、`$target$`、`$actual$` 是内层记录表一行的值。同一变量出现多次时应绑定同一内容。表头是固定结构，不换成变量。需要更多行时复制内容树中的对应行，然后重新排版；不能只替换 PDF 字节，也不能假定较长的正文仍占原来的页数。

图片在 PDF 中仍是原演示图片。在变量 JSON 中，`content` 顺序为文字 → `{ "type": "image", "src": "$image_path$" }` → 文字。填充 JSON 应使用正常 JSON 序列化，不做未转义的字符串拼接。变量模板不是 AcroForm 交互式表单。

`.short.json` 沿用仓库当前格式：`tables → rows → 单元格 → content / nested_tables`。`row`、`column`、`colspan` 表示结构；`page` 是定位信息；不含 bbox，空 `nested_tables` 不输出。跨页外层空工种/序号继承上页；内层表可能保留为同一父单元格下的多个分页片段，按页和纵坐标排列，并非多份独立业务任务。

## 五组数据怎么选的

审阅了超过五个候选，最终使用九条源记录。单条 2424 已能展示 6 页的多层结构，因此单独使用；其余四组各收录两个独立任务，保持各自项目名称、编码和原文顺序，不建立来源没有的业务依赖。筛选与排除原因见 `manifest.json`。1171、1954 等存在明显截断的条目未选用。

“模拟工种”和外层 10/20/30… 是排版演示标签，不是推断出的真实工种或原始工步号。“原文 n.” 保留原条目编号；内层 n.m 由原文 n. 下的 m) 转写，所以它不要求等于外层示范编号的小数扩展。a)/b) 列表改为“项次 / 操作与检查要求”表；4153 的分号分隔约束改为“项次 / 加压约束”表。没有补造缺失的压力测量表、测量值或工具。

这是已有 AO 数据的排版模拟，不代表经批准的实际维修作业文件。源记录引用的外部 TASK、图和未提供的表没有补全；若用于训练，应保留这一数据边界。

## 复现

在 `E:\codes\xifei2026\task1` 下执行：

```powershell
.\.build_win_venv\Scripts\python.exe -X utf8 scripts\ao_template\build_documents.py
.\.build_win_venv\Scripts\python.exe -X utf8 scripts\ao_template\build_pairs.py
.\.build_win_venv\Scripts\python.exe -X utf8 scripts\ao_template\verify_pairs.py
.\.build_win_venv\Scripts\python.exe -X utf8 scripts\ao_template\make_index.py
```

依赖清单在 `scripts/ao_template/requirements.txt`。本次使用现有 Windows Python 3.12 虚拟环境，补装了 ReportLab、pypdf 和约 20 MB 下载量的 PyMuPDF；没有使用远程服务器。仿宋/宋体使用本机 Windows 字体；换机器须在 `build_documents.py` 的 `setup()` 配置合法可用的字体文件。截图验证调用已有 `pdftoppm`。

脚本读取相邻的 `ao-testcase-generation/pdf_extract/src` 和 `data/ao_instructions/ao.jsonl`。演示图片读取 `task1/tmp/ao_visible_sample_image.png`。保留目录布局即可直接重跑；不是独立打包 App。

## 回读适配与验证边界

本次没有修改正式仓库代码。`build_pairs.py::extract_pair()` 包含两项局部适配：

- 自动表候选的首个数据行也必须与表头相接，防止前一张表的最后数据行被误认成下一张表的表头。
- `nested_tables` 按页、y、x 排序，而不是沿用面积发现顺序。

因此，仅直接运行原仓库的 `run_pipeline.py`，不一定得到这里相同的结果；后续要合入的改动集中在上述适配函数。图片回读必须启用 PyMuPDF；旧的 Pillow 备用路径会漏掉某些原始像素流。

`verification.json` 记录每个写入文本字段在回读 JSON 中的覆盖、外层序号、变量集合、图片数、字体和页面边界检查，以及来源字符顺序覆盖检查。已渲染全部页面并做视觉检查。这些检查证明当前样例的排版/回读一致性，不等同于领域内容正确性证明，也不证明提取器已对所有未知 PDF 通用。
