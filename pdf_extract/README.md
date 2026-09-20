# 矢量 PDF → 版式 HTML → 结构化文档

这是一个纯本地、无模型调用的提取流程。它读取矢量 PDF 的文字位置、直线、矩形、底色和图片，恢复表格、嵌套内容及图片链接；不进行 OCR，也不理解正文领域语义。推荐 Python 3.11。

## 快速使用

```bash
python -m pip install -r pdf_extract/requirements.txt
python pdf_extract/run_pipeline.py pdf_extract/example/sample.pdf --output-dir output/pdf_demo -ft AO
```

`sample.pdf` 是最初的示例输入：我们对照图片 PDF 手工重绘的矢量文件，另加了含内嵌图片的演示页，不是扫描件自动矢量化结果。输出目录包含 `document.cells.html`（中间版式 HTML，只保存文字、格子、坐标、底色、图片）、`document.structured.html`、`document.structured.long.json`、`document.structured.short.json`、`document.review.html/json`、`document.cards.json` 和 `images/`。

中间版式 HTML 尚未赋予表头和数据关系；long JSON 含 bbox、方向和来源信息；short JSON 去除 bbox、重复字段和空 `nested_tables`，用于生成模型。图片路径是相对 `images/` 的链接，复制结果时须一并复制图片。

## 文件类型与表头配置

维护者编辑 UTF-8 文件 `pdf_extract/file_templates.txt`，格式为“文件名正则 `::=>::` 表头”：

```text
(?:KG.*|工艺指示单) ::=>:: 任务事项,更改原因,编制依据|序号,名称,型号,更改内容,通电内容
(?:AO.*|AO工序表) ::=>:: 工种,序号,工序内容|序号,项目内容
```

左侧对 `-ft` 做不区分大小写的完整匹配；零匹配或多个匹配报错。右侧逗号分隔同组表头，`|` 分隔多个表头组；匹配时忽略空白并统一中英文冒号。使用者通常只需填写 `-ft AO`、`-ft KG0030-006` 等类型；`--headers` 显式指定时优先于 `-ft`，`--headers '*'` 自动发现候选表。`@document` 表示全文兜底模式。

纵向表默认按“首行表头”读取，横向表按“首列表头”读取。固定表头可精确识别两种方向，底色和几何仅作辅助。**如果没有表头配置，也没有可靠底色或版式线索，只凭一个对齐网格无法知道哪边是表头；此时默认纵向读取，不能保证语义方向。** 当前固定表头逻辑主要覆盖传统表单，不承诺任意复杂 rowspan/colspan；子表必须落在父表内容单元格内且通常小于父表。

未匹配到任何表时，整份文件会作为一个 `文档内容` 父块，各页文字和图片按顺序保存，不会丢弃正文；图片型 PDF 只能保留页面图片，不能凭空识别图片中的字。

## 网页核验与拆卡

与现有 `pre/server.py` 集成的网页说明见 [`pre/PDF_WORKFLOW.md`](../pre/PDF_WORKFLOW.md)。原有启动参数、Base/LoRA、检索、文本输入和 Excel 导出不变。网页可上传多个 PDF、切换文档、用原 PDF 页面预览对照、编辑某个 short JSON 块并核验。默认拆卡表头为 `工种,序号,工序内容|序号,项目内容`，每个序号生成一张卡；嵌套匹配优先取最内层并携带外层说明，非匹配表或全文兜底块各生成一张卡。核验后的卡片可逐个或批量送入原有生成接口，结果可导出 ZIP。

## 限制与验证

`python pdf_extract/run_pipeline.py --help` 可查看页码、线长/面积比例、颜色容差等参数。已回归 CP、材料状态更改单、KG、XHTZ 四类表单、44 页正文全文兜底、变量模板和五个 AO 示例；这不等于任意 PDF 都能 100% 正确转换，模糊线条、复杂合并格和阅读顺序歧义仍需人工核验。真实业务文件和联调上传数据不随源码发布。
